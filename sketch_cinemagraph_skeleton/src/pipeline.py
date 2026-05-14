from pathlib import Path

import cv2
import numpy as np

from src.types import UserInput
from src.scene_generation.generator import SceneGenerator
from src.fluid_mask.semantic_mask import build_semantic_mask
from src.fluid_mask.refine_mask import refine_fluid_mask
from src.fluid_mask.postprocess import combine_masks
from src.fluid_mask.foreground_protection import (
    build_foreground_protection_mask,
    dilate_and_feather_protection_mask,
)
from src.motion_field.sketch_parser import parse_motion_sketch
from src.motion_field.sparse_constraints import build_sparse_constraints
from src.motion_field.propagate import propagate_sparse_to_dense
from src.motion_field.smooth import smooth_motion_field, apply_motion_protection_to_flow
from src.motion_field.t2c_flow_predictor import T2CFlowPredictor
from src.synthesis.warp import warp_frames
from src.synthesis.loop import temporal_smooth_frames, blend_loop_boundary, enforce_loop
from src.synthesis.export_video import export_cinemagraph
from src.evaluation.metrics import (
    compute_motion_smoothness,
    compute_loop_consistency,
    compute_mask_leakage,
    compute_mask_valid,
    compute_psnr,
    compute_ms_ssim_loop,
    compute_temporal_consistency,
)


def _resize_to_match(array, target_hw, is_mask=False):
    """Resize any image/mask to match target (height, width)."""
    target_h, target_w = target_hw
    if array.shape[:2] == (target_h, target_w):
        return array
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(array, (target_w, target_h), interpolation=interp)


class CinemagraphPipeline:
    """Simplified end-to-end pipeline for the course project."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        scene_cfg = dict(cfg.get("scene", {}))
        scene_cfg["debug_dir"] = cfg.get("output", {}).get("debug_dir", "data/outputs/debug")
        self.scene_generator = SceneGenerator(scene_cfg)

    def run(self, user_input: UserInput) -> dict:
        scene_out = self._scene_generation(user_input)
        mask_out = self._fluid_mask_extraction(user_input, scene_out)
        motion_out = self._motion_field_estimation(user_input, scene_out, mask_out)
        synth_out = self._cinemagraph_synthesis(scene_out, mask_out, motion_out)
        metrics = self._evaluate_outputs(mask_out, motion_out, synth_out)

        return {
            "scene_out": scene_out,
            "mask_out": mask_out,
            "motion_out": motion_out,
            "synth_out": synth_out,
            "metrics": metrics,
        }

    def _scene_generation(self, user_input: UserInput) -> dict:
        scene_output = self.scene_generator.generate(
            user_input.structural_sketch,
            user_input.text_prompt,
        )
        return {
            "stylized_image": scene_output.stylized_image,
            "realistic_reference": scene_output.realistic_reference,
        }

    def _fluid_mask_extraction(self, user_input: UserInput, scene_out: dict) -> dict:
        """
        Build semantic / refined / final mask, foreground protection masks,
        and safe-moving mask — all aligned to the scene image size.
        """
        h, w = scene_out["stylized_image"].shape[:2]
        output_cfg = self.cfg.get("output", {})
        motion_cfg = self.cfg.get("motion", {})

        structural_sketch = _resize_to_match(
            user_input.structural_sketch, (h, w), is_mask=False
        )
        motion_sketch = _resize_to_match(
            user_input.motion_sketch, (h, w), is_mask=False
        )

        ref = scene_out.get("realistic_reference")
        reference_image = ref if ref is not None else scene_out["stylized_image"]
        debug_dir = Path(output_cfg.get("debug_dir", "data/outputs/debug"))

        semantic_mask = build_semantic_mask(
            structural_sketch=structural_sketch,
            motion_sketch=motion_sketch,
            reference_image=reference_image,
            debug_dir=str(debug_dir),
        )

        try:
            refined_mask = refine_fluid_mask(
                image=scene_out["stylized_image"],
                text_prompt=user_input.text_prompt,
                debug_dir=str(debug_dir),
            )
        except Exception as error:
            print(f"[Warning] refine_fluid_mask failed, using zeros as refined mask: {error}")
            refined_mask = np.zeros((h, w), dtype=np.uint8)

        refined_mask = _resize_to_match(refined_mask, (h, w), is_mask=True)

        final_fluid_mask = combine_masks(
            semantic_mask=semantic_mask,
            refined_mask=refined_mask,
            debug_dir=str(debug_dir),
        )

        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "debug_semantic.png"), semantic_mask)

        # ── Read protection params ────────────────────────────────────────────
        border_margin  = int(motion_cfg.get("border_protection_margin",      30))
        erosion_px     = int(motion_cfg.get("mask_edge_erosion",              8))
        shoreline_band = int(motion_cfg.get("shoreline_protection_band",     12))
        dilation_px    = int(motion_cfg.get("foreground_protection_dilation", 20))
        feather_px     = int(motion_cfg.get("motion_boundary_feather",        20))

        # ── Step A: Border protection mask ───────────────────────────────────
        border_protection = np.zeros((h, w), dtype=np.uint8)
        border_protection[:border_margin, :]       = 255
        border_protection[h - border_margin:, :]   = 255
        border_protection[:, :border_margin]        = 255
        border_protection[:, w - border_margin:]    = 255
        cv2.imwrite(str(debug_dir / "debug_border_protection_mask.png"), border_protection)

        # ── Step B: Shoreline protection (top edge of water per column) ──────
        water_2d = (final_fluid_mask > 0)
        has_water = water_2d.any(axis=0)                          # (W,)
        first_water_row = np.where(
            has_water, np.argmax(water_2d, axis=0), h
        )                                                          # (W,)
        row_idx = np.arange(h)[:, None]                           # (H, 1)
        dist_from_shore = row_idx - first_water_row[None, :]      # (H, W)
        shoreline_protection = (
            (dist_from_shore >= 0) & (dist_from_shore < shoreline_band) & water_2d
        ).astype(np.uint8) * 255

        # ── Step C: Boat hole detection ───────────────────────────────────────
        hole_mask = build_foreground_protection_mask(final_fluid_mask)
        dilated_mask, boat_composite_alpha, _ = dilate_and_feather_protection_mask(
            hole_mask, dilation_px=dilation_px, feather_px=feather_px
        )
        cv2.imwrite(str(debug_dir / "debug_foreground_protection_mask.png"), hole_mask)
        cv2.imwrite(str(debug_dir / "debug_dilated_foreground_protection_mask.png"), dilated_mask)

        # ── Step D: safe_moving_mask (all protections + erosion) ─────────────
        safe_moving_mask_raw = (
            (final_fluid_mask > 0)
            & (dilated_mask == 0)
            & (border_protection == 0)
            & (shoreline_protection == 0)
        ).astype(np.uint8) * 255

        if erosion_px > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
            )
            safe_moving_mask = cv2.erode(safe_moving_mask_raw, kern)
        else:
            safe_moving_mask = safe_moving_mask_raw

        cv2.imwrite(str(debug_dir / "debug_safe_moving_mask.png"), safe_moving_mask_raw)
        cv2.imwrite(str(debug_dir / "debug_safe_moving_mask_after_border.png"), safe_moving_mask)

        # ── Step E: Unified motion_alpha from distance transform ──────────────
        dist = cv2.distanceTransform(safe_moving_mask, cv2.DIST_L2, 5)
        motion_alpha = np.minimum(
            dist / max(float(feather_px), 1.0), 1.0
        ).astype(np.float32)
        # Zero outside the original fluid mask
        fluid_bin = (final_fluid_mask > 0).astype(np.float32)
        motion_alpha = motion_alpha * fluid_bin

        # ── Step F: Unified composite_alpha ──────────────────────────────────
        # boat_composite_alpha: 1 at hole centre → 0 at dilation boundary
        # (1 - motion_alpha):   1 at all protected zones → 0 in open water
        composite_alpha = np.maximum(boat_composite_alpha, 1.0 - motion_alpha)

        return {
            "semantic_mask": semantic_mask,
            "refined_mask": refined_mask,
            "final_fluid_mask": final_fluid_mask,
            "resized_motion_sketch": motion_sketch,
            "hole_mask": hole_mask,
            "dilated_protection_mask": dilated_mask,
            "border_protection_mask": border_protection,
            "shoreline_protection_mask": shoreline_protection,
            "safe_moving_mask": safe_moving_mask,
            "motion_alpha": motion_alpha,
            "composite_alpha": composite_alpha,
        }

    def _motion_field_estimation(
        self, user_input: UserInput, scene_out: dict, mask_out: dict
    ) -> dict:
        """
        Estimate dense motion field, then apply foreground protection to zero
        flow near static objects (boat, shore) with a smooth feather transition.
        """
        motion_sketch = mask_out.get("resized_motion_sketch", user_input.motion_sketch)
        mask = mask_out["final_fluid_mask"]
        motion_cfg = self.cfg.get("motion", {})

        strokes = parse_motion_sketch(motion_sketch)
        sparse_constraints = build_sparse_constraints(strokes=strokes, mask=mask)

        # ── Try T2C learned predictor ─────────────────────────────────────────
        t2c_ckpt = self.cfg.get("motion_field", {}).get("t2c_checkpoint")
        t2c = T2CFlowPredictor(checkpoint_path=t2c_ckpt)

        dense_motion_field = None
        if t2c.is_available():
            try:
                dense_motion_field = t2c.predict(
                    reference_image=scene_out["stylized_image"],
                    fluid_mask=mask,
                    motion_sketch=motion_sketch,
                )
                print("[Pipeline] Motion field: T2C neural predictor")
            except Exception as err:
                print(f"[Warning] T2C predictor failed ({err}), falling back to RBF")
                dense_motion_field = None

        # ── Fallback: RBF sparse-to-dense ────────────────────────────────────
        if dense_motion_field is None:
            dense_motion_field = propagate_sparse_to_dense(
                constraints=sparse_constraints,
                mask=mask,
            )
            print("[Pipeline] Motion field: RBF sparse-to-dense")

        max_magnitude = float(motion_cfg.get("max_displacement", 1.5))
        dense_motion_field = smooth_motion_field(
            flow=dense_motion_field, mask=mask, max_magnitude=max_magnitude
        )

        # ── Foreground motion protection ──────────────────────────────────────
        motion_alpha = mask_out.get("motion_alpha")
        debug_dir = Path(self.cfg.get("output", {}).get("debug_dir", "data/outputs/debug"))
        debug_dir.mkdir(parents=True, exist_ok=True)

        if motion_alpha is not None:
            dense_motion_field = apply_motion_protection_to_flow(
                dense_motion_field, motion_alpha
            )
            print("[Pipeline] Applied foreground motion protection")
            motion_alpha_vis = (motion_alpha * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / "debug_motion_alpha.png"), motion_alpha_vis)

        # Visualise flow magnitude after all protection
        flow_mag = np.linalg.norm(dense_motion_field, axis=-1)
        flow_max = float(flow_mag.max())
        flow_mag_vis = (
            (flow_mag / max(flow_max, 1e-6)) * 255
        ).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "debug_flow_after_protection.png"), flow_mag_vis)

        return {
            "strokes": strokes,
            "sparse_constraints": sparse_constraints,
            "dense_motion_field": dense_motion_field,
        }

    def _cinemagraph_synthesis(self, scene_out: dict, mask_out: dict, motion_out: dict) -> dict:
        synth_cfg = self.cfg.get("synthesis", {})
        output_cfg = self.cfg.get("output", {})

        num_frames = int(synth_cfg.get("num_frames", 60))
        fps = int(synth_cfg.get("fps", 20))

        composite_alpha   = mask_out.get("composite_alpha")
        safe_moving_mask  = mask_out.get("safe_moving_mask", mask_out["final_fluid_mask"])

        debug_dir = Path(output_cfg.get("debug_dir", "data/outputs/debug"))
        debug_dir.mkdir(parents=True, exist_ok=True)

        frames = warp_frames(
            image=scene_out["stylized_image"],
            flow=motion_out["dense_motion_field"],
            mask=safe_moving_mask,
            num_frames=num_frames,
            composite_alpha=composite_alpha,
            debug_dir=debug_dir,
        )

        frames = temporal_smooth_frames(frames)
        frames = blend_loop_boundary(frames)
        frames = enforce_loop(frames)

        # ── Debug frames ──────────────────────────────────────────────────────
        n = len(frames)
        for idx, label in [(0, "000"), (n // 4, "015"), (n // 2, "030"), (3 * n // 4, "045")]:
            if 0 <= idx < n:
                bgr = cv2.cvtColor(
                    np.clip(frames[idx], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(str(debug_dir / f"debug_frame_{label}.png"), bgr)

        if n > n // 4:
            diff = np.abs(
                frames[0].astype(np.float32) - frames[n // 4].astype(np.float32)
            )
            diff_vis = cv2.cvtColor(
                np.clip(diff * 5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(str(debug_dir / "debug_frame_difference_000_015.png"), diff_vis)

        # ── Left-border crop debug ────────────────────────────────────────────
        border_margin = int(self.cfg.get("motion", {}).get("border_protection_margin", 30))
        crop_w = border_margin * 3   # show protected zone + some animated water
        mid_idx = n // 2
        if mid_idx < n:
            orig_bgr = cv2.cvtColor(
                np.clip(scene_out["stylized_image"], 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            frame_bgr = cv2.cvtColor(
                np.clip(frames[mid_idx], 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            cv2.imwrite(
                str(debug_dir / "debug_left_border_crop_030.png"),
                frame_bgr[:, :crop_w, :],
            )
            left_diff = cv2.absdiff(frame_bgr[:, :crop_w, :], orig_bgr[:, :crop_w, :])
            cv2.imwrite(
                str(debug_dir / "debug_left_border_crop_difference.png"),
                np.clip(left_diff.astype(np.float32) * 5, 0, 255).astype(np.uint8),
            )

        # ── Static region change check ────────────────────────────────────────
        if composite_alpha is not None:
            orig = scene_out["stylized_image"].astype(np.float32)
            static_region = composite_alpha > 0.5
            for frame in frames:
                if static_region.any():
                    mean_diff = float(
                        np.abs(frame.astype(np.float32) - orig)[static_region].mean()
                    )
                    if mean_diff > 5.0:
                        print(
                            f"[Warning] Static foreground changed during animation "
                            f"(mean_diff={mean_diff:.2f}); compositing or protection mask failed."
                        )
                        break
            diff0 = np.abs(frames[0].astype(np.float32) - orig)
            cv2.imwrite(
                str(debug_dir / "debug_static_region_difference.png"),
                cv2.cvtColor(
                    np.clip(diff0 * 5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                ),
            )

        final_video_path = Path(
            output_cfg.get("final_video_path", "data/outputs/cinemagraphs/output.gif")
        )
        final_video_path.parent.mkdir(parents=True, exist_ok=True)
        export_cinemagraph(frames, str(final_video_path), fps=fps)

        return {
            "frames": frames,
            "final_video_path": str(final_video_path),
        }

    def _evaluate_outputs(self, mask_out: dict, motion_out: dict, synth_out: dict) -> dict:
        frames = synth_out["frames"]
        flow = motion_out["dense_motion_field"]
        mask = mask_out["final_fluid_mask"]

        return {
            # ── original metrics ──────────────────────────────
            "motion_smoothness": compute_motion_smoothness(flow, mask),
            "loop_consistency": compute_loop_consistency(frames),
            "mask_leakage": compute_mask_leakage(frames, mask),
            # ── standard metrics (no GT needed) ───────────────
            "psnr_loop": compute_psnr(frames[0], frames[-1]),
            "ssim_loop": compute_ms_ssim_loop(frames),
            "temporal_consistency_psnr": compute_temporal_consistency(frames),
            # ── mask diagnostics ──────────────────────────────
            "mask_valid": compute_mask_valid(mask),
        }
