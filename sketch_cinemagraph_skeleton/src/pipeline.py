from pathlib import Path

import cv2
import numpy as np

from src.types import UserInput
from src.scene_generation.generator import SceneGenerator
from src.fluid_mask.semantic_mask import build_semantic_mask
from src.fluid_mask.refine_mask import refine_fluid_mask
from src.fluid_mask.postprocess import combine_masks
from src.motion_field.sketch_parser import parse_motion_sketch
from src.motion_field.sparse_constraints import build_sparse_constraints
from src.motion_field.propagate import propagate_sparse_to_dense
from src.motion_field.smooth import smooth_motion_field
from src.motion_field.t2c_flow_predictor import T2CFlowPredictor
from src.synthesis.warp import warp_frames
from src.synthesis.loop import temporal_smooth_frames, blend_loop_boundary, enforce_loop
from src.synthesis.export_video import export_cinemagraph
from src.evaluation.metrics import (
    compute_motion_smoothness,
    compute_loop_consistency,
    compute_mask_leakage,
    compute_psnr,
    compute_ms_ssim_loop,
    compute_temporal_consistency,
)


def _resize_to_match(array, target_hw, is_mask=False):
    """
    Resize any image/mask to match target (height, width).
    """
    target_h, target_w = target_hw
    if array.shape[:2] == (target_h, target_w):
        return array

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(array, (target_w, target_h), interpolation=interp)


class CinemagraphPipeline:
    """
    Simplified end-to-end pipeline for the course project.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.scene_generator = SceneGenerator(cfg.get("scene", {}))

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
        Build semantic / refined / final mask, all aligned to the scene image size.
        """
        h, w = scene_out["stylized_image"].shape[:2]

        structural_sketch = _resize_to_match(
            user_input.structural_sketch, (h, w), is_mask=False
        )
        motion_sketch = _resize_to_match(
            user_input.motion_sketch, (h, w), is_mask=False
        )

        semantic_mask = build_semantic_mask(
            structural_sketch=structural_sketch,
            motion_sketch=motion_sketch,
        )

        try:
            refined_mask = refine_fluid_mask(
                image=scene_out["stylized_image"],
                text_prompt=user_input.text_prompt,
                fluid_prompt=user_input.fluid_prompt,
            )
        except Exception as error:
            print(f"[Warning] refine_fluid_mask failed, fallback to semantic mask only: {error}")
            refined_mask = semantic_mask.copy()

        refined_mask = _resize_to_match(refined_mask, (h, w), is_mask=True)

        final_fluid_mask = combine_masks(
            semantic_mask=semantic_mask,
            refined_mask=refined_mask,
        )

        return {
            "semantic_mask": semantic_mask,
            "refined_mask": refined_mask,
            "final_fluid_mask": final_fluid_mask,
            "resized_motion_sketch": motion_sketch,
        }

    def _motion_field_estimation(
        self, user_input: UserInput, scene_out: dict, mask_out: dict
    ) -> dict:
        """
        Use the resized motion sketch so flow matches scene size.

        Tries the T2C neural predictor first (when a checkpoint is configured);
        falls back to RBF sparse-to-dense propagation otherwise.
        """
        motion_sketch = mask_out.get("resized_motion_sketch", user_input.motion_sketch)
        mask = mask_out["final_fluid_mask"]

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

        dense_motion_field = smooth_motion_field(flow=dense_motion_field, mask=mask)

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

        frames = warp_frames(
            image=scene_out["stylized_image"],
            flow=motion_out["dense_motion_field"],
            mask=mask_out["final_fluid_mask"],
            num_frames=num_frames,
        )

        frames = temporal_smooth_frames(frames)
        frames = blend_loop_boundary(frames)
        frames = enforce_loop(frames)

        final_video_path = Path(output_cfg.get("final_video_path", "data/outputs/cinemagraphs/output.gif"))
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
        }