from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.scene_generation.generator import SceneGenerator
from src.fluid_mask.semantic_mask import build_semantic_mask
from src.fluid_mask.refine_mask import refine_fluid_mask
from src.fluid_mask.postprocess import combine_masks
from src.motion_field.sketch_parser import parse_motion_sketch
from src.motion_field.sparse_constraints import build_sparse_constraints
from src.motion_field.propagate import propagate_sparse_to_dense
from src.motion_field.smooth import smooth_motion_field
from src.synthesis.warp import warp_frames
from src.synthesis.loop import enforce_loop, make_pingpong_loop
from src.synthesis.export_video import export_mp4, export_gif


class CinemagraphPipeline:
    """End-to-end pipeline: scene gen → fluid mask → motion field → loop synthesis."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.scene_generator = SceneGenerator(cfg.get("scene", {}))

        motion_cfg = cfg.get("motion", {}) or {}
        self.motion_backend = str(motion_cfg.get("backend", "heuristic")).lower()
        self._learned_predictor = None
        if self.motion_backend == "learned":
            # Lazy-import so the heuristic path stays import-light.
            from src.motion_field.learned_predictor import LearnedMotionPredictor
            self._learned_predictor = LearnedMotionPredictor(motion_cfg.get("learned", {}))

    def run(
        self,
        structural_sketch: np.ndarray,
        motion_sketch: np.ndarray,
        text_prompt: str,
        pre_generated_image: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """Run all four stages; pre_generated_image enables photo-input mode (§5.6)."""
        photo_mode = pre_generated_image is not None
        if photo_mode:
            scene_out = {
                "stylized_image": pre_generated_image,
                "realistic_reference": pre_generated_image,
            }
        else:
            scene_out = self._scene_generation(structural_sketch, text_prompt)

        # Colour-based mask expansion only fires for real photos: diffusion outputs
        # unify hues across the canvas and would cause the mask to cover everything.
        mask_ref = scene_out["stylized_image"] if photo_mode else None
        mask_out = self._fluid_mask_extraction(
            structural_sketch=structural_sketch,
            motion_sketch=motion_sketch,
            stylized_image=scene_out["stylized_image"],
            text_prompt=text_prompt,
            reference_image=mask_ref,
        )
        reference_image = scene_out.get("realistic_reference")
        if reference_image is None:
            reference_image = scene_out["stylized_image"]
        motion_out = self._motion_field_estimation(
            motion_sketch=motion_sketch,
            final_fluid_mask=mask_out["final_fluid_mask"],
            reference_image=reference_image,
        )
        synth_out = self._cinemagraph_synthesis(
            stylized_image=scene_out["stylized_image"],
            dense_motion_field=motion_out["dense_motion_field"],
            final_fluid_mask=mask_out["final_fluid_mask"],
        )

        return {
            **scene_out,
            **mask_out,
            **motion_out,
            **synth_out,
        }

    def _scene_generation(self, structural_sketch: np.ndarray, text_prompt: str) -> Dict[str, Any]:
        scene_out = self.scene_generator.generate(
            structural_sketch=structural_sketch,
            text_prompt=text_prompt,
        )
        if hasattr(scene_out, "stylized_image"):
            return {
                "stylized_image": scene_out.stylized_image,
                "realistic_reference": getattr(scene_out, "realistic_reference", None),
            }
        return scene_out

    def _fluid_mask_extraction(
        self,
        structural_sketch: np.ndarray,
        motion_sketch: np.ndarray,
        stylized_image: np.ndarray,
        text_prompt: str,
        reference_image: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """semantic_mask (sketch + optional colour expansion) ∩ Grounded-SAM refined mask."""
        semantic_mask = build_semantic_mask(
            structural_sketch=structural_sketch,
            motion_sketch=motion_sketch,
            reference_image=reference_image,
        )

        refined_mask = refine_fluid_mask(
            image=stylized_image,
            text_prompt=text_prompt,
        )

        final_fluid_mask = combine_masks(
            semantic_mask=semantic_mask,
            refined_mask=refined_mask,
        )

        return {
            "semantic_mask": semantic_mask,
            "refined_mask": refined_mask,
            "final_fluid_mask": final_fluid_mask,
        }

    def _motion_field_estimation(
        self,
        motion_sketch: np.ndarray,
        final_fluid_mask: np.ndarray,
        reference_image: np.ndarray,
    ) -> Dict[str, Any]:
        """Dense flow from motion sketch via 'heuristic' KNN-IDW or 'learned' U-Net."""
        strokes = parse_motion_sketch(motion_sketch)
        sparse_constraints = build_sparse_constraints(
            strokes=strokes,
            mask=final_fluid_mask,
        )

        if self.motion_backend == "learned" and self._learned_predictor is not None:
            dense_motion_field = self._learned_predictor.predict(
                reference_image=reference_image,
                fluid_mask=final_fluid_mask,
                motion_sketch=motion_sketch,
            )
        else:
            dense_motion_field = propagate_sparse_to_dense(
                constraints=sparse_constraints,
                mask=final_fluid_mask,
            )

        dense_motion_field = smooth_motion_field(
            flow=dense_motion_field,
            mask=final_fluid_mask,
        )

        return {
            "strokes": strokes,
            "sparse_constraints": sparse_constraints,
            "dense_motion_field": dense_motion_field,
        }

    def _cinemagraph_synthesis(
        self,
        stylized_image: np.ndarray,
        dense_motion_field: np.ndarray,
        final_fluid_mask: np.ndarray,
    ) -> Dict[str, Any]:
        synth_cfg = self.cfg.get("synthesis", {})
        num_frames = int(synth_cfg.get("num_frames", 48))
        fps = int(synth_cfg.get("fps", 12))
        # 'linear' is the default — symmetric splatting already closes the loop.
        loop_mode = synth_cfg.get("loop_mode", "linear")

        frames = warp_frames(
            image=stylized_image,
            flow=dense_motion_field,
            mask=final_fluid_mask,
            num_frames=num_frames,
            motion_scale=float(synth_cfg.get("motion_scale", 1.0)),
            border_mode=synth_cfg.get("border_mode", "reflect"),
        )

        if loop_mode == "pingpong":
            frames = make_pingpong_loop(frames)
        else:
            frames = enforce_loop(frames)

        output_cfg = self.cfg.get("output", {})
        output_dir = Path(output_cfg.get("dir", "data/outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = output_dir / output_cfg.get("video_name", "result.mp4")
        gif_path = output_dir / output_cfg.get("gif_name", "result.gif")

        export_mp4(frames, str(video_path), fps=fps)
        export_gif(frames, str(gif_path), fps=fps)

        return {
            "frames": frames,
            "video_path": str(video_path),
            "gif_path": str(gif_path),
        }
