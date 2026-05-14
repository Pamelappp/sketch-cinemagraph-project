# src/scene_generation/generator.py

import json
import pathlib

import cv2
import numpy as np
from PIL import Image

from src.types import SceneOutput
from src.scene_generation.prompt_utils import (
    build_scene_prompt,
    build_reference_prompt,
    build_negative_prompt,
)
from src.scene_generation.diffusion_backend import DiffusionSceneBackend


class SceneGenerator:
    def __init__(self, cfg: dict) -> None:
        self.cfg     = cfg
        self.backend = cfg.get("backend", "placeholder")

        self.image_size    = cfg.get("image_size", None)
        raw_seed           = cfg.get("seed", 42)
        self.seed          = None if raw_seed is None else int(raw_seed)
        self.save_reference = bool(cfg.get("save_reference", True))
        self.style_prompt  = cfg.get("style_prompt") or None

        self.control_type            = cfg.get("control_type", "lineart")
        self.sketch_binary_threshold = int(cfg.get("sketch_binary_threshold", 220))
        self.use_different_ref_seed  = bool(cfg.get("use_different_reference_seed", False))
        self.scene_prior             = cfg.get("scene_prior", "")
        self.debug_dir               = cfg.get("debug_dir", None)

        # Per-mode sub-configs (fallback to empty dict → _get_gen_params uses backend defaults)
        self._stylized_cfg  = cfg.get("stylized", {})
        self._reference_cfg = cfg.get("reference", {})

        self.diffusion_backend = None
        if self.backend == "diffusion":
            self.diffusion_backend = DiffusionSceneBackend(cfg)

    # ── Public entry point ────────────────────────────────────────────────────

    def generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        if self.backend == "diffusion":
            stylized  = self.generate_stylized_diffusion(structural_sketch, text_prompt)
            reference = None
            if self.save_reference:
                reference = self.generate_reference_diffusion(structural_sketch, text_prompt)
            control_pil = self.prepare_control_image(structural_sketch)
            self._save_debug_config(text_prompt)
            self._check_alignment(control_pil, stylized, reference)
        else:
            stylized  = self.generate_stylized_placeholder(structural_sketch, text_prompt)
            reference = None
            if self.save_reference:
                reference = self.generate_reference_placeholder(structural_sketch, text_prompt)

        return SceneOutput(stylized_image=stylized, realistic_reference=reference)

    # ── Sub-config helpers ────────────────────────────────────────────────────

    def _get_gen_params(self, sub_cfg: dict) -> dict:
        """Merge sub-config with top-level backend defaults."""
        b = self.diffusion_backend
        return {
            "num_inference_steps":           sub_cfg.get("num_inference_steps",           b.num_inference_steps),
            "guidance_scale":                sub_cfg.get("guidance_scale",                b.guidance_scale),
            "controlnet_conditioning_scale": sub_cfg.get("controlnet_conditioning_scale", b.controlnet_conditioning_scale),
            "style_prompt":                  sub_cfg.get("style_prompt",                  self.style_prompt),
            "extra_positive_prompt":         sub_cfg.get("extra_positive_prompt",         ""),
            "extra_negative_prompt":         sub_cfg.get("extra_negative_prompt",         ""),
        }

    # ── Diffusion generation ──────────────────────────────────────────────────

    def generate_stylized_diffusion(self, structural_sketch, text_prompt: str):
        p = self._get_gen_params(self._stylized_cfg)
        prompt = build_scene_prompt(
            text_prompt,
            scene_prior=self.scene_prior,
            style_prompt=p["style_prompt"],
            extra_positive_prompt=p["extra_positive_prompt"],
        )
        neg = build_negative_prompt(
            base_prompt=text_prompt,
            scene_prior=self.scene_prior,
            extra_negative_prompt=p["extra_negative_prompt"],
            mode="stylized",
        )
        control_image = self.prepare_control_image(structural_sketch)

        self._save_debug_text("debug_stylized_prompt.txt", prompt)
        self._save_debug_text("debug_stylized_negative_prompt.txt", neg)

        result = self.diffusion_backend.generate(
            prompt, control_image,
            negative_prompt=neg,
            num_inference_steps=p["num_inference_steps"],
            guidance_scale=p["guidance_scale"],
            controlnet_conditioning_scale=p["controlnet_conditioning_scale"],
        )
        self._save_debug_image("debug_stylized_image.png", result)
        return result

    def generate_reference_diffusion(self, structural_sketch, text_prompt: str):
        p = self._get_gen_params(self._reference_cfg)
        prompt = build_reference_prompt(
            text_prompt,
            scene_prior=self.scene_prior,
            style_prompt=p["style_prompt"],
            extra_positive_prompt=p["extra_positive_prompt"],
        )
        neg = build_negative_prompt(
            base_prompt=text_prompt,
            scene_prior=self.scene_prior,
            extra_negative_prompt=p["extra_negative_prompt"],
            mode="reference",
        )
        control_image = self.prepare_control_image(structural_sketch)

        self._save_debug_text("debug_reference_prompt.txt", prompt)
        self._save_debug_text("debug_reference_negative_prompt.txt", neg)

        ref_seed = self.diffusion_backend.seed
        if self.use_different_ref_seed and ref_seed is not None:
            ref_seed = ref_seed + 1

        result = self.diffusion_backend.generate(
            prompt, control_image,
            seed=ref_seed,
            negative_prompt=neg,
            num_inference_steps=p["num_inference_steps"],
            guidance_scale=p["guidance_scale"],
            controlnet_conditioning_scale=p["controlnet_conditioning_scale"],
        )
        self._save_debug_image("debug_reference_image.png", result)
        return result

    # ── Placeholder generation (no diffusion model) ───────────────────────────

    def generate_stylized_placeholder(self, structural_sketch, text_prompt: str):
        return self.preprocess_sketch(structural_sketch)

    def generate_reference_placeholder(self, structural_sketch, text_prompt: str):
        return self.preprocess_sketch(structural_sketch)

    # ── Control image preparation ─────────────────────────────────────────────

    def preprocess_sketch(self, structural_sketch):
        array = np.asarray(structural_sketch)
        if array.ndim == 2:
            pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        else:
            pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")

        if self.image_size is not None:
            target_w, target_h = self.image_size
        else:
            orig_w, orig_h = pil_image.size
            target_w = _round_to_multiple_of_8(orig_w)
            target_h = _round_to_multiple_of_8(orig_h)

        pil_image = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return np.asarray(pil_image)

    def prepare_control_image(self, structural_sketch):
        image = self.preprocess_sketch(structural_sketch)
        arr   = np.asarray(image)

        # Grayscale
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr.copy()

        # Hard binary threshold: below threshold → black line, else → white background
        binary = np.where(gray < self.sketch_binary_threshold, 0, 255).astype(np.uint8)

        if self.control_type == "canny":
            # Canny ControlNet expects white edges on black background
            control_arr = cv2.Canny(binary, threshold1=50, threshold2=150)
        else:
            # lineart / scribble: clean white background, black lines
            control_arr = binary

        self._save_debug_image_raw("debug_control_image.png", control_arr)

        return Image.fromarray(control_arr).convert("RGB")

    # ── Alignment sanity check ────────────────────────────────────────────────

    def _check_alignment(self, control_pil, stylized, reference):
        """Print edge-overlap metrics; warn when images are misaligned."""
        ctrl_gray = np.asarray(control_pil.convert("L")).astype(np.uint8)
        ctrl_edges = cv2.Canny(ctrl_gray, 50, 150).astype(np.float32) / 255.0

        def _edges(arr):
            g = cv2.cvtColor(
                np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )
            return cv2.Canny(g, 50, 150).astype(np.float32) / 255.0

        def _overlap(a, b):
            return float(np.logical_and(a > 0, b > 0).sum()) / (a.sum() + 1e-6)

        sty_edges    = _edges(stylized)
        sty_ctrl_ov  = _overlap(ctrl_edges, sty_edges)
        print(f"[Alignment] stylized↔control edge overlap  : {sty_ctrl_ov:.3f}")
        if sty_ctrl_ov < 0.15:
            print("[Warning] Generated stylized image does not follow the structural "
                  "sketch well. Consider raising controlnet_conditioning_scale.")

        if reference is not None:
            ref_edges    = _edges(reference)
            ref_ctrl_ov  = _overlap(ctrl_edges, ref_edges)
            sty_ref_sim  = _overlap(sty_edges, ref_edges)
            print(f"[Alignment] reference↔control edge overlap : {ref_ctrl_ov:.3f}")
            print(f"[Alignment] stylized↔reference edge sim    : {sty_ref_sim:.3f}")
            if sty_ref_sim < 0.20:
                print("[Warning] Stylized and reference images may be spatially "
                      "misaligned — motion transfer may fail.")

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def _debug_path(self, filename: str) -> pathlib.Path | None:
        if self.debug_dir is None:
            return None
        d = pathlib.Path(self.debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / filename

    def _save_debug_text(self, filename: str, text: str):
        p = self._debug_path(filename)
        if p is not None:
            p.write_text(text, encoding="utf-8")

    def _save_debug_image(self, filename: str, arr):
        p = self._debug_path(filename)
        if p is None:
            return
        bgr = cv2.cvtColor(
            np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(str(p), bgr)

    def _save_debug_image_raw(self, filename: str, arr):
        p = self._debug_path(filename)
        if p is None:
            return
        cv2.imwrite(str(p), np.clip(arr, 0, 255).astype(np.uint8))

    def _save_debug_config(self, text_prompt: str = ""):
        p = self._debug_path("debug_generation_config.json")
        if p is None:
            return
        b = self.diffusion_backend

        def _sty_params():
            if not self._stylized_cfg:
                return {}
            ps = self._get_gen_params(self._stylized_cfg)
            return {
                "num_inference_steps":           ps["num_inference_steps"],
                "guidance_scale":                ps["guidance_scale"],
                "controlnet_conditioning_scale": ps["controlnet_conditioning_scale"],
                "style_prompt":                  ps["style_prompt"],
                "extra_positive_prompt":         ps["extra_positive_prompt"],
                "extra_negative_prompt":         ps["extra_negative_prompt"],
            }

        def _ref_params():
            if not self._reference_cfg:
                return {}
            pr = self._get_gen_params(self._reference_cfg)
            return {
                "num_inference_steps":           pr["num_inference_steps"],
                "guidance_scale":                pr["guidance_scale"],
                "controlnet_conditioning_scale": pr["controlnet_conditioning_scale"],
                "style_prompt":                  pr["style_prompt"],
                "extra_positive_prompt":         pr["extra_positive_prompt"],
                "extra_negative_prompt":         pr["extra_negative_prompt"],
            }

        data = {
            "shared": {
                "model_id":                     getattr(b, "model_id", None),
                "controlnet_id":                getattr(b, "controlnet_id", None),
                "control_type":                 self.control_type,
                "seed":                         getattr(b, "seed", None),
                "use_different_reference_seed": self.use_different_ref_seed,
                "control_guidance_start":       getattr(b, "control_guidance_start", None),
                "control_guidance_end":         getattr(b, "control_guidance_end", None),
                "sketch_binary_threshold":      self.sketch_binary_threshold,
                "scene_prior":                  self.scene_prior,
            },
            "stylized":  _sty_params(),
            "reference": _ref_params(),
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _round_to_multiple_of_8(value: int) -> int:
    return max(8, int(round(value / 8.0)) * 8)
