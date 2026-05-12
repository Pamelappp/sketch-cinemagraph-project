"""Stable Diffusion + ControlNet backend for scene generation.

Implements the Sketch2Cinemagraph paper's stylized landscape image
generation step using off-the-shelf pretrained models:

- Base LDM   : runwayml/stable-diffusion-v1-5 (or any SD 1.5 mirror)
- ControlNet : lllyasviel/sd-controlnet-scribble for sketch conditioning

The paper additionally fine-tunes the LDM with DreamBooth on the Holynski
landscape dataset for better water/waterfall textures. We skip that step
for the course-project MVP — the pretrained ControlNet still produces
visually coherent stylized landscapes from freehand sketches, which is
enough to demo the full Sketch2Cinemagraph pipeline.

Usage::

    cfg = {
        "backend": "diffusion",
        "image_size": [512, 512],
        "model_id": "runwayml/stable-diffusion-v1-5",   # optional override
        "controlnet_id": "lllyasviel/sd-controlnet-scribble",
        "device": "mps",                                  # optional override
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "controlnet_conditioning_scale": 1.1,
        "seed": 42,
    }
    generator = DiffusionSceneGenerator(cfg)
    out = generator.generate(structural_sketch, "ocean waves under blue sky")
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.types import SceneOutput


_DEFAULT_MODEL_ID = os.environ.get(
    "SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"
)
# ControlNet v1.1 lineart model: much better at interpreting hand-drawn
# structural line art (mountain silhouettes, horizon lines) compared to
# the v1.0 scribble model which was trained on HED edge maps.
_DEFAULT_CONTROLNET_ID = os.environ.get(
    "CONTROLNET_ID", "lllyasviel/control_v11p_sd15_lineart"
)


class DiffusionSceneGenerator:
    """SD + ControlNet scene generator that mirrors the proposal §3.3 spec."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.image_size = tuple(cfg.get("image_size", [512, 512]))
        self.seed = int(cfg.get("seed", 42))
        self.save_reference = bool(cfg.get("save_reference", True))

        self.model_id = str(cfg.get("model_id", _DEFAULT_MODEL_ID))
        self.controlnet_id = str(cfg.get("controlnet_id", _DEFAULT_CONTROLNET_ID))
        self._device_pref: Optional[str] = cfg.get("device")

        self.num_inference_steps = int(cfg.get("num_inference_steps", 25))
        self.guidance_scale = float(cfg.get("guidance_scale", 7.5))
        self.controlnet_conditioning_scale = float(
            cfg.get("controlnet_conditioning_scale", 1.0)
        )

        self._lock = threading.Lock()
        self._pipe = None
        self._device = None

    # ------------------------------------------------------------------ API

    def generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        """Produce a stylized landscape image plus a realistic reference."""
        sketch = self._prepare_sketch(structural_sketch)
        stylized = self.generate_stylized(sketch, text_prompt)
        realistic = None
        if self.save_reference:
            realistic = self.generate_reference(sketch, text_prompt)
        return SceneOutput(stylized_image=stylized, realistic_reference=realistic)

    def generate_stylized(self, structural_sketch, text_prompt: str) -> np.ndarray:
        """Stylized output: prompt is taken at face value (user controls style)."""
        sketch = self._prepare_sketch(structural_sketch)
        prompt = self._stylized_prompt(text_prompt)
        return self._run(sketch, prompt, seed_offset=0)

    def generate_reference(self, structural_sketch, text_prompt: str) -> np.ndarray:
        """Realistic reference for motion estimation: forces a photo-realistic style."""
        sketch = self._prepare_sketch(structural_sketch)
        prompt = self._realistic_prompt(text_prompt)
        return self._run(sketch, prompt, seed_offset=1)

    # ------------------------------------------------------------------ internals

    def _stylized_prompt(self, text_prompt: str) -> str:
        text_prompt = text_prompt.strip() or "landscape"
        # Encourage cohesive landscape rendering; user's own style words pass through.
        return f"{text_prompt}, beautiful landscape, detailed, cinematic lighting"

    def _realistic_prompt(self, text_prompt: str) -> str:
        text_prompt = text_prompt.strip() or "landscape"
        return (
            f"{text_prompt}, photo realistic, natural lighting, 4k, "
            "high detail, sharp focus, landscape photography"
        )

    def _negative_prompt(self) -> str:
        return (
            "lowres, blurry, jpeg artifacts, watermark, text, signature, "
            "deformed, distorted, ugly"
        )

    def _prepare_sketch(self, structural_sketch) -> Image.Image:
        """Convert a sketch array into the control image that ControlNet expects.

        - **lineart** (v1.1 default): expects a clean line drawing — white
          strokes on a black background. The model interprets lines as
          structural boundaries (edges, silhouettes) rather than painting
          them literally.
        - **scribble** (v1.0 fallback): same convention (white-on-black).
        """
        array = np.asarray(structural_sketch)
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
                array = (array * 255.0).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)

        # White lines on black background — the convention shared by both
        # scribble and lineart ControlNet variants.
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        ink = (gray < 200).astype(np.uint8) * 255

        # Light dilation so thin 1-px strokes survive down-sampling.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ink = cv2.dilate(ink, kernel, iterations=1)

        ink_rgb = cv2.cvtColor(ink, cv2.COLOR_GRAY2RGB)

        pil = Image.fromarray(ink_rgb)
        pil = pil.resize(self.image_size, Image.Resampling.BILINEAR)
        return pil

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        with self._lock:
            if self._pipe is not None:
                return

            import torch
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetPipeline,
                UniPCMultistepScheduler,
            )

            from src.motion_field.networks import resolve_device

            device = resolve_device(self._device_pref)
            # SD on MPS is most reliable in fp32; fp16 has unsupported ops on
            # some MPS builds. CUDA happily uses fp16 for ~2x speedup.
            dtype = torch.float16 if device.type == "cuda" else torch.float32

            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id, torch_dtype=dtype
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)

            # Memory savers — important on MPS where unified memory is shared.
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

            self._pipe = pipe
            self._device = device

    def _run(self, control_image: Image.Image, prompt: str, seed_offset: int) -> np.ndarray:
        self._ensure_loaded()
        import torch

        generator = torch.Generator(device="cpu").manual_seed(self.seed + seed_offset)
        result = self._pipe(
            prompt=prompt,
            negative_prompt=self._negative_prompt(),
            image=control_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            generator=generator,
            width=self.image_size[0],
            height=self.image_size[1],
        )
        pil_out = result.images[0]
        return np.array(pil_out.convert("RGB"))
