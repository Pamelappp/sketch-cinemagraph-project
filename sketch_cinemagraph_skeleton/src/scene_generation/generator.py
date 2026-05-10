"""Scene generation module for stylized image and realistic reference creation."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.types import SceneOutput
from src.scene_generation.prompt_utils import build_reference_prompt, build_scene_prompt


class SceneGenerator:
    """Generate scene images from structural sketches and text prompts."""

    def __init__(self, cfg: dict) -> None:
        """
        Supported backends:
        - placeholder: simple non-generative fallback
        - diffusion: ControlNet-guided Stable Diffusion inference
        """
        self.cfg = cfg
        self.backend = cfg.get("backend", "placeholder")
        self.image_size = tuple(cfg.get("image_size", [512, 512]))
        self.seed = int(cfg.get("seed", 42))
        self.save_reference = bool(cfg.get("save_reference", True))

        # Diffusion-related config
        self.model_id = cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
        self.controlnet_id = cfg.get("controlnet_id", "lllyasviel/sd-controlnet-canny")
        self.num_inference_steps = int(cfg.get("num_inference_steps", 30))
        self.guidance_scale = float(cfg.get("guidance_scale", 7.5))
        self.negative_prompt = cfg.get(
            "negative_prompt",
            "low quality, blurry, distorted, extra objects, messy composition",
        )
        self.controlnet_conditioning_scale = float(
            cfg.get("controlnet_conditioning_scale", 1.0)
        )
        self.enable_cpu_offload = bool(cfg.get("enable_cpu_offload", True))

        self._pipe = None
        self._torch = None

        # Keep deterministic local RNG for placeholder mode.
        self.rng = np.random.default_rng(self.seed)

    def generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        """
        Produce the main stylized image and an optional realistic reference image.
        """
        if self.backend == "diffusion":
            stylized_image = self.generate_stylized_diffusion(structural_sketch, text_prompt)

            realistic_reference = None
            if self.save_reference:
                realistic_reference = self.generate_reference_diffusion(
                    structural_sketch, text_prompt
                )
        else:
            stylized_image = self.generate_stylized_placeholder(structural_sketch, text_prompt)

            realistic_reference = None
            if self.save_reference:
                realistic_reference = self.generate_reference_placeholder(
                    structural_sketch, text_prompt
                )

        return SceneOutput(
            stylized_image=stylized_image,
            realistic_reference=realistic_reference,
        )

    # ------------------------------------------------------------------
    # Placeholder backend
    # ------------------------------------------------------------------

    def generate_stylized_placeholder(self, structural_sketch, text_prompt: str):
        """
        Placeholder stylized image generation.
        """
        _ = build_scene_prompt(text_prompt)

        image = self.preprocess_sketch(structural_sketch)
        pil_image = Image.fromarray(image)

        pil_image = pil_image.filter(ImageFilter.SMOOTH_MORE)
        pil_image = ImageEnhance.Contrast(pil_image).enhance(1.35)
        pil_image = ImageEnhance.Color(pil_image).enhance(1.25)

        array = np.asarray(pil_image).astype(np.float32)

        tint = np.array([1.05, 1.12, 0.92], dtype=np.float32)
        array = array * tint

        noise = self.rng.normal(loc=0.0, scale=3.0, size=array.shape)
        array = array + noise

        return np.clip(array, 0, 255).astype(np.uint8)

    def generate_reference_placeholder(self, structural_sketch, text_prompt: str):
        """
        Placeholder realistic reference generation.
        """
        _ = build_reference_prompt(text_prompt)

        image = self.preprocess_sketch(structural_sketch)
        pil_image = Image.fromarray(image)

        gray = ImageOps.grayscale(pil_image)
        gray = gray.filter(ImageFilter.SMOOTH)
        gray = ImageEnhance.Contrast(gray).enhance(1.15)

        gray_array = np.asarray(gray).astype(np.float32)

        reference = np.stack(
            [
                gray_array * 0.95 + 12.0,
                gray_array * 1.02 + 18.0,
                gray_array * 1.08 + 22.0,
            ],
            axis=-1,
        )

        return np.clip(reference, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Diffusion backend
    # ------------------------------------------------------------------

    def generate_stylized_diffusion(self, structural_sketch, text_prompt: str):
        """
        Generate the stylized landscape image with ControlNet-guided diffusion.
        """
        prompt = build_scene_prompt(text_prompt)
        control_image = self.prepare_control_image(structural_sketch)
        return self._run_diffusion(prompt, control_image)

    def generate_reference_diffusion(self, structural_sketch, text_prompt: str):
        """
        Generate the realistic reference image with the same structural sketch.
        """
        prompt = build_reference_prompt(text_prompt)
        control_image = self.prepare_control_image(structural_sketch)
        return self._run_diffusion(prompt, control_image)

    def _run_diffusion(self, prompt: str, control_image: Image.Image) -> np.ndarray:
        """
        Run one ControlNet inference and return uint8 RGB numpy image.
        """
        pipe = self._get_diffusion_pipeline()
        torch = self._torch

        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        result = pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            generator=generator,
            height=self.image_size[1],
            width=self.image_size[0],
        )

        image = result.images[0]
        return np.asarray(image.convert("RGB"))

    def _get_diffusion_pipeline(self):
        """
        Lazy-load the diffusion pipeline only when needed.
        """
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        self._torch = torch

        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch_dtype,
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        if device == "cuda":
            if self.enable_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        else:
            pipe.to(device)

        self._pipe = pipe
        return self._pipe

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def preprocess_sketch(self, structural_sketch) -> np.ndarray:
        """
        Normalize or convert the structural sketch into a stable RGB format.
        """
        array = np.asarray(structural_sketch)

        if array.ndim == 2:
            pil_image = Image.fromarray(_to_uint8(array)).convert("RGB")
        elif array.ndim == 3:
            pil_image = Image.fromarray(_to_uint8(array)).convert("RGB")
        else:
            raise ValueError("Structural sketch must be a 2D or 3D image array.")

        # PIL size uses (width, height)
        pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
        return np.asarray(pil_image)

    def prepare_control_image(self, structural_sketch) -> Image.Image:
        """
        Prepare the structural sketch as the ControlNet conditioning image.

        For the current default config, this keeps the sketch as a clean
        high-contrast line image. This works well for edge-like ControlNet
        backends and keeps the structure explicit.
        """
        image = self.preprocess_sketch(structural_sketch)
        pil_image = Image.fromarray(image).convert("L")

        # Increase contrast to make line structure more explicit.
        pil_image = ImageEnhance.Contrast(pil_image).enhance(2.0)

        # Back to RGB because the pipeline accepts PIL images as control inputs.
        pil_image = pil_image.convert("RGB")
        return pil_image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image data to uint8 so PIL can process it safely."""
    array = np.asarray(image)

    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)