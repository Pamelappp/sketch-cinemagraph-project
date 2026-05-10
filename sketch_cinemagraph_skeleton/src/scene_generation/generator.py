# src/scene_generation/generator.py

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

from src.types import SceneOutput
from src.scene_generation.prompt_utils import build_reference_prompt, build_scene_prompt
from src.scene_generation.diffusion_backend import DiffusionSceneBackend


class SceneGenerator:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.backend = cfg.get("backend", "placeholder")
        self.image_size = tuple(cfg.get("image_size", [512, 512]))
        self.seed = int(cfg.get("seed", 42))
        self.save_reference = bool(cfg.get("save_reference", True))

        self.diffusion_backend = None
        if self.backend == "diffusion":
            self.diffusion_backend = DiffusionSceneBackend(cfg)

    def generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        if self.backend == "diffusion":
            stylized_image = self.generate_stylized_diffusion(structural_sketch, text_prompt)
            realistic_reference = None
            if self.save_reference:
                realistic_reference = self.generate_reference_diffusion(structural_sketch, text_prompt)
        else:
            stylized_image = self.generate_stylized_placeholder(structural_sketch, text_prompt)
            realistic_reference = None
            if self.save_reference:
                realistic_reference = self.generate_reference_placeholder(structural_sketch, text_prompt)

        return SceneOutput(
            stylized_image=stylized_image,
            realistic_reference=realistic_reference,
        )

    def generate_stylized_diffusion(self, structural_sketch, text_prompt: str):
        prompt = build_scene_prompt(text_prompt)
        control_image = self.prepare_control_image(structural_sketch)
        return self.diffusion_backend.generate(prompt, control_image)

    def generate_reference_diffusion(self, structural_sketch, text_prompt: str):
        prompt = build_reference_prompt(text_prompt)
        control_image = self.prepare_control_image(structural_sketch)
        return self.diffusion_backend.generate(prompt, control_image)

    def generate_stylized_placeholder(self, structural_sketch, text_prompt: str):
        image = self.preprocess_sketch(structural_sketch)
        return image

    def generate_reference_placeholder(self, structural_sketch, text_prompt: str):
        image = self.preprocess_sketch(structural_sketch)
        return image

    def preprocess_sketch(self, structural_sketch):
        array = np.asarray(structural_sketch)
        if array.ndim == 2:
            pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        elif array.ndim == 3:
            pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError("Structural sketch must be a 2D or 3D image array.")
        pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
        return np.asarray(pil_image)

    def prepare_control_image(self, structural_sketch):
        image = self.preprocess_sketch(structural_sketch)
        pil_image = Image.fromarray(image).convert("L")

        # strengthen line structure
        pil_image = ImageOps.autocontrast(pil_image)
        pil_image = ImageEnhance.Contrast(pil_image).enhance(2.2)

        # optional light binarization effect
        arr = np.asarray(pil_image)
        arr = np.where(arr > 200, 255, arr)
        pil_image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

        return pil_image