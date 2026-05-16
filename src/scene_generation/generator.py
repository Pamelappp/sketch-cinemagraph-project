"""Scene generation module for stylized image and realistic reference creation."""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.types import SceneOutput
from src.scene_generation.prompt_utils import build_reference_prompt, build_scene_prompt


class SceneGenerator:
    """Generate scene images from structural sketches and text prompts."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.backend = cfg.get("backend", "placeholder")
        # image_size=None keeps the input sketch's native resolution.
        _img_size = cfg.get("image_size")
        self.image_size = tuple(_img_size) if _img_size else None
        self.seed = int(cfg.get("seed", 42))
        self.save_reference = bool(cfg.get("save_reference", True))

        self.rng = np.random.default_rng(self.seed)

    def generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        """Route to the diffusion backend or the offline placeholder generator."""
        if self.backend == "diffusion":
            return self._diffusion_generate(structural_sketch, text_prompt)

        stylized_image = self.generate_stylized(structural_sketch, text_prompt)

        realistic_reference = None
        if self.save_reference:
            realistic_reference = self.generate_reference(structural_sketch, text_prompt)

        return SceneOutput(
            stylized_image=stylized_image,
            realistic_reference=realistic_reference,
        )

    def _diffusion_generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        """Lazy-instantiate the diffusion backend; rebuild when controlnet_id changes."""
        from src.scene_generation.diffusion_backend import DiffusionSceneGenerator

        desired_cn = self.cfg.get("controlnet_id")
        if (
            not hasattr(self, "_diffusion_backend")
            or self._diffusion_backend is None
            or (desired_cn and self._diffusion_backend.controlnet_id != desired_cn)
        ):
            self._diffusion_backend = DiffusionSceneGenerator(self.cfg)
        return self._diffusion_backend.generate(structural_sketch, text_prompt)

    def generate_stylized(self, structural_sketch, text_prompt: str):
        """Placeholder stylized image: smoothed + contrast/colour-boosted sketch."""
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

    def generate_reference(self, structural_sketch, text_prompt: str):
        """Placeholder photo-like reference: grayscale sketch mapped to a muted palette."""
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

    def preprocess_sketch(self, structural_sketch):
        """Coerce a sketch into a uint8 RGB array, optionally resized to image_size."""
        array = np.asarray(structural_sketch)

        if array.ndim == 2:
            pil_image = Image.fromarray(_to_uint8(array)).convert("RGB")
        elif array.ndim == 3:
            pil_image = Image.fromarray(_to_uint8(array)).convert("RGB")
        else:
            raise ValueError("Structural sketch must be a 2D or 3D image array.")

        if self.image_size is not None:
            # PIL size is (width, height); config image_size follows the same convention.
            pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)

        return np.asarray(pil_image)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)

    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)
