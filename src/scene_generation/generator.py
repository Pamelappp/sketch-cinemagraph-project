"""Scene generation module for stylized image and realistic reference creation."""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.types import SceneOutput
from src.scene_generation.prompt_utils import build_reference_prompt, build_scene_prompt


class SceneGenerator:
    """Generate scene images from structural sketches and text prompts."""

    def __init__(self, cfg: dict) -> None:
        """
        Initialize the scene generation backend and its parameters.

        Detailed TODO:
        1. Store config values such as model name, seed, image size, and prompt settings.
        2. Prepare the actual backend later, such as ControlNet or a placeholder generator.
        3. Keep this class flexible enough to swap different generators.
        """
        self.cfg = cfg
        self.backend = cfg.get("backend", "placeholder")
        _img_size = cfg.get("image_size")
        # None / falsy: skip resize and keep the input sketch's native resolution.
        self.image_size = tuple(_img_size) if _img_size else None
        self.seed = int(cfg.get("seed", 42))
        self.save_reference = bool(cfg.get("save_reference", True))

        # Keep the random generator local and deterministic for repeatable demos.
        self.rng = np.random.default_rng(self.seed)

    def generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        """
        Produce the main stylized image and an optional realistic reference image.

        Routes to the diffusion backend (Stable Diffusion + ControlNet) when
        ``cfg["backend"] == "diffusion"`` is configured; otherwise falls back
        to the lightweight placeholder for offline / smoke-test scenarios.
        """
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
        """Lazy-instantiate and call the SD + ControlNet diffusion backend.

        If the ControlNet model ID changed since the last call (e.g. user
        switched from lineart to scribble in the GUI), rebuild the backend.
        """
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
        """
        Generate the stylized landscape image used for final cinemagraph rendering.

        Detailed TODO:
        1. Build a stylized scene prompt.
        2. Feed structural sketch + prompt into the selected generator.
        3. Return the synthesized stylized image.
        """
        # The prompt is built here so the method keeps the same shape as a real generator.
        _ = build_scene_prompt(text_prompt)

        image = self.preprocess_sketch(structural_sketch)
        pil_image = Image.fromarray(image)

        # Smooth rough sketch strokes and increase contrast for a stylized look.
        pil_image = pil_image.filter(ImageFilter.SMOOTH_MORE)
        pil_image = ImageEnhance.Contrast(pil_image).enhance(1.35)
        pil_image = ImageEnhance.Color(pil_image).enhance(1.25)

        array = np.asarray(pil_image).astype(np.float32)

        # Add a simple landscape-like tint while preserving the sketch layout.
        tint = np.array([1.05, 1.12, 0.92], dtype=np.float32)
        array = array * tint

        # Add a tiny deterministic warm variation so the placeholder is less flat.
        noise = self.rng.normal(loc=0.0, scale=3.0, size=array.shape)
        array = array + noise

        return np.clip(array, 0, 255).astype(np.uint8)

    def generate_reference(self, structural_sketch, text_prompt: str):
        """
        Generate the realistic reference image used to support motion estimation.

        Detailed TODO:
        1. Build a more realistic variant of the original prompt.
        2. Reuse the same structural sketch to preserve layout consistency.
        3. Return a realistic-looking image that aligns with the stylized result.
        """
        # Build the reference prompt for future real backends; the placeholder ignores it.
        _ = build_reference_prompt(text_prompt)

        image = self.preprocess_sketch(structural_sketch)
        pil_image = Image.fromarray(image)

        # Make a cleaner neutral reference that keeps the same layout.
        gray = ImageOps.grayscale(pil_image)
        gray = gray.filter(ImageFilter.SMOOTH)
        gray = ImageEnhance.Contrast(gray).enhance(1.15)

        gray_array = np.asarray(gray).astype(np.float32)

        # Map grayscale sketch values into a muted natural RGB palette.
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
        """
        Normalize or convert the structural sketch into the format required by the generator.

        Detailed TODO:
        1. Convert to the expected size and color format.
        2. Normalize pixel values if the backend requires it.
        3. Keep line strokes sharp enough for structural control.
        """
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
    """Convert image data to uint8 so PIL can process it safely."""
    array = np.asarray(image)

    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)
