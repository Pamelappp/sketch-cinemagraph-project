"""Utility functions for file I/O, config loading, and saving intermediate results."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.types import UserInput


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a .yaml/.yml/.json config file and return the top-level dict."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    try:
        with path.open("r", encoding="utf-8") as file:
            if suffix in {".yaml", ".yml"}:
                import yaml

                data = yaml.safe_load(file)
            elif suffix == ".json":
                data = json.load(file)
            else:
                raise ValueError(
                    f"Unsupported config file type '{suffix}'. Use .yaml, .yml, or .json."
                )
    except Exception as error:
        raise ValueError(f"Could not load config file {path}: {error}") from error

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a dictionary at the top level: {path}")

    return data


def load_user_input(input_cfg: dict[str, Any]) -> UserInput:
    """Build a UserInput from sketch paths and prompt (inline or via prompt_path)."""
    structural_path = _get_required_path(input_cfg, "structural_sketch_path")
    motion_path = _get_required_path(input_cfg, "motion_sketch_path")

    prompt_text = input_cfg.get("prompt")
    prompt_path = input_cfg.get("prompt_path")

    if prompt_text is None:
        if prompt_path is None:
            raise ValueError("Input config must include either 'prompt' or 'prompt_path'.")

        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        prompt_text = prompt_file.read_text(encoding="utf-8").strip()

    structural_sketch = _load_image(structural_path)
    motion_sketch = _load_image(motion_path)

    return UserInput(
        structural_sketch=structural_sketch,
        motion_sketch=motion_sketch,
        text_prompt=str(prompt_text).strip(),
    )


def save_image(image, output_path: str | Path) -> None:
    """Save an image array to disk via PIL, creating parent dirs as needed."""
    path = Path(output_path)
    ensure_dir(path.parent)
    image_array = _to_uint8_image(image)
    Image.fromarray(image_array).save(path)


def save_mask(mask, output_path: str | Path) -> None:
    """Save a binary mask as a 0/255 PNG."""
    path = Path(output_path)
    ensure_dir(path.parent)

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    binary = (array > 0).astype(np.uint8) * 255
    Image.fromarray(binary).save(path)


def save_motion_field(flow, output_path: str | Path) -> None:
    """Save flow as both .npy (raw) and .png (HSV colour-coded direction/magnitude)."""
    path = Path(output_path)
    ensure_dir(path.parent)

    array = np.asarray(flow, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"Expected H x W x 2 flow, got shape {array.shape}")

    npy_path = path.with_suffix(".npy")
    png_path = path.with_suffix(".png")
    np.save(npy_path, array)

    import cv2

    fx = array[..., 0]
    fy = array[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 180.0 / np.pi / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    if magnitude.max() > 0:
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    Image.fromarray(rgb).save(png_path)


def ensure_dir(path: str | Path) -> Path:
    """Make `path` (and parents) if missing; return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _get_required_path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Input config is missing required key: {key}")

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found for '{key}': {path}")

    return path


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _to_uint8_image(image: Any) -> np.ndarray:
    array = np.asarray(image)

    if array.ndim not in {2, 3}:
        raise ValueError("Image must be a 2D grayscale or 3D color array.")

    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)
