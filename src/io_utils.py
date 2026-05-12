"""Utility functions for file I/O, config loading, and saving intermediate results."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.types import UserInput


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load project configuration from YAML or JSON file.

    Detailed TODO:
    1. Normalize config_path to a Path object.
    2. Detect file type from extension (.yaml, .yml, or .json).
    3. Open the file safely and parse its content.
    4. Return a plain Python dict used by the pipeline.
    5. Add simple error handling for missing files or invalid syntax.
    """
    # Normalize the input so callers can pass either a string or a Path.
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    try:
        # Parse the file based on its extension.
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
    """
    Load structural sketch, motion sketch, and text prompt into a UserInput object.

    Detailed TODO:
    1. Read image paths for the structural and motion sketch from config.
    2. Read the prompt from either a text file or a literal config string.
    3. Load images with a consistent backend (for example cv2 or PIL).
    4. Convert image arrays into the format expected by later modules.
    5. Package the loaded data into a UserInput dataclass.
    """
    # Sketch files are required because later stages depend on both images.
    structural_path = _get_required_path(input_cfg, "structural_sketch_path")
    motion_path = _get_required_path(input_cfg, "motion_sketch_path")

    # The prompt can be provided directly or read from a text file.
    prompt_text = input_cfg.get("prompt")
    prompt_path = input_cfg.get("prompt_path")

    if prompt_text is None:
        if prompt_path is None:
            raise ValueError("Input config must include either 'prompt' or 'prompt_path'.")

        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        prompt_text = prompt_file.read_text(encoding="utf-8").strip()

    # Load sketches as RGB numpy arrays for a stable project-wide format.
    structural_sketch = _load_image(structural_path)
    motion_sketch = _load_image(motion_path)

    return UserInput(
        structural_sketch=structural_sketch,
        motion_sketch=motion_sketch,
        text_prompt=str(prompt_text).strip(),
    )


def save_image(image, output_path: str | Path) -> None:
    """
    Save an image array to disk for debugging or final outputs.

    Detailed TODO:
    1. Ensure the output directory exists.
    2. Convert the image into uint8 RGB/BGR format if needed.
    3. Save the image using a chosen backend.
    4. Preserve file extension from output_path.
    """
    path = Path(output_path)
    ensure_dir(path.parent)

    # PIL expects uint8 arrays for ordinary PNG/JPEG/GIF saving.
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
    """
    Save a dense motion field both as a raw .npy tensor and a colour-coded
    PNG visualisation. The PNG is written next to the .npy under the same
    stem, so e.g. ``flow.npy`` is paired with ``flow.png``.
    """
    path = Path(output_path)
    ensure_dir(path.parent)

    array = np.asarray(flow, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"Expected H x W x 2 flow, got shape {array.shape}")

    npy_path = path.with_suffix(".npy")
    png_path = path.with_suffix(".png")
    np.save(npy_path, array)

    # Lazy import to avoid pulling cv2 in for callers that only need .npy.
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
    """
    Create a directory if it does not exist and return the normalized Path.

    Detailed TODO:
    1. Convert input into Path.
    2. Create parents recursively when needed.
    3. Return the final Path object for reuse by callers.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _get_required_path(config: dict[str, Any], key: str) -> Path:
    """Read a required path value from a config dictionary."""
    value = config.get(key)
    if value is None:
        raise ValueError(f"Input config is missing required key: {key}")

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found for '{key}': {path}")

    return path


def _load_image(path: Path) -> np.ndarray:
    """Load an image file as a RGB numpy array."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _to_uint8_image(image: Any) -> np.ndarray:
    """Convert common image arrays into uint8 format for saving."""
    array = np.asarray(image)

    if array.ndim not in {2, 3}:
        raise ValueError("Image must be a 2D grayscale or 3D color array.")

    if array.dtype == np.uint8:
        return array

    # If values look normalized, scale them into the usual 0-255 image range.
    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)
