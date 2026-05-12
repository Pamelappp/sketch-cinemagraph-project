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

    # Optional fluid prompt — used by Grounding-SAM instead of deriving
    # keywords from the scene prompt (accepts "sea. water." style queries).
    fluid_text = str(input_cfg.get("fluid_prompt", "")).strip()
    fluid_prompt_path = input_cfg.get("fluid_prompt_path")
    if not fluid_text and fluid_prompt_path:
        fluid_file = Path(fluid_prompt_path)
        if fluid_file.exists():
            fluid_text = fluid_file.read_text(encoding="utf-8").strip()

    return UserInput(
        structural_sketch=structural_sketch,
        motion_sketch=motion_sketch,
        text_prompt=str(prompt_text).strip(),
        fluid_prompt=fluid_text,
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
    """
    Save a mask image to disk for inspection.

    Detailed TODO:
    1. Ensure the output directory exists.
    2. Convert the mask into a visible binary image (0 or 255).
    3. Save as PNG so boundaries remain clear.
    4. Optionally overlay the mask on the source image in a debug mode.
    """
    raise NotImplementedError


def save_motion_field(flow, output_path: str | Path) -> None:
    """
    Save a dense motion field in a suitable visualization or binary format.

    Detailed TODO:
    1. Save the raw flow tensor for later reuse (.npy or similar).
    2. Generate a color visualization for quick inspection.
    3. Ensure both save paths are organized under an intermediate folder.
    4. Keep naming consistent with the corresponding input sample.
    """
    raise NotImplementedError


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
