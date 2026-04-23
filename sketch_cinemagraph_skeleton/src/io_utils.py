"""Utility functions for file I/O, config loading, and saving intermediate results."""

from pathlib import Path
from typing import Any


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
    raise NotImplementedError


def load_user_input(input_cfg: dict[str, Any]):
    """
    Load structural sketch, motion sketch, and text prompt into a UserInput object.

    Detailed TODO:
    1. Read image paths for the structural and motion sketch from config.
    2. Read the prompt from either a text file or a literal config string.
    3. Load images with a consistent backend (for example cv2 or PIL).
    4. Convert image arrays into the format expected by later modules.
    5. Package the loaded data into a UserInput dataclass.
    """
    raise NotImplementedError


def save_image(image, output_path: str | Path) -> None:
    """
    Save an image array to disk for debugging or final outputs.

    Detailed TODO:
    1. Ensure the output directory exists.
    2. Convert the image into uint8 RGB/BGR format if needed.
    3. Save the image using a chosen backend.
    4. Preserve file extension from output_path.
    """
    raise NotImplementedError


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
    raise NotImplementedError
