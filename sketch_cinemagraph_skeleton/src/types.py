"""Dataclasses and type containers used across the pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ImageArray = np.ndarray
MaskArray = np.ndarray
FlowArray = np.ndarray


@dataclass
class UserInput:
    """User-provided inputs loaded from the config file."""

    structural_sketch: ImageArray
    motion_sketch: ImageArray
    text_prompt: str
    fluid_prompt: str = ""  # optional Grounding-SAM query (e.g. "sea. water.")


@dataclass
class SceneOutput:
    """Images produced by the scene generation stage."""

    stylized_image: ImageArray
    realistic_reference: ImageArray | None = None


@dataclass
class MaskOutput:
    """Masks created and refined for the animated fluid region."""

    semantic_mask: MaskArray
    refined_mask: MaskArray
    final_fluid_mask: MaskArray


@dataclass
class MotionFieldOutput:
    """Motion constraints and dense flow used for animation."""

    sparse_constraints: dict[str, Any]
    dense_motion_field: FlowArray


@dataclass
class SynthesisOutput:
    """Final frames and exported cinemagraph file path."""

    frames: list[ImageArray]
    output_path: Path
