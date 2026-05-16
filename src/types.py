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
    structural_sketch: ImageArray
    motion_sketch: ImageArray
    text_prompt: str


@dataclass
class SceneOutput:
    stylized_image: ImageArray
    realistic_reference: ImageArray | None = None


@dataclass
class MaskOutput:
    semantic_mask: MaskArray
    refined_mask: MaskArray
    final_fluid_mask: MaskArray


@dataclass
class MotionFieldOutput:
    sparse_constraints: dict[str, Any]
    dense_motion_field: FlowArray


@dataclass
class SynthesisOutput:
    frames: list[ImageArray]
    output_path: Path
