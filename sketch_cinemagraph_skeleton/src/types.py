"""Dataclasses and type containers used across the pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class UserInput:
    """
    Container for user-provided structural sketch, motion sketch, and text prompt.

    TODO meaning:
    - structural_sketch stores the scene-layout guidance image.
    - motion_sketch stores the user-drawn motion strokes.
    - text_prompt stores semantic and style description.
    """
    structural_sketch: object
    motion_sketch: object
    text_prompt: str


@dataclass
class SceneOutput:
    """
    Container for outputs from scene generation.

    TODO meaning:
    - stylized_image is the main image used for final animation.
    - realistic_reference is an auxiliary image for motion estimation.
    """
    stylized_image: object
    realistic_reference: object | None = None


@dataclass
class MaskOutput:
    """
    Container for semantic, refined, and final fluid masks.

    TODO meaning:
    - semantic_mask comes from user input and scene reasoning.
    - refined_mask comes from image-based boundary refinement.
    - final_fluid_mask is the mask actually used in motion and synthesis.
    """
    semantic_mask: object
    refined_mask: object
    final_fluid_mask: object


@dataclass
class MotionFieldOutput:
    """
    Container for sparse constraints and the final dense motion field.

    TODO meaning:
    - sparse_constraints records parsed points/vectors from strokes.
    - dense_motion_field is the H x W x 2 flow used to animate pixels.
    """
    sparse_constraints: dict
    dense_motion_field: object


@dataclass
class SynthesisOutput:
    """
    Container for synthesized frames and exported cinemagraph path.

    TODO meaning:
    - frames stores the generated looping frame list.
    - output_path points to the final GIF/MP4 export location.
    """
    frames: list
    output_path: Path
