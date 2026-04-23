"""Scene generation module for stylized image and realistic reference creation."""

from src.types import SceneOutput


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
        raise NotImplementedError

    def generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        """
        Produce the main stylized image and an optional realistic reference image.

        Detailed TODO:
        1. Preprocess the incoming structural sketch.
        2. Call generate_stylized().
        3. Call generate_reference().
        4. Wrap both outputs into SceneOutput.
        """
        raise NotImplementedError

    def generate_stylized(self, structural_sketch, text_prompt: str):
        """
        Generate the stylized landscape image used for final cinemagraph rendering.

        Detailed TODO:
        1. Build a stylized scene prompt.
        2. Feed structural sketch + prompt into the selected generator.
        3. Return the synthesized stylized image.
        """
        raise NotImplementedError

    def generate_reference(self, structural_sketch, text_prompt: str):
        """
        Generate the realistic reference image used to support motion estimation.

        Detailed TODO:
        1. Build a more realistic variant of the original prompt.
        2. Reuse the same structural sketch to preserve layout consistency.
        3. Return a realistic-looking image that aligns with the stylized result.
        """
        raise NotImplementedError

    def preprocess_sketch(self, structural_sketch):
        """
        Normalize or convert the structural sketch into the format required by the generator.

        Detailed TODO:
        1. Convert to the expected size and color format.
        2. Normalize pixel values if the backend requires it.
        3. Keep line strokes sharp enough for structural control.
        """
        raise NotImplementedError
