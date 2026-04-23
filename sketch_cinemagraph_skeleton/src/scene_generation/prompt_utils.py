"""Helpers for prompt construction and style prompt management."""


def build_scene_prompt(base_prompt: str, style_prompt: str | None = None) -> str:
    """
    Merge content description and optional style description into one scene-generation prompt.

    Detailed TODO:
    1. Keep semantic content such as river, sea, or waterfall.
    2. Append style words like Monet, oil painting, or watercolor when provided.
    3. Return one prompt string suitable for stylized generation.
    """
    raise NotImplementedError


def build_reference_prompt(base_prompt: str) -> str:
    """
    Create a more realistic prompt variant for generating the reference image.

    Detailed TODO:
    1. Remove or weaken stylization words.
    2. Add realistic descriptors if needed.
    3. Return a prompt aimed at motion-friendly realistic output.
    """
    raise NotImplementedError


def parse_prompt_components(text_prompt: str) -> dict:
    """
    Split a prompt into semantic content and style-related components.

    Detailed TODO:
    1. Identify scene object/content words.
    2. Identify style-related words.
    3. Return a dict that can be reused by other prompt builders.
    """
    raise NotImplementedError
