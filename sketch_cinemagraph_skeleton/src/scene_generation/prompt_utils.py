"""Helpers for prompt construction and style prompt management."""


def build_scene_prompt(base_prompt: str, style_prompt: str | None = None) -> str:
    """
    Merge content description and optional style description into one scene-generation prompt.

    Detailed TODO:
    1. Keep semantic content such as river, sea, or waterfall.
    2. Append style words like Monet, oil painting, or watercolor when provided.
    3. Return one prompt string suitable for stylized generation.
    """
    components = parse_prompt_components(base_prompt)

    prompt_parts = [components["content"]]

    # A provided style prompt should take priority over automatically detected style words.
    if style_prompt:
        prompt_parts.append(_clean_text(style_prompt))
    elif components["style"]:
        prompt_parts.append(components["style"])

    prompt_parts.append("detailed stylized landscape")

    return _join_prompt_parts(prompt_parts)


def build_reference_prompt(base_prompt: str) -> str:
    """
    Create a more realistic prompt variant for generating the reference image.

    Detailed TODO:
    1. Remove or weaken stylization words.
    2. Add realistic descriptors if needed.
    3. Return a prompt aimed at motion-friendly realistic output.
    """
    components = parse_prompt_components(base_prompt)

    # The reference should describe the same scene, but without strong art-style wording.
    prompt_parts = [
        components["content"],
        "realistic landscape photograph",
        "natural lighting",
        "clear scene structure",
    ]

    return _join_prompt_parts(prompt_parts)


def parse_prompt_components(text_prompt: str) -> dict:
    """
    Split a prompt into semantic content and style-related components.

    Detailed TODO:
    1. Identify scene object/content words.
    2. Identify style-related words.
    3. Return a dict that can be reused by other prompt builders.
    """
    original = _clean_text(text_prompt)
    content = original
    detected_styles: list[str] = []

    # This lightweight list is enough for a course-project prompt helper.
    style_terms = [
        "watercolor",
        "oil painting",
        "ink painting",
        "digital painting",
        "anime",
        "cartoon",
        "sketch",
        "impressionist",
        "impressionism",
        "monet",
        "van gogh",
        "ghibli",
        "stylized",
        "painterly",
    ]

    lowered = original.lower()

    for term in style_terms:
        if term in lowered:
            detected_styles.append(term)
            content = _remove_phrase_case_insensitive(content, term)

    content = _clean_prompt_separators(content)

    # If the prompt was only style words, keep the original so the result is never empty.
    if not content:
        content = original

    return {
        "content": content,
        "style": _join_prompt_parts(detected_styles),
        "original": original,
    }


def _clean_text(text: str) -> str:
    """Normalize whitespace in a prompt string."""
    return " ".join(str(text).strip().split())


def _join_prompt_parts(parts: list[str]) -> str:
    """Join prompt fragments while skipping empty values."""
    cleaned_parts = []
    for part in parts:
        cleaned = _clean_text(part)
        if cleaned and cleaned not in cleaned_parts:
            cleaned_parts.append(cleaned)

    return ", ".join(cleaned_parts)


def _remove_phrase_case_insensitive(text: str, phrase: str) -> str:
    """Remove one style phrase from text without needing regular expressions."""
    words = text.split()
    phrase_words = phrase.split()
    result: list[str] = []
    index = 0

    while index < len(words):
        current = " ".join(words[index : index + len(phrase_words)]).lower()
        if current.strip(" ,.;:") == phrase:
            index += len(phrase_words)
        else:
            result.append(words[index])
            index += 1

    return " ".join(result)


def _clean_prompt_separators(text: str) -> str:
    """Clean extra commas and spaces left after removing style terms."""
    cleaned = _clean_text(text)

    while ", ," in cleaned:
        cleaned = cleaned.replace(", ,", ",")

    return cleaned.strip(" ,.;:")
