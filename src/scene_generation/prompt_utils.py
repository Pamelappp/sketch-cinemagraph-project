"""Helpers for prompt construction and style prompt management."""


def build_scene_prompt(base_prompt: str, style_prompt: str | None = None) -> str:
    """Stylized prompt: content + explicit style (else detected style words) + landscape tag."""
    components = parse_prompt_components(base_prompt)

    prompt_parts = [components["content"]]

    if style_prompt:
        prompt_parts.append(_clean_text(style_prompt))
    elif components["style"]:
        prompt_parts.append(components["style"])

    prompt_parts.append("detailed stylized landscape")

    return _join_prompt_parts(prompt_parts)


def build_reference_prompt(base_prompt: str) -> str:
    """Realistic-photo prompt: strip style words, add photo descriptors."""
    components = parse_prompt_components(base_prompt)

    prompt_parts = [
        components["content"],
        "realistic landscape photograph",
        "natural lighting",
        "clear scene structure",
    ]

    return _join_prompt_parts(prompt_parts)


def parse_prompt_components(text_prompt: str) -> dict:
    """Split prompt into {content, style, original} via a small style-term dictionary."""
    original = _clean_text(text_prompt)
    content = original
    detected_styles: list[str] = []

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

    # If the prompt was only style words, keep the original so we never return empty.
    if not content:
        content = original

    return {
        "content": content,
        "style": _join_prompt_parts(detected_styles),
        "original": original,
    }


def _clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _join_prompt_parts(parts: list[str]) -> str:
    cleaned_parts = []
    for part in parts:
        cleaned = _clean_text(part)
        if cleaned and cleaned not in cleaned_parts:
            cleaned_parts.append(cleaned)

    return ", ".join(cleaned_parts)


def _remove_phrase_case_insensitive(text: str, phrase: str) -> str:
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
    cleaned = _clean_text(text)

    while ", ," in cleaned:
        cleaned = cleaned.replace(", ,", ",")

    return cleaned.strip(" ,.;:")
