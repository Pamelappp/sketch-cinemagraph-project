"""Helpers for prompt construction and negative-prompt management."""

# ── Shared layout anchors injected into BOTH prompts ─────────────────────────
_LAYOUT_PHRASES = [
    "same composition as the input sketch",
    "preserve object positions from the line drawing",
    "visible small wooden boat on the left side",
    "small village houses aligned along the shoreline",
    "calm lake in the foreground",
    "clear horizontal shoreline",
    "sky above, land in the middle, water in the foreground",
]

# ── Base negative — quality / global artefacts ────────────────────────────────
_BASE_NEGATIVE = (
    "low quality, blurry, distorted, repeated texture, mosaic pattern, "
    "extra objects, messy composition, "
    "deformed architecture, stretched building, warped roofs, melted structure, "
    "unrealistic perspective, inconsistent composition, wrong layout"
)

# ── Artefacts common to BOTH modes ───────────────────────────────────────────
_COMMON_ARTEFACT_NEGATIVE = (
    "visible sketch lines, copied sketch strokes, glowing outlines, grey outline strokes, "
    "line drawing texture, text, letters, watermark, logo, signature, "
    "bridge, extra bridge, extra trees, forest blocking village, "
    "black platform, hard rectangular water, hard shoreline, "
    "giant moon, oversized moon, missing boat"
)

# ── Mode-specific extras ──────────────────────────────────────────────────────
_STYLIZED_EXTRA = "photorealistic, raw photo, DSLR photograph"

_REFERENCE_EXTRA = (
    "painting, illustration, cartoon, anime, stylized, painterly, brush strokes, "
    "impressionist, watercolor, oil painting, "
    "boat removed, houses shifted to the right, floating island, "
    "desert, sand dunes, dry land"
)

_WATER_NEGATIVE = (
    "desert, sand dunes, dry land, beach, arid, no water, "
    "missing lake, missing river, dry riverbed"
)
_BOAT_NEGATIVE = "missing boat, no boat"


def build_scene_prompt(base_prompt: str,
                       style_prompt: str | None = None,
                       scene_prior: str = "",
                       extra_positive_prompt: str = "") -> str:
    """Build the stylized-image prompt with layout anchors and optional style."""
    components  = parse_prompt_components(base_prompt)
    prior_clean = _clean_text(scene_prior) if scene_prior else ""

    parts = [components["content"]]
    if prior_clean:
        parts.append(prior_clean)
    parts.extend(_LAYOUT_PHRASES)
    parts.append("convert line drawing into natural painted forms")

    if style_prompt:
        parts.append(_clean_text(style_prompt))
    elif components["style"]:
        parts.append(components["style"])
    else:
        parts.append("high quality landscape")

    if extra_positive_prompt:
        parts.append(_clean_text(extra_positive_prompt))

    return _join_prompt_parts(parts)


def build_reference_prompt(base_prompt: str,
                            scene_prior: str = "",
                            style_prompt: str | None = None,
                            extra_positive_prompt: str = "") -> str:
    """Build the realistic-reference prompt with layout anchors and photo descriptors."""
    components  = parse_prompt_components(base_prompt)
    prior_clean = _clean_text(scene_prior) if scene_prior else ""

    parts = [components["content"]]
    if prior_clean:
        parts.append(prior_clean)
    parts.extend(_LAYOUT_PHRASES)
    parts += [
        "realistic landscape photograph",
        "photorealistic",
        "natural lighting",
        "DSLR photograph",
        "high detail",
        "clear scene structure",
    ]

    if style_prompt:
        parts.append(_clean_text(style_prompt))

    if extra_positive_prompt:
        parts.append(_clean_text(extra_positive_prompt))

    return _join_prompt_parts(parts)


def build_negative_prompt(base_prompt: str = "",
                           scene_prior: str = "",
                           extra_negative_prompt: str = "",
                           mode: str = "stylized") -> str:
    """
    Build a mode-specific negative prompt.

    mode='stylized'  → blocks photorealistic terms; allows painterly artefacts
    mode='reference' → blocks painterly terms; blocks layout-breaking artefacts
    """
    combined = (base_prompt + " " + scene_prior).lower()
    parts    = [_BASE_NEGATIVE, _COMMON_ARTEFACT_NEGATIVE]

    if mode == "stylized":
        parts.append(_STYLIZED_EXTRA)
    else:
        parts.append(_REFERENCE_EXTRA)

    if any(w in combined for w in ("water", "lake", "river", "sea", "ocean")):
        parts.append(_WATER_NEGATIVE)
    if "boat" in combined:
        parts.append(_BOAT_NEGATIVE)

    if extra_negative_prompt:
        parts.append(_clean_text(extra_negative_prompt))

    return _join_prompt_parts(parts)


def parse_prompt_components(text_prompt: str) -> dict:
    """Split a user prompt into semantic content and style-related components."""
    original = _clean_text(text_prompt)
    content  = original

    style_terms = [
        "watercolor", "oil painting", "ink painting", "digital painting",
        "anime", "cartoon", "sketch", "impressionist", "impressionism",
        "monet", "van gogh", "ghibli", "stylized", "painterly",
    ]

    detected_styles: list[str] = []
    lowered = original.lower()
    for term in style_terms:
        if term in lowered:
            detected_styles.append(term)
            content = _remove_phrase_case_insensitive(content, term)

    content = _clean_prompt_separators(content)
    if not content:
        content = original

    return {
        "content": content,
        "style":   _join_prompt_parts(detected_styles),
        "original": original,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _join_prompt_parts(parts: list[str]) -> str:
    cleaned = []
    for p in parts:
        c = _clean_text(p)
        if c and c not in cleaned:
            cleaned.append(c)
    return ", ".join(cleaned)


def _remove_phrase_case_insensitive(text: str, phrase: str) -> str:
    words        = text.split()
    phrase_words = phrase.split()
    result: list[str] = []
    index = 0
    while index < len(words):
        current = " ".join(words[index: index + len(phrase_words)]).lower()
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
