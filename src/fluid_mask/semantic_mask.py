"""Build a coarse semantic mask from user sketches and prompt hints.

When a reference image is available (photo mode or after scene generation),
the mask is expanded from the stroke seed region to cover the entire
visually similar area — e.g. the whole lake surface, not just the pixels
under the brush strokes.  The strokes *guide* motion direction; the image
colour determines *where* motion happens.
"""

from __future__ import annotations

import cv2
import numpy as np


_INK_THRESHOLD = 200
_STROKE_THRESHOLD = 250
_MIN_REGION_FRACTION = 0.001
_MIN_OVERLAP_PIXELS = 5

# Colour-distance threshold (in CIE-LAB ΔE) for the image-aware expansion.
# Pixels closer to the mean stroke colour than this are considered fluid.
_LAB_DISTANCE_THRESHOLD = 35.0


def build_semantic_mask(structural_sketch, motion_sketch, reference_image=None):
    """
    Create a preliminary fluid-region mask based on sketch-defined motion areas.

    If *reference_image* is supplied (the stylised landscape or uploaded photo)
    the mask is grown from the stroke positions to cover the full connected
    region of similar colour in the image — so the entire lake animates, not
    only the pixels under the drawn strokes.

    Pipeline:
    1. connected-component segmentation of the structural sketch produces
       closed candidate regions; motion-stroke overlap selects which regions
       are intended to animate.
    2. If no closed regions are recovered **and** a reference image is
       available, the colour-based expansion is used.
    3. Otherwise dilated motion strokes serve as a fallback.
    """
    structural_array = _to_uint8(structural_sketch)
    motion_array = _to_uint8(motion_sketch)
    motion_array = _resize_to(motion_array, structural_array.shape[:2])

    candidate_regions = extract_candidate_regions(structural_array)

    if candidate_regions.max() == 0:
        # No structural regions found — try colour-based expansion first.
        if reference_image is not None:
            ref = _to_uint8(reference_image)
            ref = _resize_to(ref, structural_array.shape[:2])
            expanded = _expand_mask_by_colour(ref, motion_array)
            if expanded is not None and int(expanded.sum()) > 0:
                return expanded
        return _stroke_fallback_mask(motion_array)

    mask = associate_motion_with_regions(candidate_regions, motion_array)

    if int(mask.sum()) == 0:
        if reference_image is not None:
            ref = _to_uint8(reference_image)
            ref = _resize_to(ref, structural_array.shape[:2])
            expanded = _expand_mask_by_colour(ref, motion_array)
            if expanded is not None and int(expanded.sum()) > 0:
                return expanded
        return _stroke_fallback_mask(motion_array)

    return mask


def extract_candidate_regions(structural_sketch):
    """
    Identify candidate semantic regions separated by ink strokes in the
    structural sketch.

    In a typical landscape sketch the horizon line divides the canvas into
    a sky region (top) and a water region (bottom). Both regions touch the
    image borders, so we **do not** exclude border-touching regions —
    that would discard the very regions we need. Instead we keep all
    regions above a minimum area. The downstream ``associate_motion_with_regions``
    function picks which of these candidates are actually fluid based on
    motion-stroke overlap.

    The only region that IS excluded is the single largest one when it
    covers more than 75 % of the canvas — that is almost certainly the
    background / sky and including it would mask the whole image.

    Returns a 2-D int32 label image with the same H x W as the input.
    Label 0 marks ink, too-small blobs, and the oversized background.
    """
    gray = _to_grayscale(structural_sketch)
    height, width = gray.shape
    total_pixels = height * width

    ink = (gray < _INK_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ink = cv2.dilate(ink, kernel, iterations=1)

    interior = (1 - ink).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        interior, connectivity=4
    )

    min_area = max(int(_MIN_REGION_FRACTION * total_pixels), 1)
    max_area = int(0.75 * total_pixels)  # skip oversized "background"

    # Find the single largest region — exclude it as likely sky / background.
    largest_label = -1
    largest_area = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_area = area
            largest_label = label_id

    output = np.zeros_like(labels, dtype=np.int32)
    next_label = 1

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # Skip the single largest region if it's very big (likely sky).
        if label_id == largest_label and largest_area > max_area:
            continue
        region_pixels = labels == label_id
        output[region_pixels] = next_label
        next_label += 1

    return output


def associate_motion_with_regions(candidate_regions, motion_sketch):
    """
    Decide which candidate regions are fluid-related based on motion-stroke coverage.
    """
    gray = _to_grayscale(motion_sketch)
    stroke = (gray < _STROKE_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    stroke = cv2.dilate(stroke, kernel, iterations=1)

    fluid_mask = np.zeros(candidate_regions.shape, dtype=np.uint8)
    if stroke.sum() == 0:
        return fluid_mask

    stroke_bool = stroke.astype(bool)
    label_ids = np.unique(candidate_regions)
    for label_id in label_ids:
        if label_id == 0:
            continue
        region = candidate_regions == label_id
        overlap = int(np.logical_and(region, stroke_bool).sum())
        if overlap >= _MIN_OVERLAP_PIXELS:
            fluid_mask[region] = 255
            continue

        ys, xs = np.where(stroke_bool)
        if ys.size == 0:
            continue
        cy = int(round(ys.mean()))
        cx = int(round(xs.mean()))
        if 0 <= cy < region.shape[0] and 0 <= cx < region.shape[1] and region[cy, cx]:
            fluid_mask[region] = 255

    return fluid_mask


def _expand_mask_by_colour(
    reference_image: np.ndarray,
    motion_sketch: np.ndarray,
    lab_threshold: float = _LAB_DISTANCE_THRESHOLD,
) -> np.ndarray | None:
    """
    Grow the fluid mask from stroke seeds to all visually-similar pixels in the
    reference image, so the entire lake / sky / smoke region animates.

    Algorithm:
    1. Extract stroke seed pixels from the motion sketch.
    2. Sample the reference image at those positions → build a colour model
       in CIE-LAB space with *separate* thresholds for lightness (L) and
       chrominance (AB). Water can vary widely in brightness (depth, foam,
       reflections) but its hue/saturation stays relatively consistent, so
       the L-channel gets a generous threshold while AB stays strict.
    3. Pixels matching the colour model are candidates.
    4. Keep only connected components that overlap with the original seeds.
    5. Morphological cleanup → final expanded mask.
    """
    if reference_image.ndim == 2:
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)

    gray = _to_grayscale(motion_sketch)
    stroke = (gray < _STROKE_THRESHOLD).astype(np.uint8)
    if stroke.sum() < 10:
        return None

    # Dilate strokes slightly so we sample a neighbourhood, not just edge pixels.
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    seed_region = cv2.dilate(stroke, seed_kernel, iterations=2)
    seed_bool = seed_region > 0

    # Convert to CIE-LAB for perceptually uniform distance.
    lab = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Build colour model from seed pixels.
    seed_colours = lab[seed_bool]
    if len(seed_colours) < 10:
        return None
    mean_lab = seed_colours.mean(axis=0)
    std_lab = seed_colours.std(axis=0)

    # Separate thresholds: generous for L (brightness varies a lot in water),
    # stricter for AB (chrominance / hue stays consistent).
    l_thresh = max(lab_threshold * 1.5, std_lab[0] * 2.5)  # brightness
    ab_thresh = lab_threshold + std_lab[1:].mean() * 0.8     # chrominance

    diff = lab - mean_lab[None, None, :]
    l_ok = np.abs(diff[..., 0]) < l_thresh
    ab_dist = np.sqrt(diff[..., 1] ** 2 + diff[..., 2] ** 2)
    ab_ok = ab_dist < ab_thresh

    candidate = ((l_ok & ab_ok).astype(np.uint8)) * 255

    # Morphological close to merge nearby blobs, then open to remove speckle.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_k)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_k)

    # Keep only components that overlap with the original stroke seeds.
    num_labels, labels = cv2.connectedComponents(candidate, connectivity=8)
    stroke_dilated = cv2.dilate(stroke, seed_kernel, iterations=1) > 0

    kept = np.zeros_like(candidate)
    for label_id in range(1, num_labels):
        region = labels == label_id
        if np.logical_and(region, stroke_dilated).any():
            kept[region] = 255

    if int(kept.sum()) == 0:
        return None

    # Final smooth.
    blurred = cv2.GaussianBlur(kept, (7, 7), 0)
    _, kept = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return kept


def _stroke_fallback_mask(motion_sketch):
    """Use dilated motion strokes themselves as the semantic mask."""
    gray = _to_grayscale(motion_sketch)
    stroke = (gray < _STROKE_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    expanded = cv2.dilate(stroke, kernel, iterations=2)
    return (expanded * 255).astype(np.uint8)


def _to_grayscale(image):
    array = _to_uint8(image)
    if array.ndim == 2:
        return array
    if array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


def _to_uint8(image):
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _resize_to(image, target_shape):
    if image.shape[:2] == tuple(target_shape):
        return image
    h, w = target_shape
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
