"""Build a coarse semantic mask from user sketches.

Core logic (baseline paper §4):
  1. Connected-component segmentation of the structural sketch → candidate regions.
  2. Structural lines are removed from the motion sketch (black pixels → white).
  3. Motion-stroke overlap selects which candidate regions are fluid.

Engineering supplement (not in baseline):
  When a reference image is available, colour-based expansion via CIE-LAB
  similarity supplements the region-based mask when sketch lines are open/partial.
  The final mask is still constrained by the Grounded-SAM intersection in
  postprocess.combine_masks().
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np


_INK_THRESHOLD = 200
_STROKE_THRESHOLD = 250
_STRUCT_BLACK_THRESHOLD = 10   # near-pure-black pixels are structural ink
_MIN_REGION_FRACTION = 0.001
_MIN_OVERLAP_PIXELS = 5

# Colour-distance threshold (in CIE-LAB ΔE) for the image-aware expansion.
_LAB_DISTANCE_THRESHOLD = 35.0


def clean_motion_sketch(structural_sketch, motion_sketch):
    """
    Remove structural sketch lines from the motion sketch (baseline paper §4).

    The paper: structural pixels I(p) = (0,0,0) are identified, the structural
    mask is dilated 3×3, and those pixels are replaced with white in the motion
    sketch so only the white-to-black gradient motion strokes remain.
    """
    struct_gray = _to_grayscale(structural_sketch)
    struct_mask = (struct_gray < _STRUCT_BLACK_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    struct_mask = cv2.dilate(struct_mask, kernel, iterations=1)

    cleaned = motion_sketch.copy()
    if cleaned.ndim == 3:
        cleaned[struct_mask > 0] = 255
    else:
        cleaned[struct_mask > 0] = 255
    return cleaned


def build_semantic_mask(structural_sketch, motion_sketch,
                        reference_image=None, debug_dir=None):
    """
    Create a preliminary fluid-region mask based on sketch-defined motion areas.

    Baseline logic:
    1. Structural lines are removed from the motion sketch so gradient strokes
       are isolated (clean_motion_sketch).
    2. Connected-component segmentation of the structural sketch produces
       candidate regions.
    3. Motion-stroke overlap selects which candidate regions are fluid.

    Engineering supplement (not baseline):
    - If colour expansion covers significantly more area than the region-based
      mask, they are merged (handles open/partial sketch lines).
    - Empty mask is returned (with a warning) instead of a stroke-dilation
      fallback that would include structural lines in the semantic mask.
    """
    structural_array = _to_uint8(structural_sketch)
    motion_array = _to_uint8(motion_sketch)
    motion_array = _resize_to(motion_array, structural_array.shape[:2])

    # ── Step 1: remove structural lines from motion sketch ────────────────────
    cleaned_motion = clean_motion_sketch(structural_array, motion_array)

    if debug_dir is not None:
        d = pathlib.Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "debug_cleaned_motion_sketch.png"), cleaned_motion)
        _gray = _to_grayscale(cleaned_motion)
        _strokes = ((_gray < _STROKE_THRESHOLD).astype(np.uint8)) * 255
        cv2.imwrite(str(d / "debug_motion_stroke.png"), _strokes)

    # ── Step 2: extract candidate regions from structural sketch ──────────────
    candidate_regions = extract_candidate_regions(structural_array)

    if debug_dir is not None:
        _vis = np.zeros((*candidate_regions.shape, 3), dtype=np.uint8)
        for _lid in range(1, int(candidate_regions.max()) + 1):
            _colour = [(_lid * 67) % 256, (_lid * 131) % 256, (_lid * 197) % 256]
            _vis[candidate_regions == _lid] = _colour
        cv2.imwrite(str(pathlib.Path(debug_dir) / "debug_candidate_regions.png"), _vis)

    if candidate_regions.max() == 0:
        print("[Warning] build_semantic_mask: no closed candidate regions found in structural sketch.")
        if reference_image is not None:
            ref = _to_uint8(reference_image)
            ref = _resize_to(ref, structural_array.shape[:2])
            expanded = _expand_mask_by_colour(ref, cleaned_motion)
            if expanded is not None and int(expanded.sum()) > 0:
                return expanded
        print("[Warning] build_semantic_mask: returning empty semantic mask.")
        return np.zeros(structural_array.shape[:2], dtype=np.uint8)

    # ── Step 3: select regions overlapping cleaned motion strokes ─────────────
    mask = associate_motion_with_regions(candidate_regions, cleaned_motion)

    # ── Engineering supplement: colour expansion for open sketch lines ────────
    if reference_image is not None:
        ref = _to_uint8(reference_image)
        ref = _resize_to(ref, structural_array.shape[:2])
        expanded = _expand_mask_by_colour(ref, cleaned_motion)
        if expanded is not None:
            mask_area = int((mask > 0).sum())
            expanded_area = int((expanded > 0).sum())
            if mask_area == 0:
                mask = expanded
            elif expanded_area > mask_area * 1.5:
                mask = cv2.bitwise_or(mask, expanded)

    if int(mask.sum()) == 0:
        print("[Warning] build_semantic_mask: no candidate regions overlapped motion strokes — returning empty mask.")
        return np.zeros(structural_array.shape[:2], dtype=np.uint8)

    # ── Sanity check: diagnose obviously-wrong masks ──────────────────────────
    total = mask.size
    fg = int((mask > 0).sum())
    ratio = fg / total if total > 0 else 0.0
    if ratio > 0.8:
        print(
            f"[Warning] build_semantic_mask: foreground ratio {ratio:.1%} > 80 % — "
            "possible mask inversion or over-selection (all white-space regions selected)."
        )
    elif ratio < 0.001 and fg > 0:
        print(
            f"[Warning] build_semantic_mask: foreground ratio {ratio:.3%} < 0.1 % — "
            "mask is nearly empty; motion strokes may not overlap any candidate region."
        )

    return mask


def extract_candidate_regions(structural_sketch):
    """
    Identify candidate semantic regions separated by ink strokes in the
    structural sketch.

    All regions above a minimum area are kept as candidates.  The downstream
    ``associate_motion_with_regions`` function picks which are fluid based on
    cleaned motion-stroke overlap — a region with no strokes is not selected
    regardless of size.

    Returns a 2-D int32 label image. Label 0 marks ink pixels and tiny blobs.
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

    output = np.zeros_like(labels, dtype=np.int32)
    next_label = 1

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        output[labels == label_id] = next_label
        next_label += 1

    return output


def associate_motion_with_regions(candidate_regions, cleaned_motion_sketch):
    """
    Decide which candidate regions are fluid based on motion-stroke coverage.

    Expects a *cleaned* motion sketch (structural lines already removed via
    clean_motion_sketch) so only gradient motion strokes are detected.
    """
    gray = _to_grayscale(cleaned_motion_sketch)
    stroke = (gray < _STROKE_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    stroke = cv2.dilate(stroke, kernel, iterations=1)

    fluid_mask = np.zeros(candidate_regions.shape, dtype=np.uint8)
    if stroke.sum() == 0:
        return fluid_mask

    stroke_bool = stroke.astype(bool)
    for label_id in np.unique(candidate_regions):
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
    cleaned_motion_sketch: np.ndarray,
    lab_threshold: float = _LAB_DISTANCE_THRESHOLD,
) -> np.ndarray | None:
    """
    Grow the fluid mask from stroke seeds to visually-similar pixels in the
    reference image (engineering supplement, not baseline logic).

    Uses CIE-LAB colour similarity: separate thresholds for L (brightness)
    and AB (chrominance) so water/sky colours are matched robustly.
    """
    if reference_image.ndim == 2:
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)

    gray = _to_grayscale(cleaned_motion_sketch)
    stroke = (gray < _STROKE_THRESHOLD).astype(np.uint8)
    if stroke.sum() < 10:
        return None

    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    seed_region = cv2.dilate(stroke, seed_kernel, iterations=2)
    seed_bool = seed_region > 0

    lab = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB).astype(np.float32)

    seed_colours = lab[seed_bool]
    if len(seed_colours) < 10:
        return None
    mean_lab = seed_colours.mean(axis=0)
    std_lab = seed_colours.std(axis=0)

    l_thresh = max(lab_threshold * 1.5, std_lab[0] * 2.5)
    ab_thresh = lab_threshold + std_lab[1:].mean() * 0.8

    diff = lab - mean_lab[None, None, :]
    l_ok = np.abs(diff[..., 0]) < l_thresh
    ab_dist = np.sqrt(diff[..., 1] ** 2 + diff[..., 2] ** 2)
    ab_ok = ab_dist < ab_thresh

    candidate = ((l_ok & ab_ok).astype(np.uint8)) * 255

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_k)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_k)

    num_labels, labels = cv2.connectedComponents(candidate, connectivity=8)
    stroke_dilated = cv2.dilate(stroke, seed_kernel, iterations=1) > 0

    kept = np.zeros_like(candidate)
    for label_id in range(1, num_labels):
        region = labels == label_id
        if np.logical_and(region, stroke_dilated).any():
            kept[region] = 255

    if int(kept.sum()) == 0:
        return None

    blurred = cv2.GaussianBlur(kept, (7, 7), 0)
    _, kept = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return kept


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
