"""Post-process and combine semantic and refined masks into a final fluid mask."""

from __future__ import annotations

import cv2
import numpy as np


_MIN_INTERSECTION_FRACTION = 0.05
_FINAL_MIN_AREA = 100


def combine_masks(semantic_mask, refined_mask):
    """
    Fuse the semantic mask (user intent) with the refined mask (image boundaries).

    The fusion rule mirrors the baseline paper: take the intersection so the
    final mask respects both user-specified structural constraints and the
    accurate boundaries from the image-based segmentation. When the
    intersection is degenerate (refined backend missed the fluid region) the
    semantic mask alone is used so the pipeline still produces motion.
    """
    semantic = _ensure_2d_uint8(semantic_mask)
    refined = _ensure_2d_uint8(refined_mask)

    if refined.shape != semantic.shape:
        h, w = semantic.shape
        refined = cv2.resize(refined, (w, h), interpolation=cv2.INTER_NEAREST)

    semantic_bool = semantic > 0
    refined_bool = refined > 0

    semantic_area = int(semantic_bool.sum())
    intersection = np.logical_and(semantic_bool, refined_bool)
    intersection_area = int(intersection.sum())

    if semantic_area == 0:
        # No user hint at all — fall back to whatever SAM found.
        fused = refined_bool
    elif intersection_area >= _MIN_INTERSECTION_FRACTION * semantic_area:
        # Refine boundaries within the user-indicated region.
        # The semantic mask is the hard upper bound: refined_mask can only
        # shrink it, never expand it to areas like sky or background.
        fused = intersection
    else:
        # Grounding-SAM missed the fluid region entirely — trust the
        # semantic mask derived from the user's motion sketch.
        fused = semantic_bool

    final = (fused.astype(np.uint8)) * 255
    final = smooth_mask_edges(final)
    final = remove_small_regions(final, min_area=_FINAL_MIN_AREA)
    return final


def smooth_mask_edges(mask):
    """
    Smooth jagged mask boundaries to reduce artifacts in later warping.
    """
    binary = _ensure_2d_uint8(mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return binary


def remove_small_regions(mask, min_area: int = 0):
    """
    Drop tiny disconnected mask regions whose area is below ``min_area``.
    """
    binary = _ensure_2d_uint8(mask)
    if min_area <= 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def _ensure_2d_uint8(mask):
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.dtype == np.bool_:
        return (array.astype(np.uint8)) * 255
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array