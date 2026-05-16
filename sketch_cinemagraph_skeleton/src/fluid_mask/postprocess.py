"""Post-process and combine semantic and refined masks into a final fluid mask."""

from __future__ import annotations

import cv2
import numpy as np


_MIN_INTERSECTION_FRACTION = 0.05
_FINAL_MIN_AREA = 20


def combine_masks(semantic_mask, refined_mask, debug_dir=None):
    """Return the intersection of semantic and refined masks, cleaned. Warns if empty."""
    semantic = _ensure_2d_uint8(semantic_mask)
    refined = _ensure_2d_uint8(refined_mask)

    if refined.shape != semantic.shape:
        h, w = semantic.shape
        refined = cv2.resize(refined, (w, h), interpolation=cv2.INTER_NEAREST)

    semantic_bool = semantic > 0
    refined_bool = refined > 0

    semantic_area = int(semantic_bool.sum())
    refined_area = int(refined_bool.sum())
    intersection = np.logical_and(semantic_bool, refined_bool)
    intersection_area = int(intersection.sum())

    if semantic_area == 0:
        print("[Warning] combine_masks: semantic mask is empty — no candidate regions matched motion strokes.")
    if refined_area == 0:
        print("[Warning] combine_masks: refined mask is empty — Grounded-SAM detected no fluid regions.")
    if semantic_area > 0 and refined_area > 0 and intersection_area < _MIN_INTERSECTION_FRACTION * semantic_area:
        print(
            f"[Warning] combine_masks: intersection ({intersection_area} px) is very small relative to "
            f"semantic ({semantic_area} px). Consider lowering DINO thresholds or checking the text query."
        )

    fused = intersection
    final = (fused.astype(np.uint8)) * 255
    final = smooth_mask_edges(final)
    final = remove_small_regions(final, min_area=_FINAL_MIN_AREA)

    if debug_dir is not None:
        import pathlib
        d = pathlib.Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "debug_intersection.png"), (intersection.astype(np.uint8)) * 255)
        cv2.imwrite(str(d / "debug_final.png"), final)

    return final


def smooth_mask_edges(mask):
    """Close-only morphology + Gaussian → re-threshold. No open, to keep rivers thin."""
    binary = _ensure_2d_uint8(mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    blurred = cv2.GaussianBlur(binary, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return binary


def remove_small_regions(mask, min_area: int = 0):
    """Drop disconnected components smaller than min_area."""
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
