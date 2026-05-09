"""Build a coarse semantic mask from user sketches and prompt hints."""

from __future__ import annotations

import cv2
import numpy as np


_INK_THRESHOLD = 200
_STROKE_THRESHOLD = 250
_MIN_REGION_FRACTION = 0.001
_MIN_OVERLAP_PIXELS = 5


def build_semantic_mask(structural_sketch, motion_sketch):
    """
    Create a preliminary fluid-region mask based on sketch-defined motion areas.

    Pipeline: connected-component segmentation of the structural sketch produces
    closed candidate regions; motion-stroke overlap then selects which of those
    regions are intended to animate. If no closed regions are recovered, the
    dilated motion strokes themselves are returned as a fallback.
    """
    structural_array = _to_uint8(structural_sketch)
    motion_array = _to_uint8(motion_sketch)
    motion_array = _resize_to(motion_array, structural_array.shape[:2])

    candidate_regions = extract_candidate_regions(structural_array)

    if candidate_regions.max() == 0:
        return _stroke_fallback_mask(motion_array)

    mask = associate_motion_with_regions(candidate_regions, motion_array)

    if int(mask.sum()) == 0:
        return _stroke_fallback_mask(motion_array)

    return mask


def extract_candidate_regions(structural_sketch):
    """
    Identify candidate semantic regions enclosed by ink strokes in the structural sketch.

    Returns a 2-D int32 label image with the same H x W as the input. Label 0 marks
    background pixels (image border, ink, or regions too small to keep).
    """
    gray = _to_grayscale(structural_sketch)
    height, width = gray.shape

    ink = (gray < _INK_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ink = cv2.dilate(ink, kernel, iterations=1)

    interior = (1 - ink).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(interior, connectivity=4)

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    border_labels.add(0)

    min_area = max(int(_MIN_REGION_FRACTION * height * width), 1)
    output = np.zeros_like(labels, dtype=np.int32)
    next_label = 1

    for label_id in range(1, num_labels):
        if label_id in border_labels:
            continue
        region_pixels = labels == label_id
        if int(region_pixels.sum()) < min_area:
            continue
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
