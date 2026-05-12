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

    Pipeline:
    1. Compute zone_top from the topmost motion stroke.  Everything above this
       line is erased from the candidate interior map before connected-component
       analysis, so buildings (above the water zone) are never candidates and
       the sky/background never bleeds into the fluid region.
    2. Connected-component segmentation inside the zone produces the water-area
       strips enclosed by structural wave lines.
    3. Motion-stroke overlap confirms which regions are fluid.
    4. Morphological closing fills gaps between wave-line strips into a solid
       water-body mask.
    """
    structural_array = _to_uint8(structural_sketch)
    motion_array = _to_uint8(motion_sketch)
    motion_array = _resize_to(motion_array, structural_array.shape[:2])

    h = structural_array.shape[0]

    motion_gray = _to_grayscale(motion_array)
    motion_stroke_bin = (motion_gray < _STROKE_THRESHOLD).astype(np.uint8)
    motion_ys = np.argwhere(motion_stroke_bin > 0)[:, 0]

    if motion_ys.size == 0:
        return _stroke_fallback_mask(motion_array)

    # Start zone slightly above the topmost motion stroke so we capture the
    # full water body without including buildings or sky above.
    margin = max(h // 10, 10)
    zone_top = max(0, int(motion_ys.min()) - margin)

    candidate_regions = extract_candidate_regions(structural_array, zone_top=zone_top)

    if candidate_regions.max() == 0:
        return _stroke_fallback_mask(motion_array)

    mask = associate_motion_with_regions(candidate_regions, motion_array)

    if int(mask.sum()) == 0:
        return _stroke_fallback_mask(motion_array)

    # Fill gaps between individual wave-line strips to produce a solid water body.
    close_size = max(max(structural_array.shape[:2]) // 20, 5)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def extract_candidate_regions(structural_sketch, zone_top=0):
    """
    Identify candidate semantic regions inside the fluid zone of the structural sketch.

    Args:
        structural_sketch: the structural line drawing.
        zone_top: row index above which all pixels are treated as non-candidates.
            Pixels above this line are erased from the interior map so that the
            sky/land background region cannot bleed into the water candidates.
            Buildings that sit above zone_top are also excluded automatically.
    """
    gray = _to_grayscale(structural_sketch)
    height, width = gray.shape

    ink = (gray < _INK_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ink = cv2.dilate(ink, kernel, iterations=1)

    interior = (1 - ink).astype(np.uint8)

    # Erase everything above the zone — buildings and sky cease to exist as
    # candidates, so no border-exclusion heuristic is needed.
    if zone_top > 0:
        interior[:zone_top, :] = 0

    num_labels, labels = cv2.connectedComponents(interior, connectivity=4)

    min_area = max(int(_MIN_REGION_FRACTION * height * width), 1)
    output = np.zeros_like(labels, dtype=np.int32)
    next_label = 1

    for label_id in range(1, num_labels):
        region_pixels = labels == label_id
        if int(region_pixels.sum()) >= min_area:
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
