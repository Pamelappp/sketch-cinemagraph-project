"""Parse user motion sketches into machine-usable stroke representations."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


_STROKE_THRESHOLD = 250
_MIN_STROKE_PIXELS = 5
_DEFAULT_NUM_POINTS = 20
_SMOOTH_WINDOW = 5


def parse_motion_sketch(motion_sketch):
    """
    Extract motion strokes from the input motion sketch.

    The motion sketch follows the convention of the baseline paper: each
    stroke is drawn as a polyline shaded with a white-to-black gradient,
    where the white end marks the start of motion and the black end marks
    the destination. This function converts the raster image into a list of
    ordered, oriented point sequences ready for vectorisation.
    """
    intensity = _to_grayscale(motion_sketch)
    binary = (intensity < _STROKE_THRESHOLD).astype(np.uint8)
    if binary.sum() == 0:
        return []

    skeleton = skeletonize(binary.astype(bool))
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, num_components = ndimage.label(skeleton, structure=structure)

    strokes: List[np.ndarray] = []
    for component_id in range(1, num_components + 1):
        component = labels == component_id
        if int(component.sum()) < _MIN_STROKE_PIXELS:
            continue

        ordered = _trace_stroke(component, intensity)
        if ordered is None or len(ordered) < _MIN_STROKE_PIXELS:
            continue
        strokes.append(ordered)

    if not strokes:
        return []

    strokes = resample_strokes(strokes, num_points=_DEFAULT_NUM_POINTS)
    strokes = smooth_strokes(strokes)
    return [stroke.astype(np.int32) for stroke in strokes]


def resample_strokes(strokes, num_points: int = _DEFAULT_NUM_POINTS):
    """
    Resample each stroke to a fixed number of evenly spaced points.

    Mirrors the baseline paper, which resamples each user-drawn stroke to
    20 control points before downstream processing.
    """
    resampled: List[np.ndarray] = []
    for stroke in strokes:
        stroke = np.asarray(stroke, dtype=np.float32)
        if len(stroke) < 2:
            continue
        deltas = np.diff(stroke, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = cumulative[-1]
        if total_length <= 1e-6:
            continue

        targets = np.linspace(0.0, total_length, num_points)
        ys = np.interp(targets, cumulative, stroke[:, 0])
        xs = np.interp(targets, cumulative, stroke[:, 1])
        resampled.append(np.stack([ys, xs], axis=1))
    return resampled


def smooth_strokes(strokes):
    """
    Apply moving-average smoothing to each stroke trajectory.
    """
    smoothed: List[np.ndarray] = []
    kernel = np.ones(_SMOOTH_WINDOW, dtype=np.float32) / _SMOOTH_WINDOW
    for stroke in strokes:
        stroke = np.asarray(stroke, dtype=np.float32)
        if len(stroke) < _SMOOTH_WINDOW:
            smoothed.append(stroke)
            continue
        ys = _convolve_padded(stroke[:, 0], kernel)
        xs = _convolve_padded(stroke[:, 1], kernel)
        smoothed.append(np.stack([ys, xs], axis=1))
    return smoothed


def _trace_stroke(component_mask, intensity):
    """
    Walk the 1-pixel-wide skeleton of a stroke from one endpoint to the other.
    """
    pixel_set = {tuple(p) for p in np.argwhere(component_mask)}
    if not pixel_set:
        return None

    neighbours = _neighbour_counts(component_mask)
    endpoint_coords = [tuple(p) for p in np.argwhere((neighbours == 1) & component_mask)]

    if not endpoint_coords:
        endpoint_coords = _farthest_pair(np.argwhere(component_mask))

    start, end = _select_oriented_endpoints(endpoint_coords, intensity)
    if start is None:
        return None

    path = [start]
    visited = {start}
    current = start
    while current != end:
        next_pixel = None
        for candidate in _eight_neighbours(current):
            if candidate in pixel_set and candidate not in visited:
                next_pixel = candidate
                break
        if next_pixel is None:
            break
        path.append(next_pixel)
        visited.add(next_pixel)
        current = next_pixel

    if current != end and end in pixel_set and end not in visited:
        path.append(end)

    return np.array(path, dtype=np.float32)


def _neighbour_counts(mask):
    kernel = np.ones((3, 3), dtype=np.uint8)
    counts = cv2.filter2D(mask.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return counts - mask.astype(np.uint8)


def _eight_neighbours(point):
    y, x = point
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yield (y + dy, x + dx)


def _farthest_pair(pixels):
    if len(pixels) < 2:
        return [tuple(pixels[0]), tuple(pixels[0])] if len(pixels) == 1 else []
    sample = pixels if len(pixels) <= 256 else pixels[np.linspace(0, len(pixels) - 1, 256, dtype=int)]
    diffs = sample[:, None, :] - sample[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    a, b = np.unravel_index(np.argmax(dists), dists.shape)
    return [tuple(sample[a]), tuple(sample[b])]


def _select_oriented_endpoints(endpoints, intensity):
    if not endpoints:
        return None, None
    if len(endpoints) == 1:
        return endpoints[0], endpoints[0]

    pair = endpoints[:2]
    if len(endpoints) > 2:
        coords = np.array(endpoints)
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        a, b = np.unravel_index(np.argmax(dists), dists.shape)
        pair = [endpoints[a], endpoints[b]]

    p_start, p_end = pair
    intensity_start = float(intensity[p_start[0], p_start[1]])
    intensity_end = float(intensity[p_end[0], p_end[1]])

    if intensity_start >= intensity_end:
        return p_start, p_end
    return p_end, p_start


def _convolve_padded(signal, kernel):
    pad = len(kernel) // 2
    padded = np.pad(signal, pad, mode="edge")
    convolved = np.convolve(padded, kernel, mode="valid")
    return convolved[: len(signal)]


def _to_grayscale(image):
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return array
    if array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
