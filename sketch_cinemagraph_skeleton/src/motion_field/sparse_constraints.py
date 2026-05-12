"""Convert parsed strokes into sparse vector constraints for motion propagation."""

from __future__ import annotations

import numpy as np


def build_sparse_constraints(strokes, mask):
    """
    Combine the per-stroke point/vector pairs into a single dictionary of
    sparse motion constraints, restricted to pixels inside the fluid mask.

    The returned dictionary is shaped for downstream propagation:
        {
            "points":  np.ndarray (M, 2) int32  -- (y, x) image coordinates
            "vectors": np.ndarray (M, 2) float32 -- (dx, dy) flow vectors
        }
    """
    if not strokes:
        return _empty_constraints()

    all_points = []
    all_vectors = []
    for stroke in strokes:
        points, vectors = stroke_to_vectors(stroke)
        if len(points) == 0:
            continue
        all_points.append(points)
        all_vectors.append(vectors)

    if not all_points:
        return _empty_constraints()

    points = np.concatenate(all_points, axis=0)
    vectors = np.concatenate(all_vectors, axis=0)

    points, vectors = filter_constraints_by_mask(points, vectors, mask)
    if len(points) == 0:
        return _empty_constraints()

    return {
        "points": points.astype(np.int32),
        "vectors": vectors.astype(np.float32),
    }


def stroke_to_vectors(stroke):
    """
    Convert a single ordered stroke polyline into per-point motion vectors.

    Centred finite differences are used along the interior of the stroke,
    falling back to forward/backward differences at the endpoints. The input
    is shape ``(N, 2)`` ``(y, x)``; the returned vectors are stored as
    ``(dx, dy)`` to align with the dense flow channel order used by
    ``cv2.remap``.
    """
    stroke = np.asarray(stroke, dtype=np.float32)
    if stroke.ndim != 2 or stroke.shape[0] < 2 or stroke.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.float32)

    n = stroke.shape[0]
    deltas = np.zeros_like(stroke)
    deltas[1:-1] = (stroke[2:] - stroke[:-2]) * 0.5
    deltas[0] = stroke[1] - stroke[0]
    deltas[-1] = stroke[-1] - stroke[-2]

    vectors = np.zeros_like(deltas)
    vectors[:, 0] = deltas[:, 1]
    vectors[:, 1] = deltas[:, 0]

    # Normalise to unit direction so RBF interpolates pure direction,
    # not magnitude-contaminated blends (long strokes would otherwise
    # dominate short ones and pull the field in the wrong direction).
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    nonzero = magnitudes[:, 0] > 1e-6
    vectors[nonzero] = vectors[nonzero] / magnitudes[nonzero]

    points = np.rint(stroke).astype(np.int32)
    return points, vectors.astype(np.float32)


def filter_constraints_by_mask(points, vectors, mask):
    """
    Drop constraints whose anchor points fall outside the fluid mask.
    """
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.float32)

    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    h, w = mask_array.shape[:2]
    ys = np.clip(points[:, 0], 0, h - 1)
    xs = np.clip(points[:, 1], 0, w - 1)
    inside = mask_array[ys, xs] > 0

    return points[inside], vectors[inside]


def _empty_constraints():
    return {
        "points": np.zeros((0, 2), dtype=np.int32),
        "vectors": np.zeros((0, 2), dtype=np.float32),
    }