"""Sparse-to-dense propagation algorithms for building dense motion fields."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


_DEFAULT_K = 8
_IDW_POWER = 2.0
_EPS = 1e-6


def propagate_sparse_to_dense(constraints, mask):
    """K-nearest-neighbour inverse-distance weighting from sparse points into the mask."""
    flow = initialize_empty_flow(mask)

    points = np.asarray(constraints.get("points", []))
    vectors = np.asarray(constraints.get("vectors", []), dtype=np.float32)
    if points.size == 0 or vectors.size == 0:
        return flow

    mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
    mask_pixels = np.argwhere(mask_2d > 0)
    if mask_pixels.size == 0:
        return flow

    k = min(_DEFAULT_K, len(points))
    tree = cKDTree(points.astype(np.float32))
    dists, idx = tree.query(mask_pixels.astype(np.float32), k=k)

    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    weights = 1.0 / (dists ** _IDW_POWER + _EPS)
    weights /= weights.sum(axis=1, keepdims=True)

    propagated = (weights[:, :, None] * vectors[idx]).sum(axis=1)
    flow[mask_pixels[:, 0], mask_pixels[:, 1]] = propagated.astype(np.float32)
    return flow


def compute_distance_weights(points, query_point):
    """Normalised inverse-distance weights from a query pixel to every anchor point."""
    points = np.asarray(points, dtype=np.float32)
    query = np.asarray(query_point, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return np.zeros((0,), dtype=np.float64)

    dists = np.linalg.norm(points - query, axis=1)
    weights = 1.0 / (dists ** _IDW_POWER + _EPS)
    total = weights.sum()
    if total <= 0.0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def initialize_empty_flow(mask):
    """Allocate a zero dense motion field with the same H×W as `mask`."""
    array = np.asarray(mask)
    if array.ndim == 3:
        h, w = array.shape[:2]
    else:
        h, w = array.shape
    return np.zeros((h, w, 2), dtype=np.float32)
