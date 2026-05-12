"""Sparse-to-dense motion field propagation.

Primary method: RBF thin-plate spline interpolation via
scipy.interpolate.RBFInterpolator.  Compared with the original KNN
inverse-distance weighting (IDW), thin-plate splines give:
  - Globally smooth fields with no "star" artifacts around constraint points
  - Better behaviour in regions far from any sketch stroke
  - Direction-continuous transitions across the whole fluid mask

Fallback: if RBFInterpolator is unavailable (scipy < 1.7) or there are
too few constraint points, the code falls back to the original KNN IDW.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ── Tuning constants ──────────────────────────────────────────────────────────
_RBF_MIN_POINTS = 3      # minimum constraints needed for RBF; fewer → KNN
_KNN_K = 8
_IDW_POWER = 2.0
_EPS = 1e-6


# ── Public API ────────────────────────────────────────────────────────────────

def propagate_sparse_to_dense(constraints, mask):
    """
    Expand sparse motion constraints into a dense H x W x 2 flow field.

    Tries RBF thin-plate spline first; falls back to KNN IDW when the
    constraint set is too small or scipy is too old.

    Args:
        constraints: dict with
            "points"  (M, 2) int32  – (y, x) image coordinates
            "vectors" (M, 2) float32 – (dx, dy) flow vectors
        mask: H x W (or H x W x 1) uint8 fluid mask

    Returns:
        flow: H x W x 2 float32 dense motion field (dx, dy)
    """
    flow = initialize_empty_flow(mask)

    points = np.asarray(constraints.get("points", []))
    vectors = np.asarray(constraints.get("vectors", []), dtype=np.float32)
    if points.size == 0 or vectors.size == 0:
        return flow

    mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
    mask_pixels = np.argwhere(mask_2d > 0)
    if mask_pixels.size == 0:
        return flow

    if len(points) >= _RBF_MIN_POINTS:
        filled = _rbf_propagate(points, vectors, mask_pixels, flow)
        if filled:
            return flow

    # Fallback ─────────────────────────────────────────────────────────
    return _knn_propagate(points, vectors, mask_pixels, flow)


# ── RBF thin-plate spline propagation ────────────────────────────────────────

def _rbf_propagate(points, vectors, mask_pixels, flow):
    """
    Fill *flow* in-place using RBF thin-plate spline interpolation.

    Returns True on success, False if scipy is too old or interpolation fails.
    """
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:
        return False

    try:
        pts = points.astype(np.float64)   # (M, 2) in (y, x) order
        qpts = mask_pixels.astype(np.float64)  # (N, 2) in (y, x) order

        # Interpolate dx and dy channels independently
        rbf_dx = RBFInterpolator(pts, vectors[:, 0].astype(np.float64),
                                 kernel="thin_plate_spline", degree=1, smoothing=1.0)
        rbf_dy = RBFInterpolator(pts, vectors[:, 1].astype(np.float64),
                                 kernel="thin_plate_spline", degree=1, smoothing=1.0)

        dx = rbf_dx(qpts).astype(np.float32)
        dy = rbf_dy(qpts).astype(np.float32)

        flow[mask_pixels[:, 0], mask_pixels[:, 1], 0] = dx
        flow[mask_pixels[:, 0], mask_pixels[:, 1], 1] = dy
        return True

    except Exception:
        return False


# ── KNN IDW fallback ──────────────────────────────────────────────────────────

def _knn_propagate(points, vectors, mask_pixels, flow):
    """Fill *flow* in-place using k-nearest-neighbour inverse-distance weighting."""
    k = min(_KNN_K, len(points))
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


# ── Utilities ─────────────────────────────────────────────────────────────────

def compute_distance_weights(points, query_point):
    """
    Normalised inverse-distance weights from a query pixel to every anchor point.
    (Kept for backwards compatibility; not used by the primary propagation path.)
    """
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
    """Allocate a zero dense motion field with the same H x W as *mask*."""
    array = np.asarray(mask)
    h, w = array.shape[:2]
    return np.zeros((h, w, 2), dtype=np.float32)
