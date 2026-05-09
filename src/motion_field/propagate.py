"""Sparse-to-dense propagation algorithms for building dense motion fields."""


def propagate_sparse_to_dense(constraints, mask):
    """
    Expand sparse motion constraints into a dense 2D motion field inside the fluid mask.

    Detailed TODO:
    1. Initialize an empty H x W x 2 flow array.
    2. For each pixel inside the fluid mask, compute influence from sparse constraints.
    3. Use distance-weighted interpolation or another heuristic propagation rule.
    4. Write the resulting vector to the dense flow map.
    5. Keep flow zero outside the mask.
    """
    raise NotImplementedError


def compute_distance_weights(points, query_point):
    """
    Compute distance-based weights for propagating sparse vectors to a target pixel.

    Detailed TODO:
    1. Measure distance from query_point to each sparse point.
    2. Convert distances to normalized weights.
    3. Return weights that sum to one.
    """
    raise NotImplementedError


def initialize_empty_flow(mask):
    """
    Create an empty dense motion field with the same spatial size as the fluid mask.

    Detailed TODO:
    1. Read height and width from the mask.
    2. Create a zero array with shape H x W x 2.
    3. Use float dtype so later interpolation is stable.
    """
    raise NotImplementedError
