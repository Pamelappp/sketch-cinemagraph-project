"""Convert parsed strokes into sparse vector constraints for motion propagation."""


def build_sparse_constraints(strokes, mask):
    """
    Convert motion strokes into sparse directional constraints restricted to the fluid mask.

    Detailed TODO:
    1. Convert each stroke to local vectors.
    2. Collect all point/vector pairs from all strokes.
    3. Keep only valid constraints inside the fluid mask.
    4. Return a dict such as {points, vectors}.
    """
    raise NotImplementedError


def stroke_to_vectors(stroke):
    """
    Convert a single stroke polyline into point-wise motion vectors.

    Detailed TODO:
    1. Compute the direction from one sampled point to the next.
    2. Store the current point as the anchor and the difference as the vector.
    3. Return point/vector arrays for this stroke.
    """
    raise NotImplementedError


def filter_constraints_by_mask(points, vectors, mask):
    """
    Keep only sparse motion constraints that lie inside valid fluid regions.

    Detailed TODO:
    1. Check whether each point falls inside mask == 1.
    2. Discard points outside fluid regions.
    3. Return filtered points and vectors.
    """
    raise NotImplementedError
