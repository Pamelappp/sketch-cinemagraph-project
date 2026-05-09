"""Evaluation metrics for motion quality and cinemagraph output quality."""


def compute_motion_smoothness(flow, mask):
    """
    Measure whether the dense motion field changes smoothly inside valid fluid regions.

    Detailed TODO:
    1. Compute local differences of neighboring motion vectors.
    2. Restrict calculation to pixels inside the mask.
    3. Return a scalar smoothness score.
    """
    raise NotImplementedError


def compute_loop_consistency(frames):
    """
    Measure how similar the final frame is to the first frame for looping quality.

    Detailed TODO:
    1. Compare frame 0 and frame -1.
    2. Use a simple metric such as MSE or SSIM.
    3. Return a scalar score where a better loop is easy to identify.
    """
    raise NotImplementedError


def compute_mask_leakage(frames, mask):
    """
    Estimate whether motion artifacts spill into background regions outside the mask.

    Detailed TODO:
    1. Compare background pixels over time.
    2. Measure how much change appears outside the valid fluid region.
    3. Return a leakage score for debugging synthesis quality.
    """
    raise NotImplementedError
