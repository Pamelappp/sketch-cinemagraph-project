"""Frame warping utilities used to animate fluid regions from the dense motion field."""


def warp_frames(image, flow, mask, num_frames: int = 60):
    """
    Generate a frame sequence by warping the stylized image over time using the motion field.

    Detailed TODO:
    1. Loop over each frame index from 0 to num_frames - 1.
    2. Build a displacement field for the current timestep.
    3. Warp the base image using the displacement.
    4. Only replace pixels inside the fluid mask.
    5. Return the list of synthesized frames.
    """
    raise NotImplementedError


def warp_single_frame(image, flow, mask, t: int, num_frames: int):
    """
    Warp one frame at time step t while keeping static background regions unchanged.

    Detailed TODO:
    1. Compute displacement for timestep t.
    2. Remap fluid-region pixels.
    3. Copy original pixels for background regions.
    4. Return a single output frame.
    """
    raise NotImplementedError


def build_loop_displacement(flow, t: int, num_frames: int):
    """
    Convert the base motion field into a loop-friendly displacement field for a given frame index.

    Detailed TODO:
    1. Compute a temporal coefficient such as a sinusoidal factor.
    2. Scale the motion field by that factor.
    3. Return the resulting displacement used for warping.
    """
    raise NotImplementedError
