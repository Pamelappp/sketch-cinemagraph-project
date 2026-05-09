"""Frame warping utilities used to animate fluid regions from the dense motion field."""

import numpy as np
import cv2


def warp_frames(image, flow, mask, num_frames: int = 60):
    """
    Generate a frame sequence by warping the stylized image over time using the motion field.

    Args:
        image: H x W x 3 stylized image
        flow: H x W x 2 dense motion field
        mask: H x W binary / grayscale fluid mask
        num_frames: number of frames to generate

    Returns:
        frames: list of synthesized frames
    """
    frames = []

    for t in range(num_frames):
        frame = warp_single_frame(image, flow, mask, t, num_frames)
        frames.append(frame)

    return frames


def warp_single_frame(image, flow, mask, t: int, num_frames: int):
    """
    Warp one frame at time step t while keeping static background regions unchanged.

    Args:
        image: H x W x 3 stylized image
        flow: H x W x 2 dense motion field
        mask: H x W binary / grayscale fluid mask
        t: current frame index
        num_frames: total number of frames

    Returns:
        output_frame: one warped frame
    """
    h, w = mask.shape[:2]

    # 1. Compute displacement for current timestep
    displacement = build_loop_displacement(flow, t, num_frames)

    # 2. Build remap coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + displacement[:, :, 0]).astype(np.float32)
    map_y = (grid_y + displacement[:, :, 1]).astype(np.float32)

    # 3. Warp the full image
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # 4. Only replace pixels inside the fluid mask
    output_frame = image.copy()

    if mask.ndim == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    fluid_region = mask_2d > 0
    output_frame[fluid_region] = warped[fluid_region]

    return output_frame


def build_loop_displacement(flow, t: int, num_frames: int):
    """
    Convert the base motion field into a loop-friendly displacement field for a given frame index.

    Args:
        flow: H x W x 2 dense motion field
        t: current frame index
        num_frames: total number of frames

    Returns:
        displacement: H x W x 2 displacement field
    """
    # Use a sinusoidal factor so motion naturally comes back for looping
    alpha = np.sin(2 * np.pi * t / num_frames)

    # Scale the motion field by the temporal factor
    displacement = flow * alpha

    return displacement
