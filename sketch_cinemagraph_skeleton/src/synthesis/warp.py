"""Frame warping via Euler integration for cinemagraph synthesis.

Replaces the original sinusoidal scaling with proper Euler integration:
each pixel's displacement accumulates by sampling the flow at its
*current* (displaced) position rather than always from the origin.
The loop is closed with a ping-pong strategy — the second half of the
sequence is the forward half played in reverse — giving a seamless loop
without needing the symmetric-splatting U-Net used in the paper.
"""

import numpy as np
import cv2


def warp_frames(image, flow, mask, num_frames: int = 120):
    """
    Generate a looping cinemagraph frame sequence via Euler integration.

    The sequence is structured as a ping-pong loop:
        forward  t = 0 .. half   (half + 1 frames)
        backward t = half-1 .. 1 (half - 1 frames)
    Total = 2 * half = num_frames  (works correctly for even num_frames).

    Args:
        image:      H x W x 3 stylized image (uint8)
        flow:       H x W x 2 dense motion field  (float32, channels = dx, dy)
        mask:       H x W fluid mask (uint8 or bool)
        num_frames: total frames to generate (use an even number)

    Returns:
        frames: list of H x W x 3 uint8 arrays
    """
    H, W = image.shape[:2]
    mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
    fluid = mask_2d > 0

    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    half = num_frames // 2

    # ── Forward Euler integration ──────────────────────────────────────
    forward_frames = []
    current_disp = np.zeros((H, W, 2), dtype=np.float32)

    for t in range(half + 1):
        frame = _apply_displacement(image, current_disp, fluid, grid_x, grid_y)
        forward_frames.append(frame)
        if t < half:
            current_disp = _euler_step(flow, current_disp, grid_x, grid_y, W, H)

    # ── Ping-pong: forward[0..half] + reverse[half-1..1] ──────────────
    # Lengths: (half + 1) + (half - 1) = 2 * half = num_frames
    frames = forward_frames + forward_frames[-2:0:-1]
    return frames


# ── Internal helpers ──────────────────────────────────────────────────

def _euler_step(flow, current_disp, grid_x, grid_y, W, H):
    """
    Advance displacement by one Euler step.

    Sample the flow at each pixel's *current* displaced position so that
    successive steps compound correctly (unlike the original sinusoidal
    scaling which always pulled from the same base field).
    """
    sample_x = np.clip(grid_x + current_disp[:, :, 0], 0, W - 1).astype(np.float32)
    sample_y = np.clip(grid_y + current_disp[:, :, 1], 0, H - 1).astype(np.float32)

    # Remap each flow channel at the current displaced coordinates
    fx = cv2.remap(
        flow[:, :, 0], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    fy = cv2.remap(
        flow[:, :, 1], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return current_disp + np.stack([fx, fy], axis=-1)


def _apply_displacement(image, disp, fluid_mask, grid_x, grid_y):
    """Warp image by disp; only replace fluid pixels, keep background static."""
    # Negative sign: cv2.remap is a backward warp (dst[y,x] = src[map_y,map_x]),
    # so subtracting disp makes content appear to travel WITH the flow direction.
    map_x = (grid_x - disp[:, :, 0]).astype(np.float32)
    map_y = (grid_y - disp[:, :, 1]).astype(np.float32)

    warped = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    frame = image.copy()
    frame[fluid_mask] = warped[fluid_mask]
    return frame
