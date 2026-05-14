"""Frame warping via Euler integration for cinemagraph synthesis.

Replaces the original sinusoidal scaling with proper Euler integration:
each pixel's displacement accumulates by sampling the flow at its
*current* (displaced) position rather than always from the origin.
The loop is closed with a ping-pong strategy — the second half of the
sequence is the forward half played in reverse — giving a seamless loop
without needing the symmetric-splatting U-Net used in the paper.

Static foreground protection:
  Pass safe_moving_mask (= water minus dilation halo around holes) to restrict
  warping only to open water, and composite_alpha to restore the original image
  over protected zones after each warp step.  This prevents boat/shore texture
  from being dragged into the water region by bilinear remap interpolation.
"""

from pathlib import Path

import numpy as np
import cv2


def warp_frames(
    image,
    flow,
    mask,
    num_frames: int = 120,
    safe_moving_mask=None,
    composite_alpha=None,
    debug_dir=None,
):
    """
    Generate a looping cinemagraph frame sequence via Euler integration.

    The sequence is structured as a ping-pong loop:
        forward  t = 0 .. half   (half + 1 frames)
        backward t = half-1 .. 1 (half - 1 frames)
    Total = 2 * half = num_frames  (works correctly for even num_frames).

    Args:
        image:            H x W x 3 stylized image (uint8)
        flow:             H x W x 2 dense motion field (float32, channels = dx, dy)
        mask:             H x W fluid mask (uint8 or bool) — used if safe_moving_mask
                          is not provided
        num_frames:       total frames to generate (use an even number)
        safe_moving_mask: H x W uint8 — animatable pixels after excluding the
                          foreground protection zone.  If None, falls back to mask.
        composite_alpha:  H x W float32 — 1.0 = show original (protected object),
                          0.0 = show warped (water).  Applied after every warp step
                          to ensure static objects are never visually corrupted.
        debug_dir:        if set, saves pre/post composite frames at peak displacement
                          (debug_frame_030_before/after_composite.png).

    Returns:
        frames: list of H x W x 3 uint8 arrays
    """
    H, W = image.shape[:2]

    # Choose the animatable mask
    active_mask = safe_moving_mask if safe_moving_mask is not None else mask
    active_2d = active_mask[:, :, 0] if active_mask.ndim == 3 else active_mask
    fluid = active_2d > 0

    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    half = num_frames // 2
    _debug_dir = Path(debug_dir) if debug_dir is not None else None

    # ── Forward Euler integration ──────────────────────────────────────
    forward_frames = []
    current_disp = np.zeros((H, W, 2), dtype=np.float32)

    for t in range(half + 1):
        frame = _apply_displacement(
            image, current_disp, fluid, grid_x, grid_y,
            composite_alpha=composite_alpha,
        )
        forward_frames.append(frame)

        # Save pre/post composite debug at peak displacement (t == half)
        if _debug_dir is not None and t == half:
            frame_pre = _apply_displacement(
                image, current_disp, fluid, grid_x, grid_y, composite_alpha=None
            )
            cv2.imwrite(
                str(_debug_dir / "debug_frame_030_before_composite.png"),
                cv2.cvtColor(frame_pre, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(_debug_dir / "debug_frame_030_after_composite.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

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

    fx = cv2.remap(
        flow[:, :, 0], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    fy = cv2.remap(
        flow[:, :, 1], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return current_disp + np.stack([fx, fy], axis=-1)


def _apply_displacement(image, disp, fluid_mask, grid_x, grid_y, composite_alpha=None):
    """
    Warp image by disp; only replace fluid pixels; then composite original back
    over any protected foreground zone.

    The backward warp (negative sign on disp) means dst[y,x] = src[y-dy, x-dx],
    so content appears to travel WITH the flow direction.
    """
    map_x = (grid_x - disp[:, :, 0]).astype(np.float32)
    map_y = (grid_y - disp[:, :, 1]).astype(np.float32)

    warped = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    frame = image.copy()
    frame[fluid_mask] = warped[fluid_mask]

    # Composite the original static image back over the protection zone so that
    # bilinear-remap boundary artifacts near the boat are always overwritten.
    if composite_alpha is not None:
        alpha3 = composite_alpha[:, :, None]          # H x W x 1
        frame = (
            frame.astype(np.float32) * (1.0 - alpha3)
            + image.astype(np.float32) * alpha3
        )
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame
