"""Frame warping for sketch-guided cinemagraphs.

Pipeline: ``image + flow + mask`` -> looping animated frames.

Loop construction — Symmetric Forward Splatting
------------------------------------------------
Implements the core rendering equation from the baseline paper (Sketch2Cinemagraph
§3.4 / Holynski et al. §4) in pixel space, using pure NumPy (no CUDA required):

1. **Forward splatting** pushes every source pixel to its flow-displaced
   destination with bilinear weight distribution. A softmax z-buffer
   resolves multi-pixel collisions at the same destination — the pixel
   with higher importance (closer to camera / faster flow) wins.

2. **Symmetric blending** produces seamless loops without a cross-dissolve:
   - Forward stream: splat source by  ``+t / T * F``
   - Backward stream: splat source by ``-(T-t) / T * F``
   - Blend: ``frame_t = (1 - t/T) * Forward_t + (t/T) * Backward_t``
   At ``t = 0`` the forward stream is the untouched source (no displacement);
   at ``t = T-1`` the backward stream is nearly the untouched source. The
   visible motion is always unidirectional — the two streams complement each
   other at the boundary rather than fighting.

3. **Hole-filling** with inward erosion of neighbour colours closes the
   small pixel-gaps that forward splatting inevitably creates.

This replaces the earlier backward-warp + cross-dissolve scheme that
created a visible "rewind" feel at the loop boundary.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


_BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
    "constant": cv2.BORDER_CONSTANT,
}


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def warp_frames(
    image: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    num_frames: int = 60,
    motion_scale: float = 1.0,
    border_mode: str = "reflect",
    blend_window: float = 0.30,       # ignored in symmetric mode (kept for compat)
    method: str = "symmetric_splat",   # "symmetric_splat" or "backward"
) -> List[np.ndarray]:
    """
    Animate the fluid region of *image* along *flow*.

    Args:
        image: H x W x 3 stylized landscape image (RGB uint8).
        flow: H x W x 2 motion field, channels (dx, dy), pixels per loop.
        mask: H x W binary fluid mask. Background pixels never move.
        num_frames: number of frames in the output loop.
        motion_scale: multiplier on *flow*; controls how far water travels
            over a complete loop.
        border_mode: cv2.remap border handling (used only by backward fallback).
        blend_window: dissolve fraction (backward fallback only).
        method: ``"symmetric_splat"`` (default, paper-aligned) or
                ``"backward"`` (legacy cross-dissolve fallback).
    """
    if num_frames < 2:
        return [image.copy()]

    if method == "backward":
        return _backward_warp_frames(
            image, flow, mask, num_frames, motion_scale, border_mode, blend_window,
        )

    return _symmetric_splat_frames(
        image, flow, mask, num_frames, motion_scale,
    )


def warp_single_frame(
    image: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    t: int,
    num_frames: int,
    motion_scale: float = 1.0,
    border_mode: str = "reflect",
) -> np.ndarray:
    """Convenience wrapper for rendering a single frame at index *t*."""
    frames = warp_frames(
        image, flow, mask, num_frames=num_frames,
        motion_scale=motion_scale, border_mode=border_mode,
    )
    return frames[max(0, min(t, len(frames) - 1))]


# ---------------------------------------------------------------------------
#  Symmetric forward splatting (paper approach)
# ---------------------------------------------------------------------------


def _symmetric_splat_frames(
    image: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    num_frames: int,
    motion_scale: float,
) -> List[np.ndarray]:
    """Symmetric splatting loop following Sketch2Cinemagraph §3.4."""
    h, w = image.shape[:2]
    flow_scaled = np.asarray(flow, dtype=np.float32) * float(motion_scale)

    mask_2d = mask if mask.ndim == 2 else mask[..., 0]
    fluid = mask_2d > 0

    # Pre-compute importance map — flow magnitude gives a reasonable
    # occlusion proxy (faster-moving foreground occludes slower background).
    importance = np.linalg.norm(flow_scaled, axis=-1).astype(np.float32)
    # Normalise into a moderate range so exp() doesn't explode.
    imp_max = importance.max()
    if imp_max > 1e-3:
        importance = importance / imp_max * 3.0  # range [0, 3]

    # Only splat within the fluid region — build masked copies of source.
    src_float = image.astype(np.float32)
    # Zero out non-fluid pixels in the source so splatting never moves them.
    fluid_src = src_float.copy()
    fluid_src[~fluid] = 0.0
    fluid_imp = importance.copy()
    fluid_imp[~fluid] = -1e6  # very low importance → won't win softmax

    frames: List[np.ndarray] = []

    for t in range(num_frames):
        phase = t / num_frames  # 0  ..  (N-1)/N

        # Forward stream: splat by +phase * F
        fwd_flow = flow_scaled * phase
        fwd_rgb, fwd_weight = _forward_splat(fluid_src, fwd_flow, fluid_imp, h, w)

        # Backward stream: splat by -(1-phase) * F
        bwd_flow = flow_scaled * -(1.0 - phase)
        bwd_rgb, bwd_weight = _forward_splat(fluid_src, bwd_flow, fluid_imp, h, w)

        # Symmetric blend weight  (= phase).
        # At t=0 (phase=0): 100 % forward (which is the untouched source).
        # At t≈N  (phase≈1): 100 % backward (which is nearly the source).
        alpha = phase

        blended = np.zeros((h, w, 3), dtype=np.float32)
        denom = np.zeros((h, w, 1), dtype=np.float32)

        fwd_w = (1.0 - alpha) * fwd_weight[..., None]
        bwd_w = alpha * bwd_weight[..., None]
        total_w = fwd_w + bwd_w

        valid = total_w[..., 0] > 1e-8
        blended[valid] = (fwd_w[valid] * fwd_rgb[valid] + bwd_w[valid] * bwd_rgb[valid]) / total_w[valid]

        # Fill holes where neither stream contributed — use source pixels.
        blended[~valid] = src_float[~valid]

        # Inpaint small remaining holes inside the fluid region.
        splatted_fluid = valid & fluid
        missing_fluid = (~valid) & fluid
        if missing_fluid.any():
            blended = _inpaint_holes(blended, splatted_fluid, missing_fluid)

        # Compose: background unchanged, fluid region from splatting.
        out = image.copy()
        out[fluid] = np.clip(blended[fluid], 0, 255).astype(np.uint8)
        frames.append(out)

    return frames


def _forward_splat(
    src: np.ndarray,
    flow: np.ndarray,
    importance: np.ndarray,
    h: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward-splat *src* pixels along *flow* with a softmax z-buffer.

    Returns:
        rgb: (H, W, 3) float32 — accumulated colour (un-normalised where weight > 0).
        weight: (H, W) float32 — total accumulated softmax weight per pixel.
    """
    c = src.shape[2] if src.ndim == 3 else 1

    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")

    dst_x = xs + flow[..., 0]
    dst_y = ys + flow[..., 1]

    # Clamp the exponent to avoid overflow; the relative ranking is what matters.
    safe_z = np.clip(importance, -20.0, 20.0)
    exp_z = np.exp(safe_z)

    # Bilinear splat corners.
    x0 = np.floor(dst_x).astype(np.int32)
    y0 = np.floor(dst_y).astype(np.int32)
    fx = dst_x - x0.astype(np.float32)
    fy = dst_y - y0.astype(np.float32)

    corners = [
        (x0,     y0,     (1 - fx) * (1 - fy)),
        (x0 + 1, y0,     fx * (1 - fy)),
        (x0,     y0 + 1, (1 - fx) * fy),
        (x0 + 1, y0 + 1, fx * fy),
    ]

    # Accumulation buffers — one *contiguous* 1-D array per channel so
    # ``np.add.at`` writes in-place (a non-contiguous .ravel() would copy).
    n_pixels = h * w
    rgb_flat = [np.zeros(n_pixels, dtype=np.float64) for _ in range(c)]
    w_flat = np.zeros(n_pixels, dtype=np.float64)

    # Flatten source channels once.
    src_64 = src.astype(np.float64)
    src_ch_flat = [src_64[..., ch].ravel().copy() for ch in range(c)]

    exp_z_flat = exp_z.ravel()

    for cx, cy, bilinear_w in corners:
        valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        flat_src = valid.ravel()  # indices into flattened source
        flat_dst = (cy[valid] * w + cx[valid]).astype(np.intp)
        sw = (bilinear_w[valid] * exp_z_flat[valid.ravel()]).astype(np.float64)

        for ch in range(c):
            np.add.at(rgb_flat[ch], flat_dst, sw * src_ch_flat[ch][valid.ravel()])
        np.add.at(w_flat, flat_dst, sw)

    # Normalise.
    nonzero = w_flat > 1e-10
    for ch in range(c):
        rgb_flat[ch][nonzero] /= w_flat[nonzero]
        rgb_flat[ch][~nonzero] = 0.0

    rgb_acc = np.stack([ch_arr.reshape(h, w) for ch_arr in rgb_flat], axis=-1)
    return rgb_acc.astype(np.float32), w_flat.reshape(h, w).astype(np.float32)


def _inpaint_holes(
    image: np.ndarray,
    valid: np.ndarray,
    holes: np.ndarray,
    iterations: int = 3,
) -> np.ndarray:
    """
    Fill small holes by iteratively averaging valid 4-connected neighbours.

    This is much faster than cv2.inpaint for the sparse 1–2 px gaps that
    forward splatting typically produces.
    """
    result = image.copy()
    remaining = holes.copy()

    for _ in range(iterations):
        if not remaining.any():
            break
        # Pad to handle borders.
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_valid = np.pad(valid | (~remaining), ((1, 1), (1, 1)), mode="constant",
                              constant_values=False)

        # 4-neighbour sum.
        neighbour_sum = (
            padded[:-2, 1:-1] +   # top
            padded[2:, 1:-1] +    # bottom
            padded[1:-1, :-2] +   # left
            padded[1:-1, 2:]      # right
        )
        neighbour_count = (
            padded_valid[:-2, 1:-1].astype(np.float32) +
            padded_valid[2:, 1:-1].astype(np.float32) +
            padded_valid[1:-1, :-2].astype(np.float32) +
            padded_valid[1:-1, 2:].astype(np.float32)
        )

        fillable = remaining & (neighbour_count > 0)
        if not fillable.any():
            break

        nc = neighbour_count[fillable][..., None]
        result[fillable] = neighbour_sum[fillable] / nc
        remaining[fillable] = False
        valid[fillable] = True

    return result


# ---------------------------------------------------------------------------
#  Legacy backward-warp fallback
# ---------------------------------------------------------------------------


def _backward_warp_frames(
    image: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    num_frames: int,
    motion_scale: float,
    border_mode: str,
    blend_window: float,
) -> List[np.ndarray]:
    """Original backward-warp + cubic-dissolve — kept as a fallback."""
    cv_border = _BORDER_MODES.get(str(border_mode).lower(), cv2.BORDER_REFLECT)
    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    flow_arr = np.asarray(flow, dtype=np.float32) * float(motion_scale)
    Dx = flow_arr[..., 0]
    Dy = flow_arr[..., 1]

    mask_2d = mask if mask.ndim == 2 else mask[..., 0]
    fluid_region = mask_2d > 0

    forward_frames: List[np.ndarray] = []
    for t in range(num_frames):
        phase = t / (num_frames - 1)
        map_x = (grid_x - phase * Dx).astype(np.float32)
        map_y = (grid_y - phase * Dy).astype(np.float32)
        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv_border,
        )
        out = image.copy()
        out[fluid_region] = warped[fluid_region]
        forward_frames.append(out)

    blend_window = float(np.clip(blend_window, 0.0, 0.5))
    W = int(round(num_frames * blend_window))
    if blend_window <= 0.0 or W <= 1:
        return forward_frames

    final = list(forward_frames)
    source = forward_frames[0].astype(np.float32)

    for i in range(W):
        late_idx = num_frames - W + i
        ratio = i / (W - 1)
        alpha = ratio ** 3
        late = forward_frames[late_idx].astype(np.float32)
        blended = (1.0 - alpha) * late + alpha * source
        final[late_idx] = np.clip(blended, 0, 255).astype(np.uint8)

    final[-1] = forward_frames[0].copy()
    return final
