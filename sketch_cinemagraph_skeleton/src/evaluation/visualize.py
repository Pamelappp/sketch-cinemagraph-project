"""Visualization helpers for debugging intermediate results in the pipeline."""

import cv2
import numpy as np


def visualize_mask(mask):
    """Render a 2D mask as an HxWx3 uint8 image (binary 0/1 is rescaled to 0/255)."""
    if mask.ndim == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    mask_uint8 = np.clip(mask_2d, 0, 255).astype(np.uint8)
    if mask_uint8.max() <= 1:
        mask_uint8 = mask_uint8 * 255

    return cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)


def save_debug_mask(path, mask, invert: bool = False):
    """Save a binary mask as a single-channel PNG; set invert=True to flip foreground."""
    import pathlib
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    binary = (array > 0).astype(np.uint8) * 255
    if invert:
        binary = 255 - binary
    cv2.imwrite(str(pathlib.Path(path)), binary)


def visualize_candidate_regions(candidate_regions):
    """Colour-code a connected-component label map into HxWx3 uint8 (label 0 stays black)."""
    vis = np.zeros((*candidate_regions.shape[:2], 3), dtype=np.uint8)
    for lid in range(1, int(candidate_regions.max()) + 1):
        colour = [(lid * 67) % 256, (lid * 131) % 256, (lid * 197) % 256]
        vis[candidate_regions == lid] = colour
    return vis


def visualize_motion_field(flow):
    """Render an HxWx2 flow field as a BGR HSV-based directional colour map."""
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=False)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
    hsv[:, :, 1] = 255

    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = mag_norm.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def visualize_pipeline_summary(scene_out, mask_out, motion_out):
    """Side-by-side panel: stylized image | final fluid mask | dense motion field."""
    stylized_image = scene_out["stylized_image"]
    final_mask = mask_out["final_fluid_mask"]
    dense_motion_field = motion_out["dense_motion_field"]

    mask_vis = visualize_mask(final_mask)
    flow_vis = visualize_motion_field(dense_motion_field)

    h, w = stylized_image.shape[:2]
    mask_vis = cv2.resize(mask_vis, (w, h))
    flow_vis = cv2.resize(flow_vis, (w, h))

    scene_panel = stylized_image.copy()
    mask_panel = mask_vis.copy()
    flow_panel = flow_vis.copy()

    cv2.putText(scene_panel, "Stylized Image", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(mask_panel, "Final Fluid Mask", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(flow_panel, "Dense Motion Field", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return np.concatenate([scene_panel, mask_panel, flow_panel], axis=1)
