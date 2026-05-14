"""Export synthesized frames into GIF or MP4 cinemagraph outputs."""

import os
import cv2
import imageio.v2 as imageio
import numpy as np


def export_cinemagraph(frames, output_path: str, fps: int = 20):
    """
    Save the looping frame sequence as a cinemagraph file on disk.

    Args:
        frames: list of H x W x C numpy arrays
        output_path: output file path, e.g. "result.gif" or "result.mp4"
        fps: frames per second

    Returns:
        output_path: final saved path
    """
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".gif":
        return export_gif(frames, output_path, fps)
    elif ext == ".mp4":
        return export_mp4(frames, output_path, fps)
    else:
        raise ValueError(f"Unsupported output format: {ext}. Use .gif or .mp4")


def _drop_duplicate_tail(frames, threshold: float = 1.0):
    """
    Remove the last frame when it is near-identical to the first frame.

    enforce_loop() copies frames[0] into frames[-1] to close the loop boundary.
    Saving that duplicate tail causes a visible pause at the loop point because the
    player renders the first frame twice in a row. Dropping it produces a seamless
    loop: the player jumps from the last saved frame directly back to the first.
    """
    if len(frames) < 2:
        return frames
    first = frames[0].astype(np.float32)
    last = frames[-1].astype(np.float32)
    mad = float(np.mean(np.abs(last - first)))
    if mad < threshold:
        print(f"[Export] Dropping duplicate tail frame (first↔last MAD={mad:.4f} < {threshold})")
        return frames[:-1]
    return frames


def export_gif(frames, output_path: str, fps: int = 20):
    """
    Export the cinemagraph as a GIF file.

    Args:
        frames: list of H x W x C numpy arrays
        output_path: output gif path
        fps: frames per second

    Returns:
        output_path: saved gif path
    """
    if len(frames) == 0:
        raise ValueError("frames is empty")

    frames = _drop_duplicate_tail(frames)
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    processed_frames = []
    for frame in frames:
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        processed_frames.append(frame_uint8)

    duration = 1.0 / fps
    imageio.mimsave(output_path, processed_frames, duration=duration, loop=0)

    return output_path


def export_mp4(frames, output_path: str, fps: int = 20):
    """
    Export the cinemagraph as an MP4 file.

    Args:
        frames: list of H x W x C numpy arrays
        output_path: output mp4 path
        fps: frames per second

    Returns:
        output_path: saved mp4 path
    """
    if len(frames) == 0:
        raise ValueError("frames is empty")

    frames = _drop_duplicate_tail(frames)
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    first_frame = np.clip(frames[0], 0, 255).astype(np.uint8)
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)

        # OpenCV expects BGR
        if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_uint8

        writer.write(frame_bgr)

    writer.release()
    return output_path
