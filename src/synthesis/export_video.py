"""Export synthesized frames into GIF or MP4 cinemagraph outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image


def export_cinemagraph(frames: List[np.ndarray], output_path: str, fps: int = 20) -> str:
    """Dispatch to export_gif/export_mp4 based on the file extension."""
    suffix = Path(output_path).suffix.lower()
    if suffix == ".gif":
        return export_gif(frames, output_path, fps=fps)
    if suffix in {".mp4", ".m4v", ".mov"}:
        return export_mp4(frames, output_path, fps=fps)
    raise ValueError(f"Unsupported cinemagraph extension '{suffix}'. Use .gif or .mp4.")


def export_gif(frames: List[np.ndarray], output_path: str, fps: int = 20) -> str:
    """Save frames as a looping animated GIF."""
    if not frames:
        raise ValueError("Cannot export an empty frame list.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pil_frames = [Image.fromarray(_ensure_uint8_rgb(frame)) for frame in frames]
    duration_ms = max(1, int(round(1000 / max(fps, 1))))

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=0,   # disposal=0 (no disposal) prevents inter-frame flicker.
    )
    return str(path)


def export_mp4(frames: List[np.ndarray], output_path: str, fps: int = 20) -> str:
    """Save frames as an MP4 (mp4v codec)."""
    if not frames:
        raise ValueError("Cannot export an empty frame list.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    first = _ensure_uint8_rgb(frames[0])
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    try:
        for frame in frames:
            rgb = _ensure_uint8_rgb(frame)
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    return str(path)


def _ensure_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return array
