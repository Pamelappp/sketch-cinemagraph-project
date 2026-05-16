"""Synthetic and disk-backed datasets for the learned motion predictor."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    image_size: int = 256
    num_samples: int = 2000
    max_strokes: int = 6
    min_strokes: int = 3
    max_flow_magnitude: float = 6.0
    stroke_thickness: int = 3
    seed: int = 42


class SyntheticMotionDataset(Dataset):
    """Procedurally generated landscape-like training data."""

    def __init__(self, cfg: Optional[SyntheticConfig] = None) -> None:
        self.cfg = cfg or SyntheticConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.cfg.seed + idx)
        size = self.cfg.image_size

        image, mask = _make_landscape_scene(size, rng)
        flow = _make_smooth_flow(size, mask, self.cfg.max_flow_magnitude, rng)
        sketch = _flow_to_motion_sketch(
            flow,
            mask,
            num_strokes=int(rng.integers(self.cfg.min_strokes, self.cfg.max_strokes + 1)),
            rng=rng,
            thickness=self.cfg.stroke_thickness,
        )

        sample = _to_training_tensors(image, sketch, mask, flow)
        return sample


def _make_landscape_scene(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Render sky/ground/water layout; return (RGB image, fluid mask)."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    sky_height = int(rng.integers(int(size * 0.25), int(size * 0.55)))
    water_top = int(rng.integers(sky_height + 20, max(sky_height + 21, int(size * 0.85))))

    sky_color = np.array(
        [
            rng.integers(160, 220),
            rng.integers(180, 230),
            rng.integers(210, 255),
        ],
        dtype=np.uint8,
    )
    ground_color = np.array(
        [
            rng.integers(40, 110),
            rng.integers(60, 140),
            rng.integers(40, 110),
        ],
        dtype=np.uint8,
    )
    water_color = np.array(
        [
            rng.integers(20, 80),
            rng.integers(60, 130),
            rng.integers(120, 220),
        ],
        dtype=np.uint8,
    )

    image[:sky_height] = sky_color
    image[sky_height:water_top] = ground_color
    image[water_top:] = water_color

    noise = rng.normal(0.0, 6.0, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    mask[water_top:] = 255

    perturb = rng.integers(-5, 6, size)
    for x in range(size):
        boundary = max(0, min(size - 1, water_top + int(perturb[x])))
        mask[boundary:, x] = 255

    return image, mask


def _make_smooth_flow(
    size: int,
    mask: np.ndarray,
    max_magnitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Smooth flow field built from a sum of low-frequency sinusoids."""
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    norm_y = ys.astype(np.float32) / size
    norm_x = xs.astype(np.float32) / size

    flow = np.zeros((size, size, 2), dtype=np.float32)
    num_components = int(rng.integers(2, 5))
    for _ in range(num_components):
        kx = rng.uniform(0.5, 2.5)
        ky = rng.uniform(0.5, 2.5)
        phase_x = rng.uniform(0, 2 * math.pi)
        phase_y = rng.uniform(0, 2 * math.pi)
        amp_x = rng.uniform(-1.0, 1.0)
        amp_y = rng.uniform(-1.0, 1.0)
        flow[..., 0] += amp_x * np.sin(2 * math.pi * (kx * norm_x + ky * norm_y) + phase_x)
        flow[..., 1] += amp_y * np.sin(2 * math.pi * (kx * norm_x + ky * norm_y) + phase_y)

    # Bias toward a dominant horizontal direction (typical of water surfaces).
    dominant = rng.choice([-1.0, 1.0])
    flow[..., 0] += dominant * 0.6

    magnitude = np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-6
    target_max = rng.uniform(max_magnitude * 0.35, max_magnitude)
    flow = flow / magnitude.max() * target_max

    mask_factor = (mask > 0).astype(np.float32)[..., None]
    flow = flow * mask_factor
    return flow.astype(np.float32)


def _flow_to_motion_sketch(
    flow: np.ndarray,
    mask: np.ndarray,
    num_strokes: int,
    rng: np.random.Generator,
    thickness: int = 3,
) -> np.ndarray:
    """Render a few flow streamlines as white-to-black gradient strokes."""
    h, w = mask.shape
    sketch = np.full((h, w, 3), 255, dtype=np.uint8)
    valid = np.argwhere(mask > 0)
    if len(valid) == 0:
        return sketch

    step_size = 1.5
    max_steps = 60
    for _ in range(max(1, num_strokes)):
        seed_idx = rng.integers(0, len(valid))
        y0, x0 = valid[seed_idx]
        y, x = float(y0), float(x0)

        path: List[Tuple[float, float]] = [(y, x)]
        for _step in range(max_steps):
            iy, ix = int(round(y)), int(round(x))
            if iy < 0 or iy >= h or ix < 0 or ix >= w:
                break
            if mask[iy, ix] == 0:
                break
            vec = flow[iy, ix]
            magnitude = float(np.linalg.norm(vec))
            if magnitude < 1e-3:
                break
            dx = vec[0] / magnitude * step_size
            dy = vec[1] / magnitude * step_size
            x += dx
            y += dy
            path.append((y, x))

        if len(path) < 4:
            continue

        path_arr = np.array(path)
        for i in range(len(path_arr) - 1):
            ratio = i / max(len(path_arr) - 2, 1)
            grey = int(round(255 * (1.0 - ratio)))
            p0 = (int(round(path_arr[i, 1])), int(round(path_arr[i, 0])))
            p1 = (int(round(path_arr[i + 1, 1])), int(round(path_arr[i + 1, 0])))
            cv2.line(sketch, p0, p1, (grey, grey, grey), thickness, lineType=cv2.LINE_AA)

    return sketch


class LandscapeMotionDataset(Dataset):
    """Disk-backed dataset; one directory per sample with image.png/sketch.png/mask.png/flow.npy."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        flow_scale: float = 1.0,
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.flow_scale = flow_scale
        self.augment = augment
        self.samples = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not self.samples:
            raise FileNotFoundError(f"No sample directories found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]
        image = _read_rgb(sample_dir / "image.png")
        sketch = _read_rgb(sample_dir / "sketch.png")
        mask = _read_grayscale(sample_dir / "mask.png")
        flow = np.load(sample_dir / "flow.npy").astype(np.float32)

        image = _resize_rgb(image, self.image_size)
        sketch = _resize_rgb(sketch, self.image_size)
        mask = _resize_mask(mask, self.image_size)
        flow = _resize_flow(flow, self.image_size) * self.flow_scale

        if self.augment:
            image, sketch, mask, flow = _augment(image, sketch, mask, flow)

        return _to_training_tensors(image, sketch, mask, flow)


def _to_training_tensors(
    image: np.ndarray,
    sketch: np.ndarray,
    mask: np.ndarray,
    flow: np.ndarray,
) -> dict:
    image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    sketch_t = torch.from_numpy(sketch.astype(np.float32) / 255.0).permute(2, 0, 1)
    mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
    flow_t = torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1)

    inputs = torch.cat([image_t, sketch_t, mask_t], dim=0)
    return {
        "inputs": inputs,
        "flow": flow_t,
        "mask": mask_t,
    }


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_grayscale(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


def _resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def _resize_flow(flow: np.ndarray, size: int) -> np.ndarray:
    h, w = flow.shape[:2]
    if h == size and w == size:
        return flow
    scale_y = size / h
    scale_x = size / w
    flow_resized = cv2.resize(flow, (size, size), interpolation=cv2.INTER_LINEAR)
    flow_resized[..., 0] *= scale_x
    flow_resized[..., 1] *= scale_y
    return flow_resized.astype(np.float32)


def _augment(
    image: np.ndarray,
    sketch: np.ndarray,
    mask: np.ndarray,
    flow: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        sketch = np.ascontiguousarray(sketch[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
        flow = np.ascontiguousarray(flow[:, ::-1])
        flow[..., 0] *= -1.0
    return image, sketch, mask, flow
