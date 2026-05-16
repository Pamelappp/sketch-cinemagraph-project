"""Learned motion predictor backed by a U-Net trained on (sketch, mask, image) -> flow."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


_DEFAULT_INPUT_SIZE = 256


class LearnedMotionPredictor:
    """
    Sketch-conditioned neural motion predictor.

    The network is the lightweight ``MotionUNet`` defined in
    ``src.motion_field.networks``; it is trained by
    ``scripts/train_learned_predictor.py`` and saved as a PyTorch checkpoint.

    Configuration keys (all optional):

    - ``checkpoint_path``: path to the trained ``.pt`` file. If omitted or
      missing, ``predict`` raises ``RuntimeError`` with instructions for
      generating one.
    - ``device``: explicit torch device string (``"cuda"``, ``"mps"`` or
      ``"cpu"``). Defaults to the best available device.
    - ``input_size``: spatial resolution the network was trained at (default
      256). Inputs are resized to this resolution before inference and the
      predicted flow is rescaled back to the original size.
    - ``base_channels``: width multiplier of the U-Net (default 64). Must
      match the trained checkpoint.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        self.cfg: Dict[str, Any] = dict(cfg) if cfg else {}
        self.checkpoint_path: Optional[str] = self.cfg.get("checkpoint_path")
        self.input_size: int = int(self.cfg.get("input_size", _DEFAULT_INPUT_SIZE))
        self.base_channels: int = int(self.cfg.get("base_channels", 64))
        self._device_pref: Optional[str] = self.cfg.get("device")
        self._lock = threading.Lock()
        self._model = None
        self._device = None

    def predict(
        self,
        reference_image: np.ndarray,
        fluid_mask: np.ndarray,
        motion_sketch: np.ndarray,
    ) -> np.ndarray:
        """
        Predict a dense motion field from the three conditioning inputs.

        Args:
            reference_image: H x W x 3 uint8 RGB landscape image.
            fluid_mask: H x W binary mask (uint8 0/255 or float 0/1).
            motion_sketch: H x W x 3 uint8 RGB motion sketch with white-to-black
                gradient strokes.

        Returns:
            H x W x 2 float32 dense motion field with channels (dx, dy).
        """
        self._ensure_loaded()

        import torch  # local import keeps the module importable without torch

        original_h, original_w = reference_image.shape[:2]
        size = self.input_size

        image_resized = _resize_rgb(reference_image, size)
        sketch_resized = _resize_rgb(motion_sketch, size)
        mask_resized = _resize_mask(fluid_mask, size)

        image_t = torch.from_numpy(image_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        sketch_t = torch.from_numpy(sketch_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy((mask_resized > 0).astype(np.float32)).unsqueeze(0)
        inputs = torch.cat([image_t, sketch_t, mask_t], dim=0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            flow_low = self._model(inputs)

        flow_low = flow_low.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        flow_full = _resize_flow(flow_low, original_h, original_w)

        # Hard-enforce mask boundary so background pixels stay perfectly static.
        if fluid_mask.ndim == 3:
            mask_2d = fluid_mask[:, :, 0]
        else:
            mask_2d = fluid_mask
        mask_factor = (mask_2d > 0).astype(np.float32)[..., None]
        return flow_full * mask_factor

    def load_weights(self, checkpoint_path: str) -> None:
        """Load (or replace) the underlying model weights from a checkpoint file."""
        self.checkpoint_path = checkpoint_path
        with self._lock:
            self._model = None
            self._device = None
        self._ensure_loaded()

    def train_step(self, batch):
        """Training is performed by ``scripts/train_learned_predictor.py``."""
        raise NotImplementedError(
            "LearnedMotionPredictor.train_step is not exposed at runtime; "
            "use scripts/train_learned_predictor.py to train the U-Net."
        )

    # ------------------------------------------------------------------ utils

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            if not self.checkpoint_path:
                raise RuntimeError(
                    "LearnedMotionPredictor has no checkpoint configured. "
                    "Train one with scripts/train_learned_predictor.py and "
                    "pass cfg['checkpoint_path'] when constructing the predictor."
                )

            checkpoint_file = Path(self.checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found at {checkpoint_file}. "
                    "Run scripts/train_learned_predictor.py first."
                )

            import torch

            from src.motion_field.networks import MotionUNet, resolve_device

            device = resolve_device(self._device_pref)
            checkpoint = torch.load(checkpoint_file, map_location=device)

            base_channels = int(checkpoint.get("base_channels", self.base_channels))
            input_size = int(checkpoint.get("input_size", self.input_size))
            self.base_channels = base_channels
            self.input_size = input_size

            model = MotionUNet(in_channels=7, base_channels=base_channels)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            self._model = model
            self._device = device


def _resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def _resize_flow(flow: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = flow.shape[:2]
    if src_h == target_h and src_w == target_w:
        return flow
    scale_x = target_w / src_w
    scale_y = target_h / src_h
    resized = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= scale_x
    resized[..., 1] *= scale_y
    return resized.astype(np.float32)
