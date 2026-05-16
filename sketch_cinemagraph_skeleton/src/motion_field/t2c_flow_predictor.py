"""T2C Direction Flow Predictor — inference wrapper."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_INPUT_SIZE = 512  # network trained at 512; output is 256 (half-res), rescaled by predict().


class T2CFlowPredictor:
    """Neural flow predictor using T2C motion-direction pretrained weights."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self._device_pref    = device
        self._lock  = threading.Lock()
        self._model = None
        self._device: Optional["torch.device"] = None  # type: ignore[name-defined]

    def predict(
        self,
        reference_image: np.ndarray,
        fluid_mask: np.ndarray,
        motion_sketch: np.ndarray,
    ) -> np.ndarray:
        """Return H×W×2 (dx, dy) dense flow, zeroed outside the fluid mask."""
        self._ensure_loaded()
        import torch

        orig_h, orig_w = reference_image.shape[:2]

        img_t = _to_tensor(_INPUT_SIZE, reference_image)
        skc_t = _to_tensor(_INPUT_SIZE, motion_sketch)
        x = torch.from_numpy(
            np.concatenate([img_t, skc_t], axis=0)[None]
        ).to(self._device)

        with torch.no_grad():
            flow_raw = self._model(x)

        flow_np = (
            flow_raw.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        )
        flow_full = _resize_flow(flow_np, orig_h, orig_w)

        mask_2d = fluid_mask[:, :, 0] if fluid_mask.ndim == 3 else fluid_mask
        flow_full *= (mask_2d > 0).astype(np.float32)[..., None]
        return flow_full

    def is_available(self) -> bool:
        if not self.checkpoint_path:
            return False
        if not Path(self.checkpoint_path).exists():
            return False
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
                raise RuntimeError(
                    "T2CFlowPredictor: checkpoint not found at "
                    f"{self.checkpoint_path!r}. "
                    "Set motion_field.t2c_checkpoint in configs/default.yaml."
                )

            import torch
            from src.motion_field.t2c_networks import load_t2c_direction_net

            dev = self._device_pref or _best_device()
            self._model  = load_t2c_direction_net(self.checkpoint_path, device=dev)
            self._device = torch.device(dev)


def _best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _to_tensor(size: int, img: np.ndarray) -> np.ndarray:
    """Resize H×W×3 uint8 → (3, size, size) float32 in [-1, 1]."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


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
