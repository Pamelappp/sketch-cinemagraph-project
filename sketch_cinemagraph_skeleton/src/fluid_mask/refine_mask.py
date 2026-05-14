"""Refine coarse fluid masks using a Grounded-SAM image segmentation backend."""

from __future__ import annotations

import os
import re
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# Baseline-inspired fluid categories (waterfall, river, sea, sky, smoke).
# Extended with additional fluid-like categories (fire, lava, steam) for
# robustness beyond the baseline paper's scope.
FLUID_KEYWORDS: Tuple[str, ...] = (
    "water",
    "river",
    "lake",
    "sea",
    "ocean",
    "waterfall",
    "wave",
    "pond",
    "stream",
    "sky",
    "cloud",
    "mist",
    "fog",
    "smoke",
    "fire",
    "lava",
    "steam",
)

_DEFAULT_QUERY = "water. sky. cloud. smoke."

# Default Grounded-SAM checkpoints. Override via environment variables when
# stronger models are available locally.
_DINO_MODEL_ID = os.environ.get("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
_SAM_MODEL_ID = os.environ.get("SAM_MODEL", "facebook/sam-vit-base")

# Lower thresholds improve recall on generated/stylized images where confidence
# scores are typically lower than on natural photographs.
# Override at runtime: DINO_BOX_THRESHOLD=0.25 python main.py
_BOX_THRESHOLD = float(os.environ.get("DINO_BOX_THRESHOLD", "0.18"))
_TEXT_THRESHOLD = float(os.environ.get("DINO_TEXT_THRESHOLD", "0.15"))

_BACKEND_LOCK = threading.Lock()
_BACKEND: Optional["_GroundedSAMBackend"] = None


def refine_fluid_mask(image, text_prompt: str, debug_dir=None):
    """
    Refine fluid-region boundaries from the generated landscape image.

    The implementation follows the baseline paper: an open-vocabulary
    object detector (Grounding DINO) localises fluid categories from the
    text prompt; the resulting bounding boxes condition the Segment
    Anything Model, which produces precise instance masks. The masks are
    unioned and morphologically cleaned to obtain the refined fluid mask.
    """
    import pathlib as _pathlib
    raw_mask = run_segmentation_backend(image, text_prompt, debug_dir=debug_dir)
    cleaned = clean_refined_mask(raw_mask)
    if debug_dir is not None:
        d = _pathlib.Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "debug_refined_cleaned.png"), cleaned)
    return cleaned


def run_segmentation_backend(image, text_prompt: str, debug_dir=None):
    """
    Execute the Grounded-SAM backend (Grounding DINO + SAM) on the input
    landscape image, conditioned on the textual fluid query derived from
    the user prompt. Returns a binary uint8 mask of the union of all
    detected fluid regions.
    """
    import pathlib as _pathlib
    rgb = _to_uint8_rgb(image)
    height, width = rgb.shape[:2]
    pil_image = Image.fromarray(rgb)

    query = _build_dino_query(text_prompt)
    backend = _get_backend()

    boxes, scores = backend.detect_boxes(
        pil_image,
        query,
        box_threshold=_BOX_THRESHOLD,
        text_threshold=_TEXT_THRESHOLD,
    )
    if boxes is None or len(boxes) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    masks = backend.segment_with_boxes(pil_image, boxes)
    if masks is None or len(masks) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    fused = np.any(masks, axis=0).astype(np.uint8) * 255
    if fused.shape != (height, width):
        fused = cv2.resize(fused, (width, height), interpolation=cv2.INTER_NEAREST)
    if debug_dir is not None:
        d = _pathlib.Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "debug_refined_raw.png"), fused)
    return fused


def clean_refined_mask(mask):
    """
    Apply morphological filtering to remove speckle noise and tidy boundaries.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = (mask > 0).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= 20:
            cleaned[labels == label_id] = 255

    return cleaned


def _build_dino_query(text_prompt: str) -> str:
    """
    Build a Grounding DINO query string from the user prompt.

    Grounding DINO expects lowercase noun phrases separated by full stops.
    We extract the fluid-related terms from the prompt; if none are present
    a generic fluid vocabulary is used so the detector still has plausible
    targets to localise.
    """
    keywords: List[str] = []
    if text_prompt:
        tokens = re.findall(r"[a-zA-Z]+", text_prompt.lower())
        for token in tokens:
            if token in FLUID_KEYWORDS and token not in keywords:
                keywords.append(token)

    if not keywords:
        return _DEFAULT_QUERY
    return " ".join(f"{word}." for word in keywords)


def _get_backend() -> "_GroundedSAMBackend":
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                _BACKEND = _GroundedSAMBackend()
    return _BACKEND


class _GroundedSAMBackend:
    """Wraps Grounding DINO + SAM via Hugging Face transformers."""

    def __init__(self) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        self._torch = torch
        # HuggingFace's Grounding-DINO post-processing uses float64 internally,
        # which MPS does not support. CUDA is fine; on Apple Silicon we fall
        # back to CPU (one-shot inference, ~10–20 s per call). Override via
        # GROUNDED_SAM_DEVICE=mps if your transformers version is patched.
        forced = os.environ.get("GROUNDED_SAM_DEVICE")
        if forced:
            self.device = forced
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.dino_processor = AutoProcessor.from_pretrained(_DINO_MODEL_ID)
        self.dino_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(_DINO_MODEL_ID)
            .to(self.device)
            .eval()
        )

        self.sam_processor = SamProcessor.from_pretrained(_SAM_MODEL_ID)
        self.sam_model = SamModel.from_pretrained(_SAM_MODEL_ID).to(self.device).eval()

    def detect_boxes(
        self,
        pil_image: Image.Image,
        text_query: str,
        box_threshold: float = _BOX_THRESHOLD,
        text_threshold: float = _TEXT_THRESHOLD,
    ):
        """Run Grounding DINO and return (boxes_xyxy, scores) for the image."""
        torch = self._torch
        inputs = self.dino_processor(
            images=pil_image, text=text_query, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        try:
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )

        if not results:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        result = results[0]
        boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        return boxes, scores

    def segment_with_boxes(self, pil_image: Image.Image, boxes_xyxy: np.ndarray):
        """Run SAM conditioned on Grounding DINO boxes; return (N, H, W) bool array."""
        torch = self._torch
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return None

        boxes_list = boxes_xyxy.astype(np.float32).tolist()
        inputs = self.sam_processor(
            pil_image, input_boxes=[boxes_list], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs, multimask_output=False)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        if not masks:
            return None

        per_box_masks = masks[0].numpy()  # shape (num_boxes, num_predictions, H, W)
        if per_box_masks.ndim == 4:
            per_box_masks = per_box_masks[:, 0, :, :]
        return per_box_masks.astype(bool)


def _to_uint8_rgb(image):
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(_to_uint8(array), cv2.COLOR_GRAY2RGB)
    array = _to_uint8(array)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return array


def _to_uint8(image):
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)
