"""Optional learned motion predictor for the enhanced version of the project."""

from __future__ import annotations

from typing import Any, Dict


_NOT_IMPLEMENTED_MESSAGE = (
    "LearnedMotionPredictor is the optional enhancement path described in the "
    "proposal. The heuristic sparse-to-dense propagation pipeline is used by "
    "default; the learned predictor is not implemented in this version."
)


class LearnedMotionPredictor:
    """
    Interface placeholder for a sketch-conditioned neural motion predictor.

    The baseline paper estimates motion fields with a Latent Motion Diffusion
    Model that consumes a realistic landscape image, a fluid mask, and motion
    sketches. This class reserves the corresponding interface so the pipeline
    can later be swapped to a learned backend without changing call sites; the
    course deliverable uses the heuristic propagation module instead.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        self.cfg: Dict[str, Any] = dict(cfg) if cfg else {}
        self.checkpoint_path = self.cfg.get("checkpoint_path")
        self.device = self.cfg.get("device", "cpu")
        self._weights_loaded = False

    def predict(self, reference_image, fluid_mask, motion_sketch):
        """Predict a dense motion field from the three conditioning inputs."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pre-trained model weights from disk."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)

    def train_step(self, batch):
        """Run one training step on a batch of paired sketch/motion data."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)
