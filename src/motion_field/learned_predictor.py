"""Optional learned motion predictor for the enhanced version of the project."""


class LearnedMotionPredictor:
    """Placeholder class for a future sketch-conditioned neural motion predictor."""

    def __init__(self, cfg: dict) -> None:
        """
        Initialize model configuration and any train/inference settings.

        Detailed TODO:
        1. Store network hyperparameters and checkpoint paths.
        2. Build a lightweight encoder-decoder or U-Net later.
        3. Keep the interface consistent with the heuristic version.
        """
        raise NotImplementedError

    def predict(self, reference_image, fluid_mask, motion_sketch):
        """
        Predict a dense motion field from reference image, mask, and motion sketch.

        Detailed TODO:
        1. Encode the three conditioning inputs.
        2. Run the model forward pass.
        3. Return an H x W x 2 dense motion field.
        """
        raise NotImplementedError

    def load_weights(self, checkpoint_path: str) -> None:
        """
        Load pre-trained model weights for inference or further training.

        Detailed TODO:
        1. Read checkpoint from disk.
        2. Restore model state.
        3. Set the model to evaluation mode if appropriate.
        """
        raise NotImplementedError

    def train_step(self, batch):
        """
        Run one training step for the learned motion predictor.

        Detailed TODO:
        1. Parse input batch into image, mask, sketch, and target motion field.
        2. Compute prediction.
        3. Compute loss.
        4. Backpropagate and update model weights.
        """
        raise NotImplementedError
