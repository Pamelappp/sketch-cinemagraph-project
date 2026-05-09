"""Entry point for the sketch-guided cinemagraph project."""

from src.io_utils import load_config, load_user_input
from src.pipeline import CinemagraphPipeline


def main() -> None:
    """
    Load configuration and user inputs, then run the full pipeline.

    Detailed TODO:
    1. Read the default config file from configs/default.yaml.
    2. Parse input paths for structural sketch, motion sketch, and prompt text.
    3. Create a UserInput object using load_user_input().
    4. Initialize CinemagraphPipeline with the loaded config.
    5. Call pipeline.run(user_input) to execute all four stages.
    6. Print or log the final output path for debugging.
    7. Optionally save intermediate visualizations when debug mode is enabled.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
