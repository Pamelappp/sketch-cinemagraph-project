"""Refine coarse fluid masks using image-based segmentation or boundary cues."""


def refine_fluid_mask(image, text_prompt: str):
    """
    Refine fluid-region boundaries from the generated image content.

    Detailed TODO:
    1. Run a segmentation backend or placeholder algorithm on the image.
    2. Use the text prompt to infer the target fluid category when possible.
    3. Return a refined mask with cleaner boundaries than the semantic mask.
    """
    raise NotImplementedError


def run_segmentation_backend(image, text_prompt: str):
    """
    Call a segmentation backend or placeholder algorithm to extract fluid regions.

    Detailed TODO:
    1. Leave a hook for Grounded-SAM or any later segmentation model.
    2. Allow a simpler fallback such as thresholding or GrabCut during early development.
    3. Return the raw extracted region mask.
    """
    raise NotImplementedError


def clean_refined_mask(mask):
    """
    Apply filtering or morphology to remove noise from the refined mask.

    Detailed TODO:
    1. Fill small holes.
    2. Remove isolated noisy pixels.
    3. Make region edges smoother for downstream warping.
    """
    raise NotImplementedError
