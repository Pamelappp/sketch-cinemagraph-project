"""Build a coarse semantic mask from user sketches and prompt hints."""


def build_semantic_mask(structural_sketch, motion_sketch):
    """
    Create a preliminary fluid-region mask based on sketch-defined motion areas.

    Detailed TODO:
    1. Detect connected structural regions from the structural sketch.
    2. Detect where motion strokes are drawn.
    3. Mark structural regions overlapped by motion strokes as candidate fluid areas.
    4. Return a coarse mask before image-based refinement.
    """
    raise NotImplementedError


def extract_candidate_regions(structural_sketch):
    """
    Identify candidate semantic regions from the structural sketch layout.

    Detailed TODO:
    1. Convert sketch into binary form.
    2. Use contours or connected components to split enclosed regions.
    3. Return region proposals for later matching.
    """
    raise NotImplementedError


def associate_motion_with_regions(candidate_regions, motion_sketch):
    """
    Decide which regions are fluid-related by checking overlap with motion strokes.

    Detailed TODO:
    1. Detect motion stroke pixels.
    2. Compute overlap between strokes and region proposals.
    3. Keep only the regions touched by motion strokes.
    """
    raise NotImplementedError
