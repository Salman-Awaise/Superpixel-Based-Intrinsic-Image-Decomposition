# keeping SLIC superpixel segmentation here
from skimage.segmentation import slic

from . import config


# running SLIC on an RGB image and returning the superpixel label map
def run_slic(img_rgb, n_segments=None, compactness=None):
    n_segments = config.N_SEGMENTS if n_segments is None else n_segments
    compactness = config.COMPACTNESS if compactness is None else compactness
    return slic(img_rgb, n_segments=n_segments, compactness=compactness, start_label=0)
