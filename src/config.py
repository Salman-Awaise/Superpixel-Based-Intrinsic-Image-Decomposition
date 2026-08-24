# keeping dataset locations, SLIC/K-Means settings and decomposition constants here
import os

# getting the project root from the location of this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# setting where the dataset tarball is downloaded and extracted
ROOT_DIR = os.path.join(PROJECT_ROOT, "data_mit_intrinsic")
URL_DATA = "http://people.csail.mit.edu/rgrosse/intrinsic/intrinsic-data.tar.gz"
TAR_NAME = "intrinsic-data.tar.gz"

# setting where figures are written when saving is enabled
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
# toggling figure saving, left off so the default run matches the notebook
SAVE_FIGURES = False

# listing the image extensions treated as candidate inputs
INCLUDE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff")
# listing filename keywords that mark a file as ground truth rather than input
EXCLUDE_KEYWORDS = ("refl", "reflect", "albedo", "shade", "shading", "illum", "light")

# setting SLIC superpixel parameters
N_SEGMENTS = 500
COMPACTNESS = 12.0

# setting K-Means parameters
K = 6
KMEANS_N_INIT = 10
RANDOM_STATE = 0

# setting feature weights for the training feature vector
W_L = 0.15      # lightness weight
W_XY = 0.25     # position weight
W_TEX = 0.10    # texture weight
# the remaining weight, roughly 0.50, sits implicitly on the a* and b* colour features

# weighting lightness in the test feature vector
# the notebook hardcodes 0.3 here rather than reusing W_L, and that is kept as-is
TEST_W_L = 0.3

# ignoring very dark background pixels by lightness
L_MASK_THRESH = 5.0

# setting bilateral filter parameters used for the smooth shading estimate
BILATERAL_ITERS = 2
BILATERAL_D = 0
SIGMA_COLOR = 0.1 * 255.0
SIGMA_SPACE = 9

# setting display-side constants for the reflectance image
REFLECTANCE_PERCENTILE = 99.0
GAMMA = 1 / 2.2

# setting the fixed lightness used when drawing cluster centre colour chips
CHIP_L = 70.0
CHIP_SIZE = 50

# setting the matplotlib display dpi used by the notebook
FIGURE_DPI = 120
