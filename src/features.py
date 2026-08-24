# keeping Lab conversion and per-superpixel feature construction here
import numpy as np
from skimage import color

from . import config


# converting an RGB image to Lab and splitting the channels
def lab_channels(img_rgb):
    lab = color.rgb2lab(img_rgb).astype(np.float32)
    return lab[..., 0], lab[..., 1], lab[..., 2]


# building coordinate grids normalized to [-0.5, 0.5]
def normalized_grids(shape):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    x_norm = (xx / max(1, W - 1)) - 0.5
    y_norm = (yy / max(1, H - 1)) - 0.5
    return x_norm, y_norm


# masking low-lightness pixels to avoid background leakage
def lightness_mask(L, thresh=None):
    thresh = config.L_MASK_THRESH if thresh is None else thresh
    return L > thresh


# building the six-dimensional weighted training features per superpixel
def build_train_features(spx, L, a, b, valid, w_L=None, w_XY=None, w_Tex=None):
    w_L = config.W_L if w_L is None else w_L
    w_XY = config.W_XY if w_XY is None else w_XY
    w_Tex = config.W_TEX if w_Tex is None else w_Tex
    x_norm, y_norm = normalized_grids(spx.shape)
    spx_ids = np.unique(spx)
    feats = np.zeros((len(spx_ids), 6), dtype=np.float32)
    for i, sid in enumerate(spx_ids):
        m = (spx == sid) & valid
        if not np.any(m):
            # assigning neutral feature if superpixel is background
            feats[i, :] = 0.0
            continue
        # computing Lab-based color/texture and position features
        a_mean = a[m].mean()
        b_mean = b[m].mean()
        L_mean = (L[m].mean() - 50.0)   # centering L around mid-gray
        x_mean = x_norm[m].mean()
        y_mean = y_norm[m].mean()
        L_std = L[m].std()
        # assembling weighted feature vector
        feats[i, 0] = a_mean
        feats[i, 1] = b_mean
        feats[i, 2] = w_L * L_mean
        feats[i, 3] = w_XY * x_mean
        feats[i, 4] = w_XY * y_mean
        feats[i, 5] = w_Tex * L_std
    return feats, spx_ids


# building the three-dimensional test features per superpixel
def build_test_features(spx, L, a, b, valid, w_L=None):
    # the notebook weights lightness by a hardcoded 0.3 here rather than reusing W_L
    w_L = config.TEST_W_L if w_L is None else w_L
    spx_ids = np.unique(spx)
    feats = np.zeros((len(spx_ids), 3), dtype=np.float32)
    for i, sid in enumerate(spx_ids):
        m = (spx == sid) & valid
        if not np.any(m):
            feats[i, :] = 0.0
            continue
        feats[i, 0] = a[m].mean()
        feats[i, 1] = b[m].mean()
        feats[i, 2] = w_L * (L[m].mean() - 50.0)
    return feats, spx_ids


# expanding superpixel-level cluster labels back onto the pixel grid
def expand_to_pixel_map(spx, spx_ids, labels_sp):
    H, W = spx.shape
    group_map = np.zeros((H, W), dtype=np.int32)
    for i, sid in enumerate(spx_ids):
        group_map[spx == sid] = labels_sp[i]
    return group_map
