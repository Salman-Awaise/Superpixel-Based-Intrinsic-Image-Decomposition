# keeping the intrinsic image decomposition here
import cv2
import numpy as np

from . import config


# estimating smooth shading by bilateral filtering the lightness channel
def estimate_smooth_L(L):
    L01 = np.clip(L / 100.0, 0.0, 1.0)
    sm = L01.copy()
    for _ in range(config.BILATERAL_ITERS):
        sm = cv2.bilateralFilter(sm, d=config.BILATERAL_D,
                                 sigmaColor=config.SIGMA_COLOR,
                                 sigmaSpace=config.SIGMA_SPACE)
    return np.clip(sm * 100.0, 1e-6, 100.0)


# normalizing the lightness ratio per cluster to reduce global gain bias
def normalize_ratio_per_cluster(L, L_smooth, group_map, k=None):
    k = config.K if k is None else k
    ratio = L / np.maximum(L_smooth, 1e-6)
    ratio_adj = ratio.copy()
    for g in range(k):
        m = (group_map == g)
        if np.any(m):
            med = np.median(ratio[m])
            if med > 0:
                ratio_adj[m] = ratio[m] / med
    return ratio_adj


# deriving shading and reflectance for one image
def decompose(img_rgb, L, group_map, k=None):
    # estimating the smooth illumination component
    L_smooth = estimate_smooth_L(L)
    # removing per-cluster gain from the ratio
    ratio_adj = normalize_ratio_per_cluster(L, L_smooth, group_map, k)
    # deriving per-pixel shading L and capping to a safe range
    S_L = np.clip(L / np.maximum(ratio_adj, 1e-6), 1e-3, 100.0)
    # forming grayscale shading rgb
    S_gray = np.clip(S_L / 100.0, 1e-3, 1.0)
    S_rgb = np.stack([S_gray] * 3, axis=-1).astype(np.float32)
    # dividing the input by shading to get reflectance
    R_rgb = np.clip(img_rgb / np.maximum(S_rgb, 1e-6), 0.0, 10.0)
    # compressing reflectance dynamic range and applying gamma for visualization
    hi = np.percentile(R_rgb, config.REFLECTANCE_PERCENTILE)
    if hi > 0:
        R_rgb = np.clip(R_rgb / hi, 0.0, 1.0)
    R_rgb = R_rgb ** config.GAMMA
    return S_L, S_rgb, R_rgb
