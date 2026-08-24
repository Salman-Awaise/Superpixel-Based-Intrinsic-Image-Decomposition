# keeping dataset download, extraction and image selection here
import glob
import os
import tarfile
import urllib.request

import cv2
import numpy as np
from skimage import color
from skimage.util import img_as_float

from . import config


# downloading and extracting the dataset, then returning the nested data folder
def ensure_dataset(root_dir=None):
    root_dir = root_dir or config.ROOT_DIR
    # specifying where to keep the tar and extracted data
    os.makedirs(root_dir, exist_ok=True)
    tar_path = os.path.join(root_dir, config.TAR_NAME)
    # downloading dataset if not already present
    if not os.path.isfile(tar_path):
        print("[info] downloading:", config.URL_DATA)
        urllib.request.urlretrieve(config.URL_DATA, tar_path)
        print("[info] saved:", tar_path)
    else:
        print("[info] already present:", tar_path)
    # extracting dataset contents from tar file
    print("[info] extracting:", tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=root_dir)
    print("[info] extracted to:", root_dir)
    # locating nested data folder inside extracted directory tree
    mit_data_dir = None
    for sub in os.listdir(root_dir):
        full = os.path.join(root_dir, sub)
        nested = os.path.join(full, "data")
        if os.path.isdir(nested):
            mit_data_dir = nested
            break
    # displaying path to the extracted dataset
    print("[info] MIT intrinsic 'data' folder:", mit_data_dir)
    return mit_data_dir


# detecting ground truth or illumination-related files by filename
def is_gt(fname):
    n = os.path.basename(fname).lower()
    return any(k in n for k in config.EXCLUDE_KEYWORDS)


# collecting the base RGB inputs, dropping ground-truth maps
def collect_rgb_paths(mit_data_dir):
    # verifying that the extracted dataset folder exists
    assert mit_data_dir is not None and os.path.isdir(mit_data_dir), \
        "Dataset 'data' folder not found."
    # collecting all image paths recursively from the dataset
    all_paths = []
    for ext in config.INCLUDE_EXTS:
        all_paths.extend(glob.glob(os.path.join(mit_data_dir, "**", f"*{ext}"), recursive=True))
    # filtering out ground truth or illumination images to keep only base RGB inputs
    rgb_paths = [p for p in all_paths if not is_gt(p)]
    rgb_paths.sort()
    # printing count and sample of collected RGB images
    print(f"[info] found {len(rgb_paths)} base RGB images.")
    print("[info] sample:", [os.path.basename(p) for p in rgb_paths[:8]])
    return rgb_paths


# loading an image as RGB float32 in [0, 1]
def load_rgb_image(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"Cannot read: {path}"
    return img_as_float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).astype(np.float32)


# computing mean chroma (colorfulness) of an image in Lab space
def mean_chroma_lab(path):
    arr = load_rgb_image(path)
    lab = color.rgb2lab(arr).astype(np.float32)
    a, b = lab[..., 1], lab[..., 2]
    return float(np.mean(np.sqrt(a * a + b * b)))


# ranking every image by chroma and returning the most colorful one
def select_colorful_image(rgb_paths, top_n=10):
    print("\n=== COLORFUL IMAGE SELECTION ===")
    scores = [(p, mean_chroma_lab(p)) for p in rgb_paths]
    scores.sort(key=lambda x: x[1], reverse=True)
    print("Top colorful images:")
    for p, s in scores[:top_n]:
        print(f"{os.path.basename(p):>24}  chroma≈{s:.3f}")
    # selecting the most colorful image as the test image
    colorful_test_path = scores[0][0]
    print(f"\n[selected colorful test image] {os.path.basename(colorful_test_path)}")
    return colorful_test_path, scores
