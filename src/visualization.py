# keeping every plot from the notebook here
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.io import imread

from . import config

# adjusting display dpi for better quality
plt.rcParams["figure.dpi"] = config.FIGURE_DPI


# saving the current figure when enabled, then showing it
def _finish(name):
    if config.SAVE_FIGURES:
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        out_path = os.path.join(config.FIGURES_DIR, name)
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        print("[info] saved figure:", out_path)
    plt.show()


# displaying random image samples from the dataset
def show_dataset_samples(paths, n=6):
    # checking if image list is empty
    if not paths:
        print("[info] no images to show.")
        return
    # selecting random subset of images to visualize
    n = min(n, len(paths))
    picks = random.sample(paths, n)
    # setting up grid layout for displaying images
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 4 * rows))
    # looping through selected images and displaying each one
    for i, p in enumerate(picks):
        img = imread(p)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(os.path.basename(p))
        plt.axis("off")
    plt.tight_layout()
    _finish("dataset_samples.png")


# showing the training image next to its SLIC superpixels
def show_image_and_superpixels(img_rgb, spx, label="train"):
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title(f"{label.capitalize()} image (RGB)")
    plt.axis("off")
    _finish(f"{label}_input.png")
    plt.figure(figsize=(5, 5))
    plt.imshow(spx, cmap="nipy_spectral")
    plt.title(f"SLIC superpixels ({label})")
    plt.axis("off")
    _finish(f"{label}_superpixels.png")


# colouring a cluster map for display
def group_visualization(group_map):
    num_groups = int(group_map.max() + 1)
    # using pyplot.get_cmap, the supported replacement for the removed plt.cm.get_cmap
    cmap = plt.get_cmap("tab20", max(1, num_groups))
    group_norm = group_map / max(1, num_groups - 1) if num_groups > 1 else group_map
    return cmap(group_norm)[..., :3]


# showing input, clusters, shading and reflectance in a 2x2 grid
def show_decomposition(img_rgb, group_map, S_rgb, R_rgb, suptitle, name):
    group_viz = group_visualization(group_map)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 2, 1); ax.imshow(img_rgb); ax.set_title("Input"); ax.axis("off")
    ax = fig.add_subplot(2, 2, 2); ax.imshow(group_viz); ax.set_title("KMeans groups"); ax.axis("off")
    ax = fig.add_subplot(2, 2, 3); ax.imshow(np.clip(S_rgb, 0, 1)); ax.set_title("Shading"); ax.axis("off")
    ax = fig.add_subplot(2, 2, 4); ax.imshow(np.clip(R_rgb, 0, 1)); ax.set_title("Reflectance"); ax.axis("off")
    fig.suptitle(suptitle, y=0.98)
    plt.tight_layout()
    _finish(name)


# comparing several results as a grid of input, shading and reflectance
def show_comparison_grid(examples):
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    titles = ["Input", "Shading", "Reflectance"]
    # looping over each example and displaying its input, shading, and reflectance
    for i, (inp, shade, refl) in enumerate(examples):
        axes[0, i].imshow(np.clip(inp, 0, 1))
        axes[0, i].set_title(f"{titles[0]} {i + 1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(shade, 0, 1))
        axes[1, i].set_title(f"{titles[1]} {i + 1}")
        axes[1, i].axis("off")

        axes[2, i].imshow(np.clip(refl, 0, 1))
        axes[2, i].set_title(f"{titles[2]} {i + 1}")
        axes[2, i].axis("off")
    plt.suptitle("Multiple Results Comparison", fontsize=14, y=0.95)
    plt.tight_layout()
    _finish("comparison_grid.png")


# drawing cluster chroma as colour chips at a fixed lightness
def show_cluster_center_chips(ab_centers, k=None):
    k = config.K if k is None else k
    size = config.CHIP_SIZE
    patch = np.zeros((size, size * k, 3), dtype=np.float32)
    for i, (aa, bb) in enumerate(ab_centers):
        tile_lab = np.ones((size, size, 3), dtype=np.float32)
        tile_lab[..., 0] = config.CHIP_L
        tile_lab[..., 1] = float(aa)
        tile_lab[..., 2] = float(bb)
        # converting lab to rgb
        tile_rgb = np.clip(color.lab2rgb(tile_lab), 0, 1)
        # placing each color chip in patch
        patch[:, i * size:(i + 1) * size, :] = tile_rgb
    plt.figure(figsize=(max(3, k), 2))
    plt.imshow(patch)
    plt.title(f"Cluster centers  at L={config.CHIP_L:.0f}")
    plt.axis("off")
    _finish("cluster_center_chips.png")


# plotting the distribution of superpixels across clusters
def plot_cluster_sizes(labels, title, name):
    # getting unique cluster ids and their counts
    ids, counts = np.unique(labels, return_counts=True)
    # sorting clusters by size in descending order
    order = np.argsort(-counts)
    ids, counts = ids[order], counts[order]
    # plotting bar chart for cluster sizes
    plt.figure(figsize=(6, 3))
    plt.bar([str(i) for i in ids], counts)
    plt.title(title)
    plt.xlabel("Cluster id")
    plt.ylabel(" superpixels")
    plt.tight_layout()
    _finish(name)
