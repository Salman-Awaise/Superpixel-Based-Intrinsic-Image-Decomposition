# Superpixel-Based Intrinsic Image Decomposition

A computational photography project that decomposes images into their intrinsic components: **shading** (illumination) and **reflectance** (albedo). This implementation uses SLIC superpixel segmentation, K-Means clustering, and bilateral filtering to separate the lighting effects from the surface properties of objects.

## Overview

Intrinsic image decomposition aims to separate an image into two components:
- **Shading (S)**: The illumination component - how light falls on surfaces
- **Reflectance (R)**: The intrinsic material properties - the color/texture of surfaces

The decomposition follows the relationship: `Image = Shading × Reflectance`

## Features

- Automatic download and processing of the MIT Intrinsic Images dataset
- SLIC superpixel segmentation for efficient spatial grouping
- K-Means clustering with weighted features (color, position, texture)
- Bilateral filtering for smooth shading estimation
- Per-cluster ratio normalization to reduce global gain bias
- Visual evaluation with comparative displays
- Silhouette score metrics for clustering quality assessment

## Dataset

The project uses the **MIT Intrinsic Images** dataset, which contains:
- 80 base RGB images
- Ground truth shading and reflectance maps
- Various lighting conditions and scene types

The dataset is automatically downloaded from:
```
http://people.csail.mit.edu/rgrosse/intrinsic/intrinsic-data.tar.gz
```

## Methodology

### 1. Pre-processing
- Downloads and extracts the MIT Intrinsic Images dataset
- Selects colorful images for better visualization
- Converts images to CIELAB color space for perceptual uniformity

### 2. Feature Extraction
For each superpixel, computes:
- **Color features**: a* and b* chroma values (Lab space)
- **Spatial features**: x and y position (normalized)
- **Lightness**: L channel (centered and masked)
- **Texture**: Standard deviation of L channel

### 3. Clustering
- Uses K-Means (k=6) with weighted features:
  - Color (a*, b*): Implicit weight ~0.50
  - Lightness (L): Weight 0.15
  - Position (x, y): Weight 0.25
  - Texture: Weight 0.10

### 4. Decomposition
- Estimates smooth shading via bilateral filtering
- Normalizes ratio per cluster to reduce bias
- Derives shading and reflectance components
- Applies dynamic range compression for visualization

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended.

### Key Dependencies

- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `opencv-python`: Image I/O and bilateral filtering
- `scikit-image`: SLIC superpixels and color space conversion
- `scikit-learn`: K-Means clustering and metrics

## Usage

Run the full analysis:

```bash
python main.py
```

That will download and extract the dataset on first run, fit K-Means on the
selected training image, decompose both the training and test images, and display
every figure.

Useful flags:

```bash
python main.py --save-figures        # also write figures to results/figures/
python main.py --no-figures          # headless, print metrics only
python main.py --k 8                 # override the number of clusters
python main.py --n-segments 800      # override the superpixel count
```

Parameters live in `src/config.py`:

```python
K = 6                  # Number of clusters
W_L = 0.15             # Lightness weight
W_XY = 0.25            # Position weight
W_TEX = 0.10           # Texture weight
N_SEGMENTS = 500       # Superpixel segments
COMPACTNESS = 12.0     # SLIC compactness parameter
```

## Results

The implementation produces:
- **K-Means cluster visualization**: Shows how superpixels are grouped
- **Shading maps**: Illumination effects separated from reflectance
- **Reflectance maps**: Material properties without lighting
- **Comparison grids**: Side-by-side visualization of inputs and outputs
- **Metrics**: Inertia and silhouette scores for clustering quality

### Sample Metrics
```
KMeans (k=6)  inertia: 382.52   silhouette: 0.884
Shading range: [0.00, 41.01]
Reflectance range: [0.000, 1.000]
```

## Technical Details

### Color Space
- Uses **CIELAB** color space for perceptual uniformity
- a* and b* channels capture chromatic information
- L channel captures luminance

### Superpixels
- **SLIC (Simple Linear Iterative Clustering)** algorithm
- Default: 500 segments, compactness=12.0
- Efficiently groups spatially coherent pixels

### Shading Estimation
- **Bilateral filtering** (2 iterations) on L channel
  - `sigmaColor = 0.1*255`
  - `sigmaSpace = 9`
- Preserves edges while smoothing illumination

### Clustering Strategy
- Features standardized for K-Means stability
- Per-cluster median normalization to reduce global gain bias
- Background masking using lightness threshold (L > 5.0)

## Project Structure

```
.
├── main.py                  # CLI entry point
├── requirements.txt
├── README.md
├── src/
│   ├── config.py            # paths, SLIC/K-Means settings, decomposition constants
│   ├── dataset.py           # download, extract, image collection and chroma ranking
│   ├── superpixels.py       # SLIC segmentation
│   ├── features.py          # Lab conversion and per-superpixel feature vectors
│   ├── clustering.py        # K-Means fitting, standardization, silhouette
│   ├── decomposition.py     # bilateral shading estimate, shading/reflectance split
│   ├── visualization.py     # every plot
│   └── pipeline.py          # wires the stages into train/test runs
├── data_mit_intrinsic/      # Downloaded dataset (created automatically, gitignored)
└── results/figures/         # Written by --save-figures (gitignored)
```

### How the modules connect

```
main.py
  └── pipeline.run_all()
        ├── dataset.ensure_dataset() -> collect_rgb_paths() -> select_colorful_image()
        ├── run_train()  ── superpixels.run_slic() ─┐
        │                   features.build_train_features()
        │                   clustering.fit_train()  ├─> decomposition.decompose()
        │                   features.expand_to_pixel_map()
        ├── run_test()   ── same chain, but build_test_features() + fit_test()
        └── report_metrics() ── clustering.silhouette() + visualization chips
```

## References

- **Dataset**: MIT Intrinsic Images (Barron & Malik, 2011)
- **SLIC**: Achanta et al., "SLIC Superpixels" (PAMI 2012)
- **Intrinsic Images**: Grosse et al., "Ground truth dataset and baseline evaluations" (ICCV 2009)

## Notes

- The implementation automatically selects the most colorful images for better visualization
- Background masking prevents very dark regions from affecting clustering
- Dynamic range compression (99th percentile) is applied to reflectance for display
- Gamma correction (1/2.2) applied to final results for perceptual linearity

## License

This project is for educational and research purposes. The MIT Intrinsic Images dataset has its own licensing terms.

## Author

Advanced Perception Project - SpTp2
