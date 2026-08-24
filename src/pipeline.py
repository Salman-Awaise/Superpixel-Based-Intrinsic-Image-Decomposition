# keeping the end-to-end orchestration here
import os

from . import clustering, config, dataset, decomposition, features, superpixels, visualization


# preparing the dataset and choosing the train and test images
def prepare(show_samples=True):
    # downloading, extracting and locating the dataset
    mit_data_dir = dataset.ensure_dataset()
    # collecting the base RGB inputs
    rgb_paths = dataset.collect_rgb_paths(mit_data_dir)
    # picking the most colorful image to use as the test image
    colorful_test_path, _ = dataset.select_colorful_image(rgb_paths)
    # displaying six random RGB images from the dataset
    if show_samples:
        visualization.show_dataset_samples(rgb_paths, n=6)
    # selecting first image for k-means fitting and colorful image for testing
    train_path = rgb_paths[0]
    test_path = colorful_test_path
    # printing chosen filenames for traceability
    print("[train image]", os.path.basename(train_path))
    print("[test  image]", os.path.basename(test_path))
    return dict(rgb_paths=rgb_paths, train_path=train_path, test_path=test_path)


# running the training image through segmentation, clustering and decomposition
def run_train(train_path, show=True):
    # loading the image as RGB float32 and splitting Lab channels
    img_rgb = dataset.load_rgb_image(train_path)
    L, a, b = features.lab_channels(img_rgb)
    # computing SLIC superpixels on the train image
    spx = superpixels.run_slic(img_rgb)
    if show:
        visualization.show_image_and_superpixels(img_rgb, spx, label="train")
    # masking low-lightness pixels to avoid background leakage
    valid = features.lightness_mask(L)
    # building per-superpixel features with masking
    feats, spx_ids = features.build_train_features(spx, L, a, b, valid)
    # fitting k-means on standardized superpixel features
    km = clustering.fit_train(feats)
    # mapping superpixel labels back to pixel grid
    group_map = features.expand_to_pixel_map(spx, spx_ids, km["labels"])
    # deriving shading and reflectance
    S_L, S_rgb, R_rgb = decomposition.decompose(img_rgb, L, group_map)
    # visualizing input, clusters, shading, and reflectance
    if show:
        visualization.show_decomposition(
            img_rgb, group_map, S_rgb, R_rgb,
            suptitle=f"KMeans (k={config.K}) — inertia: {km['inertia']:.2f}",
            name="train_decomposition.png")
    # reporting the achieved ranges
    print(f"DECOMPOSITION Shading range: [{S_L.min():.2f}, {S_L.max():.2f}]")
    print(f"DECOMPOSITION Reflectance range: [{R_rgb.min():.3f}, {R_rgb.max():.3f}]")
    return dict(img_rgb=img_rgb, spx=spx, feats=feats, km=km, group_map=group_map,
                S_L=S_L, S_rgb=S_rgb, R_rgb=R_rgb)


# running the test image with its own per-image clustering
def run_test(test_path, show=True):
    # loading the image as RGB float32 and splitting Lab channels
    img_rgb = dataset.load_rgb_image(test_path)
    L, a, b = features.lab_channels(img_rgb)
    # computing SLIC superpixels for the test image
    spx = superpixels.run_slic(img_rgb)
    # masking background/very dark pixels by lightness
    valid = features.lightness_mask(L)
    # building per-superpixel Lab features - a, b, centered L
    feats, spx_ids = features.build_test_features(spx, L, a, b, valid)
    # fitting k-means per image for independent clustering on test
    km = clustering.fit_test(feats)
    # expanding superpixel labels to a full-resolution pixel map
    group_map = features.expand_to_pixel_map(spx, spx_ids, km["labels"])
    # deriving shading and reflectance for display
    S_L, S_rgb, R_rgb = decomposition.decompose(img_rgb, L, group_map)
    # visualizing input, clusters, shading, and reflectance for test image
    if show:
        visualization.show_decomposition(
            img_rgb, group_map, S_rgb, R_rgb,
            suptitle="Per-image KMeans",
            name="test_decomposition.png")
    return dict(img_rgb=img_rgb, spx=spx, feats=feats, km=km, group_map=group_map,
                S_L=S_L, S_rgb=S_rgb, R_rgb=R_rgb)


# reporting clustering quality and drawing the cluster centre chips
def report_metrics(train, test, show=True):
    # computing silhouette score on the same standardized space used for k-means
    sil = clustering.silhouette(train["km"]["feats_std"], train["km"]["labels"])
    print(f"[metrics] KMeans (k={config.K})  inertia: {train['km']['inertia']:.2f}   silhouette: {sil:.3f}")
    # picking which cluster centres feed the colour chips
    #
    # the notebook reuses a single `centers_std` variable across cells: the training
    # cell assigns the standardized 6-D train centres, then the test cell overwrites
    # it with the raw 3-D test centres. By the time the chips are drawn, the variable
    # holds the TEST centres, so the destandardize branch never runs and the chips
    # show raw a*/b* values. That behaviour is reproduced here deliberately.
    centers = test["km"]["centers"]
    if centers.shape[1] == 6:
        centers_orig = clustering.destandardize_centers(
            centers, train["km"]["mu"], train["km"]["std"])
        ab_centers = centers_orig[:, :2]
    elif centers.shape[1] >= 2:
        ab_centers = centers[:, :2]
    else:
        raise ValueError("Cluster centers do not contain a* and b* components.")
    # printing the range of a* and b* values to check chromatic spread across clusters
    print(f"[debug] a* range: [{ab_centers[:, 0].min():.5f}, {ab_centers[:, 0].max():.5f}]")
    print(f"[debug] b* range: [{ab_centers[:, 1].min():.5f}, {ab_centers[:, 1].max():.5f}]")
    # displaying the color chips for all clusters
    if show:
        visualization.show_cluster_center_chips(ab_centers)
    return dict(silhouette=sil, ab_centers=ab_centers)


# running every stage in notebook order
def run_all(show=True):
    paths = prepare(show_samples=show)
    train = run_train(paths["train_path"], show=show)
    test = run_test(paths["test_path"], show=show)
    # selecting three representative examples for comparison
    # the notebook reuses the test image for the third slot, so that is kept
    if show:
        visualization.show_comparison_grid([
            (train["img_rgb"], train["S_rgb"], train["R_rgb"]),
            (test["img_rgb"], test["S_rgb"], test["R_rgb"]),
            (test["img_rgb"], test["S_rgb"], test["R_rgb"]),])
    metrics = report_metrics(train, test, show=show)
    # plotting cluster size distribution for training and testing clusters
    if show:
        visualization.plot_cluster_sizes(train["km"]["labels"], "Cluster sizes (TRAIN)",
                                        "cluster_sizes_train.png")
        visualization.plot_cluster_sizes(test["km"]["labels"], "Cluster sizes (TEST)",
                                        "cluster_sizes_test.png")
    return dict(paths=paths, train=train, test=test, metrics=metrics)
