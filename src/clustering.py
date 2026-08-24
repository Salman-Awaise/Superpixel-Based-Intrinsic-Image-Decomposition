# keeping K-Means clustering over superpixel features here
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config


# standardizing features so k-means is not dominated by one scale
def standardize(feats):
    mu = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-6
    return (feats - mu) / std, mu, std


# fitting k-means on standardized training features
def fit_train(feats, k=None):
    k = config.K if k is None else k
    feats_std, mu, std = standardize(feats)
    km = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_STATE)
    labels = km.fit_predict(feats_std)
    return dict(labels=labels, centers=km.cluster_centers_.copy(),
                inertia=float(km.inertia_), feats_std=feats_std, mu=mu, std=std)


# fitting k-means per image on the raw test features
# the notebook does not standardize here, so the centres stay in raw a*/b* units
def fit_test(feats, k=None):
    k = config.K if k is None else k
    km = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_STATE)
    labels = km.fit_predict(feats)
    return dict(labels=labels, centers=km.cluster_centers_)


# computing the silhouette score in the same space k-means was fitted in
def silhouette(feats_std, labels):
    if len(np.unique(labels)) >= 2 and len(labels) >= 10:
        return float(silhouette_score(feats_std, labels))
    return 0.0


# recovering cluster centres in original feature units
def destandardize_centers(centers, mu, std):
    return centers * std + mu
