"""M4: clustering, dimensionality reduction and anomaly detection.

These are exploratory techniques: labels (PlacementStatus, IsAnomaly) are never used for
fitting, only to describe or evaluate the result afterwards.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from .. import contract as C
from ..estimators import optional_available
from .common import CORE_NUMERIC, LabData, numeric_matrix_pipeline, r, scatter_sample

ALL_NUMERIC = [f.name for f in C.NUMERIC_FEATURES]


def _space(data: LabData, n: int, columns: list[str] = CORE_NUMERIC, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame, Any]:
    frame = data.sample("train", n, seed=seed)
    pre = numeric_matrix_pipeline(scale=True)
    X = pre.fit_transform(frame[columns])
    return X, frame, pre


def _projection(X: np.ndarray, labels: np.ndarray, frame: pd.DataFrame, max_points: int = 1200, extra: dict[str, np.ndarray] | None = None) -> list[dict]:
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    idx = scatter_sample(len(X), max_points)
    pts = []
    for i in idx:
        p = {"x": r(xy[i, 0], 3), "y": r(xy[i, 1], 3), "cluster": int(labels[i]), "placed": int(frame[C.TARGET].iloc[i])}
        if extra:
            for k, v in extra.items():
                p[k] = v[i] if isinstance(v[i], str) else int(v[i])
        pts.append(p)
    return pts


def kmeans(data: LabData, k: int = 3, sample_size: int = 3000) -> dict[str, Any]:
    X, frame, pre = _space(data, sample_size)
    elbow = []
    for kk in range(1, 11):
        km = KMeans(n_clusters=kk, n_init=5, random_state=0).fit(X)
        elbow.append(
            {
                "k": kk,
                "inertia": r(km.inertia_, 2),
                "silhouette": r(silhouette_score(X, km.labels_, sample_size=min(2000, len(X)), random_state=0)) if kk > 1 else None,
            }
        )
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    scaler = pre.named_steps["scale"]
    centroids = scaler.inverse_transform(km.cluster_centers_)
    clusters = []
    for c in range(k):
        m = km.labels_ == c
        clusters.append(
            {
                "cluster": c,
                "size": int(m.sum()),
                "placed_rate": r(frame.loc[m, C.TARGET].mean()),
                "centroid": {f: r(centroids[c, j], 2) for j, f in enumerate(CORE_NUMERIC)},
            }
        )
    return {
        "split": data.split_info(len(frame), 0, "Standardised numeric features from a stratified training sample."),
        "features": CORE_NUMERIC,
        "k": k,
        "elbow": elbow,
        "silhouette": r(silhouette_score(X, km.labels_, sample_size=min(2000, len(X)), random_state=0)),
        "clusters": clusters,
        "projection": _projection(X, km.labels_, frame),
        "note": "Placement rates are attached to clusters after fitting; K-means never sees the label.",
    }


def hierarchical(data: LabData, linkage: str = "ward", n_clusters: int = 3, sample_size: int = 500) -> dict[str, Any]:
    if linkage not in ("ward", "complete", "average", "single"):
        raise ValueError("linkage must be ward, complete, average or single")
    X, frame, _ = _space(data, sample_size)
    D = pdist(X)
    comparison = []
    chosen: dict[str, Any] = {}
    for method in ("ward", "complete", "average", "single"):
        Z = scipy_linkage(X, method=method)
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        sizes = np.bincount(labels)[1:]
        coph, _ = cophenet(Z, D)
        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
        comparison.append(
            {
                "linkage": method,
                "silhouette": r(sil),
                "cophenetic_correlation": r(coph),
                "largest_cluster_fraction": r(sizes.max() / sizes.sum()),
                "cluster_sizes": sorted([int(s) for s in sizes], reverse=True),
            }
        )
        if method == linkage:
            dn = dendrogram(Z, truncate_mode="lastp", p=30, no_plot=True, color_threshold=Z[-(n_clusters - 1), 2] if n_clusters > 1 else 0)
            cut_height = float((Z[-n_clusters, 2] + Z[-(n_clusters - 1), 2]) / 2) if n_clusters > 1 else float(Z[-1, 2] * 1.05)
            chosen = {
                "dendrogram": {
                    "icoord": [[r(v, 2) for v in seg] for seg in dn["icoord"]],
                    "dcoord": [[r(v, 3) for v in seg] for seg in dn["dcoord"]],
                    "leaf_labels": dn["ivl"],
                    "colors": dn["color_list"],
                    "cut_height": r(cut_height, 3),
                },
                "projection": _projection(X, labels - 1, frame, max_points=sample_size),
                "placed_rate_by_cluster": [
                    {"cluster": int(c - 1), "size": int((labels == c).sum()), "placed_rate": r(frame.loc[labels == c, C.TARGET].mean())}
                    for c in np.unique(labels)
                ],
            }
    return {
        "split": data.split_info(len(frame), 0, f"Hierarchical clustering is O(n^2) in memory, so it uses a {len(frame)}-row sample."),
        "linkage": linkage,
        "n_clusters": n_clusters,
        "comparison": comparison,
        **chosen,
        "note": "Single linkage tends to chain points into one giant cluster; Ward minimises within-cluster variance and requires Euclidean distances.",
    }


def dbscan(data: LabData, eps: float = 1.8, min_samples: int = 10, sample_size: int = 2500) -> dict[str, Any]:
    X, frame, _ = _space(data, sample_size)
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist, _ = nn.kneighbors(X)
    kd = np.sort(dist[:, -1])
    kidx = np.unique(np.linspace(0, len(kd) - 1, 150).round().astype(int))
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = model.labels_
    core = np.zeros(len(X), dtype=bool)
    core[model.core_sample_indices_] = True
    kind = np.where(labels == -1, "noise", np.where(core, "core", "border"))
    clusters = sorted(set(labels) - {-1})
    by_cluster = [
        {"cluster": int(c), "size": int((labels == c).sum()), "placed_rate": r(frame.loc[labels == c, C.TARGET].mean())}
        for c in clusters
    ]
    noise = labels == -1
    sil = None
    if len(clusters) > 1 and (~noise).sum() > len(clusters):
        sil = r(silhouette_score(X[~noise], labels[~noise]))
    return {
        "split": data.split_info(len(frame), 0, "Standardised numeric features from a stratified training sample."),
        "params": {"eps": eps, "min_samples": min_samples},
        "n_clusters": len(clusters),
        "n_core": int(core.sum()),
        "n_border": int(((~core) & (~noise)).sum()),
        "n_noise": int(noise.sum()),
        "noise_placed_rate": r(frame.loc[noise, C.TARGET].mean()) if noise.any() else None,
        "silhouette_excluding_noise": sil,
        "clusters": by_cluster,
        "k_distance": [{"rank": int(i), "distance": r(kd[i], 3)} for i in kidx],
        "k_distance_hint": r(float(np.quantile(kd, 0.9)), 3),
        "projection": _projection(X, labels, frame, extra={"point_type": kind}),
        "note": "Choose eps near the 'knee' of the sorted k-distance curve (k = min_samples).",
    }


def pca(data: LabData, sample_size: int = 20000) -> dict[str, Any]:
    X, frame, _ = _space(data, sample_size, columns=ALL_NUMERIC)
    p = PCA(random_state=0).fit(X)
    ratio = p.explained_variance_ratio_
    cum = np.cumsum(ratio)
    xy = p.transform(X)[:, :2]
    idx = scatter_sample(len(X), 1500)
    loadings = [
        {"feature": f, "pc1": r(p.components_[0, j]), "pc2": r(p.components_[1, j])} for j, f in enumerate(ALL_NUMERIC)
    ]
    return {
        "split": data.split_info(len(frame), 0, "PCA fitted on standardised numeric features of a training sample."),
        "features": ALL_NUMERIC,
        "explained_variance": [
            {"component": f"PC{i + 1}", "ratio": r(v), "cumulative": r(cum[i])} for i, v in enumerate(ratio)
        ],
        "components_for_90": int(np.searchsorted(cum, 0.90) + 1),
        "components_for_95": int(np.searchsorted(cum, 0.95) + 1),
        "loadings": sorted(loadings, key=lambda d: -abs(d["pc1"])),
        "projection": [
            {"x": r(xy[i, 0], 3), "y": r(xy[i, 1], 3), "placed": int(frame[C.TARGET].iloc[i])} for i in idx
        ],
    }


def embedding(data: LabData, method: str = "tsne", perplexity: float = 30.0, n_neighbors: int = 15, sample_size: int = 1500) -> dict[str, Any]:
    X, frame, _ = _space(data, sample_size, columns=ALL_NUMERIC)
    t = time.perf_counter()
    if method == "tsne":
        emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(X)
        params = {"perplexity": perplexity}
    elif method == "umap":
        if not optional_available("umap"):
            raise ValueError("UMAP is not installed (pip install umap-learn)")
        import umap

        emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=0).fit_transform(X)
        params = {"n_neighbors": n_neighbors}
    else:
        raise ValueError("method must be tsne or umap")
    elapsed = time.perf_counter() - t
    tw = trustworthiness(X, emb, n_neighbors=10)
    return {
        "split": data.split_info(len(frame), 0, "Non-linear embeddings are fitted on a small sample for speed."),
        "method": method,
        "params": params,
        "seconds": r(elapsed, 2),
        "trustworthiness": r(tw),
        "umap_available": optional_available("umap"),
        "points": [
            {"x": r(emb[i, 0], 3), "y": r(emb[i, 1], 3), "placed": int(frame[C.TARGET].iloc[i]), "tier": str(frame["CollegeTier"].iloc[i])}
            for i in range(len(emb))
        ],
        "note": "Distances between far-apart groups and group sizes in t-SNE/UMAP plots are not meaningful; trustworthiness measures how well local neighbourhoods are preserved.",
    }


def anomaly(data: LabData, contamination: float = 0.035, nu: float = 0.035, train_rows: int = 10000, ocsvm_rows: int = 3000) -> dict[str, Any]:
    tr = data.sample("train", train_rows, stratify=False)
    va = data.validation
    pre = numeric_matrix_pipeline(scale=True)
    X_tr = pre.fit_transform(tr[ALL_NUMERIC])
    X_va = pre.transform(va[ALL_NUMERIC])
    y_va = va[C.ANOMALY_LABEL].to_numpy().astype(int)

    def evaluate(scores: np.ndarray, flagged: np.ndarray) -> dict[str, Any]:
        bins = np.linspace(scores.min(), scores.max(), 31)
        h0, _ = np.histogram(scores[y_va == 0], bins=bins)
        h1, _ = np.histogram(scores[y_va == 1], bins=bins)
        return {
            "flagged": int(flagged.sum()),
            "precision": r(precision_score(y_va, flagged, zero_division=0)),
            "recall": r(recall_score(y_va, flagged, zero_division=0)),
            "f1": r(f1_score(y_va, flagged, zero_division=0)),
            "roc_auc_of_score": r(roc_auc_score(y_va, scores)),
            "score_histogram": [
                {"score": r((bins[i] + bins[i + 1]) / 2, 3), "normal": int(h0[i]), "labelled_anomaly": int(h1[i])} for i in range(30)
            ],
        }

    t = time.perf_counter()
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=0, n_jobs=-1).fit(X_tr)
    iso_scores = -iso.score_samples(X_va)  # higher = more anomalous
    iso_res = evaluate(iso_scores, iso.predict(X_va) == -1)
    iso_res["seconds"] = r(time.perf_counter() - t, 2)

    t = time.perf_counter()
    oc = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(X_tr[:ocsvm_rows])
    oc_scores = -oc.decision_function(X_va)
    oc_res = evaluate(oc_scores, oc.predict(X_va) == -1)
    oc_res["seconds"] = r(time.perf_counter() - t, 2)
    oc_res["train_rows"] = int(min(ocsvm_rows, len(X_tr)))

    profile = (
        va.groupby(C.ANOMALY_LABEL)[["CGPA", "AttendancePercent", "CodingTestScore", "AptitudeTestScore", "MockInterviewScore"]]
        .mean()
        .round(2)
    )
    return {
        "split": data.split_info(len(tr), len(va), "Detectors are fitted without labels on training rows; the dataset's IsAnomaly flag is used only to score them on validation rows."),
        "params": {"contamination": contamination, "nu": nu},
        "labelled_anomaly_rate_validation": r(y_va.mean()),
        "isolation_forest": iso_res,
        "one_class_svm": oc_res,
        "labelled_profile": {str(int(k)): {c: r(v, 2) for c, v in row.items()} for k, row in profile.iterrows()},
        "autoencoder": "Conceptual only: an autoencoder learns to reconstruct normal rows and flags rows with high reconstruction error. It is not implemented here (deep learning is out of scope).",
        "note": "Anomaly detection is exploratory. The IsAnomaly flag comes from the dataset and may describe anomalies that are not visible in these features.",
    }
