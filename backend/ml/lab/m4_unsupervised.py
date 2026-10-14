"""M4: clustering, dimensionality reduction and anomaly detection.

These are exploratory techniques: ``Class`` is never used for fitting, only to describe or
evaluate the result afterwards. Inputs are V1-V28 plus log(1 + Amount), standardised on the
sample being analysed. Clustering samples over-represent fraud (5 %) so that fraud rows are
visible at all; the real rate is 0.17 %.
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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .. import contract as C
from ..estimators import optional_available
from .common import CLUSTER_FEATURES, LabData, r, scaled_matrix, scatter_sample

FRAUD_SHARE = 0.05


def _space(data: LabData, n: int, columns: list[str] = CLUSTER_FEATURES, seed: int = 0, scale: bool = True) -> tuple[np.ndarray, pd.DataFrame, Any]:
    n_fraud = max(1, int(round(n * FRAUD_SHARE)))
    frame = data.enriched("train", n - n_fraud, n_fraud=n_fraud, seed=seed)
    sc = StandardScaler().fit(frame[columns]) if scale else None
    X = sc.transform(frame[columns]) if sc is not None else frame[columns].to_numpy(float)
    return X, frame, sc


def _sample_note(frame: pd.DataFrame) -> str:
    return f"Training sample of {len(frame):,} rows with {int(frame[C.TARGET].sum())} fraud rows over-represented for visibility (real rate 0.17 %)."


def _projection(X: np.ndarray, labels: np.ndarray, frame: pd.DataFrame, max_points: int = 1200, extra: dict[str, np.ndarray] | None = None) -> list[dict]:
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    y = frame[C.TARGET].to_numpy()
    idx = scatter_sample(len(X), max_points, must_include=np.flatnonzero(y == 1))
    pts = []
    for i in idx:
        p = {"x": r(xy[i, 0], 3), "y": r(xy[i, 1], 3), "cluster": int(labels[i]), "fraud": int(y[i])}
        if extra:
            for k, v in extra.items():
                p[k] = v[i] if isinstance(v[i], str) else int(v[i])
        pts.append(p)
    return pts


def kmeans(data: LabData, k: int = 4, sample_size: int = 3000) -> dict[str, Any]:
    X, frame, sc = _space(data, sample_size)
    elbow = []
    for kk in range(1, 11):
        km = KMeans(n_clusters=kk, n_init=5, random_state=0).fit(X)
        elbow.append({"k": kk, "inertia": r(km.inertia_, 2), "silhouette": r(silhouette_score(X, km.labels_, sample_size=min(2000, len(X)), random_state=0)) if kk > 1 else None})
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    centroids = sc.inverse_transform(km.cluster_centers_)
    spread = np.std(km.cluster_centers_, axis=0)
    shown = [CLUSTER_FEATURES[j] for j in np.argsort(-spread)[:6]]
    clusters = []
    for c in range(k):
        m = km.labels_ == c
        clusters.append(
            {
                "cluster": c,
                "size": int(m.sum()),
                "fraud_rows": int(frame.loc[m, C.TARGET].sum()),
                "fraud_share": r(frame.loc[m, C.TARGET].mean()),
                "centroid": {f: r(centroids[c, CLUSTER_FEATURES.index(f)], 3) for f in shown},
            }
        )
    return {
        "split": data.split_info(frame, 0, _sample_note(frame)),
        "features": CLUSTER_FEATURES,
        "centroid_features_shown": shown,
        "k": k,
        "elbow": elbow,
        "silhouette": r(silhouette_score(X, km.labels_, sample_size=min(2000, len(X)), random_state=0)),
        "clusters": clusters,
        "projection": _projection(X, km.labels_, frame),
        "note": "Fraud shares are attached to clusters after fitting; K-means never sees the label. Centroids are shown for the six features whose standardised centroids differ most.",
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
                "fraud_by_cluster": [
                    {"cluster": int(c - 1), "size": int((labels == c).sum()), "fraud_rows": int(frame.loc[labels == c, C.TARGET].sum()), "fraud_share": r(frame.loc[labels == c, C.TARGET].mean())}
                    for c in np.unique(labels)
                ],
            }
    return {
        "split": data.split_info(frame, 0, f"Hierarchical clustering is O(n^2) in memory, so it uses {len(frame)} rows. " + _sample_note(frame)),
        "linkage": linkage,
        "n_clusters": n_clusters,
        "comparison": comparison,
        **chosen,
        "note": "Single linkage tends to chain points into one giant cluster; Ward minimises within-cluster variance and requires Euclidean distances.",
    }


def dbscan(data: LabData, eps: float = 4.0, min_samples: int = 10, sample_size: int = 2500) -> dict[str, Any]:
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
        {"cluster": int(c), "size": int((labels == c).sum()), "fraud_rows": int(frame.loc[labels == c, C.TARGET].sum()), "fraud_share": r(frame.loc[labels == c, C.TARGET].mean())}
        for c in clusters
    ]
    noise = labels == -1
    y = frame[C.TARGET].to_numpy()
    sil = None
    if len(clusters) > 1 and (~noise).sum() > len(clusters):
        sil = r(silhouette_score(X[~noise], labels[~noise]))
    return {
        "split": data.split_info(frame, 0, _sample_note(frame)),
        "params": {"eps": eps, "min_samples": min_samples},
        "n_clusters": len(clusters),
        "n_core": int(core.sum()),
        "n_border": int(((~core) & (~noise)).sum()),
        "n_noise": int(noise.sum()),
        "noise_fraud_share": r(y[noise].mean()) if noise.any() else None,
        "share_of_fraud_in_noise": r(np.sum(noise & (y == 1)) / max(int(y.sum()), 1)),
        "silhouette_excluding_noise": sil,
        "clusters": by_cluster,
        "k_distance": [{"rank": int(i), "distance": r(kd[i], 3)} for i in kidx],
        "k_distance_hint": r(float(np.quantile(kd, 0.9)), 3),
        "projection": _projection(X, labels, frame, extra={"point_type": kind}),
        "note": "Choose eps near the 'knee' of the sorted k-distance curve (k = min_samples). Points labelled noise are density outliers, which is why fraud concentrates there.",
    }


def pca(data: LabData, sample_size: int = 20000, standardize: bool = True) -> dict[str, Any]:
    X, frame, _ = _space(data, sample_size, scale=standardize)
    p = PCA(random_state=0).fit(X)
    ratio = p.explained_variance_ratio_
    cum = np.cumsum(ratio)
    xy = p.transform(X)[:, :2]
    y = frame[C.TARGET].to_numpy()
    idx = scatter_sample(len(X), 1500, must_include=np.flatnonzero(y == 1))
    loadings = [{"feature": f, "pc1": r(p.components_[0, j]), "pc2": r(p.components_[1, j])} for j, f in enumerate(CLUSTER_FEATURES)]
    raw_var = frame[CLUSTER_FEATURES].var().sort_values(ascending=False)
    return {
        "split": data.split_info(frame, 0, ("PCA on standardised features. " if standardize else "PCA on unstandardised features. ") + _sample_note(frame)),
        "features": CLUSTER_FEATURES,
        "standardized": standardize,
        "explained_variance": [{"component": f"PC{i + 1}", "ratio": r(v), "cumulative": r(cum[i])} for i, v in enumerate(ratio)],
        "components_for_90": int(np.searchsorted(cum, 0.90) + 1),
        "components_for_95": int(np.searchsorted(cum, 0.95) + 1),
        "loadings": sorted(loadings, key=lambda d: -abs(d["pc1"])),
        "largest_raw_variances": [{"feature": str(k), "variance": r(v)} for k, v in raw_var.head(5).items()],
        "projection": [{"x": r(xy[i, 0], 3), "y": r(xy[i, 1], 3), "fraud": int(y[i])} for i in idx],
        "note": (
            "V1-V28 are already principal components, so they are uncorrelated. After standardisation every input has variance 1, "
            "and PCA has little left to compress except the extra structure introduced by the fraud rows and log(1 + Amount). "
            "Without standardisation PCA mostly re-orders the inputs by their raw variance."
        ),
    }


def embedding(data: LabData, method: str = "tsne", perplexity: float = 30.0, n_neighbors: int = 15, sample_size: int = 1500) -> dict[str, Any]:
    X, frame, _ = _space(data, sample_size)
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
    y = frame[C.TARGET].to_numpy()
    return {
        "split": data.split_info(frame, 0, "Non-linear embeddings are fitted on a small sample for speed. " + _sample_note(frame)),
        "method": method,
        "params": params,
        "seconds": r(elapsed, 2),
        "trustworthiness": r(tw),
        "umap_available": optional_available("umap"),
        "points": [{"x": r(emb[i, 0], 3), "y": r(emb[i, 1], 3), "fraud": int(y[i])} for i in range(len(emb))],
        "note": "Distances between far-apart groups and group sizes in t-SNE/UMAP plots are not meaningful; trustworthiness measures how well local neighbourhoods are preserved.",
    }


def anomaly(data: LabData, contamination: float = 0.002, nu: float = 0.002, train_rows: int = 60000, ocsvm_rows: int = 5000) -> dict[str, Any]:
    tr = data.sample("train", train_rows)
    va = data.validation
    X_tr, X_va = scaled_matrix(tr, CLUSTER_FEATURES, va)
    y_va = va[C.TARGET].to_numpy().astype(int)

    def evaluate(scores: np.ndarray, flagged: np.ndarray) -> dict[str, Any]:
        lo, hi = np.quantile(scores, [0.001, 0.9999])
        bins = np.linspace(lo, hi, 31)
        h0, _ = np.histogram(np.clip(scores[y_va == 0], lo, hi), bins=bins)
        h1, _ = np.histogram(np.clip(scores[y_va == 1], lo, hi), bins=bins)
        return {
            "flagged": int(flagged.sum()),
            "precision": r(precision_score(y_va, flagged, zero_division=0)),
            "recall": r(recall_score(y_va, flagged, zero_division=0)),
            "f1": r(f1_score(y_va, flagged, zero_division=0)),
            "roc_auc_of_score": r(roc_auc_score(y_va, scores)),
            "average_precision_of_score": r(average_precision_score(y_va, scores)),
            "score_histogram": [{"score": r((bins[i] + bins[i + 1]) / 2, 3), "legitimate": int(h0[i]), "fraud": int(h1[i])} for i in range(30)],
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

    profile = va.groupby(C.TARGET)[["amount_log", "V14", "V17", "V12", "V10"]].mean().round(3)
    return {
        "split": data.split_info(tr, va, "Detectors are fitted without labels on training rows (natural fraud rate); Class is used only to score them on the validation split."),
        "params": {"contamination": contamination, "nu": nu},
        "fraud_rate_validation": r(y_va.mean(), 6),
        "isolation_forest": iso_res,
        "one_class_svm": oc_res,
        "class_profile": {C.CLASS_LABELS[int(k)]: {c: r(v, 3) for c, v in row.items()} for k, row in profile.iterrows()},
        "autoencoder": "Conceptual only: an autoencoder learns to reconstruct normal rows and flags rows with high reconstruction error. It is not implemented here (deep learning is out of scope).",
        "note": "Anomaly is not the same as fraud: many unusual transactions are legitimate and some frauds look ordinary, which is why the production system combines the anomaly score with a supervised model.",
    }
