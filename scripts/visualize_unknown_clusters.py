from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize unknown text clusters (2D).")
    ap.add_argument("--input", default="reports/unknown_clusters.csv")
    ap.add_argument("--out", default="reports/unknown_clusters_plot.png")
    ap.add_argument("--method", choices=["svd", "tsne"], default="tsne")
    ap.add_argument("--max-texts", type=int, default=5000)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    if "cluster_id" not in df.columns or "text" not in df.columns:
        raise ValueError("Input CSV must contain columns: cluster_id, text")

    df = df.copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)
    df["text"] = df["text"].astype("string").fillna("")
    df = df[df["text"].astype(str).str.len() > 0].copy()
    df = df.drop_duplicates(subset=["text"], keep="first").copy()

    if len(df) > int(args.max_texts):
        df = df.sample(n=int(args.max_texts), random_state=int(args.random_state)).copy()

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=60000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(df["text"].tolist())

    if args.method == "svd":
        svd = TruncatedSVD(n_components=2, random_state=int(args.random_state))
        XY = svd.fit_transform(X)
        title = "Unknown texts: TF-IDF + SVD (2D) by cluster_id"
        xlab, ylab = "SVD-1", "SVD-2"
    else:
        # SVD -> t-SNE gives much better visible separation than 2D SVD alone
        # on high-dimensional sparse text data.
        svd = TruncatedSVD(n_components=min(50, max(2, X.shape[1] - 1)), random_state=int(args.random_state))
        Z = svd.fit_transform(X)
        perplexity = max(5, min(40, (len(df) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=int(args.random_state),
        )
        XY = tsne.fit_transform(Z)
        title = "Unknown texts: TF-IDF + SVD(50) + t-SNE (2D) by cluster_id"
        xlab, ylab = "t-SNE-1", "t-SNE-2"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 7))
    clusters = sorted(df["cluster_id"].unique().tolist())
    cmap = plt.get_cmap("tab20", max(len(clusters), 1))

    for i, cid in enumerate(clusters):
        mask = df["cluster_id"] == cid
        pts = XY[mask.values]
        if len(pts) == 0:
            continue
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=12,
            alpha=0.75,
            color=cmap(i % max(len(clusters), 1)),
            label=f"{cid} (n={int(mask.sum())})",
            linewidths=0,
        )

    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    # Legend can be huge; keep it outside and small.
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

