from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def top_terms_per_cluster(tfidf: TfidfVectorizer, centers: np.ndarray, top_k: int = 12) -> dict[int, list[str]]:
    terms = np.asarray(tfidf.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for i, row in enumerate(centers):
        idx = np.argsort(-row)[:top_k]
        out[i] = [str(t) for t in terms[idx].tolist()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Cluster unknown texts for HITL triage.")
    ap.add_argument("--input", default="data/labeled/data_labeled_final.csv")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--label-col", default="label_final")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--unknown", default="unknown")
    ap.add_argument("--max-texts", type=int, default=5000)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"Expected columns `{args.text_col}` and `{args.label_col}` in {in_path}")

    txt = df[args.text_col].astype("string").fillna("")
    lab = df[args.label_col].astype("string").fillna("")
    unk_mask = lab.str.lower().eq(str(args.unknown).lower())
    unk = df.loc[unk_mask, [args.text_col, args.label_col]].copy()
    unk = unk.rename(columns={args.text_col: "text", args.label_col: "label"})

    if unk.empty:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "unknown_clusters.md").write_text("No `unknown` rows found.\n", encoding="utf-8")
        return 0

    # Dedupe for clustering (keeps cluster summaries stable)
    unk = unk.drop_duplicates(subset=["text"], keep="first").copy()
    if len(unk) > int(args.max_texts):
        unk = unk.sample(n=int(args.max_texts), random_state=int(args.random_state)).copy()

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(unk["text"].astype("string").fillna("").tolist())

    k = int(args.n_clusters)
    if k < 2:
        raise ValueError("--n-clusters must be >= 2")
    if X.shape[0] < k:
        k = max(2, min(X.shape[0], k))

    km = KMeans(n_clusters=k, random_state=int(args.random_state), n_init="auto")
    cluster_id = km.fit_predict(X)
    unk["cluster_id"] = cluster_id.astype(int)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV for labeling in bulk
    csv_path = out_dir / "unknown_clusters.csv"
    unk[["cluster_id", "text"]].to_csv(csv_path, index=False, encoding="utf-8")

    # Human-friendly summary
    centers = km.cluster_centers_
    top_terms = top_terms_per_cluster(vectorizer, centers, top_k=12)
    sizes = unk["cluster_id"].value_counts().sort_values(ascending=False)

    md_lines: list[str] = []
    md_lines.append("# Unknown clustering (for HITL)")
    md_lines.append("")
    md_lines.append(f"Input: `{in_path.as_posix()}`")
    md_lines.append(f"Rows clustered (deduped/sample): **{len(unk)}**")
    md_lines.append(f"Clusters: **{int(k)}**")
    md_lines.append("")
    md_lines.append("## How to use")
    md_lines.append("- Open `unknown_clusters.csv` and scan per `cluster_id`.")
    md_lines.append("- Assign a disease label to the whole cluster when it’s coherent.")
    md_lines.append("")

    for cid, n in sizes.items():
        cid_int = int(cid)
        md_lines.append(f"## Cluster {cid_int} (n={int(n)})")
        md_lines.append(f"Top terms: `{', '.join(top_terms.get(cid_int, []))}`")
        md_lines.append("")
        examples = (
            unk.loc[unk["cluster_id"] == cid_int, "text"]
            .astype("string")
            .head(8)
            .tolist()
        )
        for ex in examples:
            ex = str(ex).replace("\n", " ").strip()
            md_lines.append(f"- {ex[:300]}")
        md_lines.append("")

    (out_dir / "unknown_clusters.md").write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    meta = {
        "input": str(in_path),
        "rows_clustered": int(len(unk)),
        "clusters": int(k),
        "cluster_sizes": {str(int(k)): int(v) for k, v in sizes.to_dict().items()},
        "top_terms": {str(int(k)): v for k, v in top_terms.items()},
    }
    (out_dir / "unknown_clusters_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

