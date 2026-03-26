from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from agents.al_agent import ActiveLearningAgent
from agents.annotation_agent import AnnotationAgent
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.utils import ensure_dir, sha256_bytes, stable_json_dumps, utc_now_iso


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--resume-hitl", action="store_true", help="If set, will try to apply *_corrected.csv if present.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(args.artifacts_dir or cfg.get("output", {}).get("artifacts_dir", "artifacts"))
    ensure_dir(artifacts_dir)

    run_meta = {
        "run_name": cfg.get("project", {}).get("run_name", "run"),
        "started_at": utc_now_iso(),
        "config_path": args.config,
        "config": cfg,
    }
    run_id = sha256_bytes(stable_json_dumps(run_meta).encode("utf-8"))[:12]

    run_dir = artifacts_dir / f"run_{run_id}"
    ensure_dir(run_dir)

    # ---------- Step 1: collect ----------
    collector = DataCollectionAgent(config=args.config)
    sources = cfg["collection"]["sources"]
    raw_df = collector.run(sources=sources)
    save_df(raw_df, run_dir / "data_raw.csv")

    # ---------- Step 2: quality ----------
    dq = DataQualityAgent()
    dq_report = dq.detect_issues(raw_df)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "reports" / "quality_report.json").write_text(json.dumps(dq_report, ensure_ascii=False, indent=2), encoding="utf-8")

    clean_df = dq.fix(raw_df, strategy=cfg.get("quality", {}).get("fix", {}).get("strategy", {}))
    comp = dq.compare(raw_df, clean_df)
    save_df(clean_df, run_dir / "data_clean.csv")
    save_df(comp, run_dir / "reports" / "quality_compare.csv")

    # ---------- Step 3: annotation + HITL ----------
    ann_cfg = cfg.get("annotation", {})
    annotator = AnnotationAgent(modality=ann_cfg.get("modality", "text"), label_space=ann_cfg.get("label_space"))
    labeled_df = annotator.auto_label(clean_df)
    save_df(labeled_df, run_dir / "data_auto_labeled.csv")

    spec_md = annotator.generate_spec(labeled_df, task="text_classification")
    (run_dir / "reports" / "annotation_spec.md").write_text(spec_md, encoding="utf-8")

    metrics = annotator.check_quality(labeled_df)
    (run_dir / "reports" / "annotation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # HITL queue
    thr = float(ann_cfg.get("confidence_threshold", 0.7))
    review_queue = annotator.build_review_queue(labeled_df, threshold=thr)
    review_path = run_dir / "review_queue.csv"
    corrected_path = run_dir / "review_queue_corrected.csv"
    save_df(review_queue, review_path)

    final_labeled = labeled_df.copy()
    if args.resume_hitl and corrected_path.exists():
        corrected = pd.read_csv(corrected_path)
        corrected = corrected.rename(columns={"label_human": "label"})
        corrected = corrected[corrected["label"].notna() & (corrected["label"].astype(str).str.len() > 0)]
        corrected = corrected[["text", "label"]]
        # apply corrections by exact text match (last write wins)
        final_labeled["label_final"] = final_labeled.get("label_auto", final_labeled["label"]).astype("string")
        corr = corrected.drop_duplicates(subset=["text"], keep="last").copy()
        corr["text"] = corr["text"].astype("string")
        corr["label"] = corr["label"].astype("string")
        corr_map = dict(zip(corr["text"].tolist(), corr["label"].tolist()))
        base = final_labeled.get("label_auto", final_labeled["label"]).astype("string")
        corrected_labels = final_labeled["text"].astype("string").map(corr_map)
        final_labeled["label_final"] = corrected_labels.combine_first(base)
    else:
        final_labeled["label_final"] = final_labeled.get("label_auto", final_labeled["label"])

    save_df(final_labeled, run_dir / "data_labeled_final.csv")

    # LabelStudio export
    ls_path = run_dir / "labelstudio_import.json"
    ls_df = final_labeled.copy()
    ls_df["label_auto"] = ls_df["label_final"]
    annotator.export_to_labelstudio(ls_df, ls_path)

    # ---------- Step 4: Active Learning ----------
    # Use only rows with a final label that isn't unknown
    al_df = final_labeled[final_labeled["label_final"].astype("string").str.lower().ne("unknown")].copy()
    al_df = al_df[al_df["text"].notna() & al_df["label_final"].notna()].copy()
    # Avoid duplicate 'label' columns by using a dedicated column.
    al_df["label_for_al"] = al_df["label_final"].astype("string")
    if len(al_df) >= 200 and int(al_df["label_for_al"].nunique()) >= 2:
        train_df, test_df = train_test_split(al_df, test_size=0.2, random_state=42, stratify=al_df["label_for_al"])
        train_df = train_df[["text", "label_for_al"]].rename(columns={"label_for_al": "label"}).copy()
        test_df = test_df[["text", "label_for_al"]].rename(columns={"label_for_al": "label"}).copy()
        n_initial = int(cfg.get("active_learning", {}).get("n_initial", 50))
        labeled0 = train_df.sample(n=min(n_initial, len(train_df)), random_state=42)
        pool = train_df.drop(index=labeled0.index)

        al_agent = ActiveLearningAgent(model=cfg.get("active_learning", {}).get("model", "logreg"))
        history_all: list[dict[str, Any]] = []
        for strat in cfg.get("active_learning", {}).get("strategies", ["entropy", "random"]):
            hist = al_agent.run_cycle(
                labeled_df=labeled0.reset_index(drop=True),
                pool_df=pool.reset_index(drop=True),
                test_df=test_df.reset_index(drop=True),
                strategy=strat,
                n_iterations=int(cfg.get("active_learning", {}).get("n_iterations", 5)),
                batch_size=int(cfg.get("active_learning", {}).get("batch_size", 20)),
            )
            history_all.extend(hist)

        history_df = pd.DataFrame(history_all)
        save_df(history_df, run_dir / "reports" / "al_history.csv")
        al_agent.report(history_all, run_dir / "reports" / "learning_curve.png")
    else:
        (run_dir / "reports" / "al_skipped.txt").write_text(
            "Active Learning skipped: not enough labeled data or only one class.\n",
            encoding="utf-8",
        )

    # ---------- Manifest ----------
    manifest = {
        "run_id": run_id,
        "finished_at": utc_now_iso(),
        "paths": {
            "data_raw": str(run_dir / "data_raw.csv"),
            "data_clean": str(run_dir / "data_clean.csv"),
            "data_auto_labeled": str(run_dir / "data_auto_labeled.csv"),
            "review_queue": str(review_path),
            "review_queue_corrected": str(corrected_path),
            "data_labeled_final": str(run_dir / "data_labeled_final.csv"),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK. run_id={run_id} artifacts={run_dir}")
    print(f"HITL: edit {review_path} and save as {corrected_path}, then rerun with --resume-hitl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

