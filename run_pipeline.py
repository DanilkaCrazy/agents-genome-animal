from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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


def write_text_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def markdown_from_quality_report(dq_report: dict[str, Any]) -> str:
    missing = dq_report.get("missing", {})
    duplicates = dq_report.get("duplicates", None)
    outliers = dq_report.get("outliers", {})
    imbalance = dq_report.get("imbalance", {})

    lines: list[str] = []
    lines.append("# Quality Report")
    lines.append("")
    lines.append("## Missing Values")
    for col, cnt in sorted((missing.get("count") or {}).items()):
        pct = (missing.get("pct") or {}).get(col, None)
        lines.append(f"- `{col}`: {cnt} ({pct}%)")
    lines.append("")
    lines.append("## Duplicates")
    lines.append(f"- duplicates (text): {duplicates}")
    lines.append("")
    lines.append("## Outliers (IQR on text length)")
    lines.append(f"- feature: {outliers.get('feature')}")
    lines.append(f"- bounds: lo={outliers.get('bounds', {}).get('lo')} hi={outliers.get('bounds', {}).get('hi')}")
    lines.append(f"- n: {outliers.get('n')}")
    lines.append("")
    lines.append("## Class Imbalance")
    n_classes = imbalance.get("n_classes", 0)
    lines.append(f"- n_classes: {n_classes}")
    dist = imbalance.get("dist", {})
    if dist:
        top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:20]
        for k, v in top:
            lines.append(f"- `{k}`: {v}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def markdown_from_annotation_report(spec_md: str, metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Annotation Report")
    lines.append("")
    lines.append("## Spec (generated)")
    lines.append("")
    lines.append("```markdown")
    lines.append(spec_md.strip())
    lines.append("```")
    lines.append("")
    lines.append("## Metrics")
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def markdown_from_al_report(history_df: pd.DataFrame, saved_examples: int | None, learning_curve_rel: str) -> str:
    lines: list[str] = []
    lines.append("# Active Learning Report")
    lines.append("")
    lines.append(f"Learning curve: `{learning_curve_rel}`")
    lines.append("")
    if saved_examples is not None:
        lines.append(f"Saved examples (entropy vs random to reach random peak F1): {saved_examples}")
        lines.append("")
    lines.append("## Learning curve summary (per strategy)")
    if history_df is not None and not history_df.empty:
        for strat in history_df["strategy"].unique():
            sub = history_df[history_df["strategy"] == strat].sort_values("n_labeled")
            if "f1" in sub:
                best_row = sub.loc[sub["f1"].idxmax()]
                lines.append(f"- `{strat}`: best F1={best_row['f1']} at n_labeled={best_row['n_labeled']}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def prepare_al_dataset(df: pd.DataFrame, al_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare AL dataset with leakage-safe dedup and configurable unknown handling.
    """
    meta: dict[str, Any] = {}
    out = df.copy()

    out = out[out["text"].notna() & out["label_final"].notna()].copy()
    out["label_for_al"] = out["label_final"].astype("string")

    before_rows = len(out)
    if bool(al_cfg.get("dedup_before_split", True)):
        out = out.drop_duplicates(subset=["text"], keep="first").copy()
    meta["rows_before_dedup"] = int(before_rows)
    meta["rows_after_dedup"] = int(len(out))

    unknown_policy = str(al_cfg.get("unknown_policy", "drop")).lower()
    if unknown_policy == "drop":
        out = out[out["label_for_al"].str.lower().ne("unknown")].copy()
    elif unknown_policy == "cap":
        cap_ratio = float(al_cfg.get("unknown_cap_ratio", 0.2))
        known = out[out["label_for_al"].str.lower().ne("unknown")].copy()
        unk = out[out["label_for_al"].str.lower().eq("unknown")].copy()
        max_unk = int(len(known) * cap_ratio)
        if len(unk) > max_unk:
            unk = unk.sample(n=max_unk, random_state=42)
        out = pd.concat([known, unk], ignore_index=True)
    elif unknown_policy == "keep":
        pass
    else:
        raise ValueError(f"Unknown active_learning.unknown_policy: {unknown_policy}")
    meta["unknown_policy"] = unknown_policy
    meta["rows_after_unknown_policy"] = int(len(out))

    min_class_count = int(al_cfg.get("min_class_count", 1))
    if min_class_count > 1:
        vc = out["label_for_al"].value_counts()
        keep_labels = vc[vc >= min_class_count].index.astype("string")
        out = out[out["label_for_al"].isin(keep_labels)].copy()
    meta["min_class_count"] = min_class_count
    meta["rows_after_min_class_filter"] = int(len(out))
    meta["n_classes_final"] = int(out["label_for_al"].nunique()) if len(out) else 0

    return out, meta


def _value_counts_dict(s: pd.Series) -> dict[str, int]:
    vc = s.astype("string").fillna("missing").value_counts()
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def _write_model_diagnostics(
    out_path: Path,
    *,
    al_prep_meta: dict[str, Any],
    train_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    y_true: list[str] | None,
    y_pred: list[str] | None,
) -> None:
    diag: dict[str, Any] = {
        "al_prep_meta": al_prep_meta,
    }
    if train_df is not None and not train_df.empty:
        diag["train_label_counts"] = _value_counts_dict(train_df["label"])
    if test_df is not None and not test_df.empty:
        diag["test_label_counts"] = _value_counts_dict(test_df["label"])
    if y_pred is not None:
        diag["pred_label_counts"] = _value_counts_dict(pd.Series(y_pred))
        unique_pred = sorted(set(y_pred))
        diag["unique_pred_labels"] = unique_pred
        diag["prediction_collapse"] = bool(len(unique_pred) <= 1)
    if y_true is not None and y_pred is not None and len(y_true) == len(y_pred) and len(y_true) > 0:
        diag["accuracy"] = float(accuracy_score(y_true, y_pred))
        diag["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--resume-hitl", action="store_true", help="If set, will try to apply *_corrected.csv if present.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(args.artifacts_dir or cfg.get("output", {}).get("artifacts_dir", "artifacts"))
    ensure_dir(artifacts_dir)

    # Required project structure (latest outputs)
    data_raw_dir = Path("data/raw")
    data_labeled_dir = Path("data/labeled")
    models_dir = Path("models")
    reports_dir = Path("reports")
    ensure_dir(data_raw_dir)
    ensure_dir(data_labeled_dir)
    ensure_dir(models_dir)
    ensure_dir(reports_dir)

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
    save_df(raw_df, data_raw_dir / "data_raw.csv")

    # ---------- Step 2: quality ----------
    dq = DataQualityAgent()
    dq_report = dq.detect_issues(raw_df)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "reports" / "quality_report.json").write_text(json.dumps(dq_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_md(reports_dir / "quality_report.md", markdown_from_quality_report(dq_report))

    clean_df = dq.fix(raw_df, strategy=cfg.get("quality", {}).get("fix", {}).get("strategy", {}))
    comp = dq.compare(raw_df, clean_df)
    save_df(clean_df, run_dir / "data_clean.csv")
    save_df(comp, run_dir / "reports" / "quality_compare.csv")
    save_df(clean_df, data_labeled_dir / "data_clean.csv")

    # ---------- Step 3: annotation + HITL ----------
    ann_cfg = cfg.get("annotation", {})
    annotator = AnnotationAgent(modality=ann_cfg.get("modality", "text"), label_space=ann_cfg.get("label_space"))
    labeled_df = annotator.auto_label(clean_df)
    save_df(labeled_df, run_dir / "data_auto_labeled.csv")
    save_df(labeled_df, data_labeled_dir / "data_auto_labeled.csv")

    spec_md = annotator.generate_spec(labeled_df, task="text_classification")
    (run_dir / "reports" / "annotation_spec.md").write_text(spec_md, encoding="utf-8")

    metrics = annotator.check_quality(labeled_df)
    (run_dir / "reports" / "annotation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # HITL queue
    thr = float(ann_cfg.get("confidence_threshold", 0.7))
    train_thr = float(ann_cfg.get("train_confidence_threshold", thr))
    prioritize_rare = bool(ann_cfg.get("prioritize_rare", True))
    review_queue = annotator.build_review_queue(labeled_df, threshold=thr, prioritize_rare=prioritize_rare)
    review_path = run_dir / "review_queue.csv"
    corrected_path = run_dir / "review_queue_corrected.csv"
    save_df(review_queue, review_path)
    save_df(review_queue, Path("review_queue.csv"))

    final_labeled = labeled_df.copy()
    # Default: use auto-label prediction, but do not treat low-confidence predictions as
    # ground-truth. This prevents the model from being trained on likely-wrong labels.
    base_pred = final_labeled.get("label_auto", final_labeled.get("label")).astype("string")
    final_labeled["label_final"] = base_pred
    if "confidence" in final_labeled.columns:
        conf = pd.to_numeric(final_labeled["confidence"], errors="coerce")
        # Use a potentially lower threshold for training inclusion to avoid
        # throwing away too many examples before HITL corrections are applied.
        low_conf_mask = conf < train_thr
        final_labeled.loc[low_conf_mask, "label_final"] = "unknown"
    metrics_after_hitl: dict[str, Any] | None = None
    if args.resume_hitl:
        # Convenience: allow editing root-level HITL file instead of the run folder.
        root_corrected = Path("review_queue_corrected.csv")
        if not corrected_path.exists() and root_corrected.exists():
            corrected_path.write_bytes(root_corrected.read_bytes())

    if args.resume_hitl and corrected_path.exists():
        corrected_all = pd.read_csv(corrected_path)
        if "label_human" not in corrected_all.columns:
            raise ValueError("review_queue_corrected.csv must contain column `label_human`.")

        corrected_all = corrected_all.copy()
        corrected_all["label_human"] = corrected_all["label_human"].astype("string")
        corrected_all = corrected_all[
            corrected_all["label_human"].notna() & (corrected_all["label_human"].astype(str).str.len() > 0)
        ].copy()
        corrected_all = corrected_all[["text", "label_human"]]
        # apply corrections by exact text match (last write wins)
        # (base label_final already contains `unknown` for low-confidence auto-labels)
        corr = corrected_all.drop_duplicates(subset=["text"], keep="last").copy()
        corr["text"] = corr["text"].astype("string")
        corr["label_human"] = corr["label_human"].astype("string")
        corr_map = dict(zip(corr["text"].tolist(), corr["label_human"].tolist()))
        base = final_labeled["label_final"].astype("string")
        corrected_labels = final_labeled["text"].astype("string").map(corr_map).astype("string")
        # Treat empty strings as missing to avoid dtype/concat warnings in newer pandas.
        corrected_labels = corrected_labels.mask(corrected_labels.astype(str).str.len() == 0, pd.NA)
        final_labeled["label_final"] = corrected_labels.fillna(base).astype("string")

        # Recompute Cohen's kappa after human corrections (only reviewed subset)
        reviewed_texts = set(corr["text"].tolist())
        df_metrics = final_labeled[final_labeled["text"].astype("string").isin(reviewed_texts)].copy()
        df_metrics["label_human"] = df_metrics["text"].astype("string").map(corr_map)
        df_metrics["label_auto"] = df_metrics["label_final"].astype("string")
        if not df_metrics.empty:
            metrics_after_hitl = annotator.check_quality(df_metrics)
            (run_dir / "reports" / "annotation_metrics_after_hitl.json").write_text(
                json.dumps(metrics_after_hitl, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # Convenience: keep a root-level copy for the final folder structure.
        try:
            Path("review_queue_corrected.csv").write_bytes(corrected_path.read_bytes())
        except Exception:
            pass
    else:
        pass

    save_df(final_labeled, run_dir / "data_labeled_final.csv")

    # LabelStudio export
    ls_path = run_dir / "labelstudio_import.json"
    ls_df = final_labeled.copy()
    ls_df["label_auto"] = ls_df["label_final"]
    annotator.export_to_labelstudio(ls_df, ls_path)

    # ---------- Step 4: Active Learning ----------
    al_cfg = cfg.get("active_learning", {})
    al_df, al_prep_meta = prepare_al_dataset(final_labeled, al_cfg)
    history_all: list[dict[str, Any]] = []
    history_df = pd.DataFrame()
    saved_examples: int | None = None
    final_model_metrics: dict[str, float] | None = None
    lc_src = run_dir / "reports" / "learning_curve.png"
    diag_path = run_dir / "reports" / "model_diagnostics.json"

    if len(al_df) >= 200 and int(al_df["label_for_al"].nunique()) >= 2:
        train_df_raw, test_df_raw = train_test_split(
            al_df,
            test_size=0.2,
            random_state=42,
            stratify=al_df["label_for_al"],
        )

        train_df_cycle = train_df_raw[["text", "label_for_al"]].rename(columns={"label_for_al": "label"}).copy()
        test_df_cycle = test_df_raw[["text", "label_for_al"]].rename(columns={"label_for_al": "label"}).copy()

        n_initial = int(al_cfg.get("n_initial", 250))
        labeled0 = train_df_cycle.sample(n=min(n_initial, len(train_df_cycle)), random_state=42)
        pool = train_df_cycle.drop(index=labeled0.index)

        al_agent = ActiveLearningAgent(model=al_cfg.get("model", "logreg"))
        for strat in al_cfg.get("strategies", ["entropy", "random"]):
            hist = al_agent.run_cycle(
                labeled_df=labeled0.reset_index(drop=True),
                pool_df=pool.reset_index(drop=True),
                test_df=test_df_cycle.reset_index(drop=True),
                strategy=strat,
                n_iterations=int(al_cfg.get("n_iterations", 5)),
                batch_size=int(al_cfg.get("batch_size", 20)),
            )
            history_all.extend(hist)

        history_df = pd.DataFrame(history_all)
        save_df(history_df, run_dir / "reports" / "al_history.csv")
        al_agent.report(history_all, lc_src)

        # "Saved examples": how many fewer labels entropy needs to reach random peak F1
        if (
            not history_df.empty
            and set(history_df["strategy"].unique()) >= {"entropy", "random"}
            and "f1" in history_df.columns
        ):
            rand = history_df[history_df["strategy"] == "random"].sort_values("n_labeled")
            ent = history_df[history_df["strategy"] == "entropy"].sort_values("n_labeled")
            target_f1 = float(rand["f1"].max())
            required_random_rows = rand[rand["f1"] >= target_f1].sort_values("n_labeled")
            required_entropy_rows = ent[ent["f1"] >= target_f1].sort_values("n_labeled")
            if len(required_random_rows) > 0 and len(required_entropy_rows) > 0:
                required_random = int(required_random_rows.iloc[0]["n_labeled"])
                required_entropy = int(required_entropy_rows.iloc[0]["n_labeled"])
                saved_examples = required_random - required_entropy

        # Final model trained on all labeled data (after HITL)
        final_train_df = al_df[["text", "label_for_al"]].rename(columns={"label_for_al": "label"}).copy()
        final_test_df = test_df_raw[["text", "label_for_al"]].rename(columns={"label_for_al": "label"}).copy()

        final_model = al_agent.fit(final_train_df)
        pred = final_model.predict(final_test_df["text"].astype("string").tolist())
        final_model_metrics = {
            "accuracy": float(accuracy_score(final_test_df["label"].astype("string").tolist(), pred)),
            "f1_macro": float(f1_score(final_test_df["label"].astype("string").tolist(), pred, average="macro")),
        }

        # Per-class diagnostics for imbalance analysis.
        y_true = final_test_df["label"].astype("string").tolist()
        y_pred = [str(x) for x in pred.tolist()]
        _write_model_diagnostics(
            diag_path,
            al_prep_meta=al_prep_meta,
            train_df=final_train_df,
            test_df=final_test_df,
            y_true=y_true,
            y_pred=y_pred,
        )
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        (run_dir / "reports" / "classification_report.json").write_text(
            json.dumps(report_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(report_dict).transpose().to_csv(run_dir / "reports" / "classification_report.csv", index=True)
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(run_dir / "reports" / "confusion_matrix.csv", index=True)

        with (models_dir / "final_model.pkl").open("wb") as f:
            pickle.dump(final_model, f)
        tfidf = getattr(final_model, "named_steps", {}).get("tfidf")
        if tfidf is not None:
            with (models_dir / "vectorizer.pkl").open("wb") as f:
                pickle.dump(tfidf, f)
    else:
        _write_model_diagnostics(
            diag_path,
            al_prep_meta=al_prep_meta,
            train_df=None,
            test_df=None,
            y_true=None,
            y_pred=None,
        )
        (run_dir / "reports" / "al_skipped.txt").write_text(
            "Active Learning skipped: not enough labeled data or only one class.\n",
            encoding="utf-8",
        )

    (run_dir / "reports" / "al_prep_meta.json").write_text(
        json.dumps(al_prep_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---------- Top-level reports ----------
    annotation_metrics_to_write = metrics_after_hitl or metrics
    write_text_md(
        reports_dir / "annotation_report.md",
        markdown_from_annotation_report(spec_md=spec_md, metrics=annotation_metrics_to_write),
    )
    write_text_md(
        reports_dir / "al_report.md",
        markdown_from_al_report(
            history_df=history_df,
            saved_examples=saved_examples,
            learning_curve_rel="learning_curve.png",
        ),
    )
    if lc_src.exists():
        (reports_dir / "learning_curve.png").write_bytes(lc_src.read_bytes())
    for fname in ["classification_report.csv", "classification_report.json", "confusion_matrix.csv", "al_prep_meta.json"]:
        src = run_dir / "reports" / fname
        if src.exists():
            (reports_dir / fname).write_bytes(src.read_bytes())
    diag_src = run_dir / "reports" / "model_diagnostics.json"
    if diag_src.exists():
        (reports_dir / "model_diagnostics.json").write_bytes(diag_src.read_bytes())

    final_report_lines: list[str] = []
    final_report_lines.append("# Final Report")
    final_report_lines.append("")
    final_report_lines.append("## Run folders")
    final_report_lines.append(f"- artifacts: `artifacts/run_<run_id>/` (see `artifacts/run_manifest.json`) ")
    final_report_lines.append("")
    final_report_lines.append("## Metrics snapshot")
    final_report_lines.append(
        f"- AL prep: dedup {al_prep_meta.get('rows_before_dedup')} -> {al_prep_meta.get('rows_after_dedup')}, "
        f"unknown_policy={al_prep_meta.get('unknown_policy')}, classes={al_prep_meta.get('n_classes_final')}"
    )
    if metrics_after_hitl is not None:
        final_report_lines.append(f"- annotation kappa (after HITL): {metrics_after_hitl.get('kappa')}")
    if final_model_metrics is not None:
        final_report_lines.append(f"- final model accuracy: {final_model_metrics.get('accuracy')}")
        final_report_lines.append(f"- final model f1_macro: {final_model_metrics.get('f1_macro')}")
    else:
        final_report_lines.append("- final model: not trained (AL skipped)")
    final_report_lines.append("")
    final_report_lines.append("## How to run HITL")
    final_report_lines.append("1) Open `review_queue.csv`.")
    final_report_lines.append("2) Fill `label_human` and save as `artifacts/run_<run_id>/review_queue_corrected.csv`.")
    final_report_lines.append("3) Rerun with `python run_pipeline.py --resume-hitl`.")
    write_text_md(reports_dir / "final_report.md", "\n".join(final_report_lines).strip() + "\n")

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

