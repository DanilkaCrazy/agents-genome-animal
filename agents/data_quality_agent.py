from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


MissingStrategy = Literal["drop", "fill_unknown"]
DuplicateStrategy = Literal["drop", "keep_first"]
OutlierStrategy = Literal["clip_iqr", "remove_iqr", "none"]


@dataclass
class DataQualityAgent:
    def detect_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        report: dict[str, Any] = {}

        # For a text-only pipeline, `audio`/`image` are expected to be null.
        # To keep the quality report meaningful, we treat columns that are
        # entirely-missing across the whole dataset as "not applicable".
        is_na = df.isna()
        missing = is_na.sum().to_dict()
        missing_pct = (is_na.mean() * 100).round(2).to_dict()
        for col in ["audio", "image"]:
            if col in df.columns and bool(is_na[col].all()):
                missing[col] = 0
                missing_pct[col] = 0.0
        report["missing"] = {"count": missing, "pct": missing_pct}

        dup_n = int(df.duplicated(subset=["text"]).sum()) if "text" in df.columns else int(df.duplicated().sum())
        report["duplicates"] = dup_n

        if "text" in df.columns:
            lengths = df["text"].astype("string").str.len().fillna(0).astype(int)
            q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
            iqr = q3 - q1
            lo = float(q1 - 1.5 * iqr)
            hi = float(q3 + 1.5 * iqr)
            outlier_mask = (lengths < lo) | (lengths > hi)
            report["outliers"] = {
                "feature": "text_length",
                "method": "iqr",
                "bounds": {"lo": lo, "hi": hi},
                "n": int(outlier_mask.sum()),
                "examples_idx": df.index[outlier_mask].to_list()[:20],
            }
        else:
            report["outliers"] = {"feature": None, "method": None, "bounds": None, "n": 0, "examples_idx": []}

        if "label" in df.columns:
            vc = df["label"].astype("string").fillna("missing").value_counts()
            total = int(vc.sum())
            dist = (vc / max(total, 1)).round(4).to_dict()
            labels_wo_missing = [x for x in vc.index.tolist() if str(x) != "missing"]
            min_non_missing = min((int(vc.loc[x]) for x in labels_wo_missing), default=0)
            imbalance = {
                "n_classes": int(len(labels_wo_missing)),
                "counts": vc.to_dict(),
                "dist": dist,
                "max_min_ratio": float((int(vc.max()) / max(min_non_missing, 1)) if labels_wo_missing else np.inf),
            }
            report["imbalance"] = imbalance
        else:
            report["imbalance"] = {"n_classes": 0, "counts": {}, "dist": {}, "max_min_ratio": None}

        return report

    def fix(
        self,
        df: pd.DataFrame,
        strategy: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        strategy = strategy or {}
        missing_s: MissingStrategy = strategy.get("missing", "drop")
        dup_s: DuplicateStrategy = strategy.get("duplicates", "drop")
        out_s: OutlierStrategy = strategy.get("outliers", "clip_iqr")

        out = df.copy()

        if missing_s == "drop":
            out = out.dropna(subset=["text"])
        elif missing_s == "fill_unknown":
            if "label" in out.columns:
                out["label"] = out["label"].fillna("unknown")
            if "text" in out.columns:
                out["text"] = out["text"].fillna("")

        if dup_s in ("drop", "keep_first"):
            if "text" in out.columns:
                out = out.drop_duplicates(subset=["text"], keep="first")
            else:
                out = out.drop_duplicates(keep="first")

        if "text" in out.columns and out_s != "none":
            lengths = out["text"].astype("string").str.len().fillna(0).astype(int)
            q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            if out_s == "remove_iqr":
                out = out[(lengths >= lo) & (lengths <= hi)]
            elif out_s == "clip_iqr":
                clipped = lengths.clip(lower=int(max(lo, 0)), upper=int(max(hi, 0)))
                # If clipping changes length, keep text but store derived feature for downstream EDA
                out["text_length"] = clipped

        out = out.reset_index(drop=True)
        return out

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        rep_before = self.detect_issues(df_before)
        rep_after = self.detect_issues(df_after)

        rows = []
        rows.append(
            {
                "metric": "rows",
                "before": len(df_before),
                "after": len(df_after),
            }
        )
        rows.append({"metric": "duplicates(text)", "before": rep_before["duplicates"], "after": rep_after["duplicates"]})
        rows.append(
            {
                "metric": "outliers(text_length)",
                "before": rep_before["outliers"]["n"],
                "after": rep_after["outliers"]["n"],
            }
        )

        for col in sorted(set(df_before.columns) | set(df_after.columns)):
            b = int(df_before[col].isna().sum()) if col in df_before.columns else None
            a = int(df_after[col].isna().sum()) if col in df_after.columns else None
            if b is not None or a is not None:
                rows.append({"metric": f"missing({col})", "before": b, "after": a})

        return pd.DataFrame(rows)

