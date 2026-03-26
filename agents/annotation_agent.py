from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


Modality = Literal["text", "audio", "image"]


@dataclass
class AnnotationAgent:
    modality: Modality = "text"
    label_space: list[str] | None = None

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.modality != "text":
            raise NotImplementedError("This template implements auto_label for text only.")

        out = df.copy()
        out["text"] = out["text"].astype("string").fillna("")
        out["label"] = out["label"].astype("string")

        is_unlabeled = out["label"].isna() | (out["label"].str.lower().isin(["unknown", "none", "nan", ""]))
        labeled = out[~is_unlabeled].copy()
        unlabeled = out[is_unlabeled].copy()

        out["confidence"] = np.nan
        out["label_auto"] = out["label"]

        if len(unlabeled) == 0:
            out["confidence"] = 1.0
            return out

        # If no labeled data exists, assign a default label with zero confidence.
        if len(labeled) < 50 or labeled["label"].nunique() < 2:
            default_label = (self.label_space or ["unknown"])[0]
            out.loc[is_unlabeled, "label_auto"] = default_label
            out.loc[is_unlabeled, "confidence"] = 0.0
            return out

        X = labeled["text"].tolist()
        y = labeled["label"].tolist()

        # Quick baseline classifier for auto-labeling.
        clf: Pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(min_df=2, max_features=20000, ngram_range=(1, 2))),
                ("logreg", LogisticRegression(max_iter=200, n_jobs=1, class_weight="balanced")),
            ]
        )
        clf.fit(X, y)

        probs = clf.predict_proba(unlabeled["text"].tolist())
        pred_idx = probs.argmax(axis=1)
        pred = np.asarray(clf.classes_)[pred_idx]
        conf = probs.max(axis=1)

        out.loc[is_unlabeled, "label_auto"] = pred
        out.loc[is_unlabeled, "confidence"] = conf

        # For already-labeled rows, set confidence=1.
        out.loc[~is_unlabeled, "confidence"] = 1.0
        return out

    def generate_spec(self, df: pd.DataFrame, task: str) -> str:
        labels = (
            df.get("label_auto", df.get("label"))
            .astype("string")
            .dropna()
            .loc[lambda s: ~s.str.lower().isin(["unknown", "none", "nan", ""])]
            .value_counts()
        )
        classes = labels.index.tolist()

        lines: list[str] = []
        lines.append(f"# Annotation spec: {task}")
        lines.append("")
        lines.append("## Task")
        lines.append(task)
        lines.append("")
        lines.append("## Classes")
        if not classes:
            lines.append("- (no classes detected yet)")
        else:
            for c in classes:
                lines.append(f"- **{c}**: (define here)")
        lines.append("")
        lines.append("## Examples (3+ per class)")
        for c in classes:
            ex = (
                df.loc[(df.get("label_auto", df["label"]).astype("string") == c) & df["text"].notna(), "text"]
                .astype("string")
                .head(3)
                .tolist()
            )
            lines.append(f"### {c}")
            if ex:
                for t in ex:
                    t = t.replace("\n", " ").strip()
                    lines.append(f"- {t[:300]}")
            else:
                lines.append("- (add examples)")
            lines.append("")
        lines.append("## Edge cases")
        lines.append("- Short/ambiguous text")
        lines.append("- Mixed sentiment / multiple topics")
        lines.append("- Sarcasm / negation")
        lines.append("")

        return "\n".join(lines).strip() + "\n"

    def check_quality(self, df_labeled: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        label_col = "label_auto" if "label_auto" in df_labeled.columns else "label"

        dist = df_labeled[label_col].astype("string").value_counts().to_dict()
        out["label_dist"] = dist
        out["confidence_mean"] = float(pd.to_numeric(df_labeled.get("confidence", pd.Series(dtype=float)), errors="coerce").mean())

        if "label_human" in df_labeled.columns:
            y1 = df_labeled["label_human"].astype("string").fillna("missing")
            y2 = df_labeled[label_col].astype("string").fillna("missing")
            out["kappa"] = float(cohen_kappa_score(y1, y2))
        else:
            out["kappa"] = None

        return out

    def export_to_labelstudio(self, df: pd.DataFrame, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        label_col = "label_auto" if "label_auto" in df.columns else "label"
        conf = pd.to_numeric(df.get("confidence", pd.Series([None] * len(df))), errors="coerce")

        tasks = []
        for i, row in df.reset_index(drop=True).iterrows():
            text = (row.get("text") or "").strip()
            if not text:
                continue
            label = row.get(label_col)
            if isinstance(label, pd.Series):
                label = label.iloc[0] if len(label) else None
            score = conf.iloc[i]
            task = {
                "id": int(i),
                "data": {"text": text},
            }
            if pd.notna(label):
                task["predictions"] = [
                    {
                        "model_version": "auto_label_v1",
                        "score": float(score) if pd.notna(score) else 0.0,
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [str(label)]},
                            }
                        ],
                    }
                ]
            tasks.append(task)

        out_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    @staticmethod
    def build_review_queue(df_labeled: pd.DataFrame, threshold: float) -> pd.DataFrame:
        df = df_labeled.copy()
        df["confidence"] = pd.to_numeric(df.get("confidence", np.nan), errors="coerce")
        low = df[df["confidence"] < float(threshold)].copy()
        cols = [c for c in ["text", "label_auto", "confidence", "source", "collected_at"] if c in low.columns]
        low = low[cols].reset_index(drop=True)
        low = low.rename(columns={"label_auto": "label_suggested"})
        low["label_human"] = ""
        return low

