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
            # Never invent a non-unknown label when we don't have enough signal.
            # This avoids polluting downstream training with arbitrary labels (e.g. "neg"/"pos").
            default_label = "unknown"
            out.loc[is_unlabeled, "label_auto"] = default_label
            out.loc[is_unlabeled, "confidence"] = 0.0
            return out

        X = labeled["text"].tolist()
        y = labeled["label"].tolist()

        # Quick baseline classifier for auto-labeling.
        clf: Pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000)),
                ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga", C=4.0)),
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

        def class_definition(label: str) -> str:
            s = str(label).strip().lower()
            if s in {"pos", "positive", "1", "true"} or "pos" in s:
                return "Positive: indicates disease/illness is present (define clinical meaning)."
            if s in {"neg", "negative", "0", "false"} or "neg" in s:
                return "Negative: indicates health/absence of disease (define clinical meaning)."
            return f"Category '{label}': define what this label means for annotators."

        def example_texts_for_class(label: str, max_examples: int = 5, min_examples: int = 3) -> list[str]:
            ex = (
                df.loc[
                    (df.get("label_auto", df["label"]).astype("string") == str(label)) & df["text"].notna(),
                    "text",
                ]
                .astype("string")
                .tolist()
            )
            ex_clean = [t.replace("\n", " ").strip() for t in ex if str(t).strip()]
            if not ex_clean:
                return []
            ex_clean = ex_clean[:max_examples]
            if len(ex_clean) >= min_examples:
                return ex_clean
            # Pad deterministically so spec always shows 3+ examples when any data exists.
            padded = ex_clean[:]
            while len(padded) < min_examples:
                padded.append(ex_clean[0])
            return padded

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
                lines.append(f"- **{c}**: {class_definition(str(c))}")
        lines.append("")
        lines.append("## Examples (3+ per class)")
        for c in classes:
            ex = example_texts_for_class(c)
            lines.append(f"### {c}")
            if ex:
                for t in ex:
                    t = t.replace("\n", " ").strip()
                    lines.append(f"- {t[:300]}")
                if len(set(ex)) < 3:
                    lines.append("- Note: fewer than 3 unique examples exist; examples may repeat in the spec.")
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
    def build_review_queue(
        df_labeled: pd.DataFrame,
        threshold: float,
        prioritize_rare: bool = True,
    ) -> pd.DataFrame:
        """
        Build HITL queue with low-confidence suggestions.
        If `prioritize_rare=True`, sort rare predicted classes first to improve macro-F1 faster.
        """
        df = df_labeled.copy()
        df["confidence"] = pd.to_numeric(df.get("confidence", np.nan), errors="coerce")

        label_col = "label_auto" if "label_auto" in df.columns else ("label" if "label" in df.columns else None)
        if label_col is None:
            raise ValueError("df_labeled must contain `label_auto` or `label` for review queue building.")

        low = df[df["confidence"] < float(threshold)].copy()

        # Compute rarity from the whole labeled set distribution (not only low-confidence rows).
        # This is intentionally aligned with macro-F1 optimization: we want to spend human effort
        # on tail classes earlier.
        if prioritize_rare:
            vc = df[label_col].astype("string").fillna("missing").value_counts()
            low["label_frequency"] = low[label_col].astype("string").fillna("missing").map(vc).fillna(0).astype(int)
            # Rare first, then lowest confidence first.
            low = low.sort_values(by=["label_frequency", "confidence"], ascending=[True, True])
        else:
            low = low.sort_values(by=["confidence"], ascending=[True])

        cols = [c for c in ["text", "label_auto", "confidence", "source", "collected_at", "label_frequency"] if c in low.columns]
        low = low[cols].reset_index(drop=True)
        low = low.rename(columns={"label_auto": "label_suggested"})
        low["label_human"] = ""
        return low

