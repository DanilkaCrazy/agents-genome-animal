from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


Strategy = Literal["entropy", "margin", "random"]


@dataclass
class ActiveLearningAgent:
    model: str = "logreg"
    random_state: int = 42

    def fit(self, labeled_df: pd.DataFrame) -> Pipeline:
        X = labeled_df["text"].astype("string").fillna("").tolist()
        y = labeled_df["label"].astype("string").tolist()
        clf: Pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(min_df=2, max_features=30000, ngram_range=(1, 2))),
                ("logreg", LogisticRegression(max_iter=300, n_jobs=1, class_weight="balanced")),
            ]
        )
        clf.fit(X, y)
        return clf

    def query(self, model: Pipeline, pool_df: pd.DataFrame, strategy: Strategy, k: int) -> list[int]:
        if len(pool_df) == 0:
            return []
        if strategy == "random":
            return pool_df.sample(n=min(k, len(pool_df)), random_state=self.random_state).index.tolist()

        Xp = pool_df["text"].astype("string").fillna("").tolist()
        probs = model.predict_proba(Xp)

        if strategy == "entropy":
            ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
            order = np.argsort(-ent)
        elif strategy == "margin":
            part = np.partition(-probs, 1, axis=1)
            top1 = -part[:, 0]
            top2 = -part[:, 1]
            margin = top1 - top2
            order = np.argsort(margin)  # smallest margin first
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        chosen = pool_df.iloc[order[: min(k, len(pool_df))]].index.tolist()
        return chosen

    def evaluate(self, model: Pipeline, labeled_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
        Xt = test_df["text"].astype("string").fillna("").tolist()
        yt = test_df["label"].astype("string").tolist()
        pred = model.predict(Xt)
        return {
            "accuracy": float(accuracy_score(yt, pred)),
            "f1": float(f1_score(yt, pred, average="macro")),
        }

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: Strategy,
        n_iterations: int,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        labeled = labeled_df.copy()
        pool = pool_df.copy()
        history: list[dict[str, Any]] = []

        for it in range(n_iterations + 1):
            model = self.fit(labeled)
            metrics = self.evaluate(model, labeled, test_df)
            history.append(
                {
                    "iteration": it,
                    "n_labeled": int(len(labeled)),
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "strategy": strategy,
                }
            )
            if it == n_iterations:
                break
            idx = self.query(model, pool, strategy=strategy, k=batch_size)
            if not idx:
                break
            newly = pool.loc[idx].copy()
            labeled = pd.concat([labeled, newly], ignore_index=True)
            pool = pool.drop(index=idx)

        return history

    def report(self, history: list[dict[str, Any]], out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(history)
        plt.figure(figsize=(8, 5))
        for strat in df["strategy"].unique():
            sub = df[df["strategy"] == strat].sort_values("n_labeled")
            plt.plot(sub["n_labeled"], sub["f1"], marker="o", label=strat)
        plt.xlabel("n_labeled")
        plt.ylabel("F1 (macro)")
        plt.title("Active Learning: learning curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

