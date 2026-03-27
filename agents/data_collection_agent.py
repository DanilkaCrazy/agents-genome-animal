from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from datasets import load_dataset

try:
    import kagglehub
except Exception:  # pragma: no cover
    kagglehub = None


SourceType = Literal["hf_dataset", "kaggle_dataset", "scrape", "api"]


@dataclass
class DataCollectionAgent:
    config: str = "config.yaml"

    def __post_init__(self) -> None:
        with open(self.config, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def scrape(self, url: str, selector: str, max_items: int | None = None) -> pd.DataFrame:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        els = soup.select(selector)
        if max_items is not None:
            els = els[: int(max_items)]
        rows = [{"text": el.get_text(" ", strip=True)} for el in els]
        df = pd.DataFrame(rows)
        df["source"] = f"scrape:{url}"
        df["label"] = "unknown"
        df["collected_at"] = self._now()
        return self._canonicalize(df)

    def fetch_api(self, endpoint: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        r = requests.get(endpoint, params=params or {}, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = [data]
        df = pd.json_normalize(data)
        return df

    def load_dataset(self, name: str, source: Literal["hf"] = "hf", **kwargs: Any) -> pd.DataFrame:
        split = kwargs.get("split", "train")
        sample = kwargs.get("sample")
        text_col = kwargs.get("text_col", "text")
        label_col = kwargs.get("label_col", "label")
        feature_cols = kwargs.get("feature_cols")  # optional for tabular datasets
        text_prefix = kwargs.get("text_prefix", "")

        ds = load_dataset(name, split=split)
        if sample is not None:
            ds = ds.shuffle(seed=42).select(range(min(int(sample), len(ds))))
        df = ds.to_pandas()

        # If dataset doesn't have a natural text column, build a text description from features.
        if text_col in df.columns:
            df = df.rename(columns={text_col: "text"})
        else:
            cols = feature_cols or [c for c in df.columns if c != label_col]
            cols = [c for c in cols if c in df.columns]
            df["text"] = df.apply(lambda r: self._row_to_text(r, cols, prefix=text_prefix), axis=1)

        if label_col in df.columns:
            df = df.rename(columns={label_col: "label"})
        else:
            df["label"] = "unknown"

        # Map numeric labels to strings if possible
        try:
            features = ds.features
            if "label" in features and getattr(features["label"], "names", None) is not None:
                names = features["label"].names
                df["label"] = df["label"].map(lambda x: names[int(x)] if pd.notna(x) else x)
        except Exception:
            pass

        df["source"] = f"hf:{name}:{split}"
        df["collected_at"] = self._now()
        return self._canonicalize(df)

    @staticmethod
    def _strip_html(text: Any) -> str:
        if text is None:
            return ""
        s = str(text)
        if "<" in s and ">" in s:
            try:
                return BeautifulSoup(s, "lxml").get_text(" ", strip=True)
            except Exception:
                return s
        return s

    @staticmethod
    def _read_kaggle_csv(local_dir: str, file_name: str | None) -> pd.DataFrame:
        from pathlib import Path

        p = Path(local_dir)
        if file_name:
            csv_path = p / file_name
            if not csv_path.exists():
                raise FileNotFoundError(f"Kaggle file not found: {csv_path}")
            return pd.read_csv(csv_path)

        # pick first CSV
        csvs = sorted(p.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in Kaggle download dir: {p}")
        return pd.read_csv(csvs[0])

    @staticmethod
    def _row_to_text(row: pd.Series, cols: list[str], prefix: str = "") -> str:
        parts = []
        for c in cols:
            v = row.get(c)
            if pd.isna(v):
                continue
            parts.append(f"{c}={v}")
        txt = "; ".join(parts)
        if prefix:
            return f"{prefix}{txt}"
        return txt

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        if not sources:
            return pd.DataFrame(columns=["text", "audio", "image", "label", "source", "collected_at"])
        df = pd.concat(sources, ignore_index=True, sort=False)
        return self._canonicalize(df)

    def run(self, sources: list[dict[str, Any]]) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for src in sources:
            t: SourceType = src["type"]
            if t == "hf_dataset":
                df = self.load_dataset(
                    name=src["name"],
                    source="hf",
                    split=src.get("split", "train"),
                    sample=src.get("sample"),
                    text_col=src.get("text_col", "text"),
                    label_col=src.get("label_col", "label"),
                )
                dfs.append(df)
            elif t == "kaggle_dataset":
                if kagglehub is None:
                    raise RuntimeError("kagglehub is required for kaggle_dataset sources. Install it and configure Kaggle creds.")
                local_dir = kagglehub.dataset_download(src["dataset"])
                raw = self._read_kaggle_csv(local_dir, src.get("file"))
                text_col = src.get("text_col", "text")
                label_col = src.get("label_col", "label")
                feature_cols = src.get("feature_cols")
                text_prefix = src.get("text_prefix", "")

                if text_col in raw.columns:
                    raw = raw.rename(columns={text_col: "text"})
                else:
                    cols = feature_cols or [c for c in raw.columns if c != label_col]
                    cols = [c for c in cols if c in raw.columns]
                    raw["text"] = raw.apply(lambda r: self._row_to_text(r, cols, prefix=text_prefix), axis=1)

                if label_col in raw.columns:
                    raw = raw.rename(columns={label_col: "label"})
                else:
                    raw["label"] = "unknown"
                if src.get("sample") is not None and len(raw) > int(src["sample"]):
                    raw = raw.sample(n=int(src["sample"]), random_state=42)
                raw["source"] = src.get("source_name") or f"kaggle:{src['dataset']}"
                raw["collected_at"] = self._now()
                dfs.append(self._canonicalize(raw[["text", "label", "source", "collected_at"]]))
            elif t == "scrape":
                dfs.append(self.scrape(url=src["url"], selector=src["selector"], max_items=src.get("max_items")))
            elif t == "api":
                params = dict(src.get("params") or {})
                if "q" in src:
                    params["q"] = src["q"]
                raw = self.fetch_api(endpoint=src["endpoint"], params=params)
                if "text_path" in src:
                    raw["text"] = raw[src["text_path"]]
                elif "text" not in raw.columns:
                    # best-effort: find first stringish column
                    str_cols = [c for c in raw.columns if raw[c].dtype == "object"]
                    raw["text"] = raw[str_cols[0]] if str_cols else raw.astype(str).agg(" ".join, axis=1)
                if src.get("html_text"):
                    raw["text"] = raw["text"].map(self._strip_html)
                raw["label"] = "unknown"
                raw["source"] = src.get("source_name") or f"api:{src['endpoint']}"
                raw["collected_at"] = self._now()
                if src.get("sample") is not None and len(raw) > int(src["sample"]):
                    raw = raw.sample(n=int(src["sample"]), random_state=42)
                dfs.append(self._canonicalize(raw[["text", "label", "source", "collected_at"]]))
            else:
                raise ValueError(f"Unknown source type: {t}")
        return self.merge(dfs)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
        # This function may receive a slice/view from upstream.
        # Copy first to avoid chained-assignment / SettingWithCopy warnings.
        df = df.copy()
        # required output columns: text/audio/image, label, source, collected_at
        for col in ["text", "audio", "image", "label", "source", "collected_at"]:
            if col not in df.columns:
                df[col] = None
        out = df[["text", "audio", "image", "label", "source", "collected_at"]].copy()

        out["text"] = out["text"].astype("string")
        out["label"] = out["label"].astype("string")
        out["source"] = out["source"].astype("string")
        out["collected_at"] = out["collected_at"].astype("string")
        return out

