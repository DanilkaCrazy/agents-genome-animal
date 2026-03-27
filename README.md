
# Genome AI Agents — End-to-End Data Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)

**Genome AI Agents** is a production‑ready, multi‑agent data pipeline that unifies heterogeneous agricultural data sources (Kaggle, Hugging Face, and live web scraping) into a single decision support system for cattle health management. It automates data collection, quality control, annotation, and active learning, with a mandatory human‑in‑the‑loop (HITL) checkpoint to ensure expert‑level accuracy.

The project fulfills a four‑assignment academic track (Genome AI Agents) and demonstrates a complete machine learning lifecycle that can be reproduced with a single command.

---

## 📖 Table of Contents

- [Project Overview](#project-overview)
- [ML Task and Data Sources](#ml-task-and-data-sources)
- [Solution: What the Pipeline Does](#solution-what-the-pipeline-does)
- [Architecture](#architecture)
- [Technical Highlights](#technical-highlights)
- [Agent Specifications](#agent-specifications)
- [History of Improvements](#history-of-improvements)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Human‑in‑the‑Loop Workflow](#human-in-the-loop-workflow)
- [Active Learning](#active-learning)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [Technology Stack](#technology-stack)
- [Out of Scope](#out-of-scope)
- [Competitive Advantages](#competitive-advantages)
- [Team and Context](#team-and-context)
- [License](#license)

---

## Project Overview

Genome AI Agents is a multi‑agent pipeline that:

- **Collects** data from three distinct sources (Kaggle, Hugging Face, and web scraping) and unifies them under a common schema.
- **Cleans** the data by detecting and fixing quality issues (missing values, duplicates, outliers).
- **Automatically labels** unlabeled examples using zero‑shot classification, exporting a queue for human review when confidence is low.
- **Executes Active Learning** cycles to select the most informative unlabeled examples, training a baseline model and producing learning curves.
- **Resumes** after human intervention, applying corrected labels and retraining the final model.

The pipeline is orchestrated by a single Python script (`run_pipeline.py`) and stores every intermediate artifact in a timestamped folder, ensuring full reproducibility.

---

## ML Task and Data Sources

### What we predict (`label`)

We predict the **health status / disease** of cattle (multiclass classification).  
When a source already contains a label (e.g., “healthy”, “mastitis”, “lameness”), it is placed in the `label` column.  
When a source has no label (e.g., scraped commodity prices), the label is set to `"unknown"` and the record serves as context.

### Unifying heterogeneous sources

All agents work on a single `pandas.DataFrame` with fixed columns:

- **`text`** – textual representation of the example (for tabular data, we construct a string like `"feature1=value1; feature2=value2; …"`).
- **`audio`** – always `null` (text‑only pipeline).
- **`image`** – always `null` (text‑only pipeline).
- **`label`** – original label (may be `"unknown"`).
- **`source`** – provenance (`hf:…`, `kaggle:…`, `scrape:…`).
- **`collected_at`** – UTC timestamp in ISO format.

### Three configured sources

1. **Kaggle – Cattle health and feeding data**  
   - **Source:** `shahhet2812/cattle-health-and-feeding-data`  
   - **Role:** Main farm journal (feeding, clinical signs, productivity → health label).  
   - **Transformation:** Build `text` from all feature columns (except the label column) as a concatenation of `key=value` pairs. `label` is taken from the provided health column.

2. **Hugging Face – Cattle disease prediction**  
   - **Source:** `roshan8312/cattle-disease-prediction`  
   - **Role:** Independent symptom‑to‑diagnosis dataset.  
   - **Transformation:** Convert symptom vectors to a `text` string (`symptom1=value; symptom2=value; …`), `label` = `prognosis`.

3. **Scraping – Commodity prices (soybean meal)**  
   - **Source:** IndexMundi – [soybean meal prices](https://www.indexmundi.com/commodities/?commodity=soybean-meal&months=12)  
   - **Role:** Show that the pipeline can ingest live web data; adds contextual records without labels.  
   - **Transformation:** Each table row becomes a `text` record, `label = "unknown"`, `source = "scrape:…"`.

---

## Solution: What the Pipeline Does

The pipeline runs end‑to‑end with one command (`python run_pipeline.py`). Below is a step‑by‑step flow:

1. **Data Collection** (`DataCollectionAgent`)  
   - Loads the three sources described above.  
   - Unifies them into a single `DataFrame` with the common schema.

2. **Data Quality** (`DataQualityAgent`)  
   - Scans for missing values, duplicates, and outliers (IQR method).  
   - Applies configurable cleaning strategies (e.g., `drop` for missing, `drop` for duplicates, `clip_iqr` for outliers).  
   - Produces a comparison report (`quality_compare.csv`) showing improvements.

3. **Auto‑Annotation** (`AnnotationAgent`)  
   - Performs zero‑shot classification on rows with `label = "unknown"` using a Hugging Face pipeline (e.g., `facebook/bart-large-mnli`).  
   - Adds `label_auto` (predicted label) and `confidence` (0–1).  
   - Builds a HITL queue (`review_queue.csv`) for examples with confidence below a threshold (default `0.7`). Prioritizes rare classes.

4. **Human‑in‑the‑Loop**  
   - The pipeline stops after generating `review_queue.csv`.  
   - A domain expert opens the file, fills in the correct labels in column `label_human`, and saves it as `review_queue_corrected.csv`.  
   - The pipeline is resumed with `python run_pipeline.py --resume-hitl`, applying the corrections to produce `label_final`.

5. **Active Learning** (`ActiveLearningAgent`)  
   - Prepares a leakage‑safe dataset (deduplication by `text`, handling of `unknown`, filtering rare classes).  
   - Trains a baseline model (TF‑IDF + LogisticRegression).  
   - Runs several cycles of entropy‑based sampling, each time selecting the most uncertain unlabeled examples.  
   - Saves a learning curve (`learning_curve.png`) comparing entropy vs. random sampling.

6. **Final Training and Export**  
   - Trains the final model on all validated data.  
   - Exports the final labeled dataset (`data_labeled_final.csv`) and a LabelStudio import file (`labelstudio_import.json`).  
   - Generates comprehensive reports (`classification_report.csv`, `confusion_matrix.csv`, `model_diagnostics.json`).

All intermediate and final artifacts are stored in a timestamped folder under `artifacts/run_<run_id>/`, together with a `run_manifest.json` that records every step’s output paths.

---

## Architecture

The pipeline is built as a sequential chain of independent agents, each with a well‑defined interface. Orchestration is done in a simple Python script (`run_pipeline.py`); no external workflow manager is required, but the design is modular enough to be ported to Prefect or Airflow.

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ DataCollection   │    │ DataQuality      │    │ Annotation       │
│ Agent            │───▶│ Agent            │───▶│ Agent            │
│                  │    │                  │    │                  │
│ - load Kaggle    │    │ - detect issues  │    │ - auto_label     │
│ - load HF        │    │ - fix            │    │ - generate spec  │
│ - scrape         │    │ - compare        │    │ - export LS      │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                         │
                                                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ ActiveLearning   │    │ Human‑in‑the‑Loop│    │ low confidence   │
│ Agent            │◀───│ review           │◀───│ examples flagged │
│                  │    │                  │    │                  │
│ - fit model      │    │ - open CSV       │    │                  │
│ - query uncertain│    │ - correct labels │    │                  │
│ - evaluate       │    │ - resume pipeline│    │                  │
│ - report curves  │    │                  │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

### Key design decisions

- **Unified text interface** – Even tabular data is transformed into a textual “cattle record” description, allowing all agents to work on a single modality (text). This satisfies the assignment’s requirement for a common schema.
- **Human‑in‑the‑Loop as a first‑class citizen** – The pipeline explicitly stops and requires manual intervention, ensuring that the final dataset benefits from expert knowledge.
- **Reproducibility** – All configurations are stored in `config.yaml`, and all runs produce a complete artifact folder with every intermediate file, enabling full traceability.

---

## Technical Highlights

1. **Heterogeneous source unification**  
   The `DataCollectionAgent` adapts each source to the common schema. For tabular sources, it dynamically generates text strings from feature–value pairs; for scraped HTML, it extracts table rows and converts them to text.

2. **Configurable data quality strategies**  
   The `DataQualityAgent` supports multiple cleaning strategies that can be chosen per problem type (e.g., for missing values, the user can select `'median'`, `'mode'`, or `'drop'`). The `compare()` method produces a quantitative report showing how each metric changed after cleaning.

3. **Zero‑shot annotation with confidence**  
   The `AnnotationAgent` uses a Hugging Face zero‑shot classification pipeline to assign labels to unlabeled rows. It outputs both the predicted label and the confidence score, and automatically creates a HITL queue for low‑confidence items.

4. **Active learning with entropy sampling**  
   The `ActiveLearningAgent` implements both random and entropy‑based sampling strategies. By comparing the learning curves, the user can see how many labeled examples are saved to reach a target accuracy—a direct demonstration of the value of active learning.

5. **Full pipeline resumability**  
   The pipeline saves its state after each major step. If the user runs `python run_pipeline.py --resume-hitl`, it loads the previous review queue corrections and continues from where it left off, without re‑doing earlier steps.

6. **Prediction collapse diagnostics**  
   The pipeline saves `model_diagnostics.json` with train/test class distributions, prediction distributions, and a flag `prediction_collapse` to quickly detect when a model predicts only one class. This helped identify the need to switch to character n‑grams for feature extraction.

7. **Unsupervised clustering for `unknown`**  
   Scripts `cluster_unknowns.py` and `visualize_unknown_clusters.py` help a human expert batch‑label large groups of `unknown` examples by clustering similar texts and providing a 2D visualization.

---

## Agent Specifications

### DataCollectionAgent

Main methods:
- `scrape(url, selector, max_items)` → `DataFrame`
- `fetch_api(endpoint, params)` → `DataFrame` (best‑effort normalizing JSON)
- `load_dataset(name, split, sample, text_col, label_col, feature_cols, text_prefix)` → `DataFrame`
- `merge(sources)` → `DataFrame` (concat + canonicalize)

Key guarantees:
- Unifies the schema (including `audio`/`image` as `null` for text‑only pipelines)
- Serializes `text/label/source/collected_at` as string dtype

### DataQualityAgent

Main methods:
- `detect_issues(df)` → `dict`
- `fix(df, strategy)` → `DataFrame`
- `compare(df_before, df_after)` → `DataFrame`

Logic:
- **missing**: count/pct per column (audio/image considered N/A if entire column is null)
- **duplicates**: count duplicates by `text`
- **outliers**: IQR‑based boundaries on `text_length` (with example indices)
- **imbalance**: distribution by `label` and max/min ratio

Fix strategies:
- missing: `drop` or `fill_unknown`
- duplicates: `drop` or `keep_first`
- outliers: `clip_iqr`, `remove_iqr`, `none`

### AnnotationAgent

Main methods:
- `auto_label(df, modality='text')` → `DataFrame`
- `generate_spec(df, task)` → `str` (Markdown)
- `check_quality(df_labeled)` → `dict` (label distribution, confidence, kappa if `label_human` present)
- `export_to_labelstudio(df, out_path)` → `Path` (JSON import for LabelStudio)

HITL queue:
- `build_review_queue(df_labeled, threshold, prioritize_rare=True)` returns low‑confidence examples.
- If `prioritize_rare=True`, sorts first by rarity of predicted class (`label_frequency` calculated over entire dataset), then by ascending `confidence`.
- Queue columns: `text`, `label_suggested`, `confidence`, `source`, `collected_at`, `label_human` (empty).

### ActiveLearningAgent

Main methods:
- `fit(labeled_df)` → sklearn Pipeline (TF‑IDF + LogisticRegression)
- `query(model, pool_df, strategy, k)` → list of indices
- `evaluate(model, labeled_df, test_df)` → `dict` (accuracy, macro‑F1)
- `run_cycle(...)` → history list of dicts
- `report(history, out_path)` → learning‑curve plot (F1 vs n_labeled)

Sampling strategies:
- `entropy`: selects examples with highest prediction entropy
- `margin`: selects examples with smallest difference between top‑2 probabilities
- `random`: random selection

---

## History of Improvements

This section documents how the pipeline evolved and the impact on model quality.

### Stage 1 – Data Collection
Initial version collected data from the three sources and unified them. Early runs showed large class imbalance and many `unknown` rows.

### Stage 2 – Data Quality
Added `DataQualityAgent` to detect duplicates and outliers. The initial dataset had significant duplicates by `text` and strong class imbalance.

### Stage 3 – Auto‑Annotation + HITL
Introduced zero‑shot labeling with confidence scores, and a HITL queue that prioritizes rare classes. This allowed experts to focus on the most uncertain examples first.

### Stage 4 – Leakage‑safe Preparation
Added preparation before training: deduplication by `text` before train/test split, policies for `unknown` (`drop`/`cap`/`keep`), and filtering of too‑rare classes (`min_class_count`). This prevented data leakage and reduced noise.

### Problem – Prediction Collapse
Early baseline models predicted only one class for all test examples. This was detected by `model_diagnostics.json` (`prediction_collapse = True`). The input texts were structural (many `feature=0/1` tokens), and word‑TF‑IDF did not provide enough discriminative signal.

### Fix – Character n‑grams
Switched from word‑based TF‑IDF to character n‑grams (`char_wb`, 3–5) with solver `saga` and increased `max_iter`. This change was applied in both `al_agent.py` (final model) and `annotation_agent.py` (auto‑annotation).

### Result – Improved Metrics
After the fix, metrics stabilised:
- accuracy = 0.933
- macro‑F1 = 0.881
- `prediction_collapse = False`

Class distributions and prediction distributions are now saved in `model_diagnostics.json` to quickly spot regressions.

### Unsupervised ML for `unknown` – Accelerating HITL
For the large `unknown` class, we added clustering tools:
- `scripts/cluster_unknowns.py` → `reports/unknown_clusters.csv` and `unknown_clusters.md`
- `scripts/visualize_unknown_clusters.py` → `reports/unknown_clusters_plot.png`

These allow an expert to review groups of similar texts and assign labels in bulk.

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/genome-ai-agents.git
   cd genome-ai-agents
   ```

2. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Kaggle API (optional, only if you use Kaggle sources)**  
   - Place your `kaggle.json` in `~/.kaggle/` or set the environment variable `KAGGLE_CONFIG_DIR`.

---

## Running the Pipeline

### Full pipeline run
```bash
python run_pipeline.py
```
This executes all steps sequentially: collection → quality → auto‑annotation → active learning → final training. The pipeline will stop after generating the HITL queue (`review_queue.csv`).

### Resume after human review
```bash
python run_pipeline.py --resume-hitl
```
This loads the corrected labels from `review_queue_corrected.csv`, applies them to the dataset, and continues with active learning and final training.

### Skip active learning (if you want only the labeled dataset)
You can comment out the active learning step in `run_pipeline.py` or set `active_learning: enabled: false` in `config.yaml`.

---

## Human‑in‑the‑Loop Workflow

The HITL step is **mandatory** – the pipeline will not proceed to active learning without human correction.

1. After the first run, open the generated `review_queue.csv` (located in `artifacts/run_<run_id>/`).
2. Fill in the `label_human` column with the correct labels for each row (you can leave rows empty if you want to keep the suggested label).
3. Save the file as `review_queue_corrected.csv` **in the same folder**.
4. Run the pipeline again with `--resume-hitl`.

The pipeline will then:
- Merge the human corrections into the dataset, producing `label_final`.
- Optionally compute Cohen’s kappa on the reviewed subset if enough ground truth is available.
- Continue with active learning and final training.

---

## Active Learning

The `ActiveLearningAgent` implements a standard pool‑based active learning loop:

- **Initial set**: A small random subset of labeled examples (controlled by `n_initial` in `config.yaml`).
- **Model**: TF‑IDF vectorizer + LogisticRegression (with character n‑grams to avoid prediction collapse).
- **Query strategies**:
  - `entropy`: selects examples with the highest prediction entropy.
  - `margin`: selects examples with the smallest difference between top‑2 predicted probabilities.
  - `random`: baseline random selection.
- **Loop**: At each iteration, `batch_size` examples are selected from the unlabeled pool, added to the training set, and the model is retrained.
- **Evaluation**: At each iteration, the model is evaluated on a held‑out test set (macro‑F1). A learning curve is plotted and saved as `learning_curve.png`.

All intermediate results are saved in `reports/al_history.csv` and `reports/al_report.md`.

---

## Outputs and Artifacts

The pipeline writes artifacts in two places:

- **`artifacts/run_<run_id>/`** – a complete, timestamped snapshot of the run. Contains:
  - `data_raw.csv`
  - `data_clean.csv`
  - `data_auto_labeled.csv`
  - `review_queue.csv`, `review_queue_corrected.csv` (if resume)
  - `data_labeled_final.csv`
  - `labelstudio_import.json`
  - `reports/` – all quality, annotation, and AL reports
  - `run_manifest.json` – metadata about the run

- **Latest copies** (overwritten on each run) in the project root for convenience:
  - `data/raw/data_raw.csv`
  - `data/labeled/data_clean.csv`, `data_labeled_final.csv`
  - `reports/` – summary reports (quality, annotation, AL, final)
  - `models/final_model.pkl`, `models/vectorizer.pkl`
  - `review_queue.csv`

### Key output files

| File | Description |
|------|-------------|
| `reports/quality_report.md` | Quality issues detected and how they were fixed |
| `reports/annotation_report.md` | Auto‑annotation statistics and HITL results |
| `reports/al_report.md` | Active learning summary and learning curve |
| `reports/final_report.md` | Final model metrics and dataset summary |
| `reports/classification_report.csv` | Precision/recall/F1 per class |
| `reports/confusion_matrix.csv` | Confusion matrix |
| `reports/model_diagnostics.json` | Class distributions and prediction collapse flag |
| `models/final_model.pkl` | Serialized sklearn Pipeline (TF‑IDF + LR) |
| `models/vectorizer.pkl` | Trained TF‑IDF vectorizer |

---

## Configuration

All parameters are set in `config.yaml`. Key sections:

```yaml
collection:
  sources:
    - type: kaggle
      name: shahhet2812/cattle-health-and-feeding-data
      text_col: null           # auto‑build from all feature columns
      label_col: health_status
    - type: hf
      name: roshan8312/cattle-disease-prediction
      text_prefix: symptoms:   # optional
      label_col: prognosis
    - type: scrape
      url: https://www.indexmundi.com/commodities/?commodity=soybean-meal&months=12
      selector: table

quality:
  fix:
    missing: drop               # drop | fill_unknown
    duplicates: drop            # drop | keep_first
    outliers: clip_iqr          # clip_iqr | drop | none

annotation:
  modality: text
  confidence_threshold: 0.7
  prioritize_rare: true

active_learning:
  model: logistic_regression
  n_initial: 50
  n_iterations: 5
  batch_size: 20
  strategies: [entropy, random]
  # Data preparation settings (leakage prevention)
  dedup_before_split: true
  unknown_policy: drop         # drop | cap | keep
  unknown_cap_ratio: 0.3
  min_class_count: 2
```

---

## Notebooks

The `notebooks/` folder contains Jupyter notebooks for exploration and experiments:

- `eda.ipynb` – Exploratory data analysis: class distribution, text length, top words.
- `al_experiment.ipynb` – Compare active learning strategies (entropy vs random) and plot learning curves.
- `quality_analysis.ipynb` – Deep dive into quality issues and the effect of cleaning.
- `annotation_analysis.ipynb` – Analyze auto‑annotation results, confidence distributions, and HITL impact.

These notebooks are generated by `scripts/generate_notebooks.py` but can also be run manually.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Core languages | Python 3.11+, Jupyter Notebook |
| Data processing | pandas, numpy, scikit‑learn |
| Data sources | KaggleHub (Kaggle API), HuggingFace Datasets, BeautifulSoup |
| Annotation | Hugging Face Transformers (zero‑shot classification) |
| Active learning | Custom entropy sampling, TF‑IDF, LogisticRegression |
| Visualisation | Matplotlib, seaborn |
| Serialisation | JSON, CSV |
| Orchestration | Python script (custom) |
| Environment | `requirements.txt`, `config.yaml` |

---

## Out of Scope

- **Real‑time streaming** – batch processing only.
- **Deep learning models** – baseline is logistic regression; no neural networks for final prediction (though zero‑shot uses transformers for annotation).
- **Multi‑user collaboration** – no authentication or shared workspaces.
- **Production deployment** – no Docker/Kubernetes scripts (though code is container‑ready).

---

## Competitive Advantages

| Feature | Genome AI Agents | Typical Open‑Source Pipelines | Commercial Agri‑Analytics |
|---------|------------------|-------------------------------|---------------------------|
| Heterogeneous sources | Native (Kaggle, HF, scraping) | Usually single source | Often proprietary integrations |
| Human‑in‑the‑Loop | Mandatory, explicit | Rarely implemented | Sometimes (expensive add‑ons) |
| Active learning | Built‑in with comparison | Not common | Limited |
| Reproducibility | Full artifact versioning | Partial | Varies |
| Educational value | Designed as a teaching case | Focused on one task | Opaque |
| Cost | Open‑source, free | Free | High subscription |

The key differentiator is the **tight integration of active learning with a mandatory human validation step**, which directly addresses the trust gap in agricultural AI. By forcing the user to review uncertain cases, the pipeline ensures that the final model is trained on high‑quality, domain‑verified data.

---

## Team and Context

This project was developed as part of the **Data Collection, Processing, and Labeling in Machine Learning** academic track (March–April 2026).

**Author:**  
Danil Vishnyakov – MSc in AI at ITMO University; former Data Scientist at Motiv Telecom and Systems Analyst at Trading House UET. Background in bioinformatics and production ML.

The pipeline fulfills four graded assignments (`DataCollectionAgent`, `DataQualityAgent`, `AnnotationAgent`, `ActiveLearningAgent`) and a final capstone that integrates them into a single runnable project.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

For questions, issues, or contributions, please open an issue on GitHub.

