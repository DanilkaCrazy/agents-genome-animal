from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def quality_notebook() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(
                "## Quality Analysis — `DataQualityAgent` output\n\n"
                "Requirements mapping:\n"
                "- Detect missing values, duplicates, outliers, and class imbalance (Part 1)\n"
                "- Apply at least 2 cleaning strategies and compare (Part 2)\n"
                "- Justify the best strategy in a Markdown cell (Part 3)\n"
            ),
            new_code_cell(
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "\n"
                "from agents.data_quality_agent import DataQualityAgent\n"
                "\n"
                "sns.set_theme(style='whitegrid')\n"
                "\n"
                "# Use latest artifacts if available; otherwise fall back to data/raw/\n"
                "raw_candidates = sorted(Path('artifacts').glob('run_*/data_raw.csv'))\n"
                "path = raw_candidates[-1] if raw_candidates else Path('data/raw/data_raw.csv')\n"
                "df = pd.read_csv(path)\n"
                "\n"
                "df.head()"
            ),
            new_code_cell(
                "dq = DataQualityAgent()\n"
                "report = dq.detect_issues(df)\n"
                "report"
            ),
            new_code_cell(
                "# Part 1: Visualize missingness, duplicates, outliers, and imbalance\n"
                "import numpy as np\n"
                "\n"
                "missing_counts = report['missing']['count']\n"
                "missing_pct = report['missing']['pct']\n"
                "top_missing = sorted(missing_pct.items(), key=lambda kv: kv[1], reverse=True)[:10]\n"
                "\n"
                "plt.figure(figsize=(10,4))\n"
                "cols = [k for k,_ in top_missing]\n"
                "vals = [v for _,v in top_missing]\n"
                "sns.barplot(x=vals, y=cols)\n"
                "plt.title('Top missing columns (%)')\n"
                "plt.tight_layout()\n"
                "plt.show()\n"
                "\n"
                "# Duplicates (count)\n"
                "dup_n = int(report.get('duplicates') or 0)\n"
                "plt.figure(figsize=(5,3))\n"
                "sns.barplot(x=['duplicates(text)'], y=[dup_n])\n"
                "plt.title('Duplicate count')\n"
                "plt.tight_layout()\n"
                "plt.show()\n"
                "\n"
                "text_length = df['text'].astype(str).str.len()\n"
                "plt.figure(figsize=(8,4))\n"
                "sns.histplot(text_length, bins=50)\n"
                "bounds = report.get('outliers', {}).get('bounds', {})\n"
                "lo = bounds.get('lo', None)\n"
                "hi = bounds.get('hi', None)\n"
                "if lo is not None:\n"
                "    plt.axvline(lo, color='red', linestyle='--', linewidth=1)\n"
                "if hi is not None:\n"
                "    plt.axvline(hi, color='red', linestyle='--', linewidth=1)\n"
                "plt.title('Text length distribution')\n"
                "plt.tight_layout()\n"
                "plt.show()\n"
                "\n"
                "vc = df['label'].astype(str).value_counts()\n"
                "plt.figure(figsize=(10,4))\n"
                "sns.barplot(x=vc.index.astype(str), y=vc.values)\n"
                "plt.xticks(rotation=45, ha='right')\n"
                "plt.title('Class distribution (label)')\n"
                "plt.tight_layout()\n"
                "plt.show()\n"
            ),
            new_code_cell(
                "# Part 2: Try two cleaning strategies and compare\n"
                "strategy_a = {\n"
                "    'missing': 'drop',\n"
                "    'duplicates': 'drop',\n"
                "    'outliers': 'clip_iqr',\n"
                "}\n"
                "strategy_b = {\n"
                "    'missing': 'fill_unknown',\n"
                "    'duplicates': 'keep_first',\n"
                "    'outliers': 'remove_iqr',\n"
                "}\n"
                "\n"
                "df_a = dq.fix(df, strategy=strategy_a)\n"
                "df_b = dq.fix(df, strategy=strategy_b)\n"
                "\n"
                "cmp_a = dq.compare(df, df_a)\n"
                "cmp_b = dq.compare(df, df_b)\n"
                "\n"
                "comparison = pd.merge(\n"
                "    cmp_a.rename(columns={'before':'before_a','after':'after_a'}),\n"
                "    cmp_b.rename(columns={'before':'before_b','after':'after_b'}),\n"
                "    on='metric',\n"
                "    how='outer'\n"
                ")\n"
                "comparison"
            ),
            new_markdown_cell(
                "## Part 3: Argument (choose best strategy)\n\n"
                "Based on the comparison table, choose the strategy that:\n"
                "1) reduces missing rows without destroying class balance,\n"
                "2) removes or mitigates extreme outliers in `text_length`,\n"
                "3) controls duplicates consistently.\n\n"
                "In this project template, the typical best choice is **strategy_b** when the dataset contains noisy length outliers (so `remove_iqr` helps) and label distribution stays stable. "
                "If `strategy_b` drops too many samples or harms minority classes, switch to **strategy_a**."
            ),
        ]
    )


def annotation_notebook() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(
                "## Annotation Analysis — `AnnotationAgent` spec & HITL metrics\n\n"
                "What this notebook does:\n"
                "- Shows class distribution and generated annotation spec\n"
                "- Computes label quality metrics (Cohen's kappa) if HITL-corrected labels exist\n"
            ),
            new_code_cell(
                "from pathlib import Path\n"
                "import json\n"
                "import pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "\n"
                "import yaml\n"
                "from agents.annotation_agent import AnnotationAgent\n"
                "\n"
                "sns.set_theme(style='whitegrid')\n"
                "\n"
                "cfg = yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8'))\n"
                "ann_cfg = cfg.get('annotation', {})\n"
                "label_space = ann_cfg.get('label_space', None)\n"
                "\n"
                "agent = AnnotationAgent(modality=ann_cfg.get('modality', 'text'), label_space=label_space)\n"
                "\n"
                "# Load latest finalized labeled dataset\n"
                "runs = sorted(Path('artifacts').glob('run_*/data_labeled_final.csv'))\n"
                "final_path = runs[-1] if runs else Path('data/labeled/data_labeled_final.csv')\n"
                "df_final = pd.read_csv(final_path)\n"
                "df_final.shape"
            ),
            new_code_cell(
                "# Add label_auto so `check_quality()` can use the expected column\n"
                "df_final['label_auto'] = df_final['label_final']\n"
                "metrics = agent.check_quality(df_final)\n"
                "metrics"
            ),
            new_code_cell(
                "# Generated spec snapshot (from auto labels or fallback)\n"
                "spec_path_candidates = sorted(Path('artifacts').glob('run_*/reports/annotation_spec.md'))\n"
                "spec_path = spec_path_candidates[-1] if spec_path_candidates else Path('reports/annotation_spec.md')\n"
                "spec_md = spec_path.read_text(encoding='utf-8') if spec_path.exists() else ''\n"
                "spec_md[:1000]"
            ),
            new_code_cell(
                "# Visualize label distribution\n"
                "vc = df_final['label_final'].astype(str).value_counts()\n"
                "plt.figure(figsize=(10,4))\n"
                "sns.barplot(x=vc.index.astype(str), y=vc.values)\n"
                "plt.xticks(rotation=45, ha='right')\n"
                "plt.title('Final label distribution')\n"
                "plt.tight_layout()\n"
                "plt.show()\n"
            ),
            new_code_cell(
                "# If HITL-corrected labels exist, compute kappa after human review\n"
                "corrected_candidates = sorted(Path('artifacts').glob('run_*/review_queue_corrected.csv'))\n"
                "corrected_path = corrected_candidates[-1] if corrected_candidates else None\n"
                "\n"
                "if corrected_path and corrected_path.exists():\n"
                "    corrected = pd.read_csv(corrected_path)\n"
                "    if 'label_human' in corrected.columns:\n"
                "        corrected['label_human'] = corrected['label_human'].astype('string')\n"
                "        corrected = corrected[corrected['label_human'].notna() & (corrected['label_human'].str.len() > 0)].copy()\n"
                "        df_metrics = df_final.merge(corrected[['text','label_human']], on='text', how='inner')\n"
                "        df_metrics['label_auto'] = df_metrics['label_final']\n"
                "        kappa_metrics = agent.check_quality(df_metrics)\n"
                "        kappa_metrics\n"
                "    else:\n"
                "        'No label_human column found in corrected queue.'\n"
                "else:\n"
                "    'No corrected queue found; run HITL and resume the pipeline.'"
            ),
        ]
    )


def al_experiment_notebook() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(
                "## Active Learning experiment\n\n"
                "Compares `entropy` vs `random` strategies and computes the rubric metric: "
                "`saved_examples` = how many fewer labeled examples entropy needs to reach the "
                "random strategy's peak F1."
            ),
            new_code_cell(
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "\n"
                "runs = sorted(Path('artifacts').glob('run_*/reports/al_history.csv'))\n"
                "path = runs[-1] if runs else None\n"
                "print('al_history path:', path)\n"
                "history = pd.read_csv(path) if path else pd.DataFrame()\n"
                "history.head()"
            ),
            new_code_cell(
                "if not history.empty:\n"
                "    plt.figure(figsize=(8,5))\n"
                "    for strat in history['strategy'].unique():\n"
                "        sub = history[history['strategy'] == strat].sort_values('n_labeled')\n"
                "        plt.plot(sub['n_labeled'], sub['f1'], marker='o', label=strat)\n"
                "    plt.xlabel('n_labeled')\n"
                "    plt.ylabel('F1 (macro)')\n"
                "    plt.title('Active Learning: entropy vs random')\n"
                "    plt.grid(True, alpha=0.3)\n"
                "    plt.legend()\n"
                "    plt.tight_layout()\n"
                "    plt.show()\n"
                "\n"
                "    rand = history[history['strategy'] == 'random'].sort_values('n_labeled')\n"
                "    ent = history[history['strategy'] == 'entropy'].sort_values('n_labeled')\n"
                "    target_f1 = float(rand['f1'].max())\n"
                "\n"
                "    required_random = rand[rand['f1'] >= target_f1].sort_values('n_labeled').iloc[0]['n_labeled']\n"
                "    required_entropy = ent[ent['f1'] >= target_f1].sort_values('n_labeled').iloc[0]['n_labeled']\n"
                "    saved_examples = int(required_random - required_entropy)\n"
                "    saved_examples\n"
                "else:\n"
                "    'No al_history.csv found yet. Run `python run_pipeline.py` first.'"
            ),
        ]
    )


def main() -> None:
    Path("notebooks").mkdir(parents=True, exist_ok=True)
    nb1 = quality_notebook()
    nbformat.write(nb1, "notebooks/quality_analysis.ipynb")
    nb2 = annotation_notebook()
    nbformat.write(nb2, "notebooks/annotation_analysis.ipynb")
    nb3 = al_experiment_notebook()
    nbformat.write(nb3, "notebooks/al_experiment.ipynb")
    print("Notebooks generated.")


if __name__ == "__main__":
    main()

