# AnnotationAgent Skill

Use this skill to automatically label data, generate a human-readable annotation specification, compute basic annotation quality metrics, and export tasks for LabelStudio.

## Available skills (methods)
- `auto_label(df, modality='text') -> pd.DataFrame`
  - For this template: text-only auto-labeling using a lightweight TF-IDF + Logistic Regression baseline.
- `generate_spec(df, task) -> str`
  - Generates `annotation_spec.md` with:
    - task description
    - class list + definitions
    - 3+ example texts per class
    - edge cases
- `check_quality(df_labeled) -> dict`
  - Computes:
    - label distribution
    - mean confidence
    - Cohen's kappa if `label_human` exists
- `export_to_labelstudio(df, out_path) -> Path`
  - Exports a JSON file in LabelStudio import format.

## HITL workflow
The pipeline creates a `review_queue.csv` with low-confidence suggestions. Human fills `label_human` and saves it as `review_queue_corrected.csv`.

If configured, the queue can prioritize rare predicted classes first (to improve macro-F1 sooner).

