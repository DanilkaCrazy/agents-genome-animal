# DataQualityAgent Skill

Use this skill to detect data quality issues (missing values, duplicates, outliers, and class imbalance) and apply deterministic cleaning strategies.

## Available skills (methods)
- `detect_issues(df) -> dict`
  - Returns a `QualityReport` containing:
    - `missing`: counts & percentages per column
    - `duplicates`: number of duplicate rows (based on `text` when available)
    - `outliers`: IQR-based bounds on `text_length` and example indices
    - `imbalance`: label distribution metrics
- `fix(df, strategy: dict) -> pd.DataFrame`
  - Applies cleaning:
    - missing: `drop` or `fill_unknown`
    - duplicates: `drop` or `keep_first`
    - outliers: `clip_iqr`, `remove_iqr`, or `none`
- `compare(df_before, df_after) -> pd.DataFrame`
  - Produces a table of metric deltas between the two datasets.

## Output contract
All methods must work on the unified dataset schema containing at least:
`text`, `label`, `source`, `collected_at` (and optionally `audio`, `image`).

