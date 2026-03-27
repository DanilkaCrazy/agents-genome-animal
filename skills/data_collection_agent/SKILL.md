# DataCollectionAgent Skill

Use this skill to collect raw data from multiple sources (Hugging Face datasets, Kaggle datasets, scraping pages, or simple JSON APIs) and unify them into the project dataset schema.

## Available skills (methods)
- `scrape(url, selector, max_items=None) -> pd.DataFrame`
  - Scrape HTML elements matching `selector` and return a `DataFrame` with `text/source/label/collected_at`.
- `fetch_api(endpoint, params=None) -> pd.DataFrame`
  - Fetch JSON from an API endpoint and return a normalized `DataFrame`.
- `load_dataset(name, source='hf', split='train', sample=None, text_col='text', label_col='label', feature_cols=None, text_prefix='') -> pd.DataFrame`
  - Load an open dataset via `datasets.load_dataset()` and convert it into a unified schema (`text/label/source/collected_at`).
- `merge(sources: list[pd.DataFrame]) -> pd.DataFrame`
  - Concatenate multiple source DataFrames and canonicalize the unified schema.

## Output contract
The agent must return a `pd.DataFrame` with fixed columns:
`text`, `audio`, `image`, `label`, `source`, `collected_at`

For this text-only pipeline, `audio` and `image` should be null.

## Example usage
```python
agent = DataCollectionAgent(config="config.yaml")
df = agent.run(sources=[
  {"type": "hf_dataset", "name": "roshan8312/cattle-disease-prediction", "split": "train", "sample": 2000, "label_col": "prognosis"},
  {"type": "scrape", "url": "https://example.com", "selector": "table tr", "max_items": 40},
])
```

