# ActiveLearningAgent Skill (Track A)

Use this skill to implement Active Learning on the labeled dataset:
train a base model, select uncertain/informative examples from a pool, iteratively add them to the labeled set, and evaluate progress.

## Available skills (methods)
- `fit(labeled_df) -> sklearn Pipeline`
- `query(pool, model, strategy, k) -> indices`
  - strategies supported: `entropy`, `margin`, `random`
- `evaluate(model, labeled_df, test_df) -> metrics`
  - returns accuracy and macro-F1
- `run_cycle(...) -> history`
  - runs N iterations, adding a batch of size `batch_size` each iteration
- `report(history, out_path) -> Path`
  - saves a learning-curve plot

