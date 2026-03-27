# Final Report

## Run folders
- artifacts: `artifacts/run_<run_id>/` (see `artifacts/run_manifest.json`) 

## Metrics snapshot
- AL prep: dedup 3457 -> 3457, unknown_policy=drop, classes=26
- final model accuracy: 0.9333333333333333
- final model f1_macro: 0.8814713064713064

## How to run HITL
1) Open `review_queue.csv`.
2) Fill `label_human` and save as `artifacts/run_<run_id>/review_queue_corrected.csv`.
3) Rerun with `python run_pipeline.py --resume-hitl`.
