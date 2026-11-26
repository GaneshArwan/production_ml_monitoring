# Model Retraining & Incident Response Playbook

## 1. Alert Severity Matrix

| Signal | Warning Trigger | Critical Trigger | Action Required |
|---|---|---|---|
| Feature PSI | `PSI >= 0.10` | `PSI >= 0.20` | Inspect feature pipeline & upstream schema |
| KS / Chi-square p-value | `p < 0.05` | `p < 0.05` (with separate `PSI >= 0.20` critical alert) | Verify covariate shift vs seasonality |
| Missing Data Rate | `>= 3.0%` | `>= 5.0%` | Audit ingestion ETL before retraining |
| Out-of-Range Rate | `>= 1.0%` | `>= 3.0%` | Block bad upstream batch & quarantine records |
| AUC-ROC Decay vs Baseline | Drop `>= 0.03` | Drop `>= 0.05` | Initiate labeled data refresh & retraining |

## 2. When to Retrain vs Fix Data Pipelines

- **Fix upstream pipeline first** if `missing_rate >= 0.05` or `out_of_range_rate >= 0.03` without corresponding ground-truth concept shift. Retraining on corrupted records propagates ingestion bugs.
- **Retrain immediately** when `auc_drop >= 0.05` persists across two consecutive batches or when `PSI >= 0.20` coincides with confirmed label shift.
- **Adjust decision threshold / calibrate** if ranking (`AUC-ROC`) remains stable while `F1` or `prediction_rate` shifts due to prior-probability shift.

## 3. Retraining Workflow

1. **Quarantine & Snapshot**: Archive the latest clean labeled production batches alongside the reference window.
2. **Validate Schema**: Enforce `FeatureRecord` bounds (`src/schemas.py`) and reject out-of-range records before fitting.
3. **Train Candidate**: Fit a new `HistGradientBoostingClassifier` pipeline on a rolling window (`reference` + recent clean labeled production slices) via `src/train.py`.
4. **Shadow Evaluation**: Score the candidate model in parallel against the incumbent model on the newest holdout batch (`2026-06` equivalent).
5. **Promotion Gate**:
   - Candidate `AUC-ROC` exceeds incumbent by `>= 0.02` on the latest clean evaluation slice.
   - Zero `CRITICAL` data quality violations in the training snapshot.
   - Generate and store a new SHA-256 checksum (`models/baseline_model.sha256`).

## 4. Rollback Procedure

1. Keep the previous verified `.joblib` and `.sha256` pair versioned in the model registry.
2. If post-deployment shadow or canary monitoring triggers a `CRITICAL` AUC drop (`>= 0.05`) or unexpected prediction-rate spike (`> 10%` shift), restore the prior artifact and SHA-256 digest immediately.
3. Re-run `python src/monitor.py` to confirm metric stability after rollback.
