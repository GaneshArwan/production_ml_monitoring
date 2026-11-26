# Production ML Monitoring Dashboard

## Problem Statement
Detect covariate shift, categorical distribution drift, upstream data-quality degradation, and classification performance decay in a deployed credit-risk model before silent accuracy loss impacts production decisions.

## Dataset
- **Source:** Deterministic synthetic credit-risk dataset (`src/data_generator.py`, `seed=42`) simulating a clean training reference window and six monthly production batches with progressive covariate shift, label/concept shift, missingness, and out-of-range corruption.
- **Size:** `22,000` rows x `8` features (`10,000` reference rows; `12,000` production rows across `2026-01` through `2026-06`, `2,000` rows per batch).
- **Features:**
  - Numeric (`6`): `income`, `debt_to_income`, `credit_score`, `utilization_rate`, `account_age_months`, `num_inquiries`
  - Categorical (`2`): `employment_type`, `region`
  - Target (`1`): binary `default` indicator (`28.4%` positive rate on reference holdout)

## Skills Demonstrated
Model monitoring, statistical drift detection (PSI, Kolmogorov-Smirnov, Chi-square), data-quality guardrails, threshold alerting, MLOps retraining playbooks, localized reporting (`en` / `id`), and interactive Streamlit visualization.

## Deliverables
- Statistical data drift engine (`src/drift.py`) using quantile-binned Population Stability Index (PSI), two-sample KS test, and Chi-square test
- Data quality checks tracking missingness, duplicate rate, and schema range violations
- Runtime security and validation guardrails (`src/schemas.py`, `src/guardrails.py`) with strict Pydantic v2 models, path traversal blocking, and SHA-256 verification before `joblib.load`
- Configurable alerting engine (`src/alerting.py`) with `Babel`-backed `en` and `id` localization (`src/i18n.py`)
- Interactive Streamlit dashboard (`app.py`) and batch trend plot (`reports/drift_performance_trends.png`)
- Operational retraining and rollback guide (`docs/RETRAINING_PLAYBOOK.md`)

## Approach
1. **Data Cleaning & EDA** (`notebooks/01_eda.ipynb`) - Quantified missingness growth (`0.3%` to `4.4%` across feature cells), out-of-range boundary violations, and feature distribution shifts across monthly slices.
2. **Feature Engineering & Preprocessing** (`src/train.py`) - Built an `sklearn.pipeline.Pipeline` with `ColumnTransformer` applying median imputation + `StandardScaler` on numeric features and most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` on categorical features.
3. **Modeling** (`src/train.py`, `notebooks/02_modeling.ipynb`) - Trained a `HistGradientBoostingClassifier` (`max_iter=150`, `learning_rate=0.06`, `max_leaf_nodes=31`, `random_state=42`) on an 80/20 stratified split of the reference window.
4. **Evaluation** - Evaluated with 5-fold `StratifiedKFold` cross-validation on the training split and verified on the 20% holdout set (`2,000` rows).

## Key Results

| Evaluation Split | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| 5-Fold Stratified CV (mean) | 0.7823 +/- 0.0135 | 0.6328 | 0.4269 | 0.5092 |
| Reference Holdout (`n=2,000`) | 0.7913 | 0.6463 | 0.4665 | 0.5419 |

## Batch Monitoring Trajectory

| Batch | Status | Max Feature PSI | AUC-ROC | Missing Rate | Alerts |
|---|---|---|---|---|---|
| `2026-01` | OK | 0.008 | 0.7924 | 0.003 | 0 |
| `2026-02` | WARNING | 0.013 | 0.8003 | 0.005 | 3 |
| `2026-03` | CRITICAL | 0.364 | 0.7644 | 0.011 | 8 |
| `2026-04` | CRITICAL | 1.737 | 0.6553 | 0.019 | 12 |
| `2026-05` | CRITICAL | 4.262 | 0.5228 | 0.032 | 15 |
| `2026-06` | CRITICAL | 5.736 | 0.4118 | 0.044 | 15 |

## Key Findings
1. Early covariate shifts trigger `WARNING` KS/Chi-square p-value alerts in `2026-02` (`max_psi=0.013`) and breach the `0.20` critical PSI ceiling in `2026-03` (`max_psi=0.364`) before severe AUC collapse occurs.
2. Combined covariate and concept shift in `2026-04` through `2026-06` degrades `AUC-ROC` from `0.7913` baseline down to `0.6553`, `0.5228`, and `0.4118`, exceeding the `0.05` critical AUC drop threshold.
3. Upstream ingestion defects (`4.4%` feature missingness and out-of-range values in `2026-06`) must be isolated before retraining per `docs/RETRAINING_PLAYBOOK.md`.

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/monitor.py
pytest -v
streamlit run app.py
```

## Files
- `src/schemas.py` - Strict Pydantic v2 data, threshold, and report models
- `src/guardrails.py` - Path traversal, row bounds, finite array, and SHA-256 verification
- `src/i18n.py` - `Babel`-based `en` and `id` localization catalog and formatters
- `src/data_generator.py` - Deterministic credit-risk reference and drifted batch generator
- `src/drift.py` - PSI, KS, Chi-square, and data-quality evaluation
- `src/performance.py` - Classification metrics (`AUC-ROC`, `Precision`, `Recall`, `F1`)
- `src/alerting.py` - Configurable localized threshold alerting
- `src/train.py` - Cross-validated training pipeline and artifact hashing
- `src/monitor.py` - Batch monitoring CLI, JSON report export, and trend plotting
- `app.py` - Interactive Streamlit monitoring dashboard
- `docs/RETRAINING_PLAYBOOK.md` - Retraining triggers, shadow evaluation, and rollback steps
- `notebooks/01_eda.ipynb` - Exploratory drift and data-quality analysis
- `notebooks/02_modeling.ipynb` - Baseline training and monitoring verification
- `tests/test_monitoring.py` - Unit and end-to-end integration test suite

## Methodology & Limitations
Monitoring decouples statistical feature drift (quantile-binned PSI on reference edges, two-sample KS for numeric features, Chi-square for categorical features) from upstream data-quality checks (missingness, duplicates, schema bounds) and labeled model performance (`AUC-ROC`, `Precision`, `Recall`, `F1`). Model artifacts are verified against a stored SHA-256 digest prior to `joblib.load` deserialization.

Performance monitoring assumes delayed ground-truth labels (`default`) become available for batch evaluation; in live credit workflows where outcomes take 30 to 90 days to mature, covariate drift (PSI/KS) and prediction-rate shifts serve as leading indicators while labeled evaluation lags. Synthetic drift magnitudes are accelerated across six batches to exercise every alert tier deterministically.
