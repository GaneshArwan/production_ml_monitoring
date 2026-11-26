"""Train baseline credit-risk classifier, evaluate CV/holdout metrics, and persist artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from pydantic import TypeAdapter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    generate_reference_and_production_batches,
)
from src.guardrails import (  # noqa: E402
    compute_file_sha256,
    resolve_safe_path,
    validate_dataframe_bounds,
)
from src.performance import compute_classification_metrics  # noqa: E402
from src.schemas import FeatureRecord  # noqa: E402

_RECORD_LIST_ADAPTER = TypeAdapter(list[FeatureRecord])


def build_pipeline(seed: int = 42) -> Pipeline:
    """Build preprocessing + gradient-boosted tree classification pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(NUMERIC_FEATURES)),
            ("cat", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ]
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.06,
        max_leaf_nodes=31,
        random_state=seed,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def run_training(
    project_root: Path | None = None,
    reference_rows: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate datasets, cross-validate, fit final baseline model, and save artifacts."""
    if reference_rows < 50:
        raise ValueError("reference_rows must be >= 50 for stratified splitting and 5-fold CV.")

    root = project_root or PROJECT_ROOT
    data_dir = resolve_safe_path(root, Path("data"))
    models_dir = resolve_safe_path(root, Path("models"))
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    ref_df, prod_df = generate_reference_and_production_batches(
        reference_rows=reference_rows,
        seed=seed,
    )
    feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    validate_dataframe_bounds(ref_df, required_columns=feature_cols + [TARGET_COLUMN])
    validate_dataframe_bounds(prod_df, required_columns=["batch_id"] + feature_cols + [TARGET_COLUMN])

    _RECORD_LIST_ADAPTER.validate_python(ref_df.to_dict(orient="records"))

    ref_df.to_csv(data_dir / "reference.csv", index=False)
    prod_df.to_csv(data_dir / "production_batches.csv", index=False)

    X = ref_df[feature_cols]
    y = ref_df[TARGET_COLUMN].to_numpy(dtype=int)
    _, class_counts = np.unique(y, return_counts=True)
    if class_counts.size < 2 or int(class_counts.min()) < 10:
        raise ValueError("Reference target must contain at least 10 samples per class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_validate(
        build_pipeline(seed=seed),
        X_train,
        y_train,
        cv=cv,
        scoring=("roc_auc", "precision", "recall", "f1"),
        n_jobs=1,
    )

    model = build_pipeline(seed=seed)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    holdout_metrics = compute_classification_metrics(y_test, y_prob)

    model_path = resolve_safe_path(root, Path("models/baseline_model.joblib"))
    sha_path = resolve_safe_path(root, Path("models/baseline_model.sha256"))
    metrics_path = resolve_safe_path(root, Path("models/baseline_metrics.json"))

    joblib.dump(model, model_path)
    digest = compute_file_sha256(model_path)
    sha_path.write_text(digest + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "reference_rows": int(len(ref_df)),
        "production_rows": int(len(prod_df)),
        "cv_5fold": {
            "auc_roc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "auc_roc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
            "precision_mean": round(float(np.mean(cv_scores["test_precision"])), 4),
            "recall_mean": round(float(np.mean(cv_scores["test_recall"])), 4),
            "f1_mean": round(float(np.mean(cv_scores["test_f1"])), 4),
        },
        "holdout": holdout_metrics.model_dump(),
        "model_sha256": digest,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_training()
    print(json.dumps(result, indent=2))
