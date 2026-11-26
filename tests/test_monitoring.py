"""Unit and integration tests for schemas, i18n, guardrails, drift, alerting, and UI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import app as dashboard_app
from src.alerting import evaluate_alerts
from src.data_generator import CATEGORICAL_FEATURES, NUMERIC_BOUNDS, NUMERIC_FEATURES
from src.drift import (
    compute_categorical_drift,
    compute_ks_test,
    compute_psi,
    evaluate_batch_drift,
    evaluate_data_quality,
)
from src.guardrails import (
    compute_file_sha256,
    resolve_safe_path,
    sanitize_numeric_array,
    validate_dataframe_bounds,
    verify_model_artifact,
)
from src.i18n import fmt_decimal, fmt_percent, normalize_locale, t
from src.monitor import _worst_severity, main as monitor_main, run_monitoring
from src.performance import compute_classification_metrics
from src.schemas import (
    AlertThresholds,
    DataQualityResult,
    FeatureDriftResult,
    FeatureRecord,
    PerformanceMetrics,
)
from src.train import run_training


def test_schemas_reject_invalid_inputs() -> None:
    with pytest.raises(ValidationError):
        AlertThresholds(psi_warning=0.30, psi_critical=0.10)
    with pytest.raises(ValidationError):
        AlertThresholds(auc_drop_warning=0.10, auc_drop_critical=0.05)
    with pytest.raises(ValidationError):
        AlertThresholds(missing_rate_warning=0.10, missing_rate_critical=0.05)
    with pytest.raises(ValidationError):
        AlertThresholds(out_of_range_warning=0.05, out_of_range_critical=0.01)

    with pytest.raises(ValidationError):
        FeatureRecord(
            income=50_000.0,
            debt_to_income=0.3,
            credit_score=950.0,
            utilization_rate=0.2,
            account_age_months=24,
            num_inquiries=1,
            employment_type="salaried",
            region="north",
            default=0,
        )

    with pytest.raises(ValidationError):
        FeatureRecord(
            income=50_000.0,
            debt_to_income=0.3,
            credit_score=700.0,
            utilization_rate=0.2,
            account_age_months=24,
            num_inquiries=1,
            employment_type="salaried",
            region="north",
            default=0,
            unexpected_field="bad",
        )


def test_i18n_en_id_and_fallbacks() -> None:
    assert normalize_locale(None) == "en"
    assert normalize_locale("   ") == "en"
    assert normalize_locale(" id-ID ") == "id"
    assert normalize_locale("fr") == "en"
    assert "Dashboard" in t("dashboard_title", locale="en")
    assert "Dasbor" in t("dashboard_title", locale="id")
    assert t("missing_translation_key", locale="en") == "missing_translation_key"
    assert fmt_decimal(0.25, locale="en", digits=2) == "0.25"
    assert fmt_percent(0.125, locale="en", digits=1) == "12.5%"


def test_guardrails_path_bounds_and_sha(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Path traversal"):
        resolve_safe_path(tmp_path, Path("../outside.txt"))
    with pytest.raises(ValueError, match="Path traversal"):
        resolve_safe_path(tmp_path, Path("C:/Windows/System32/drivers/etc/hosts"))

    safe_file = tmp_path / "model.bin"
    safe_file.write_bytes(b"safe-model-bytes")
    digest = compute_file_sha256(safe_file)
    assert verify_model_artifact(safe_file, tmp_path, expected_sha256=digest) == safe_file.resolve()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_model_artifact(safe_file, tmp_path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="non-empty"):
        verify_model_artifact(safe_file, tmp_path, expected_sha256=" ")
    with pytest.raises(FileNotFoundError):
        verify_model_artifact(tmp_path / "missing.bin", tmp_path, expected_sha256=digest)
    with pytest.raises(ValueError, match="outside valid range"):
        verify_model_artifact(safe_file, tmp_path, expected_sha256=digest, max_bytes=4)

    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="outside valid range"):
        verify_model_artifact(empty_file, tmp_path, expected_sha256=digest)

    cleaned = sanitize_numeric_array(np.array([1.0, np.nan, np.inf, 2.0]))
    assert np.array_equal(cleaned, np.array([1.0, 2.0]))

    nullable_series = pd.Series([1.0, pd.NA, "bad", 3.5], dtype="object")
    assert np.array_equal(sanitize_numeric_array(nullable_series), np.array([1.0, 3.5]))

    with pytest.raises(ValueError, match="1-dimensional"):
        sanitize_numeric_array(np.ones((2, 2)))

    with pytest.raises(ValueError, match="empty"):
        validate_dataframe_bounds(pd.DataFrame())
    with pytest.raises(ValueError, match="max_rows"):
        validate_dataframe_bounds(pd.DataFrame({"a": [1]}), max_rows=0)
    with pytest.raises(ValueError, match="duplicate column"):
        validate_dataframe_bounds(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(ValueError, match="exceeds safety cap"):
        validate_dataframe_bounds(pd.DataFrame({"a": [1, 2, 3]}), max_rows=2)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe_bounds(pd.DataFrame({"a": [1]}), required_columns=["b"])


def test_psi_ks_and_categorical_drift() -> None:
    rng = np.random.default_rng(7)
    ref = rng.normal(0.0, 1.0, size=3_000)
    same = rng.normal(0.0, 1.0, size=3_000)
    shifted = rng.normal(2.0, 1.0, size=3_000)

    assert compute_psi(ref, same) < 0.05
    assert compute_psi(ref, shifted) > 0.50
    assert compute_psi(np.ones(100), np.ones(100)) == 0.0

    with pytest.raises(ValueError, match="bins"):
        compute_psi(ref, same, bins=0)
    with pytest.raises(ValueError, match="finite"):
        compute_psi(np.array([np.nan, np.inf]), same)
    with pytest.raises(ValueError, match="finite"):
        compute_ks_test(np.array([np.nan]), same)

    ks_stat_same, ks_p_same = compute_ks_test(ref, same)
    ks_stat_shift, ks_p_shift = compute_ks_test(ref, shifted)
    assert ks_p_same > 0.01
    assert ks_stat_shift > ks_stat_same
    assert ks_p_shift < 1e-6

    ref_cat = pd.Series(["a"] * 700 + ["b"] * 300)
    cur_cat = pd.Series(["a"] * 200 + ["b"] * 800)
    cat_psi, _, chi2_p = compute_categorical_drift(ref_cat, cur_cat)
    assert cat_psi > 0.20
    assert chi2_p < 1e-6

    single_cat = pd.Series(["only"] * 50)
    s_psi, s_stat, s_p = compute_categorical_drift(single_cat, single_cat)
    assert s_psi == 0.0 and s_stat == 0.0 and s_p == 1.0

    with pytest.raises(ValueError, match="non-null"):
        compute_categorical_drift(pd.Series([None, None]), cur_cat)


def test_data_quality_performance_and_alerts() -> None:
    thresholds = AlertThresholds()
    bad_df = pd.DataFrame(
        {
            "income": [50_000.0, np.nan, "corrupt", 50_000.0],
            "debt_to_income": [0.2, 0.3, -0.5, 0.2],
            "credit_score": [700.0, 710.0, 950.0, 700.0],
            "utilization_rate": [0.2, 0.3, 0.4, 0.2],
            "account_age_months": [12.0, 24.0, 36.0, 12.0],
            "num_inquiries": [1, 2, 1, 1],
            "employment_type": ["salaried", "contract", "salaried", "salaried"],
            "region": ["north", "south", "east", "north"],
        }
    )
    quality = evaluate_data_quality(bad_df, NUMERIC_BOUNDS, thresholds)
    assert quality.missing_rate > 0.0
    assert quality.out_of_range_rate > 0.05
    assert quality.severity == "CRITICAL"

    with pytest.raises(ValueError, match="at least one feature"):
        evaluate_data_quality(pd.DataFrame({"batch_id": ["b1"], "default": [0]}), NUMERIC_BOUNDS, thresholds)

    perf = compute_classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert perf.auc_roc == 1.0
    assert perf.f1 == 1.0

    single_class_perf = compute_classification_metrics(
        np.array([0, 0, 0, 0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
    )
    assert single_class_perf.auc_roc == 0.5

    with pytest.raises(ValueError, match="decision_threshold"):
        compute_classification_metrics(np.array([0, 1]), np.array([0.2, 0.8]), decision_threshold=1.5)
    with pytest.raises(ValueError, match="finite probabilities"):
        compute_classification_metrics(np.array([0, 1]), np.array([0.2, np.nan]))
    with pytest.raises(ValueError, match="binary labels"):
        compute_classification_metrics(np.array([0, 2]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="1D arrays"):
        compute_classification_metrics(np.array([0, 1]), np.array([0.2]))

    clean_df = bad_df.copy()
    clean_df["income"] = [50_000.0, np.nan, 40_000.0, 50_000.0]
    drift = evaluate_batch_drift(
        clean_df,
        clean_df,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        thresholds,
    )
    assert all(d.severity == "OK" for d in drift)

    # Verify WARNING vs CRITICAL tiers for PSI, missingness, OOR, and AUC drop.
    synth_drift = [
        FeatureDriftResult(
            feature="income",
            feature_type="numeric",
            psi=0.15,
            ks_statistic=0.04,
            ks_pvalue=0.20,
            severity="WARNING",
        ),
        FeatureDriftResult(
            feature="credit_score",
            feature_type="numeric",
            psi=0.25,
            ks_statistic=0.20,
            ks_pvalue=0.001,
            severity="CRITICAL",
        ),
    ]
    warn_quality = DataQualityResult(
        row_count=100,
        missing_rate=0.035,
        duplicate_rate=0.0,
        out_of_range_rate=0.015,
        severity="WARNING",
    )
    warn_perf = PerformanceMetrics(
        auc_roc=0.76, precision=0.6, recall=0.5, f1=0.55, positive_rate=0.3, prediction_rate=0.25
    )
    alerts = evaluate_alerts(
        batch_id="tier-check",
        drift_results=synth_drift,
        quality_result=warn_quality,
        perf_metrics=warn_perf,
        baseline_auc=0.80,  # 0.04 drop -> WARNING
        thresholds=thresholds,
        locale="id",
    )
    by_metric = {a.metric_name: a.severity for a in alerts}
    assert by_metric["income_psi"] == "WARNING"
    assert by_metric["credit_score_psi"] == "CRITICAL"
    assert by_metric["missing_rate"] == "WARNING"
    assert by_metric["out_of_range_rate"] == "WARNING"
    assert by_metric["auc_drop"] == "WARNING"
    assert _worst_severity(["OK"]) == "OK"


def test_end_to_end_training_monitoring_and_dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="reference_rows"):
        run_training(project_root=tmp_path, reference_rows=20)

    summary = run_training(project_root=tmp_path, reference_rows=1_200, seed=42)
    assert summary["holdout"]["auc_roc"] > 0.70
    assert (tmp_path / "models" / "baseline_model.sha256").is_file()

    report = run_monitoring(project_root=tmp_path, locale="en", save_outputs=True)
    assert len(report.batches) == 6
    assert (tmp_path / "reports" / "monitoring_report.json").is_file()
    assert (tmp_path / "reports" / "drift_performance_trends.png").is_file()
    assert report.batches[-1].max_psi > report.batches[0].max_psi
    assert report.batches[-1].overall_severity == "CRITICAL"

    monkeypatch.setattr(dashboard_app, "PROJECT_ROOT", tmp_path)
    loaded = dashboard_app.load_report(AlertThresholds(), "en")
    assert loaded.locale == "en"
    assert len(loaded.batches) == 6

    # Corrupt cached JSON triggers clean in-memory recomputation without overwriting.
    (tmp_path / "reports" / "monitoring_report.json").write_text("{corrupt", encoding="utf-8")
    recomputed = dashboard_app.load_report(AlertThresholds(), "id")
    assert recomputed.locale == "id"
    assert recomputed.batches[-1].overall_severity == "CRITICAL"

    # CLI entrypoint check.
    monkeypatch.setattr("src.monitor.PROJECT_ROOT", tmp_path)
    monitor_main(["--locale", "en"])
    assert json.loads((tmp_path / "reports" / "monitoring_report.json").read_text(encoding="utf-8"))["locale"] == "en"
