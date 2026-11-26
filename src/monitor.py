"""Batch monitoring runner: evaluates drift, data quality, decay, alerts, and trend plots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerting import evaluate_alerts  # noqa: E402
from src.data_generator import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_BOUNDS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.drift import evaluate_batch_drift, evaluate_data_quality  # noqa: E402
from src.guardrails import (  # noqa: E402
    resolve_safe_path,
    validate_dataframe_bounds,
    verify_model_artifact,
)
from src.i18n import normalize_locale  # noqa: E402
from src.performance import compute_classification_metrics  # noqa: E402
from src.schemas import (  # noqa: E402
    AlertThresholds,
    BatchMonitoringReport,
    MonitoringRunReport,
    PerformanceMetrics,
    SeverityLevel,
)


def _worst_severity(levels: list[SeverityLevel]) -> SeverityLevel:
    if "CRITICAL" in levels:
        return "CRITICAL"
    if "WARNING" in levels:
        return "WARNING"
    return "OK"


def generate_trend_plot(report: MonitoringRunReport, output_path: Path) -> None:
    """Render PSI and classification metric trends across production batches."""
    batch_ids = [b.batch_id for b in report.batches]
    max_psis = [b.max_psi for b in report.batches]
    aucs = [b.performance.auc_roc for b in report.batches]
    f1s = [b.performance.f1 for b in report.batches]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(batch_ids, max_psis, marker="o", linewidth=2, color="#1f77b4", label="Max PSI")
    ax1.axhline(report.thresholds.psi_warning, color="#ff7f0e", linestyle="--", label="PSI Warning")
    ax1.axhline(report.thresholds.psi_critical, color="#d62728", linestyle="--", label="PSI Critical")
    ax1.set_ylabel("Max Feature PSI")
    ax1.set_title("Production Feature Drift & Model Performance Trends")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    auc_floor = report.baseline_metrics.auc_roc - report.thresholds.auc_drop_critical
    y_min = max(0.0, min(aucs + f1s + [auc_floor], default=0.0) - 0.05)

    ax2.plot(batch_ids, aucs, marker="o", linewidth=2, color="#2ca02c", label="AUC-ROC")
    ax2.plot(batch_ids, f1s, marker="s", linewidth=2, color="#9467bd", label="F1")
    ax2.axhline(auc_floor, color="#d62728", linestyle="--", label="Critical AUC Floor")
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Production Batch")
    ax2.set_ylim(y_min, 1.0)
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_monitoring(
    project_root: Path | None = None,
    thresholds: AlertThresholds | None = None,
    locale: str = "en",
    save_outputs: bool = True,
) -> MonitoringRunReport:
    """Run end-to-end monitoring over saved reference data, production batches, and model."""
    root = project_root or PROJECT_ROOT
    active_thresholds = thresholds or AlertThresholds()
    loc = normalize_locale(locale)

    ref_path = resolve_safe_path(root, Path("data/reference.csv"))
    prod_path = resolve_safe_path(root, Path("data/production_batches.csv"))
    metrics_path = resolve_safe_path(root, Path("models/baseline_metrics.json"))
    sha_path = resolve_safe_path(root, Path("models/baseline_model.sha256"))
    model_path = resolve_safe_path(root, Path("models/baseline_model.joblib"))

    ref_df = pd.read_csv(ref_path)
    prod_df = pd.read_csv(prod_path)

    feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    validate_dataframe_bounds(ref_df, required_columns=feature_cols + [TARGET_COLUMN])
    validate_dataframe_bounds(prod_df, required_columns=["batch_id"] + feature_cols + [TARGET_COLUMN])
    if prod_df["batch_id"].isna().any():
        raise ValueError("batch_id column must not contain null values.")

    expected_sha = sha_path.read_text(encoding="utf-8").strip()
    verified_model_path = verify_model_artifact(model_path, root, expected_sha256=expected_sha)
    model = joblib.load(verified_model_path)

    if not hasattr(model, "classes_") or len(model.classes_) != 2:
        raise ValueError("Loaded model must be a binary classifier with two classes.")

    baseline_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "holdout" not in baseline_payload:
        raise ValueError("baseline_metrics.json missing required 'holdout' metrics.")
    baseline_metrics = PerformanceMetrics(**baseline_payload["holdout"])

    batch_reports: list[BatchMonitoringReport] = []
    for batch_id, batch_df in prod_df.groupby("batch_id", sort=True):
        if batch_df[TARGET_COLUMN].isna().any():
            raise ValueError(f"Batch '{batch_id}' contains null target values.")
        drift_results = evaluate_batch_drift(
            ref_df,
            batch_df,
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
            active_thresholds,
        )
        quality_result = evaluate_data_quality(batch_df, NUMERIC_BOUNDS, active_thresholds)
        y_prob = model.predict_proba(batch_df[feature_cols])[:, 1]
        perf_metrics = compute_classification_metrics(
            batch_df[TARGET_COLUMN].to_numpy(dtype=int),
            y_prob,
        )
        alerts = evaluate_alerts(
            batch_id=str(batch_id),
            drift_results=drift_results,
            quality_result=quality_result,
            perf_metrics=perf_metrics,
            baseline_auc=baseline_metrics.auc_roc,
            thresholds=active_thresholds,
            locale=loc,
        )

        severities: list[SeverityLevel] = [quality_result.severity] + [d.severity for d in drift_results]
        severities.extend(a.severity for a in alerts)
        max_psi = max((d.psi for d in drift_results), default=0.0)
        drifted_count = sum(1 for d in drift_results if d.severity != "OK")

        batch_reports.append(
            BatchMonitoringReport(
                batch_id=str(batch_id),
                overall_severity=_worst_severity(severities),
                max_psi=round(max_psi, 6),
                drifted_feature_count=drifted_count,
                feature_drift=drift_results,
                data_quality=quality_result,
                performance=perf_metrics,
                auc_drop=round(baseline_metrics.auc_roc - perf_metrics.auc_roc, 4),
                alerts=alerts,
            )
        )

    if not batch_reports:
        raise ValueError("No production batches found to monitor.")

    report = MonitoringRunReport(
        generated_at=datetime.now(timezone.utc),
        locale=loc,
        reference_rows=len(ref_df),
        baseline_metrics=baseline_metrics,
        thresholds=active_thresholds,
        batches=batch_reports,
    )

    if save_outputs:
        reports_dir = resolve_safe_path(root, Path("reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_json = resolve_safe_path(root, Path("reports/monitoring_report.json"))
        plot_path = resolve_safe_path(root, Path("reports/drift_performance_trends.png"))
        report_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        generate_trend_plot(report, plot_path)

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run production ML monitoring checks.")
    parser.add_argument("--locale", default="en", choices=["en", "id"], help="Alert localization.")
    args = parser.parse_args(argv)

    report = run_monitoring(locale=args.locale)
    for batch in report.batches:
        print(
            f"[{batch.batch_id}] severity={batch.overall_severity} "
            f"max_psi={batch.max_psi:.3f} auc={batch.performance.auc_roc:.4f} "
            f"missing={batch.data_quality.missing_rate:.3f} alerts={len(batch.alerts)}"
        )


if __name__ == "__main__":
    main()
