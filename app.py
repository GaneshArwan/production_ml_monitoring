"""Interactive Streamlit dashboard for production ML drift, data quality, and decay."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from pydantic import ValidationError
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails import resolve_safe_path  # noqa: E402
from src.i18n import fmt_decimal, fmt_percent, t  # noqa: E402
from src.monitor import run_monitoring  # noqa: E402
from src.schemas import AlertThresholds, MonitoringRunReport  # noqa: E402
from src.train import run_training  # noqa: E402


@st.cache_data(show_spinner=False)
def _compute_cached_report(thresholds_payload: dict[str, float], locale: str) -> dict[str, Any]:
    thresholds = AlertThresholds(**thresholds_payload)
    model_file = resolve_safe_path(PROJECT_ROOT, Path("models/baseline_model.joblib"))
    if not model_file.is_file():
        run_training(project_root=PROJECT_ROOT)
    report = run_monitoring(
        project_root=PROJECT_ROOT,
        thresholds=thresholds,
        locale=locale,
        save_outputs=False,
    )
    return report.model_dump(mode="json")


def load_report(thresholds: AlertThresholds, locale: str) -> MonitoringRunReport:
    """Load matching precomputed report JSON or compute cached in-memory report."""
    report_path = resolve_safe_path(PROJECT_ROOT, Path("reports/monitoring_report.json"))
    if report_path.is_file():
        try:
            cached = MonitoringRunReport(**json.loads(report_path.read_text(encoding="utf-8")))
            if cached.thresholds == thresholds and cached.locale == locale:
                return cached
        except (OSError, json.JSONDecodeError, ValidationError):
            pass
    payload = _compute_cached_report(thresholds.model_dump(), locale)
    return MonitoringRunReport(**payload)


def main() -> None:
    st.set_page_config(page_title="Production ML Monitoring", layout="wide")

    locale = st.sidebar.selectbox("Language / Bahasa", options=["en", "id"], index=0)
    st.sidebar.header(t("sidebar_controls", locale))
    st.sidebar.subheader(t("threshold_header", locale))

    psi_warning = st.sidebar.slider(t("psi_warn_label", locale), 0.05, 0.30, 0.10, 0.01)
    psi_critical = st.sidebar.slider(
        t("psi_crit_label", locale), max(psi_warning, 0.10), 0.50, max(0.20, psi_warning), 0.01
    )
    auc_warning = st.sidebar.slider(t("auc_warn_label", locale), 0.01, 0.10, 0.03, 0.005)
    auc_critical = st.sidebar.slider(
        t("auc_crit_label", locale), max(auc_warning, 0.02), 0.20, max(0.05, auc_warning), 0.005
    )

    thresholds = AlertThresholds(
        psi_warning=psi_warning,
        psi_critical=psi_critical,
        auc_drop_warning=auc_warning,
        auc_drop_critical=auc_critical,
    )

    report = load_report(thresholds, locale)

    st.title(t("dashboard_title", locale))
    st.caption(t("dashboard_subtitle", locale))

    latest = report.batches[-1]
    total_critical = sum(1 for b in report.batches for a in b.alerts if a.severity == "CRITICAL")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("kpi_batches", locale), str(len(report.batches)))
    col2.metric(t("kpi_baseline_auc", locale), fmt_decimal(report.baseline_metrics.auc_roc, locale))
    col3.metric(
        t("kpi_latest_auc", locale),
        fmt_decimal(latest.performance.auc_roc, locale),
        delta=fmt_decimal(-latest.auc_drop, locale),
    )
    col4.metric(t("kpi_critical_alerts", locale), str(total_critical))

    st.subheader(t("chart_trends", locale))
    trend_df = pd.DataFrame(
        [
            {
                "batch_id": b.batch_id,
                "max_psi": b.max_psi,
                "auc_roc": b.performance.auc_roc,
                "f1": b.performance.f1,
                "missing_rate": b.data_quality.missing_rate,
            }
            for b in report.batches
        ]
    ).set_index("batch_id")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown(f"**{t('chart_psi', locale)}**")
        st.line_chart(trend_df[["max_psi"]])
    with chart_col2:
        st.markdown(f"**{t('chart_auc', locale)}**")
        st.line_chart(trend_df[["auc_roc", "f1"]])

    st.subheader(t("batch_inspector", locale))
    batch_map = {b.batch_id: b for b in report.batches}
    selected_id = st.selectbox(
        t("select_batch", locale),
        options=list(batch_map.keys()),
        index=len(batch_map) - 1,
    )
    selected = batch_map[selected_id]

    q_cols = st.columns(4)
    q_cols[0].metric(t("metric_rows", locale), str(selected.data_quality.row_count))
    q_cols[1].metric(
        t("metric_missing_rate", locale),
        fmt_percent(selected.data_quality.missing_rate, locale),
    )
    q_cols[2].metric(
        t("metric_oor_rate", locale),
        fmt_percent(selected.data_quality.out_of_range_rate, locale),
    )
    q_cols[3].metric(
        t("metric_status", locale),
        t(f"severity_{selected.overall_severity.lower()}", locale),
    )

    st.markdown(f"**{t('feature_drift_table', locale)}**")
    drift_df = pd.DataFrame([d.model_dump() for d in selected.feature_drift])
    st.dataframe(drift_df, width="stretch", hide_index=True)

    st.markdown(f"**{t('alerts_section', locale)}**")
    if selected.alerts:
        alerts_df = pd.DataFrame([a.model_dump() for a in selected.alerts])
        st.dataframe(alerts_df, width="stretch", hide_index=True)
    else:
        st.info(t("no_alerts", locale))


if __name__ == "__main__":
    main()
