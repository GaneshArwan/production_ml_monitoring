"""Configurable threshold alerting with localized alert messages."""

from __future__ import annotations

from typing import Literal

from src.i18n import fmt_decimal, fmt_percent, t
from src.schemas import (
    AlertEvent,
    AlertThresholds,
    DataQualityResult,
    FeatureDriftResult,
    PerformanceMetrics,
)


def _pick_level(
    value: float,
    warning_limit: float,
    critical_limit: float,
) -> tuple[Literal["WARNING", "CRITICAL"], float] | None:
    if value >= critical_limit:
        return "CRITICAL", critical_limit
    if value >= warning_limit:
        return "WARNING", warning_limit
    return None


def evaluate_alerts(
    batch_id: str,
    drift_results: list[FeatureDriftResult],
    quality_result: DataQualityResult,
    perf_metrics: PerformanceMetrics,
    baseline_auc: float,
    thresholds: AlertThresholds,
    locale: str = "en",
) -> list[AlertEvent]:
    """Generate structured WARNING/CRITICAL alerts for drift, quality, and AUC decay."""
    alerts: list[AlertEvent] = []

    for item in drift_results:
        psi_level = _pick_level(item.psi, thresholds.psi_warning, thresholds.psi_critical)
        if psi_level is not None:
            sev, limit = psi_level
            alerts.append(
                AlertEvent(
                    batch_id=batch_id,
                    category="data_drift",
                    severity=sev,
                    metric_name=f"{item.feature}_psi",
                    metric_value=item.psi,
                    threshold=limit,
                    message=t(
                        "alert_psi",
                        locale=locale,
                        feature=item.feature,
                        value=fmt_decimal(item.psi, locale),
                        severity=t(f"severity_{sev.lower()}", locale),
                        threshold=fmt_decimal(limit, locale),
                    ),
                )
            )

        for pval, metric_suffix, msg_key in (
            (item.ks_pvalue, "ks_pvalue", "alert_ks"),
            (item.chi2_pvalue, "chi2_pvalue", "alert_chi2"),
        ):
            if pval is not None and pval < thresholds.ks_alpha:
                alerts.append(
                    AlertEvent(
                        batch_id=batch_id,
                        category="data_drift",
                        severity="WARNING",
                        metric_name=f"{item.feature}_{metric_suffix}",
                        metric_value=pval,
                        threshold=thresholds.ks_alpha,
                        message=t(
                            msg_key,
                            locale=locale,
                            feature=item.feature,
                            value=fmt_decimal(pval, locale, digits=4),
                            threshold=fmt_decimal(thresholds.ks_alpha, locale),
                        ),
                    )
                )

    for metric_name, value, warn_limit, crit_limit, msg_key in (
        (
            "missing_rate",
            quality_result.missing_rate,
            thresholds.missing_rate_warning,
            thresholds.missing_rate_critical,
            "alert_missing",
        ),
        (
            "out_of_range_rate",
            quality_result.out_of_range_rate,
            thresholds.out_of_range_warning,
            thresholds.out_of_range_critical,
            "alert_oor",
        ),
    ):
        q_level = _pick_level(value, warn_limit, crit_limit)
        if q_level is not None:
            sev, limit = q_level
            alerts.append(
                AlertEvent(
                    batch_id=batch_id,
                    category="data_quality",
                    severity=sev,
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=limit,
                    message=t(
                        msg_key,
                        locale=locale,
                        value=fmt_percent(value, locale),
                        severity=t(f"severity_{sev.lower()}", locale),
                        threshold=fmt_percent(limit, locale),
                    ),
                )
            )

    auc_drop = round(baseline_auc - perf_metrics.auc_roc, 4)
    auc_level = _pick_level(auc_drop, thresholds.auc_drop_warning, thresholds.auc_drop_critical)
    if auc_level is not None:
        sev, limit = auc_level
        alerts.append(
            AlertEvent(
                batch_id=batch_id,
                category="performance",
                severity=sev,
                metric_name="auc_drop",
                metric_value=auc_drop,
                threshold=limit,
                message=t(
                    "alert_auc_drop",
                    locale=locale,
                    value=fmt_decimal(auc_drop, locale),
                    threshold=fmt_decimal(limit, locale),
                ),
            )
        )

    return alerts
