"""Internationalization and locale-aware number formatting via Babel."""

from __future__ import annotations

from babel.numbers import format_decimal, format_percent

SUPPORTED_LOCALES = ("en", "id")

_CATALOG: dict[str, dict[str, str]] = {
    "en": {
        "dashboard_title": "Production ML Monitoring Dashboard",
        "dashboard_subtitle": "Feature drift (PSI / KS / Chi2), data quality, and classification decay across monthly batches.",
        "sidebar_controls": "Monitoring Controls",
        "locale_label": "Language / Bahasa",
        "threshold_header": "Alert Thresholds",
        "psi_warn_label": "PSI Warning",
        "psi_crit_label": "PSI Critical",
        "auc_warn_label": "AUC Drop Warning",
        "auc_crit_label": "AUC Drop Critical",
        "kpi_batches": "Monitored Batches",
        "kpi_baseline_auc": "Baseline AUC-ROC",
        "kpi_latest_auc": "Latest Batch AUC-ROC",
        "kpi_critical_alerts": "Total Critical Alerts",
        "chart_trends": "Drift & Performance Trends",
        "chart_psi": "Max Feature PSI per Batch",
        "chart_auc": "AUC-ROC & F1 Over Time",
        "batch_inspector": "Batch Diagnostics",
        "select_batch": "Select Batch",
        "quality_section": "Data Quality",
        "metric_rows": "Rows",
        "metric_missing_rate": "Missing Rate",
        "metric_oor_rate": "Out-of-Range Rate",
        "metric_status": "Status",
        "feature_drift_table": "Feature-Level Drift",
        "alerts_section": "Triggered Alerts",
        "no_alerts": "No alerts triggered for current thresholds.",
        "severity_ok": "OK",
        "severity_warning": "WARNING",
        "severity_critical": "CRITICAL",
        "alert_psi": "Feature '{feature}' PSI={value} exceeds {severity} threshold ({threshold}).",
        "alert_ks": "Feature '{feature}' KS test p-value={value} below alpha ({threshold}).",
        "alert_chi2": "Feature '{feature}' Chi-square p-value={value} below alpha ({threshold}).",
        "alert_missing": "Batch missing rate {value} exceeds {severity} threshold ({threshold}).",
        "alert_oor": "Batch out-of-range rate {value} exceeds {severity} threshold ({threshold}).",
        "alert_auc_drop": "AUC-ROC dropped by {value} vs baseline ({threshold} limit).",
    },
    "id": {
        "dashboard_title": "Dasbor Pemantauan ML Produksi",
        "dashboard_subtitle": "Pergeseran fitur (PSI / KS / Chi2), kualitas data, dan penurunan performa klasifikasi per batch bulanan.",
        "sidebar_controls": "Kontrol Pemantauan",
        "locale_label": "Bahasa / Language",
        "threshold_header": "Ambang Batas Peringatan",
        "psi_warn_label": "Peringatan PSI",
        "psi_crit_label": "Kritis PSI",
        "auc_warn_label": "Peringatan Penurunan AUC",
        "auc_crit_label": "Kritis Penurunan AUC",
        "kpi_batches": "Batch Dipantau",
        "kpi_baseline_auc": "AUC-ROC Baseline",
        "kpi_latest_auc": "AUC-ROC Batch Terbaru",
        "kpi_critical_alerts": "Total Peringatan Kritis",
        "chart_trends": "Tren Pergeseran & Performa",
        "chart_psi": "PSI Maksimum Fitur per Batch",
        "chart_auc": "AUC-ROC & F1 Seiring Waktu",
        "batch_inspector": "Diagnostik Batch",
        "select_batch": "Pilih Batch",
        "quality_section": "Kualitas Data",
        "metric_rows": "Jumlah Baris",
        "metric_missing_rate": "Rasio Data Hilang",
        "metric_oor_rate": "Rasio di Luar Rentang",
        "metric_status": "Status",
        "feature_drift_table": "Pergeseran Tingkat Fitur",
        "alerts_section": "Peringatan Aktif",
        "no_alerts": "Tidak ada peringatan untuk ambang batas saat ini.",
        "severity_ok": "AMAN",
        "severity_warning": "PERINGATAN",
        "severity_critical": "KRITIS",
        "alert_psi": "PSI fitur '{feature}'={value} melampaui ambang {severity} ({threshold}).",
        "alert_ks": "Nilai-p uji KS fitur '{feature}'={value} di bawah alfa ({threshold}).",
        "alert_chi2": "Nilai-p Chi-square fitur '{feature}'={value} di bawah alfa ({threshold}).",
        "alert_missing": "Rasio data hilang {value} melampaui ambang {severity} ({threshold}).",
        "alert_oor": "Rasio nilai di luar rentang {value} melampaui ambang {severity} ({threshold}).",
        "alert_auc_drop": "AUC-ROC turun sebesar {value} dari baseline (batas {threshold}).",
    },
}


def normalize_locale(locale: str | None) -> str:
    """Return a supported locale code, falling back to 'en'."""
    if not locale or not locale.strip():
        return "en"
    short = locale.strip().split("-")[0].split("_")[0].lower()
    return short if short in SUPPORTED_LOCALES else "en"


def t(key: str, locale: str = "en", **kwargs: object) -> str:
    """Translate a catalog key into the requested locale and format placeholders."""
    loc = normalize_locale(locale)
    template = _CATALOG[loc].get(key) or _CATALOG["en"].get(key, key)
    return template.format(**kwargs) if kwargs else template


def fmt_decimal(value: float, locale: str = "en", digits: int = 3) -> str:
    """Format a floating-point number using Babel locale rules."""
    loc = normalize_locale(locale)
    pattern = f"#,##0.{'0' * digits}" if digits > 0 else "#,##0"
    return format_decimal(value, format=pattern, locale=loc)


def fmt_percent(value: float, locale: str = "en", digits: int = 1) -> str:
    """Format a ratio in [0, 1] as a localized percentage string."""
    loc = normalize_locale(locale)
    pattern = f"#,##0.{'0' * digits}%" if digits > 0 else "#,##0%"
    return format_percent(value, format=pattern, locale=loc)
