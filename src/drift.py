"""Statistical data drift (PSI, KS, Chi-square) and batch data-quality checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.guardrails import sanitize_numeric_array, validate_dataframe_bounds
from src.schemas import (
    AlertThresholds,
    DataQualityResult,
    FeatureDriftResult,
    SeverityLevel,
)

EPSILON = 1e-6


def compute_psi(reference: np.ndarray | pd.Series, current: np.ndarray | pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index using reference quantile bin edges."""
    if bins < 1:
        raise ValueError("bins must be >= 1.")
    ref_clean = sanitize_numeric_array(reference)
    cur_clean = sanitize_numeric_array(current)
    if ref_clean.size == 0 or cur_clean.size == 0:
        raise ValueError("Reference and current arrays must contain finite values.")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref_clean, quantiles))
    if edges.size < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=edges)
    cur_counts, _ = np.histogram(cur_clean, bins=edges)

    ref_dist = np.clip(ref_counts / ref_counts.sum(), EPSILON, None)
    cur_dist = np.clip(cur_counts / cur_counts.sum(), EPSILON, None)
    psi = float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))
    return max(0.0, psi)


def compute_ks_test(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Run two-sample Kolmogorov-Smirnov test and return (statistic, p_value)."""
    ref_clean = sanitize_numeric_array(reference)
    cur_clean = sanitize_numeric_array(current)
    if ref_clean.size == 0 or cur_clean.size == 0:
        raise ValueError("Reference and current arrays must contain finite values.")

    result = ks_2samp(ref_clean, cur_clean)
    return float(result.statistic), float(result.pvalue)


def compute_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float, float]:
    """Compute categorical PSI and Chi-square test (psi, chi2_stat, p_value)."""
    ref_str = reference.dropna().astype(str)
    cur_str = current.dropna().astype(str)
    if ref_str.empty or cur_str.empty:
        raise ValueError("Reference and current categorical series must contain non-null values.")

    ref_vc = ref_str.value_counts()
    cur_vc = cur_str.value_counts()
    categories = sorted(ref_vc.index.union(cur_vc.index))

    ref_counts = ref_vc.reindex(categories, fill_value=0).to_numpy(dtype=np.float64)
    cur_counts = cur_vc.reindex(categories, fill_value=0).to_numpy(dtype=np.float64)

    ref_dist = np.clip(ref_counts / max(ref_counts.sum(), 1.0), EPSILON, None)
    cur_dist = np.clip(cur_counts / max(cur_counts.sum(), 1.0), EPSILON, None)
    psi = max(0.0, float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))))

    contingency = np.vstack([ref_counts, cur_counts])
    nonzero_cols = contingency.sum(axis=0) > 0
    contingency = contingency[:, nonzero_cols]
    if contingency.shape[1] < 2:
        return psi, 0.0, 1.0

    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    return psi, float(chi2_stat), float(p_val)


def evaluate_data_quality(
    df: pd.DataFrame,
    numeric_bounds: dict[str, tuple[float, float]],
    thresholds: AlertThresholds,
) -> DataQualityResult:
    """Compute missingness, duplicate rate, and out-of-range violation rate for a batch."""
    validate_dataframe_bounds(df)
    feature_cols = [c for c in df.columns if c not in ("batch_id", "default")]
    if not feature_cols:
        raise ValueError("Input DataFrame must contain at least one feature column.")
    sub = df[feature_cols]

    missing_rate = round(float(sub.isna().sum().sum() / sub.size), 6)
    duplicate_rate = round(float(sub.duplicated().mean()), 6)

    total_numeric_cells = 0
    oor_cells = 0
    for col, (low, high) in numeric_bounds.items():
        if col in df.columns:
            non_null = df[col].dropna()
            total_numeric_cells += int(non_null.size)
            numeric = pd.to_numeric(non_null, errors="coerce")
            invalid = numeric.isna() | (numeric < low) | (numeric > high)
            oor_cells += int(invalid.sum())

    oor_rate = round(float(oor_cells / total_numeric_cells) if total_numeric_cells > 0 else 0.0, 6)

    severity: SeverityLevel = "OK"
    if missing_rate >= thresholds.missing_rate_critical or oor_rate >= thresholds.out_of_range_critical:
        severity = "CRITICAL"
    elif missing_rate >= thresholds.missing_rate_warning or oor_rate >= thresholds.out_of_range_warning:
        severity = "WARNING"

    return DataQualityResult(
        row_count=len(df),
        missing_rate=missing_rate,
        duplicate_rate=duplicate_rate,
        out_of_range_rate=oor_rate,
        severity=severity,
    )


def evaluate_batch_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    thresholds: AlertThresholds,
) -> list[FeatureDriftResult]:
    """Evaluate PSI and hypothesis tests across all numeric and categorical features."""
    required = list(numeric_features) + list(categorical_features)
    validate_dataframe_bounds(reference_df, required_columns=required)
    validate_dataframe_bounds(current_df, required_columns=required)

    results: list[FeatureDriftResult] = []

    for col in numeric_features:
        ref_clean = sanitize_numeric_array(reference_df[col])
        cur_clean = sanitize_numeric_array(current_df[col])
        psi = round(compute_psi(ref_clean, cur_clean), 6)
        ks_stat, ks_p = compute_ks_test(ref_clean, cur_clean)
        ks_stat_r = round(ks_stat, 6)
        ks_p_r = round(ks_p, 6)

        sev: SeverityLevel = "OK"
        if psi >= thresholds.psi_critical:
            sev = "CRITICAL"
        elif psi >= thresholds.psi_warning or ks_p_r < thresholds.ks_alpha:
            sev = "WARNING"

        results.append(
            FeatureDriftResult(
                feature=col,
                feature_type="numeric",
                psi=psi,
                ks_statistic=ks_stat_r,
                ks_pvalue=ks_p_r,
                severity=sev,
            )
        )

    for col in categorical_features:
        psi, chi2_stat, chi2_p = compute_categorical_drift(reference_df[col], current_df[col])
        psi_r = round(psi, 6)
        chi2_stat_r = round(chi2_stat, 6)
        chi2_p_r = round(chi2_p, 6)

        sev = "OK"
        if psi_r >= thresholds.psi_critical:
            sev = "CRITICAL"
        elif psi_r >= thresholds.psi_warning or chi2_p_r < thresholds.ks_alpha:
            sev = "WARNING"

        results.append(
            FeatureDriftResult(
                feature=col,
                feature_type="categorical",
                psi=psi_r,
                chi2_statistic=chi2_stat_r,
                chi2_pvalue=chi2_p_r,
                severity=sev,
            )
        )

    return results
