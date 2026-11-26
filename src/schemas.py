"""Strict Pydantic v2 schemas for ML monitoring inputs, outputs, and alerts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SeverityLevel = Literal["OK", "WARNING", "CRITICAL"]
FeatureType = Literal["numeric", "categorical"]
AlertCategory = Literal["data_drift", "data_quality", "performance"]

_STRICT_CONFIG = ConfigDict(extra="forbid", allow_inf_nan=False)


class FeatureRecord(BaseModel):
    """Single credit-risk feature record validated at ingestion boundaries."""

    model_config = _STRICT_CONFIG

    income: float = Field(ge=0.0, le=10_000_000.0)
    debt_to_income: float = Field(ge=0.0, le=5.0)
    credit_score: float = Field(ge=300.0, le=850.0)
    utilization_rate: float = Field(ge=0.0, le=2.0)
    account_age_months: float = Field(ge=0.0, le=1_200.0)
    num_inquiries: int = Field(ge=0, le=100)
    employment_type: Literal["salaried", "self_employed", "contract", "unemployed"]
    region: Literal["north", "south", "east", "west"]
    default: int = Field(ge=0, le=1)


class AlertThresholds(BaseModel):
    """Configurable monitoring thresholds for drift, data quality, and model decay."""

    model_config = _STRICT_CONFIG

    psi_warning: float = Field(default=0.10, gt=0.0, le=2.0)
    psi_critical: float = Field(default=0.20, gt=0.0, le=5.0)
    ks_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    auc_drop_warning: float = Field(default=0.03, gt=0.0, le=0.5)
    auc_drop_critical: float = Field(default=0.05, gt=0.0, le=0.5)
    missing_rate_warning: float = Field(default=0.03, gt=0.0, le=1.0)
    missing_rate_critical: float = Field(default=0.05, gt=0.0, le=1.0)
    out_of_range_warning: float = Field(default=0.01, gt=0.0, le=1.0)
    out_of_range_critical: float = Field(default=0.03, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ordering(self) -> AlertThresholds:
        """Ensure critical thresholds strictly exceed or equal warning thresholds."""
        if self.psi_critical < self.psi_warning:
            raise ValueError("psi_critical must be >= psi_warning")
        if self.auc_drop_critical < self.auc_drop_warning:
            raise ValueError("auc_drop_critical must be >= auc_drop_warning")
        if self.missing_rate_critical < self.missing_rate_warning:
            raise ValueError("missing_rate_critical must be >= missing_rate_warning")
        if self.out_of_range_critical < self.out_of_range_warning:
            raise ValueError("out_of_range_critical must be >= out_of_range_warning")
        return self


class FeatureDriftResult(BaseModel):
    """Statistical drift diagnostics for one feature in a production batch."""

    model_config = _STRICT_CONFIG

    feature: str = Field(min_length=1)
    feature_type: FeatureType
    psi: float = Field(ge=0.0)
    ks_statistic: float | None = Field(default=None, ge=0.0, le=1.0)
    ks_pvalue: float | None = Field(default=None, ge=0.0, le=1.0)
    chi2_statistic: float | None = Field(default=None, ge=0.0)
    chi2_pvalue: float | None = Field(default=None, ge=0.0, le=1.0)
    severity: SeverityLevel


class DataQualityResult(BaseModel):
    """Data integrity summary for a production batch."""

    model_config = _STRICT_CONFIG

    row_count: int = Field(ge=1)
    missing_rate: float = Field(ge=0.0, le=1.0)
    duplicate_rate: float = Field(ge=0.0, le=1.0)
    out_of_range_rate: float = Field(ge=0.0, le=1.0)
    severity: SeverityLevel


class PerformanceMetrics(BaseModel):
    """Classification metrics evaluated on a labeled batch or split."""

    model_config = _STRICT_CONFIG

    auc_roc: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    positive_rate: float = Field(ge=0.0, le=1.0)
    prediction_rate: float = Field(ge=0.0, le=1.0)


class AlertEvent(BaseModel):
    """Single alert emitted by the monitoring engine."""

    model_config = _STRICT_CONFIG

    batch_id: str = Field(min_length=1)
    category: AlertCategory
    severity: Literal["WARNING", "CRITICAL"]
    metric_name: str = Field(min_length=1)
    metric_value: float
    threshold: float
    message: str = Field(min_length=1)


class BatchMonitoringReport(BaseModel):
    """Complete monitoring snapshot for a single production batch."""

    model_config = _STRICT_CONFIG

    batch_id: str = Field(min_length=1)
    overall_severity: SeverityLevel
    max_psi: float = Field(ge=0.0)
    drifted_feature_count: int = Field(ge=0)
    feature_drift: list[FeatureDriftResult]
    data_quality: DataQualityResult
    performance: PerformanceMetrics
    auc_drop: float = Field(ge=-1.0, le=1.0)
    alerts: list[AlertEvent]


class MonitoringRunReport(BaseModel):
    """Top-level report serialized to disk and consumed by the dashboard."""

    model_config = _STRICT_CONFIG

    generated_at: datetime
    locale: str = Field(min_length=2)
    reference_rows: int = Field(ge=1)
    baseline_metrics: PerformanceMetrics
    thresholds: AlertThresholds
    batches: list[BatchMonitoringReport] = Field(min_length=1)
