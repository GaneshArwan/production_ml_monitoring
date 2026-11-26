"""Deterministic synthetic credit-risk dataset with progressive production drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMERIC_FEATURES: tuple[str, ...] = (
    "income",
    "debt_to_income",
    "credit_score",
    "utilization_rate",
    "account_age_months",
    "num_inquiries",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("employment_type", "region")
TARGET_COLUMN = "default"

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "income": (0.0, 10_000_000.0),
    "debt_to_income": (0.0, 5.0),
    "credit_score": (300.0, 850.0),
    "utilization_rate": (0.0, 2.0),
    "account_age_months": (0.0, 1_200.0),
    "num_inquiries": (0.0, 100.0),
}


@dataclass(frozen=True)
class BatchShiftSpec:
    """Drift and data-quality corruption parameters for one production batch."""

    batch_id: str
    n_rows: int
    income_scale: float = 1.0
    dti_shift: float = 0.0
    score_shift: float = 0.0
    util_shift: float = 0.0
    missing_rate: float = 0.0
    oor_rate: float = 0.0
    concept_shift: float = 0.0
    employment_probs: tuple[float, float, float, float] = (0.60, 0.20, 0.15, 0.05)


DEFAULT_BATCH_SPECS: tuple[BatchShiftSpec, ...] = (
    BatchShiftSpec("2026-01", 2_000, missing_rate=0.005),
    BatchShiftSpec("2026-02", 2_000, score_shift=-4.0, dti_shift=0.01, missing_rate=0.008),
    BatchShiftSpec(
        "2026-03",
        2_000,
        income_scale=0.92,
        dti_shift=0.07,
        score_shift=-18.0,
        util_shift=0.08,
        missing_rate=0.025,
        oor_rate=0.005,
        employment_probs=(0.52, 0.23, 0.18, 0.07),
    ),
    BatchShiftSpec(
        "2026-04",
        2_000,
        income_scale=0.86,
        dti_shift=0.12,
        score_shift=-30.0,
        util_shift=0.14,
        missing_rate=0.038,
        oor_rate=0.015,
        concept_shift=0.35,
        employment_probs=(0.45, 0.25, 0.20, 0.10),
    ),
    BatchShiftSpec(
        "2026-05",
        2_000,
        income_scale=0.78,
        dti_shift=0.19,
        score_shift=-48.0,
        util_shift=0.22,
        missing_rate=0.062,
        oor_rate=0.035,
        concept_shift=0.65,
        employment_probs=(0.38, 0.26, 0.22, 0.14),
    ),
    BatchShiftSpec(
        "2026-06",
        2_000,
        income_scale=0.72,
        dti_shift=0.24,
        score_shift=-62.0,
        util_shift=0.28,
        missing_rate=0.085,
        oor_rate=0.050,
        concept_shift=0.90,
        employment_probs=(0.32, 0.26, 0.24, 0.18),
    ),
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _simulate_slice(
    rng: np.random.Generator,
    spec: BatchShiftSpec,
    include_batch_id: bool,
) -> pd.DataFrame:
    n = spec.n_rows
    income = np.clip(
        rng.lognormal(mean=10.9, sigma=0.42, size=n) * spec.income_scale,
        12_000.0,
        600_000.0,
    )
    dti = np.clip(rng.normal(loc=0.34 + spec.dti_shift, scale=0.12, size=n), 0.02, 1.50)
    credit_score = np.clip(
        rng.normal(loc=695.0 + spec.score_shift, scale=55.0, size=n), 320.0, 850.0
    )
    utilization = np.clip(
        rng.beta(a=2.2, b=4.5, size=n) + spec.util_shift, 0.01, 1.35
    )
    account_age = np.clip(rng.gamma(shape=4.0, scale=18.0, size=n), 1.0, 480.0)
    inquiries = rng.poisson(lam=1.6 + max(0.0, spec.dti_shift * 5.0), size=n)

    employment = rng.choice(
        ["salaried", "self_employed", "contract", "unemployed"],
        size=n,
        p=spec.employment_probs,
    )
    region = rng.choice(["north", "south", "east", "west"], size=n, p=[0.28, 0.26, 0.24, 0.22])

    emp_risk = np.select(
        [employment == "unemployed", employment == "contract", employment == "self_employed"],
        [1.10, 0.35, 0.20],
        default=0.0,
    )

    # Concept shift progressively alters the true risk drivers (post-shift regime).
    base_signal = (
        2.45 * (dti - 0.34)
        - 0.024 * (credit_score - 695.0)
        + 2.15 * (utilization - 0.32)
        - 0.35 * np.log1p(income / 50_000.0)
        + 0.16 * inquiries
        + emp_risk
    )
    shifted_signal = (
        -1.20 * (dti - 0.45)
        + 0.012 * (credit_score - 650.0)
        - 1.00 * (utilization - 0.45)
        + 0.45 * ((account_age < 24.0).astype(float) - 0.2)
    )
    alpha = min(max(spec.concept_shift, 0.0), 1.0)
    logit = (
        -1.35
        + (1.0 - alpha) * base_signal
        + alpha * shifted_signal
        + rng.normal(0.0, 0.35 + 0.35 * alpha, size=n)
    )
    prob = _sigmoid(logit)
    target = (rng.uniform(0.0, 1.0, size=n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "income": np.round(income, 2),
            "debt_to_income": np.round(dti, 4),
            "credit_score": np.round(credit_score, 1),
            "utilization_rate": np.round(utilization, 4),
            "account_age_months": np.round(account_age, 1),
            "num_inquiries": inquiries.astype(int),
            "employment_type": employment,
            "region": region,
            TARGET_COLUMN: target,
        }
    )

    if spec.oor_rate > 0:
        oor_mask = rng.uniform(0.0, 1.0, size=n) < spec.oor_rate
        df.loc[oor_mask, "credit_score"] = 920.0  # Invalid score > 850
        df.loc[oor_mask, "debt_to_income"] = -0.25  # Invalid negative ratio

    if spec.missing_rate > 0:
        for col in ("income", "credit_score", "utilization_rate", "employment_type"):
            miss_mask = rng.uniform(0.0, 1.0, size=n) < spec.missing_rate
            df.loc[miss_mask, col] = np.nan

    if include_batch_id:
        df.insert(0, "batch_id", spec.batch_id)

    return df


def generate_reference_and_production_batches(
    reference_rows: int = 10_000,
    seed: int = 42,
    batch_specs: tuple[BatchShiftSpec, ...] = DEFAULT_BATCH_SPECS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate clean reference training data and drifted monthly production batches."""
    rng = np.random.default_rng(seed)
    ref_spec = BatchShiftSpec(batch_id="reference", n_rows=reference_rows)
    reference_df = _simulate_slice(rng, ref_spec, include_batch_id=False)
    prod_frames = [_simulate_slice(rng, spec, include_batch_id=True) for spec in batch_specs]
    production_df = pd.concat(prod_frames, ignore_index=True)
    return reference_df, production_df
