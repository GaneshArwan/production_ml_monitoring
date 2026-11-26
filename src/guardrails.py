"""Security and reliability guardrails for paths, tabular bounds, and model artifacts."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import hmac
from pathlib import Path

import numpy as np
import pandas as pd

MAX_ROWS_DEFAULT = 500_000
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024  # 100 MB ceiling before deserialization


def resolve_safe_path(base_dir: Path, target: Path) -> Path:
    """Resolve target path and verify it stays strictly within base_dir."""
    resolved_base = base_dir.resolve()
    candidate = target if target.is_absolute() else (resolved_base / target)
    resolved_target = candidate.resolve()
    if not resolved_target.is_relative_to(resolved_base):
        raise ValueError(f"Path traversal blocked: {target} escapes {resolved_base}")
    return resolved_target


def validate_dataframe_bounds(
    df: pd.DataFrame,
    max_rows: int = MAX_ROWS_DEFAULT,
    required_columns: Sequence[str] = (),
) -> None:
    """Validate non-empty DataFrame, unique columns, row-count cap, and required columns."""
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1.")
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if df.columns.has_duplicates:
        raise ValueError("Input DataFrame must not contain duplicate column names.")
    if len(df) > max_rows:
        raise ValueError(f"Row count {len(df)} exceeds safety cap of {max_rows}.")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sanitize_numeric_array(values: np.ndarray | pd.Series) -> np.ndarray:
    """Coerce 1D array-like input to float64 and strip NaN, Inf, and non-numeric values."""
    if isinstance(values, pd.Series):
        coerced = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError("Input numeric array must be 1-dimensional.")
        if raw.dtype.kind in ("U", "S", "O"):
            coerced = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
        else:
            coerced = np.asarray(raw, dtype=np.float64)
    return coerced[np.isfinite(coerced)]


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_artifact(
    artifact_path: Path,
    base_dir: Path,
    expected_sha256: str,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> Path:
    """Verify artifact path safety, size limit, and SHA-256 digest before deserialization."""
    if not expected_sha256 or not expected_sha256.strip():
        raise ValueError("expected_sha256 must be a non-empty SHA-256 hex digest.")

    safe_path = resolve_safe_path(base_dir, artifact_path)
    if not safe_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {safe_path}")
    size = safe_path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"Model artifact size {size} bytes outside valid range (1..{max_bytes}).")

    actual = compute_file_sha256(safe_path)
    if not hmac.compare_digest(actual.lower(), expected_sha256.strip().lower()):
        raise ValueError(
            f"SHA-256 mismatch for {safe_path.name}: expected {expected_sha256}, got {actual}"
        )
    return safe_path
