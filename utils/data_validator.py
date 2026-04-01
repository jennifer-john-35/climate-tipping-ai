from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS: dict[str, list[str]] = {
    "global_temperature.csv": [],   # flexible — skiprows=1 handles the header row
    "co2_levels.csv": ["year", "month", "average"],
    "ocean_heat.csv": ["Mean"],
    "rainfall.csv": [],
    "sea_ice.csv": [],
    "climate_training_data.csv": ["temp", "co2", "sea_ice", "ocean_heat"],
    "earth_grid.csv": ["latitude", "longitude"],
    "global_risk_scores.csv": ["latitude", "longitude", "risk_score", "name"],
}

NAN_THRESHOLD = 0.05
_STALE_DAYS = 30


@dataclass
class ValidationResult:
    filename: str
    valid: bool
    missing_columns: list[str] = field(default_factory=list)
    nan_violations: list[str] = field(default_factory=list)
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_stale: bool = False


class DataValidator:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_file(self, filepath: str) -> ValidationResult:
        """Validate a single CSV file.

        Checks required columns, NaN threshold per numeric column, and
        staleness (> 30 days from file mtime).  Never raises — always
        returns a ValidationResult.  If a numeric column exceeds the NaN
        threshold, forward-fill imputation is applied in-place and the
        column name is recorded in nan_violations.
        """
        filename = os.path.basename(filepath)

        # ── Staleness / mtime ──────────────────────────────────────────
        last_modified, is_stale = self._get_mtime(filepath)

        # ── Load CSV ───────────────────────────────────────────────────
        try:
            df = pd.read_csv(filepath, comment="#")
        except Exception:
            return ValidationResult(
                filename=filename,
                valid=False,
                missing_columns=[],
                nan_violations=[],
                last_modified=last_modified,
                is_stale=is_stale,
            )

        # ── Required-column check ──────────────────────────────────────
        required = REQUIRED_COLUMNS.get(filename, [])
        actual_cols = list(df.columns)
        missing = [c for c in required if c not in actual_cols]

        # ── NaN threshold check + imputation ───────────────────────────
        nan_violations: list[str] = []
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            nan_ratio = df[col].isna().mean()
            if nan_ratio > NAN_THRESHOLD:
                nan_violations.append(col)
                df[col] = df[col].ffill()

        valid = len(missing) == 0

        return ValidationResult(
            filename=filename,
            valid=valid,
            missing_columns=missing,
            nan_violations=nan_violations,
            last_modified=last_modified,
            is_stale=is_stale,
        )

    def validate_all(self) -> list[ValidationResult]:
        """Validate every file listed in REQUIRED_COLUMNS."""
        results: list[ValidationResult] = []
        for filename in REQUIRED_COLUMNS:
            filepath = os.path.join(self.data_dir, filename)
            results.append(self.validate_file(filepath))
        return results

    def get_freshness_info(self) -> dict[str, datetime]:
        """Return {filename: last_modified} for all data files."""
        info: dict[str, datetime] = {}
        for filename in REQUIRED_COLUMNS:
            filepath = os.path.join(self.data_dir, filename)
            last_modified, _ = self._get_mtime(filepath)
            info[filename] = last_modified
        return info

    def impute_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill NaN values in all columns."""
        return df.ffill()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mtime(self, filepath: str) -> tuple[datetime, bool]:
        """Return (last_modified, is_stale).  Falls back to now() if the
        file does not exist."""
        try:
            mtime = os.path.getmtime(filepath)
            last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc)
        except OSError:
            last_modified = datetime.now(timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - last_modified).total_seconds() / 86_400
        is_stale = age_days > _STALE_DAYS

        return last_modified, is_stale
