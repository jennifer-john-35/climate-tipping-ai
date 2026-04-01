from __future__ import annotations

import csv
import io
from pathlib import Path

from inference import ScanResult, FEATURE_NAMES

MODEL_PATHS = {
    "forecast_model": "models/forecast_model.keras",
    "tipping_model": "models/tipping_model.pkl",
}


def generate_csv_report(scan_result: ScanResult) -> str:
    """Generate a CSV report string for the given ScanResult.

    Returns a CSV string with:
    1. Metadata header rows (# prefixed): model versions (file mtime), scan timestamp
    2. Data rows: region_name, scan_timestamp, risk_score, anomaly_flag_count,
       shap values for each of 5 features, forecast values per signal per day
    """
    buf = io.StringIO()

    # ── Metadata comment rows ──────────────────────────────────────────────
    buf.write(f"# scan_timestamp: {scan_result.scan_timestamp.isoformat()}\n")

    for key in ("forecast_model", "tipping_model"):
        p = Path(MODEL_PATHS[key])
        mtime = p.stat().st_mtime if p.exists() else "unknown"
        buf.write(f"# {key}_mtime: {mtime}\n")

    # ── Build header and data row ──────────────────────────────────────────
    writer = csv.writer(buf)

    # Determine forecast columns from the scan result
    forecast_cols: list[str] = []
    forecast_vals: list[float] = []
    for signal, values in scan_result.forecasts.items():
        for day_idx, val in enumerate(values):
            forecast_cols.append(f"forecast_{signal}_day_{day_idx}")
            forecast_vals.append(float(val))

    header = (
        ["region_name", "scan_timestamp", "risk_score", "anomaly_flag_count"]
        + [f"shap_{feat}" for feat in FEATURE_NAMES]
        + forecast_cols
    )
    writer.writerow(header)

    data_row = (
        [
            scan_result.region_name,
            scan_result.scan_timestamp.isoformat(),
            scan_result.risk_score,
            sum(scan_result.anomaly_flags),
        ]
        + [scan_result.shap_values.get(feat, 0.0) for feat in FEATURE_NAMES]
        + forecast_vals
    )
    writer.writerow(data_row)

    return buf.getvalue()
