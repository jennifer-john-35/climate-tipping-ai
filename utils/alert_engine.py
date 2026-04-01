from __future__ import annotations

import csv
import os
from datetime import datetime

from inference import AlertEvent, ScanResult


def filter_alerts(
    scan_results: dict[str, ScanResult], threshold: float
) -> list[ScanResult]:
    """Return regions with risk_score > threshold, sorted descending by risk_score."""
    filtered = [r for r in scan_results.values() if r.risk_score > threshold]
    return sorted(filtered, key=lambda r: r.risk_score, reverse=True)


def log_alert(event: AlertEvent, path: str = "logs/alert_log.csv") -> None:
    """Append AlertEvent to CSV. Creates file with header if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "region_name", "risk_score", "threshold"])
        writer.writerow([
            event.timestamp,
            event.region_name,
            event.risk_score,
            event.threshold,
        ])
