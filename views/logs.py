from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from utils.data_validator import DataValidator, _STALE_DAYS


def render_logs() -> None:
    """Render the System Logs page."""
    st.markdown("<h2>📋 System Logs</h2>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_alert, tab_interaction = st.tabs(["Alert Log", "Interaction Log"])

    with tab_alert:
        try:
            alert_df = pd.read_csv("logs/alert_log.csv")
            st.dataframe(alert_df, width="stretch")
        except FileNotFoundError:
            st.info("No alerts logged yet.")
        except Exception as exc:
            st.error(f"Failed to load alert log: {exc}")

    with tab_interaction:
        try:
            click_df = pd.read_csv("logs/click_log.csv")
            st.dataframe(click_df, width="stretch")
        except FileNotFoundError:
            st.info("No interactions logged yet.")
        except Exception as exc:
            st.error(f"Failed to load interaction log: {exc}")

    # ── Data Freshness ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Data Freshness")

    validator = DataValidator()
    freshness_info = validator.get_freshness_info()

    now = datetime.now(timezone.utc)
    rows = []
    for filename, last_modified in freshness_info.items():
        age_days = (now - last_modified).total_seconds() / 86_400
        is_stale = age_days > _STALE_DAYS
        status = "⚠️ Stale" if is_stale else "✅ Fresh"
        rows.append({
            "File": filename,
            "Last Modified": last_modified.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Status": status,
        })

    freshness_df = pd.DataFrame(rows)
    st.dataframe(freshness_df, width="stretch")
