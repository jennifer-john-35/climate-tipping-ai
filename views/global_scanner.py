from __future__ import annotations

import pandas as pd
import streamlit as st

from components import render_globe, render_gauge, render_glass_card
from export.report_generator import generate_csv_report
from inference import ModelLoadError
from utils.alert_engine import filter_alerts
from config.theme_config import get_severity_colour


def render_global_scanner(pipeline) -> None:
    last_scan = st.session_state.get("last_scan_time")
    last_scan_str = last_scan.strftime("%Y-%m-%d %H:%M:%S") if last_scan else "Never"
    st.markdown(
        f"<h2>🌍 Global Climate Scanner</h2>"
        f"<p style='color:rgba(230,237,243,0.5);font-size:0.85rem;'>Last scanned: {last_scan_str}</p>",
        unsafe_allow_html=True,
    )

    # Curated real-world climate hotspot locations — not the grid CSV
    risk_df = pd.DataFrame([
        {"name": "Arctic Sea Ice",         "latitude": 80.0,   "longitude": 0.0,    "risk_score": 0.92},
        {"name": "Amazon Rainforest",       "latitude": -5.0,   "longitude": -60.0,  "risk_score": 0.87},
        {"name": "West Antarctic Ice",      "latitude": -80.0,  "longitude": -90.0,  "risk_score": 0.85},
        {"name": "Greenland Ice Sheet",     "latitude": 72.0,   "longitude": -40.0,  "risk_score": 0.83},
        {"name": "Great Barrier Reef",      "latitude": -18.0,  "longitude": 147.0,  "risk_score": 0.81},
        {"name": "Siberian Permafrost",     "latitude": 65.0,   "longitude": 100.0,  "risk_score": 0.79},
        {"name": "Sahel Region",            "latitude": 14.0,   "longitude": 10.0,   "risk_score": 0.74},
        {"name": "Indian Ocean Monsoon",    "latitude": 15.0,   "longitude": 75.0,   "risk_score": 0.71},
        {"name": "Boreal Forest Canada",    "latitude": 58.0,   "longitude": -100.0, "risk_score": 0.68},
        {"name": "Congo Rainforest",        "latitude": -2.0,   "longitude": 24.0,   "risk_score": 0.66},
        {"name": "Himalayan Glaciers",      "latitude": 30.0,   "longitude": 85.0,   "risk_score": 0.64},
        {"name": "Mediterranean Basin",     "latitude": 38.0,   "longitude": 15.0,   "risk_score": 0.61},
        {"name": "East African Rift",       "latitude": -2.0,   "longitude": 36.0,   "risk_score": 0.58},
        {"name": "Mekong Delta",            "latitude": 11.0,   "longitude": 105.0,  "risk_score": 0.55},
        {"name": "Patagonian Ice Fields",   "latitude": -50.0,  "longitude": -73.0,  "risk_score": 0.52},
        {"name": "North Sea",               "latitude": 56.0,   "longitude": 3.0,    "risk_score": 0.49},
        {"name": "Maldives Atolls",         "latitude": 4.0,    "longitude": 73.0,   "risk_score": 0.47},
        {"name": "Yellow River Basin",      "latitude": 36.0,   "longitude": 110.0,  "risk_score": 0.44},
        {"name": "Murray-Darling Basin",    "latitude": -34.0,  "longitude": 142.0,  "risk_score": 0.41},
        {"name": "Danube River Basin",      "latitude": 47.0,   "longitude": 19.0,   "risk_score": 0.38},
    ])
    region_names = risk_df["name"].tolist()

    col_globe, col_controls = st.columns([2, 1])

    with col_controls:
        st.markdown("#### Controls")
        region_name = st.selectbox("Region", region_names, key="scanner_region")
        projection = st.radio("Projection", ["natural earth", "orthographic"], key="scanner_projection")
        st.slider(
            "Alert Threshold", min_value=0.0, max_value=1.0,
            value=st.session_state.get("alert_threshold", 0.75),
            step=0.01, key="alert_threshold",
        )
        horizon = st.session_state.get("projection_horizon", 30)
        scan_in_progress = st.session_state.get("scan_in_progress", False)
        scan_clicked = st.button(
            "▶ Run Global Climate Scan", disabled=scan_in_progress, key="run_scan_btn",
        )

    # ── Scan ──────────────────────────────────────────────────────────────
    if scan_clicked:
        st.session_state["scan_in_progress"] = True
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(pct: int, msg: str) -> None:
            progress_bar.progress(pct)
            status_text.text(msg)

        try:
            pipeline.load_models()
        except ModelLoadError as exc:
            st.error(f"Model load failed: {exc}")
            st.session_state["scan_in_progress"] = False
            return

        try:
            # Scan selected region
            pipeline.run_scan(region_name, horizon, progress_callback)
            # Update globe with live grid scores
            grid_scores = pipeline.run_grid_scan()
            st.session_state["grid_scores"] = grid_scores
            status_text.text("✅ Scan complete.")
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
        finally:
            st.session_state["scan_in_progress"] = False

    # ── Globe ──────────────────────────────────────────────────────────────
    grid_scores = st.session_state.get("grid_scores")
    globe_data = grid_scores if grid_scores is not None else risk_df

    with col_globe:
        if not globe_data.empty:
            st.plotly_chart(render_globe(globe_data, projection=projection), width="stretch")
        else:
            st.info("No globe data available.")

    # ── Alert banner ───────────────────────────────────────────────────────
    scan_results = st.session_state.get("scan_results", {})
    alert_threshold = st.session_state.get("alert_threshold", 0.75)

    if scan_results:
        alerts = filter_alerts(scan_results, alert_threshold)
        if alerts:
            alert_html = (
                "<div class='alert-banner'><strong>⚠️ Climate Alerts</strong><br>"
                + "<br>".join(
                    f"• <b>{r.region_name}</b>: {r.risk_score:.3f}" for r in alerts
                )
                + "</div>"
            )
            st.markdown(alert_html, unsafe_allow_html=True)
        else:
            st.success(f"No regions exceed the alert threshold ({alert_threshold:.2f}).")

    # ── Results grid — show all scanned regions ────────────────────────────
    if scan_results:
        st.markdown("---")
        st.markdown("#### Scan Results")

        # Show gauge + export for selected region if scanned
        if region_name in scan_results:
            result = scan_results[region_name]
            col_g, col_info = st.columns([1, 2])
            with col_g:
                st.plotly_chart(render_gauge(result.risk_score, region_name), width="content")
            with col_info:
                colour = get_severity_colour(result.risk_score)
                severity = next(
                    k for k, (lo, hi, _) in __import__("config.theme_config", fromlist=["SEVERITY_THRESHOLDS"]).SEVERITY_THRESHOLDS.items()
                    if lo <= result.risk_score < hi or (result.risk_score >= 0.8 and k == "extreme")
                )
                st.markdown(render_glass_card(
                    title=region_name,
                    content=f"""
                        <p><b>Risk Score:</b> <span style='color:{colour};font-size:1.4rem;font-weight:bold;'>{result.risk_score:.3f}</span></p>
                        <p><b>Severity:</b> <span style='color:{colour};'>{severity.upper()}</span></p>
                        <p><b>Anomalies detected:</b> {sum(result.anomaly_flags)}</p>
                        <p><b>Scanned:</b> {result.scan_timestamp.strftime('%H:%M:%S')}</p>
                    """,
                    severity=severity,
                ), unsafe_allow_html=True)

                csv_data = generate_csv_report(result)
                st.download_button(
                    label="📥 Export CSV Report",
                    data=csv_data,
                    file_name=f"scan_{region_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                )

        # Summary table of all scanned regions
        if len(scan_results) > 1:
            st.markdown("**All Scanned Regions**")
            rows = [
                {
                    "Region": r.region_name,
                    "Risk Score": round(r.risk_score, 3),
                    "Severity": next(
                        k for k, (lo, hi, _) in __import__("config.theme_config", fromlist=["SEVERITY_THRESHOLDS"]).SEVERITY_THRESHOLDS.items()
                        if lo <= r.risk_score < hi or (r.risk_score >= 0.8 and k == "extreme")
                    ).upper(),
                    "Anomalies": sum(r.anomaly_flags),
                }
                for r in sorted(scan_results.values(), key=lambda x: x.risk_score, reverse=True)
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")

    elif not scan_results:
        st.info("Configure settings and click 'Run Global Climate Scan' to begin.")
