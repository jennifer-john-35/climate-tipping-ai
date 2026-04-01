from __future__ import annotations

import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.theme_config import (
    ACCENT_COLOR, PLOTLY_LAYOUT_DEFAULTS, PLOTLY_TEMPLATE,
    get_severity_colour, SEVERITY_THRESHOLDS,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def _load_climate() -> pd.DataFrame:
    return _clean(pd.read_csv("data/climate_training_data.csv"))


# ── 1. Heatmap Upload ──────────────────────────────────────────────────────────

def _render_heatmap_upload() -> None:
    st.markdown("### 🌡️ Heatmap Hotspot Analyzer")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>Upload a thermal or satellite "
        "heatmap image. The system detects red/yellow heat blobs and maps them onto "
        "the globe as risk zones.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload heatmap image", type=["png", "jpg", "jpeg", "webp"], key="heatmap_upload"
    )

    if not uploaded:
        return

    col_img, col_map = st.columns([1, 2])

    with col_img:
        st.image(uploaded, caption="Uploaded Heatmap", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    from utils.heatmap_analyzer import detect_hotspots
    hotspots = detect_hotspots(tmp_path)

    if not hotspots:
        st.warning("No hotspots detected. Try an image with clear red/yellow heat zones.")
        return

    lats = [h[0] for h in hotspots]
    lons = [h[1] for h in hotspots]
    intensities = [h[2] for h in hotspots]

    hs_df = pd.DataFrame({
        "latitude": lats, "longitude": lons,
        "intensity": intensities,
        "risk_score": intensities,
        "name": [f"Hotspot {i+1}" for i in range(len(hotspots))],
    })

    with col_map:
        colours = [get_severity_colour(s) for s in intensities]
        sizes = [6 + s * 14 for s in intensities]

        fig = go.Figure(go.Scattergeo(
            lat=lats, lon=lons,
            text=[f"Hotspot {i+1}<br>Intensity: {intensities[i]:.2f}" for i in range(len(hotspots))],
            hoverinfo="text",
            mode="markers",
            marker=dict(size=sizes, color=colours, opacity=0.85,
                        line=dict(width=1, color="white")),
        ))
        fig.update_geos(
            projection_type="natural earth",
            bgcolor="#0d1117", landcolor="#1a2f1a",
            oceancolor="#0a1628", showocean=True, showland=True,
            showcountries=True, countrycolor="rgba(255,255,255,0.3)",
        )
        layout = dict(PLOTLY_LAYOUT_DEFAULTS)
        layout["height"] = 380
        layout["margin"] = dict(t=10, b=10, l=10, r=10)
        layout["title"] = f"Detected {len(hotspots)} Hotspot(s)"
        fig.update_layout(**layout)
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"**{len(hotspots)} hotspot(s) detected**")
    st.dataframe(hs_df[["name", "latitude", "longitude", "intensity"]]
                 .round(3), width="stretch")


# ── 2. What-If Simulator ───────────────────────────────────────────────────────

def _render_whatif(pipeline) -> None:
    st.markdown("### 🧪 Climate What-If Simulator")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>Adjust climate variables and "
        "see how the tipping point risk score changes in real time.</p>",
        unsafe_allow_html=True,
    )

    tipping_model = st.session_state.get("tipping_model")
    if tipping_model is None:
        st.info("Load models first by running a scan from Global Scanner.")
        return

    try:
        climate_df = _load_climate()
        baseline = climate_df.tail(1).iloc[0]
    except Exception as exc:
        st.error(f"Could not load baseline data: {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        temp_delta = st.slider("🌡️ Temperature Anomaly (°C)", -2.0, 4.0,
                               float(round(baseline.get("temp", 0.5), 2)), 0.1)
        co2_val = st.slider("💨 CO₂ Level (ppm)", 300.0, 500.0,
                            float(round(baseline.get("co2", 380.0), 1)), 1.0)
    with col2:
        sea_ice_val = st.slider("🧊 Sea Ice Extent (10⁶ km²)", 3.0, 14.0,
                                float(round(baseline.get("sea_ice", 10.0), 1)), 0.1)
        ocean_heat = st.slider("🌊 Ocean Heat Content", -1.0, 1.0,
                               float(round(baseline.get("ocean_heat", 0.0), 2)), 0.01)

    row = pd.DataFrame([[temp_delta, co2_val, sea_ice_val, ocean_heat]],
                       columns=["temp", "co2", "sea_ice", "ocean_heat"])
    raw = tipping_model.decision_function(row)[0]
    risk = float(np.clip(1 - raw, 0, 1))
    colour = get_severity_colour(risk)

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        number={"valueformat": ".3f", "font": {"size": 36, "color": colour}},
        title={"text": "Simulated Risk Score", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": colour, "thickness": 0.3},
            "steps": [
                {"range": [0.0, 0.4], "color": SEVERITY_THRESHOLDS["low"][2]},
                {"range": [0.4, 0.6], "color": SEVERITY_THRESHOLDS["medium"][2]},
                {"range": [0.6, 0.8], "color": SEVERITY_THRESHOLDS["high"][2]},
                {"range": [0.8, 1.0], "color": SEVERITY_THRESHOLDS["extreme"][2]},
            ],
        },
    ))
    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["height"] = 260
    layout["margin"] = dict(t=40, b=10, l=30, r=30)
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="content")

    label = next(k for k, (lo, hi, _) in SEVERITY_THRESHOLDS.items() if lo <= risk < hi or (risk >= 0.8 and k == "extreme"))
    st.markdown(
        f"<div class='glass-card'><b>Risk Level:</b> "
        f"<span style='color:{colour};font-weight:bold;'>{label.upper()}</span> "
        f"— Score: <span style='color:{colour};'>{risk:.3f}</span></div>",
        unsafe_allow_html=True,
    )


# ── 3. Tipping Point Countdown ─────────────────────────────────────────────────

def _render_countdown() -> None:
    st.markdown("### ⏳ Tipping Point Countdown")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>Estimates days until each region's "
        "risk score crosses the extreme threshold (0.8), based on current trend.</p>",
        unsafe_allow_html=True,
    )

    scan_results = st.session_state.get("scan_results", {})
    if not scan_results:
        st.info("Run a scan from Global Scanner first.")
        return

    rows = []
    for region, result in scan_results.items():
        trend = result.rolling_risk_trend
        if len(trend) < 2:
            days = "N/A"
            colour = "#8b949e"
        else:
            slope = float(np.polyfit(range(len(trend)), trend, 1)[0])
            current = result.risk_score
            if slope <= 0 or current >= 0.8:
                days = "Already critical" if current >= 0.8 else "Stable ✅"
                colour = "#ff2244" if current >= 0.8 else "#00ff88"
            else:
                remaining = (0.8 - current) / slope
                days = f"{int(remaining)} days"
                colour = "#ff8c00" if remaining < 90 else "#ffd700"
        rows.append({"Region": region, "Current Risk": round(result.risk_score, 3),
                     "Trend": "Rising" if len(trend) > 1 and np.polyfit(range(len(trend)), trend, 1)[0] > 0 else "Stable/Falling",
                     "Days to Critical": days, "_colour": colour})

    for row in sorted(rows, key=lambda r: r["Current Risk"], reverse=True):
        col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 2])
        col_a.markdown(f"**{row['Region']}**")
        col_b.markdown(f"<span style='color:{get_severity_colour(row['Current Risk'])}'>{row['Current Risk']}</span>", unsafe_allow_html=True)
        col_c.markdown(row["Trend"])
        col_d.markdown(f"<span style='color:{row['_colour']};font-weight:bold;'>{row['Days to Critical']}</span>", unsafe_allow_html=True)


# ── 4. Anomaly Calendar ────────────────────────────────────────────────────────

def _render_anomaly_calendar() -> None:
    st.markdown("### 📅 Historical Anomaly Calendar")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>Anomaly density per time period — "
        "darker = more anomalies detected.</p>",
        unsafe_allow_html=True,
    )

    scan_results = st.session_state.get("scan_results", {})
    if not scan_results:
        st.info("Run a scan from Global Scanner first.")
        return

    result = next(iter(scan_results.values()))
    flags = result.anomaly_flags
    n = len(flags)
    if n == 0:
        st.info("No anomaly data available.")
        return

    # Group into chunks of ~30 (months proxy)
    chunk = 30
    n_chunks = max(1, n // chunk)
    densities = []
    for i in range(n_chunks):
        chunk_flags = flags[i * chunk: (i + 1) * chunk]
        densities.append(sum(chunk_flags) / len(chunk_flags) if chunk_flags else 0)

    # Build a grid (rows=years proxy, cols=months proxy)
    cols_per_row = 12
    n_rows = max(1, (n_chunks + cols_per_row - 1) // cols_per_row)
    grid = np.zeros((n_rows, cols_per_row))
    for idx, d in enumerate(densities):
        r, c = divmod(idx, cols_per_row)
        if r < n_rows:
            grid[r, c] = d

    fig = px.imshow(
        grid,
        color_continuous_scale=[[0, "#0d1117"], [0.3, "#00ff88"],
                                 [0.6, "#ffd700"], [1.0, "#ff2244"]],
        labels={"x": "Month", "y": "Year", "color": "Anomaly Density"},
        title="Anomaly Density Heatmap",
        template=PLOTLY_TEMPLATE,
    )
    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["height"] = 280
    layout["margin"] = dict(t=50, b=40, l=60, r=20)
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch")


# ── 5. Correlation Matrix ──────────────────────────────────────────────────────

def _render_correlation() -> None:
    st.markdown("### 🔗 Signal Correlation Matrix")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>How correlated are the climate "
        "signals with each other? Strong correlations suggest co-movement.</p>",
        unsafe_allow_html=True,
    )

    try:
        df = _load_climate()
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return

    corr = df.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Climate Signal Correlation Matrix",
        template=PLOTLY_TEMPLATE,
    )
    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["height"] = 380
    layout["margin"] = dict(t=50, b=40, l=80, r=20)
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch")


# ── 6. Sparkline Risk Trends ───────────────────────────────────────────────────

def _render_sparklines() -> None:
    st.markdown("### 📈 Risk Score Trend Sparklines")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>Rolling risk trend for each "
        "scanned region.</p>",
        unsafe_allow_html=True,
    )

    scan_results = st.session_state.get("scan_results", {})
    if not scan_results:
        st.info("Run a scan from Global Scanner first.")
        return

    cols = st.columns(min(3, len(scan_results)))
    for idx, (region, result) in enumerate(scan_results.items()):
        trend = result.rolling_risk_trend
        if len(trend) == 0:
            continue
        colour = get_severity_colour(result.risk_score)
        fig = go.Figure(go.Scatter(
            y=trend, mode="lines",
            line=dict(color=colour, width=2),
            fill="tozeroy",
            fillcolor=colour.replace("#", "rgba(").rstrip(")") + ",0.15)" if colour.startswith("#") else colour,
        ))
        layout = dict(PLOTLY_LAYOUT_DEFAULTS)
        layout["height"] = 120
        layout["margin"] = dict(t=30, b=10, l=10, r=10)
        layout["title"] = dict(text=f"{region[:20]}<br><sup>{result.risk_score:.3f}</sup>",
                               font=dict(size=11))
        layout["xaxis"] = dict(showticklabels=False, showgrid=False, zeroline=False)
        layout["yaxis"] = dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False)
        layout["showlegend"] = False
        fig.update_layout(**layout)
        with cols[idx % 3]:
            st.plotly_chart(fig, width="stretch")


# ── 7. Scenario Comparison ─────────────────────────────────────────────────────

def _render_scenario_comparison() -> None:
    st.markdown("### ⚖️ Scenario Comparison")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>Compare risk scores across "
        "different projection horizons for the same region.</p>",
        unsafe_allow_html=True,
    )

    scan_results = st.session_state.get("scan_results", {})
    if not scan_results:
        st.info("Run a scan from Global Scanner first.")
        return

    region = st.selectbox("Region", list(scan_results.keys()), key="scenario_region")
    result = scan_results[region]

    horizons = [7, 14, 30, 60, 90]
    scenario_scores = []
    for h in horizons:
        fc = result.forecasts
        if fc:
            sig = next(iter(fc.values()))
            if len(sig) >= h:
                projected = float(np.mean(sig[:h]))
            else:
                projected = float(np.mean(sig))
        else:
            projected = result.risk_score
        scenario_scores.append(projected)

    colours = [get_severity_colour(s) for s in scenario_scores]
    fig = go.Figure(go.Bar(
        x=[f"{h}d" for h in horizons],
        y=scenario_scores,
        marker_color=colours,
        text=[f"{s:.3f}" for s in scenario_scores],
        textposition="outside",
    ))
    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = f"Projected Risk — {region}"
    layout["yaxis"] = dict(range=[0, 1.1], gridcolor=PLOTLY_LAYOUT_DEFAULTS["yaxis"]["gridcolor"])
    layout["height"] = 320
    layout["margin"] = dict(t=50, b=40, l=60, r=20)
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch")


# ── Main render ────────────────────────────────────────────────────────────────

def render_analysis_tools(pipeline) -> None:
    st.markdown("<h2>🔧 Analysis Tools</h2>", unsafe_allow_html=True)

    tool = st.selectbox(
        "Select Tool",
        [
            "🌡️ Heatmap Hotspot Analyzer",
            "🧪 Climate What-If Simulator",
            "⏳ Tipping Point Countdown",
            "📅 Historical Anomaly Calendar",
            "🔗 Signal Correlation Matrix",
            "📈 Risk Score Sparklines",
            "⚖️ Scenario Comparison",
        ],
        key="analysis_tool_select",
    )

    st.divider()

    if tool.startswith("🌡️"):
        _render_heatmap_upload()
    elif tool.startswith("🧪"):
        _render_whatif(pipeline)
    elif tool.startswith("⏳"):
        _render_countdown()
    elif tool.startswith("📅"):
        _render_anomaly_calendar()
    elif tool.startswith("🔗"):
        _render_correlation()
    elif tool.startswith("📈"):
        _render_sparklines()
    elif tool.startswith("⚖️"):
        _render_scenario_comparison()
