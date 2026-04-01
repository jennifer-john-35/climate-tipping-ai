# app.py — Advanced Climate Tipping Point Detection System
# Entry point: sidebar navigation, session state init, view routing

import streamlit as st

# MUST be the very first Streamlit call
st.set_page_config(
    page_title="AI Climate Tipping Point Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.theme_config import GLOBAL_CSS, ACCENT_COLOR
from inference import ModelLoadError
from inference.pipeline import InferencePipeline
from views import (
    render_global_scanner,
    render_risk_index,
    render_cascade_simulator,
    render_logs,
    render_analysis_tools,
)

# ── Inject global CSS ──────────────────────────────────────────────────────────
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
SESSION_DEFAULTS = {
    "tipping_model": None,
    "forecast_model": None,
    "forecast_scaler": None,
    "scan_results": {},
    "grid_scores": None,
    "alert_threshold": 0.75,
    "last_scan_time": None,
    "active_view": "Global Scanner",
    "scan_in_progress": False,
    "selected_region": None,
    "comparison_regions": [],
    "projection_horizon": 30,
}
for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f"""
    <div style="padding: 1rem 0.5rem 0.5rem 0.5rem;">
        <div style="color: {ACCENT_COLOR}; font-size: 1.15rem; font-weight: bold;
                    font-family: 'Courier New', monospace; letter-spacing: 0.08em;">
            🌍 CLIMATE TIPPING AI
        </div>
        <div style="color: #8b949e; font-size: 0.78rem; font-family: 'Courier New', monospace;
                    margin-top: 0.2rem;">
            Advanced Detection System
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

VIEWS = [
    "🌍 Global Scanner",
    "📊 Risk Index",
    "🌊 Cascade Simulator",
    "🔧 Analysis Tools",
    "📋 Logs",
]

active_view = st.session_state["active_view"]

for view in VIEWS:
    # Strip emoji prefix to get the bare view name stored in session state
    view_name = view.split(" ", 1)[1]
    is_active = active_view == view_name

    # Highlight active nav item
    if is_active:
        st.sidebar.markdown(
            f"""
            <div style="border-left: 3px solid {ACCENT_COLOR};
                        background: rgba(0,212,255,0.08);
                        padding: 0.5rem 1rem;
                        color: {ACCENT_COLOR};
                        font-family: 'Courier New', monospace;
                        font-size: 0.9rem;
                        margin-bottom: 2px;">
                {view}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if st.sidebar.button(view, key=f"nav_{view_name}", use_container_width=True):
            st.session_state["active_view"] = view_name
            st.rerun()

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Session", use_container_width=True):
    for key in ("scan_results", "grid_scores", "last_scan_time",
                "tipping_model", "forecast_model", "forecast_scaler"):
        st.session_state[key] = SESSION_DEFAULTS[key]
    st.rerun()

# ── Pipeline (model loading) ───────────────────────────────────────────────────
try:
    pipeline = InferencePipeline(st.session_state)
    pipeline.load_models()
except ModelLoadError as exc:
    st.sidebar.error(f"⚠️ Model load failed: {exc}")
    pipeline = InferencePipeline(st.session_state)

# ── View routing ───────────────────────────────────────────────────────────────
active_view = st.session_state["active_view"]

if active_view == "Global Scanner":
    render_global_scanner(pipeline)
elif active_view == "Risk Index":
    render_risk_index()
elif active_view == "Cascade Simulator":
    render_cascade_simulator()
elif active_view == "Analysis Tools":
    render_analysis_tools(pipeline)
elif active_view == "Logs":
    render_logs()
