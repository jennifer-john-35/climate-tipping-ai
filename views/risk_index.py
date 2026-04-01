from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components import render_risk_bar, render_comparison
from config.theme_config import PLOTLY_TEMPLATE, get_severity_colour


def _clean_climate(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def render_risk_index() -> None:
    st.markdown("<h2>📊 Global Risk Index</h2>", unsafe_allow_html=True)

    try:
        risk_df = pd.read_csv("data/global_risk_scores.csv")
    except Exception as exc:
        st.error(f"Failed to load risk scores: {exc}")
        return

    risk_df = risk_df.drop_duplicates(subset=["name"])
    risk_df = risk_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    risk_df.insert(0, "rank", range(1, len(risk_df) + 1))
    risk_df["severity"] = risk_df["risk_score"].apply(get_severity_colour)

    st.dataframe(risk_df, width="stretch")

    fig_hist = px.histogram(
        risk_df, x="risk_score", nbins=20,
        title="Risk Score Distribution",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#00d4ff"],
    )
    fig_hist.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="rgba(13,17,23,0.8)",
        font=dict(color="#e6edf3"),
    )
    st.plotly_chart(fig_hist, width="stretch")

    st.divider()
    st.markdown("#### Region Comparison")

    all_regions = risk_df["name"].tolist()
    selected = st.multiselect(
        "Select 2–5 regions to compare",
        options=all_regions,
        default=st.session_state.get("comparison_regions", []),
        key="comparison_regions",
    )

    if len(selected) < 2:
        st.info("Select at least 2 regions to compare.")
        return
    if len(selected) > 5:
        st.warning("Please select no more than 5 regions.")
        return

    scores_map = {
        row["name"]: row["risk_score"]
        for _, row in risk_df[risk_df["name"].isin(selected)].iterrows()
    }
    st.plotly_chart(render_risk_bar(scores_map), width="stretch")

    try:
        climate_df = _clean_climate(pd.read_csv("data/climate_training_data.csv"))
    except Exception as exc:
        st.error(f"Failed to load climate data: {exc}")
        return

    for signal in ["temp", "co2", "sea_ice", "ocean_heat"]:
        if signal not in climate_df.columns:
            continue
        regions_data = {region: climate_df for region in selected}
        st.plotly_chart(render_comparison(regions_data, signal), width="stretch")
