import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import numpy as np
from datetime import datetime

st.title("🌍 AI Climate Tipping Point Detection System")

# Load risk dataset
risk_df = pd.read_csv("data/global_risk_scores.csv")

# Ensure numeric risk scores
risk_df["risk_score"] = pd.to_numeric(risk_df["risk_score"], errors="coerce")

st.markdown("""
    <style>

    body {
    background-color:#0a0f1c;
    color:white;
    }

    [data-testid="stSidebar"] {
    background-color:#0a0f1c;
    }

    </style>
    """, unsafe_allow_html=True)
st.set_page_config(
    page_title="AI Climate Tipping Point Detection",
    layout="wide"
)

tab1, tab2, tab3 = st.tabs(["Global Scanner", "Risk Index", "Logs"])


# ---------- TAB 1 GLOBAL MAP ----------
with tab1:

    st.subheader("Global Climate Risk Scanner")

    data = pd.read_csv("data/global_risk_scores.csv")

    # keep only highest risk points
    data = data.sort_values("risk_score", ascending=False).head(15)

    data["location"] = data["name"]
    scan = st.button("Run Global Climate Scan")
    if scan:

        progress = st.progress(0)

        import time

        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)

        st.success("Global climate scan complete. Risk zones detected.")
    fig = px.scatter_geo(
        data,
        lat="latitude",
        lon="longitude",
        hover_name="name",
        color="risk_score",
        size="risk_score",
        projection="natural earth",
        color_continuous_scale="Turbo"
    )

    fig.update_geos(
        showland=True,
        landcolor="rgb(40,120,40)",
        showocean=True,
        oceancolor="rgb(10,20,50)",
        showcountries=True,
        countrycolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Upload Global Climate Heatmap")

    uploaded = st.file_uploader("Upload heatmap image", type=["png", "jpg", "jpeg"])

    if uploaded:

        import tempfile
        from utils.heatmap_analyzer import detect_hotspots

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        hotspots = detect_hotspots(path)

        st.write("Detected Risk Zones:", len(hotspots))

        if hotspots:

            lat = [p[0] for p in hotspots]
            lon = [p[1] for p in hotspots]

            import plotly.express as px

            fig = px.scatter_geo(
                lat=lat,
                lon=lon,
                projection="natural earth"
            )

            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Select Location")

    selected_name = st.selectbox(
        "Select Location",
        risk_df["name"]
    )

    selected_row = risk_df[risk_df["name"] == selected_name].iloc[0]

    st.write("### Location Details")

    st.write("Name:", selected_row["name"])
    st.write("Latitude:", selected_row["latitude"])
    st.write("Longitude:", selected_row["longitude"])
    st.write("Risk Score:", round(selected_row["risk_score"], 3))
    # ---------- Load climate history ----------
    climate = pd.read_csv("data/climate_training_data.csv")

    # clean dataset
    climate = climate.apply(pd.to_numeric, errors="coerce")
    climate = climate.dropna()
    climate.reset_index(drop=True, inplace=True)

    # ---------- Variance Graph ----------
    st.subheader("Climate Variance")

    variance = climate.var()

    var_df = variance.reset_index()
    var_df.columns = ["feature", "variance"]

    fig_var = px.bar(
        var_df,
        x="feature",
        y="variance",
        title="Climate Signal Variance"
    )

    st.plotly_chart(fig_var)

    # ---------- Historical Trend ----------
    st.subheader("Climate History")

    fig_hist = px.line(
        climate,
        title="Climate Signals Over Time"
    )

    st.plotly_chart(fig_hist)

    # ---------- 30 Day Projection ----------
    st.subheader("30 Day Projection")

    last = climate.tail(30)

    future = last.mean() * 1.02

    proj_df = pd.DataFrame({
        "day": range(1, 31),
        "projection": [future.mean()]*30
    })

    fig_proj = px.line(
        proj_df,
        x="day",
        y="projection",
        title="Projected Climate Trend"
    )

    st.plotly_chart(fig_proj)

    # ---------- Risk Explanation ----------
    st.subheader("AI Risk Explanation")

    risk = selected_row["risk_score"]

    if risk > 0.8:
        st.error("Extreme climate instability detected. Possible tipping point approaching.")

    elif risk > 0.6:
        st.warning("High climate variance detected. Monitoring recommended.")

    else:
        st.success("Low climate tipping risk currently.")
    temp_var = climate["temp"].var()
    co2_var = climate["co2"].var()
    ice_var = climate["sea_ice"].var()

    reasons = []

    if temp_var > 0.5:
        reasons.append("High temperature anomaly")

    if co2_var > 0.2:
        reasons.append("Rapid CO₂ increase")

    if ice_var > 0.3:
        reasons.append("Sea ice loss detected")

    st.write("### AI Risk Explanation")

    for r in reasons:
        st.write("•", r)
    # ---------- Logging ----------

        log = pd.DataFrame([{
            "timestamp": datetime.now(),
            "latitude": selected_row["latitude"],
            "longitude": selected_row["longitude"]
        }])

    log.to_csv("logs/click_log.csv", mode="a", header=False, index=False)


# ---------- TAB 2 RISK INDEX ----------
with tab2:

    st.subheader("Global Climate Risk Ranking")

    data = pd.read_csv("data/global_risk_scores.csv")
    data = data.sort_values("risk_score", ascending=False).head(15)

    ranked = data.sort_values("risk_score", ascending=False)

    ranked["rank"] = range(1, len(ranked) + 1)

    st.dataframe(
        ranked[["rank", "name", "latitude", "longitude", "risk_score"]].head(50),
        use_container_width=True
    )

    fig = px.histogram(ranked, x="risk_score")

    st.plotly_chart(fig)


# ---------- TAB 3 LOG SYSTEM ----------
with tab3:

    st.subheader("Interaction Logs")

    log_file = "logs/click_log.csv"

    try:

        logs = pd.read_csv("logs/click_log.csv")

        logs["timestamp"] = pd.to_datetime(logs["timestamp"])

        st.dataframe(logs, use_container_width=True)

    except:

        st.info("No interactions recorded yet.")
