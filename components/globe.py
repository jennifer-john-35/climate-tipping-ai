from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config.theme_config import PLOTLY_LAYOUT_DEFAULTS, get_marker_size, get_severity_colour


def render_globe(
    risk_df: pd.DataFrame,
    projection: str = "orthographic",
    animate_rotation: bool = True,
) -> go.Figure:
    """Return a Plotly scatter_geo figure styled as a dark animated globe.

    risk_df columns: latitude, longitude, risk_score, name (or subset)
    Only the top 20 highest-risk named locations are shown.
    """
    # Keep only named, unique locations — top 20 by risk score
    df = risk_df.copy()
    if "name" not in df.columns:
        df["name"] = df.apply(lambda r: f"{r['latitude']:.1f},{r['longitude']:.1f}", axis=1)
    df = df.dropna(subset=["risk_score"])
    df = df.drop_duplicates(subset=["name"])
    df = df.sort_values("risk_score", ascending=False).head(20).reset_index(drop=True)
    marker_sizes = df["risk_score"].apply(get_marker_size).tolist()
    marker_colours = df["risk_score"].apply(get_severity_colour).tolist()
    tooltips = [
        f"name: {row['name']}\nRisk: {row['risk_score']:.3f}"
        for _, row in df.iterrows()
    ]

    scatter = go.Scattergeo(
        lat=df["latitude"],
        lon=df["longitude"],
        text=tooltips,
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=marker_colours,
            opacity=0.85,
            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
        ),
    )

    geo_layout = dict(
        projection_type=projection,
        bgcolor="#0d1117",
        landcolor="#1a2f1a",
        oceancolor="#0a1628",
        showocean=True,
        showland=True,
        showcountries=True,
        countrycolor="rgba(255,255,255,0.3)",
        showframe=False,
        coastlinecolor="rgba(255,255,255,0.2)",
    )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["geo"] = geo_layout
    layout["margin"] = dict(t=10, b=10, l=10, r=10)
    layout["height"] = 500

    fig = go.Figure(data=[scatter])

    if animate_rotation and projection == "orthographic":
        frames = []
        for i, lon_center in enumerate(range(0, 360, 10)):
            frames.append(
                go.Frame(
                    layout=dict(geo=dict(projection_rotation_lon=lon_center)),
                    name=str(i),
                )
            )
        fig.frames = frames

        layout["updatemenus"] = [
            dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Rotate",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ]

    fig.update_layout(**layout)
    return fig
