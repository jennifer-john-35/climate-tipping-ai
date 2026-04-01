from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from config.theme_config import ACCENT_COLOR, PLOTLY_LAYOUT_DEFAULTS


def render_forecast(
    signal_name: str,
    historical: np.ndarray,
    forecast_mean: np.ndarray,
    forecast_lower: np.ndarray,
    forecast_upper: np.ndarray,
    horizon: int,
) -> go.Figure:
    """Line chart: historical (solid), forecast mean (dashed), CI band (filled area).

    Vertical dashed line at history/forecast boundary.
    """
    n_hist = len(historical)
    x_hist = list(range(n_hist))
    x_fore = list(range(n_hist, n_hist + horizon))

    fig = go.Figure()

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=x_hist,
            y=historical,
            mode="lines",
            name="Historical",
            line=dict(color=ACCENT_COLOR, width=2),
        )
    )

    # CI band (filled area)
    fig.add_trace(
        go.Scatter(
            x=x_fore + x_fore[::-1],
            y=list(forecast_upper) + list(forecast_lower[::-1]),
            fill="toself",
            fillcolor="rgba(0,212,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="90% CI",
            showlegend=True,
        )
    )

    # Forecast mean line
    fig.add_trace(
        go.Scatter(
            x=x_fore,
            y=forecast_mean,
            mode="lines",
            name="Forecast",
            line=dict(color="white", width=2, dash="dash"),
        )
    )

    # Vertical boundary line
    fig.add_vline(
        x=n_hist,
        line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        annotation_text="Forecast →",
        annotation_position="top right",
    )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = f"{signal_name} Forecast"
    layout["xaxis_title"] = "Time Step"
    layout["yaxis_title"] = signal_name
    layout["margin"] = dict(t=50, b=40, l=60, r=20)
    layout["legend"] = dict(orientation="h", y=-0.15)
    fig.update_layout(**layout)
    return fig
