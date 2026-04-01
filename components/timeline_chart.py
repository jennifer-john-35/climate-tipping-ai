from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import zscore

from config.theme_config import PLOTLY_LAYOUT_DEFAULTS


_SIGNAL_COLOURS = [
    "#00d4ff", "#00ff88", "#ffd700", "#ff8c00", "#c084fc",
    "#f472b6", "#34d399", "#fb923c",
]


def render_timeline(
    climate_df: pd.DataFrame,
    anomaly_flags: list[bool],
    rolling_risk: np.ndarray,
    signal_names: list[str],
) -> go.Figure:
    """Multi-signal timeline with anomaly shading and rolling risk on secondary y-axis."""
    fig = go.Figure()

    x = list(range(len(climate_df)))

    # Z-score normalise each signal and add as a line trace
    for idx, col in enumerate(signal_names):
        if col not in climate_df.columns:
            continue
        series = climate_df[col].values.astype(float)
        if series.std() > 0:
            normalised = zscore(series)
        else:
            normalised = series - series.mean()

        colour = _SIGNAL_COLOURS[idx % len(_SIGNAL_COLOURS)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=normalised,
                mode="lines",
                name=col,
                line=dict(color=colour, width=1.5),
                yaxis="y1",
            )
        )

    # Anomaly shaded bands
    in_band = False
    band_start = None
    flags = list(anomaly_flags) + [False]  # sentinel to close last band
    for i, flag in enumerate(flags):
        if flag and not in_band:
            band_start = i
            in_band = True
        elif not flag and in_band:
            fig.add_vrect(
                x0=band_start,
                x1=i,
                fillcolor="rgba(255,34,68,0.15)",
                line_width=0,
                layer="below",
            )
            in_band = False

    # Rolling risk on secondary y-axis
    if rolling_risk is not None and len(rolling_risk) > 0:
        risk_x = list(range(len(rolling_risk)))
        fig.add_trace(
            go.Scatter(
                x=risk_x,
                y=rolling_risk,
                mode="lines",
                name="Rolling Risk",
                line=dict(color="#ff8c00", width=2, dash="dash"),
                yaxis="y2",
            )
        )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = "Climate Signal Timeline"
    layout["xaxis_title"] = "Time Step"
    layout["yaxis"] = dict(
        title="Z-Score",
        gridcolor=PLOTLY_LAYOUT_DEFAULTS["yaxis"]["gridcolor"],
        zerolinecolor=PLOTLY_LAYOUT_DEFAULTS["yaxis"]["zerolinecolor"],
    )
    layout["yaxis2"] = dict(
        title="Rolling Risk",
        overlaying="y",
        side="right",
        range=[0, 1],
        gridcolor="rgba(0,0,0,0)",
        showgrid=False,
    )
    layout["legend"] = dict(orientation="h", y=-0.15)
    layout["margin"] = dict(t=50, b=60, l=60, r=60)
    fig.update_layout(**layout)
    return fig
