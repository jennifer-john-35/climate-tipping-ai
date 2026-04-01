from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from scipy.stats import zscore

from config.theme_config import PLOTLY_LAYOUT_DEFAULTS, get_severity_colour


_PALETTE = [
    "#00d4ff", "#00ff88", "#ffd700", "#ff8c00", "#c084fc",
    "#f472b6", "#34d399", "#fb923c", "#60a5fa", "#a78bfa",
]


def render_comparison(regions_data: dict[str, pd.DataFrame], signal_name: str) -> go.Figure:
    """Overlaid z-score normalised line chart, one line per region."""
    fig = go.Figure()

    for idx, (region, df) in enumerate(regions_data.items()):
        if signal_name not in df.columns:
            continue
        series = df[signal_name].values.astype(float)
        if series.std() > 0:
            normalised = zscore(series)
        else:
            normalised = series - series.mean()

        colour = _PALETTE[idx % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(normalised))),
                y=normalised,
                mode="lines",
                name=region,
                line=dict(color=colour, width=2),
            )
        )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = f"{signal_name} Comparison (z-score)"
    layout["xaxis_title"] = "Time Step"
    layout["yaxis_title"] = "Z-Score"
    layout["legend"] = dict(orientation="h", y=-0.15)
    layout["margin"] = dict(t=50, b=60, l=60, r=20)
    fig.update_layout(**layout)
    return fig


def render_risk_bar(regions_scores: dict[str, float]) -> go.Figure:
    """Grouped bar chart of risk scores per region, coloured by severity."""
    regions = list(regions_scores.keys())
    scores = list(regions_scores.values())
    colours = [get_severity_colour(s) for s in scores]

    fig = go.Figure(
        go.Bar(
            x=regions,
            y=scores,
            marker_color=colours,
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = "Regional Risk Scores"
    layout["xaxis_title"] = "Region"
    layout["yaxis_title"] = "Risk Score"
    layout["yaxis"] = dict(
        range=[0, 1],
        gridcolor=PLOTLY_LAYOUT_DEFAULTS["yaxis"]["gridcolor"],
        zerolinecolor=PLOTLY_LAYOUT_DEFAULTS["yaxis"]["zerolinecolor"],
    )
    layout["margin"] = dict(t=50, b=80, l=60, r=20)
    fig.update_layout(**layout)
    return fig
