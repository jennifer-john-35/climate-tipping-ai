from __future__ import annotations

import plotly.graph_objects as go

from config.theme_config import SEVERITY_THRESHOLDS, PLOTLY_LAYOUT_DEFAULTS


def render_gauge(risk_score: float, region_name: str, width: int = 300) -> go.Figure:
    """Return a Plotly indicator (gauge) figure for the given risk score."""
    _, _, green  = SEVERITY_THRESHOLDS["low"]
    _, _, yellow = SEVERITY_THRESHOLDS["medium"]
    _, _, orange = SEVERITY_THRESHOLDS["high"]
    _, _, red    = SEVERITY_THRESHOLDS["extreme"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            number={"valueformat": ".3f", "font": {"size": 28}},
            delta={"reference": 0.5, "valueformat": ".3f"},
            title={"text": region_name, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": "#00d4ff", "thickness": 0.25},
                "steps": [
                    {"range": [0.0, 0.4], "color": green},
                    {"range": [0.4, 0.6], "color": yellow},
                    {"range": [0.6, 0.8], "color": orange},
                    {"range": [0.8, 1.0], "color": red},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": risk_score,
                },
            },
        )
    )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["width"] = width
    layout["height"] = 250
    layout["margin"] = dict(t=40, b=10, l=20, r=20)
    fig.update_layout(**layout)
    return fig
