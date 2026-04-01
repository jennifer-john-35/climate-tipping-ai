from __future__ import annotations

import plotly.graph_objects as go

from config.theme_config import PLOTLY_LAYOUT_DEFAULTS


def render_shap_bar(shap_values: dict[str, float], base_value: float) -> go.Figure:
    """Horizontal bar chart sorted by |shap_value| descending.

    Red (#ff2244) for positive SHAP values (increase risk).
    Blue (#0066ff) for negative SHAP values (decrease risk).
    Vertical reference line at base_value.
    """
    sorted_items = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colours = ["#ff2244" if v > 0 else "#0066ff" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=base_value,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text=f"base={base_value:.3f}",
        annotation_position="top right",
    )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = "SHAP Feature Contributions"
    layout["xaxis_title"] = "SHAP Value"
    layout["margin"] = dict(t=40, b=40, l=140, r=20)
    fig.update_layout(**layout)
    return fig


def render_shap_waterfall(
    shap_values: dict[str, float],
    base_value: float,
    final_score: float,
) -> go.Figure:
    """Waterfall chart showing cumulative shift from base_value to final_score."""
    sorted_items = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    measures = ["relative"] * len(features) + ["total"]
    x_labels = features + ["Final Score"]
    y_values = values + [final_score]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_values,
            base=base_value,
            connector={"line": {"color": "rgba(255,255,255,0.3)"}},
            increasing={"marker": {"color": "#ff2244"}},
            decreasing={"marker": {"color": "#0066ff"}},
            totals={"marker": {"color": "#00d4ff"}},
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = "SHAP Waterfall"
    layout["yaxis_title"] = "Risk Score"
    layout["margin"] = dict(t=40, b=60, l=60, r=20)
    fig.update_layout(**layout)
    return fig
