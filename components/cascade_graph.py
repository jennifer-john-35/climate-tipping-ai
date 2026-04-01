from __future__ import annotations

import math

import plotly.graph_objects as go

from config.theme_config import PLOTLY_LAYOUT_DEFAULTS

_NODE_COLOURS = {
    "inactive":  "#334155",
    "at_risk":   "#ffd700",
    "activated": "#ff2244",
}
_NODE_SIZE = 30


def render_cascade_graph(
    topology: dict,
    node_states: dict[str, str],
    selected_node: str | None = None,
) -> go.Figure:
    """Plotly scatter + lines network graph for the cascade simulator."""
    nodes = topology.get("nodes", [])
    edges = topology.get("edges", [])

    # Layout nodes in a circle
    n = len(nodes)
    positions: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        positions[node["id"]] = (math.cos(angle), math.sin(angle))

    fig = go.Figure()

    # Draw edges as lines
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src not in positions or tgt not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[tgt]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        weight = edge.get("weight", 0.0)

        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.25)", width=max(1, weight * 3)),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Edge weight annotation
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=f"{weight:.2f}",
            showarrow=False,
            font=dict(size=9, color="rgba(255,255,255,0.5)"),
        )

    # Draw nodes
    node_x, node_y, node_text, node_colours = [], [], [], []
    marker_line_widths, marker_line_colours = [], []

    for node in nodes:
        nid = node["id"]
        x, y = positions[nid]
        state = node_states.get(nid, "inactive")
        colour = _NODE_COLOURS.get(state, _NODE_COLOURS["inactive"])

        node_x.append(x)
        node_y.append(y)
        node_text.append(node.get("label", nid))
        node_colours.append(colour)

        if nid == selected_node:
            marker_line_widths.append(4)
            marker_line_colours.append("#00d4ff")
        else:
            marker_line_widths.append(1)
            marker_line_colours.append("rgba(255,255,255,0.2)")

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=_NODE_SIZE,
                color=node_colours,
                line=dict(width=marker_line_widths, color=marker_line_colours),
            ),
            showlegend=False,
        )
    )

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = "Tipping Cascade Network"
    layout["xaxis"] = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        gridcolor="rgba(0,0,0,0)",
    )
    layout["yaxis"] = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        gridcolor="rgba(0,0,0,0)",
    )
    layout["margin"] = dict(t=50, b=20, l=20, r=20)
    layout["height"] = 520
    fig.update_layout(**layout)
    return fig
