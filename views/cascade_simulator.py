from __future__ import annotations

import json

import streamlit as st

from components import render_cascade_graph


def render_cascade_simulator() -> None:
    """Render the Tipping Cascade Simulator page."""
    st.markdown("<h2>🌊 Tipping Cascade Simulator</h2>", unsafe_allow_html=True)

    # ── Load topology ──────────────────────────────────────────────────────
    try:
        with open("config/cascade_config.json") as f:
            topology = json.load(f)
    except Exception as exc:
        st.error(f"Failed to load cascade config: {exc}")
        return

    nodes = topology.get("nodes", [])

    # ── Derive node states from scan results ───────────────────────────────
    scan_results = st.session_state.get("scan_results", {})
    node_states: dict[str, str] = {}

    if not scan_results:
        for node in nodes:
            node_states[node["id"]] = "inactive"
        st.info("Run a scan to activate live risk propagation.")
    else:
        # Use the latest scan result (highest risk_score among all results)
        latest_result = max(scan_results.values(), key=lambda r: r.risk_score)
        risk_score = latest_result.risk_score

        for node in nodes:
            if risk_score > 0.7:
                node_states[node["id"]] = "activated"
            elif risk_score > 0.5:
                node_states[node["id"]] = "at_risk"
            else:
                node_states[node["id"]] = "inactive"

    # ── Node selector ──────────────────────────────────────────────────────
    node_ids = [node["id"] for node in nodes]
    selected_node = st.selectbox(
        "Select node for details",
        node_ids,
        key="cascade_selected_node",
    )

    # ── Cascade graph ──────────────────────────────────────────────────────
    fig = render_cascade_graph(topology, node_states, selected_node)
    st.plotly_chart(fig, width="stretch")

    # ── Node detail panel ──────────────────────────────────────────────────
    selected_node_data = next((n for n in nodes if n["id"] == selected_node), None)
    if selected_node_data:
        state = node_states.get(selected_node, "inactive")
        state_colours = {
            "inactive": "#334155",
            "at_risk": "#ffd700",
            "activated": "#ff2244",
        }
        badge_colour = state_colours.get(state, "#334155")

        st.markdown(
            f"""
            <div class='glass-card'>
                <h4>{selected_node_data.get('label', selected_node)}</h4>
                <p><b>Signal:</b> {selected_node_data.get('signal', 'N/A')}</p>
                <p><b>Description:</b> Tipping element linked to {selected_node_data.get('signal', 'N/A')} signal.</p>
                <p><b>State:</b> <span style='color:{badge_colour}; font-weight:bold;'>{state.upper()}</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Downstream edges
        edges = topology.get("edges", [])
        downstream = [e for e in edges if e["source"] == selected_node]
        if downstream:
            st.markdown("**Downstream Connections**")
            import pandas as pd
            st.dataframe(
                pd.DataFrame(downstream)[["target", "weight"]].rename(
                    columns={"target": "Target Node", "weight": "Weight"}
                ),
                width="stretch",
            )
        else:
            st.markdown("_No downstream connections._")
