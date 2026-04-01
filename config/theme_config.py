# config/theme_config.py — Colours, CSS, and theme helpers

# ── Severity thresholds: name → (min, max, hex_colour) ──
SEVERITY_THRESHOLDS = {
    "low":     (0.0, 0.4, "#00ff88"),
    "medium":  (0.4, 0.6, "#ffd700"),
    "high":    (0.6, 0.8, "#ff8c00"),
    "extreme": (0.8, 1.0, "#ff2244"),
}

# ── Colour constants ──
BACKGROUND_COLOR = "#0d1117"
CARD_BG_COLOR    = "rgba(255,255,255,0.05)"
ACCENT_COLOR     = "#00d4ff"
TEXT_COLOR       = "#e6edf3"
GRID_COLOR       = "rgba(255,255,255,0.08)"
PLOTLY_TEMPLATE  = "plotly_dark"

# ── Plotly layout defaults ──
PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BACKGROUND_COLOR,
    plot_bgcolor="rgba(13,17,23,0.8)",
    font=dict(color=TEXT_COLOR, family="'Courier New', monospace"),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
)

# ── Global CSS ──
GLOBAL_CSS = """
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117 !important;
    color: #e6edf3;
    font-family: 'Courier New', monospace;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,212,255,0.2);
}
/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
/* ── Severity badges ── */
.badge-low      { color: #00ff88; border: 1px solid #00ff88; }
.badge-medium   { color: #ffd700; border: 1px solid #ffd700; }
.badge-high     { color: #ff8c00; border: 1px solid #ff8c00; }
.badge-extreme  { color: #ff2244; border: 1px solid #ff2244; }
.severity-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: bold;
}
/* ── Alert banner ── */
.alert-banner {
    background: rgba(255,34,68,0.15);
    border: 1px solid #ff2244;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
/* ── Scan button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00d4ff22, #0066ff22);
    border: 1px solid #00d4ff;
    color: #00d4ff;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.1em;
    transition: all 0.2s;
}
[data-testid="stButton"] > button:hover {
    background: rgba(0,212,255,0.2);
    box-shadow: 0 0 20px rgba(0,212,255,0.4);
}
/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00d4ff, #0066ff);
}
/* ── Sidebar nav buttons ── */
.nav-button {
    width: 100%; text-align: left; padding: 0.6rem 1rem;
    background: transparent; border: none; color: #e6edf3;
    font-family: 'Courier New', monospace; cursor: pointer;
    border-left: 3px solid transparent; transition: all 0.15s;
}
.nav-button.active {
    border-left-color: #00d4ff; color: #00d4ff;
    background: rgba(0,212,255,0.08);
}
</style>
"""


def get_severity_colour(score: float) -> str:
    """Map a risk score in [0, 1] to a severity hex colour.

    [0.0, 0.4)  → green  #00ff88
    [0.4, 0.6)  → yellow #ffd700
    [0.6, 0.8)  → orange #ff8c00
    [0.8, 1.0]  → red    #ff2244
    """
    for _name, (low, high, colour) in SEVERITY_THRESHOLDS.items():
        if low <= score < high:
            return colour
    # score == 1.0 falls through the last half-open interval; return extreme colour
    return SEVERITY_THRESHOLDS["extreme"][2]


def get_marker_size(score: float) -> float:
    """Linearly map a risk score in [0, 1] to a marker size in [4, 20], clamped."""
    clamped = max(0.0, min(1.0, score))
    return 4.0 + clamped * 16.0
