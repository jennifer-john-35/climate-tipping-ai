from __future__ import annotations


def render_glass_card(title: str, content: str, severity: str | None = None) -> str:
    """Return HTML string for a glassmorphism card.

    severity: 'low' | 'medium' | 'high' | 'extreme' | None
    """
    badge_html = ""
    if severity is not None:
        badge_html = (
            f'<span class="severity-badge badge-{severity}" '
            f'style="position:absolute;top:1rem;right:1.5rem;">'
            f"{severity.upper()}</span>"
        )

    return (
        f'<div class="glass-card" style="position:relative;">'
        f"{badge_html}"
        f"<h4 style=\"margin:0 0 0.6rem 0;color:#00d4ff;font-family:'Courier New',monospace;\">"
        f"{title}</h4>"
        f"<div>{content}</div>"
        f"</div>"
    )
