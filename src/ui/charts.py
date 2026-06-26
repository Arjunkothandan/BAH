"""Plotly chart styling and scientific visualizations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go


DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani, sans-serif", color="#E2E8F0", size=12),
    title_font=dict(family="Orbitron, sans-serif", color="#00D4FF", size=14),
    margin=dict(l=48, r=24, t=48, b=48),
    xaxis=dict(
        gridcolor="rgba(0, 212, 255, 0.12)",
        zerolinecolor="rgba(0, 212, 255, 0.25)",
        tickfont=dict(color="#94A3B8"),
    ),
    yaxis=dict(
        gridcolor="rgba(0, 212, 255, 0.12)",
        zerolinecolor="rgba(0, 212, 255, 0.25)",
        tickfont=dict(color="#94A3B8"),
    ),
    hoverlabel=dict(
        bgcolor="rgba(15, 23, 42, 0.95)",
        bordercolor="#00D4FF",
        font=dict(color="#E2E8F0"),
    ),
)


def style_plotly_figure(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the futuristic dark neon theme to any Plotly figure."""
    fig.update_layout(**DARK_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    fig.update_layout(transition=dict(duration=800, easing="cubic-in-out"))
    return fig


def create_feature_ranking_chart(
    shap_dict: dict[str, float],
    top_n: int = 10,
) -> go.Figure:
    """Horizontal bar chart for feature importance ranking."""
    sorted_items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#00FFAA" if v >= 0 else "#FF5A7A" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(0, 212, 255, 0.3)", width=1),
            ),
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="SHAP Value",
        height=max(320, top_n * 36),
    )
    return style_plotly_figure(fig, "Feature Importance Ranking")


def create_orbital_viz(
    period: float | None = None,
    depth: float | None = None,
) -> go.Figure:
    """Interactive solar-system style orbital visualization."""
    period = period or 12.5
    depth = depth or 0.02

    t = np.linspace(0, 2 * math.pi, 200)
    orbit_r = 1.0
    ox = orbit_r * np.cos(t)
    oy = orbit_r * np.sin(t)

    planet_angle = math.pi * 0.35
    px, py = orbit_r * math.cos(planet_angle), orbit_r * math.sin(planet_angle)

    star_size = 30 + depth * 500

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ox, y=oy,
            mode="lines",
            line=dict(color="rgba(0, 212, 255, 0.25)", width=1, dash="dot"),
            name="Orbit Path",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(
                size=star_size,
                color="#FFC857",
                line=dict(color="rgba(255, 200, 87, 0.5)", width=2),
                symbol="circle",
            ),
            name="Host Star",
            hovertemplate="Host Star<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[px], y=[py],
            mode="markers",
            marker=dict(
                size=18,
                color="#22D3EE",
                line=dict(color="#00D4FF", width=2),
                symbol="circle",
            ),
            name="Candidate Planet",
            hovertemplate=f"Candidate<br>Period: {period:.2f} d<extra></extra>",
        )
    )

    # Glow trail
    trail_t = np.linspace(planet_angle - 0.5, planet_angle, 20)
    fig.add_trace(
        go.Scatter(
            x=orbit_r * np.cos(trail_t),
            y=orbit_r * np.sin(trail_t),
            mode="lines",
            line=dict(color="rgba(34, 211, 238, 0.4)", width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(0, 212, 255, 0.3)",
            font=dict(color="#E2E8F0"),
        ),
        height=380,
    )
    return style_plotly_figure(fig, f"Orbital Model · P = {period:.2f} days")


def style_shap_waterfall(fig: go.Figure) -> go.Figure:
    """Restyle the backend SHAP waterfall for the HUD theme."""
    fig.update_traces(
        connector=dict(line=dict(color="rgba(0, 212, 255, 0.3)")),
        increasing=dict(marker=dict(color="#00FFAA")),
        decreasing=dict(marker=dict(color="#FF5A7A")),
        totals=dict(marker=dict(color="#8B5CF6")),
    )
    return style_plotly_figure(fig, "SHAP Explainability Waterfall")
