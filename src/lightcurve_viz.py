"""Light-curve visualization helpers for the EXOAI dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .detrending import detrend
from .phase_folding import phase_fold
from .sigma_clipping import sigma_clip

_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0", family="Rajdhani"),
    margin=dict(l=48, r=24, t=48, b=40),
)


def preprocess_lightcurve(lc: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw, detrended, and cleaned time/flux arrays."""
    n_original = len(lc)
    lc = lc.remove_nans()
    n_after_nan = len(lc)
    lc = lc.normalize()
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)
    flux_detrended = detrend(flux)
    time_clean, flux_clean = sigma_clip(time, flux_detrended)
    return time, flux, flux_detrended, time_clean, flux_clean


def build_bls_periodogram_figure(periods: List[float], powers: List[float]) -> go.Figure:
    """BLS periodogram for transit detection."""
    fig = go.Figure(
        go.Scatter(
            x=periods,
            y=powers,
            mode="lines",
            line=dict(color="#00D4FF", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.15)",
            name="BLS Power",
        )
    )
    fig.update_layout(
        title="BLS Periodogram",
        xaxis_title="Period (days)",
        yaxis_title="Detection Power",
        xaxis=dict(gridcolor="rgba(0,212,255,0.12)"),
        yaxis=dict(gridcolor="rgba(0,212,255,0.12)"),
        **_DARK,
    )
    return fig


def build_transit_highlight_figure(
    time: np.ndarray,
    flux: np.ndarray,
    epoch: float,
    period: float,
    duration: float,
) -> go.Figure:
    """Light curve with detected transit epochs highlighted."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time, y=flux, mode="lines", line=dict(color="#22D3EE", width=1), name="Flux")
    )
    if period > 0 and not np.isnan(epoch):
        t_min, t_max = float(time.min()), float(time.max())
        transit_times = []
        t = epoch
        while t >= t_min:
            transit_times.append(t)
            t -= period
        t = epoch + period
        while t <= t_max:
            transit_times.append(t)
            t += period
        half_d = duration / 2.0
        for tc in transit_times:
            fig.add_vrect(
                x0=tc - half_d,
                x1=tc + half_d,
                fillcolor="rgba(236,72,153,0.25)",
                line_width=0,
                layer="below",
            )
    fig.update_layout(
        title="Detected Transit Events",
        xaxis_title="Time (days)",
        yaxis_title="Normalized Flux",
        **_DARK,
    )
    return fig


def build_phase_folded_figure(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float = 0.1,
) -> go.Figure:
    """Phase-folded curve with transit region highlighted."""
    phase, folded = phase_fold(time, flux, period)
    fig = go.Figure(
        go.Scatter(
            x=phase,
            y=folded,
            mode="markers",
            marker=dict(color="#8B5CF6", size=3, opacity=0.75),
            name="Folded",
        )
    )
    half_width = (duration / period) / 2.0 if period > 0 else 0.05
    fig.add_vrect(
        x0=0.5 - half_width,
        x1=0.5 + half_width,
        fillcolor="rgba(0,255,170,0.2)",
        line_width=0,
        layer="below",
    )
    fig.update_layout(
        title=f"Phase-Folded Transit (P={period:.4f} d)",
        xaxis_title="Phase",
        yaxis_title="Flux",
        **_DARK,
    )
    return fig


def build_overview_figure(
    time: np.ndarray,
    flux: np.ndarray,
    flux_detrended: np.ndarray,
    time_clean: np.ndarray,
    flux_clean: np.ndarray,
) -> go.Figure:
    """3-panel: raw, detrended, cleaned."""
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Raw Light Curve", "Detrended Light Curve", "Cleaned Light Curve"),
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(x=time, y=flux, mode="lines", line=dict(color="#22D3EE", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=time, y=flux_detrended, mode="lines", line=dict(color="#FFC857", width=1)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_clean, y=flux_clean, mode="lines", line=dict(color="#00FFAA", width=1)),
        row=3, col=1,
    )
    fig.update_layout(
        height=680,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        showlegend=False,
    )
    return fig


def build_all_figures(viz: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Build all publication-quality figures from stored viz payload."""
    time = np.array(viz["time_raw"])
    flux = np.array(viz["flux_raw"])
    detrended = np.array(viz.get("flux_detrended", viz["flux_clean"]))
    time_clean = np.array(viz["time_clean"])
    flux_clean = np.array(viz["flux_clean"])
    period = viz.get("period")
    duration = viz.get("duration", 0.1)
    epoch = viz.get("transit_epoch", 0.0)

    figs: Dict[str, go.Figure] = {
        "overview": build_overview_figure(time, flux, detrended, time_clean, flux_clean),
    }

    if viz.get("bls_periods") and viz.get("bls_powers"):
        figs["bls"] = build_bls_periodogram_figure(viz["bls_periods"], viz["bls_powers"])

    if period and period > 0:
        figs["transit"] = build_transit_highlight_figure(time_clean, flux_clean, epoch, period, duration)
        figs["folded"] = build_phase_folded_figure(time_clean, flux_clean, period, duration)

    return figs
