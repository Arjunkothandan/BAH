"""HTML component builders for the futuristic HUD."""

from __future__ import annotations

from typing import Any


def render_status_bar() -> str:
  return """
<div class="exo-nav">
  <div class="exo-nav-brand">EXO<span>AI</span> · MISSION CONTROL</div>
  <div class="exo-status">
    <div class="exo-status-dot"></div>
    SYSTEM ONLINE · DEEP SPACE LINK ACTIVE
  </div>
</div>
"""


def render_hero(
    accuracy: float = 97.4,
    planets_processed: int = 5842,
    targets_analyzed: int = 12847,
) -> str:
    return f"""
<div class="exo-hero">
  <div class="exo-hero-text">
    <h1>AI EXOPLANET<br>DETECTION PLATFORM</h1>
    <div class="exo-hero-tagline">
      <em>Detect.</em> &nbsp; <em>Analyze.</em> &nbsp; <em>Explain.</em>
    </div>
    <div class="exo-hero-stats">
      <div class="exo-stat">
        <div class="exo-stat-value">{accuracy:.1f}%</div>
        <div class="exo-stat-label">Model Accuracy</div>
      </div>
      <div class="exo-stat">
        <div class="exo-stat-value">{planets_processed:,}</div>
        <div class="exo-stat-label">Planets Processed</div>
      </div>
      <div class="exo-stat">
        <div class="exo-stat-value">{targets_analyzed:,}</div>
        <div class="exo-stat-label">Stellar Targets</div>
      </div>
    </div>
  </div>
  <div class="exo-planet-wrap">
    <div class="exo-planet-orbit"></div>
    <div class="exo-planet"></div>
  </div>
</div>
"""


def render_prediction_card(
    target_id: str,
    is_planet: bool,
    probability: float,
    confidence_label: str,
) -> str:
    label = "LIKELY EXOPLANET" if is_planet else "NON-PLANET SIGNAL"
    css_class = "planet" if is_planet else "non-planet"
    pct = min(max(probability * 100, 0), 100)
    offset = 283 - (283 * pct / 100)

    return f"""
<div class="exo-panel exo-prediction-card exo-section">
  <div class="exo-target-id">TARGET · {target_id}</div>
  <div class="exo-ring-wrap">
    <svg class="exo-ring-svg" viewBox="0 0 100 100">
      <defs>
        <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#00D4FF"/>
          <stop offset="100%" style="stop-color:#8B5CF6"/>
        </linearGradient>
      </defs>
      <circle class="exo-ring-bg" cx="50" cy="50" r="45"/>
      <circle class="exo-ring-fill" cx="50" cy="50" r="45"
              style="stroke-dashoffset:{offset}"/>
    </svg>
    <div class="exo-ring-text">
      <div class="exo-ring-pct">{pct:.1f}%</div>
      <div class="exo-ring-sub">Confidence</div>
    </div>
  </div>
  <div class="exo-prediction-label {css_class}">{label}</div>
  <div style="color:#94A3B8;font-size:0.8rem;letter-spacing:2px;margin-top:8px;">
    {confidence_label}
  </div>
</div>
"""


def render_confidence_gauges(
    explanation_confidence: float,
    reliability: float,
    raw_prob: float,
    cal_prob: float,
) -> str:
    def _cls(val: float) -> str:
        if val >= 0.8:
            return "high"
        if val >= 0.5:
            return "mid"
        return "low"

    gauges = [
        ("Model Confidence", cal_prob, _cls(cal_prob)),
        ("Explanation Trust", explanation_confidence, _cls(explanation_confidence)),
        ("Reliability (ECE)", reliability, _cls(reliability)),
        ("Raw Probability", raw_prob, _cls(raw_prob)),
    ]
    items = ""
    for label, val, cls in gauges:
        items += f"""
<div class="exo-panel exo-gauge">
  <div class="exo-gauge-label">{label}</div>
  <div class="exo-gauge-value {cls}">{val * 100:.1f}%</div>
</div>"""
    return f'<div class="exo-gauge-grid">{items}</div>'


def render_feature_bars(
    features: list[tuple[str, float]],
    positive: bool = True,
    delay_start: int = 0,
) -> str:
    if not features:
        return '<div style="color:#94A3B8;padding:12px;">No features available</div>'

    max_val = max(abs(v) for _, v in features) or 1.0
    fill_class = "positive" if positive else "negative"
    items = ""
    for i, (name, val) in enumerate(features):
        width = min(abs(val) / max_val * 100, 100)
        delay = (delay_start + i) * 0.08
        color = "#00FFAA" if positive else "#FF5A7A"
        items += f"""
<div class="exo-feature-bar" style="animation-delay:{delay}s">
  <div class="exo-feature-name">{name}</div>
  <div class="exo-feature-track">
    <div class="exo-feature-fill {fill_class}" style="width:{width}%"></div>
  </div>
  <div class="exo-feature-val" style="color:{color}">{val:+.3f}</div>
</div>"""
    return items


def render_briefing(
    summary: str,
    strengths: list[str],
    warnings: list[str],
) -> str:
    strength_items = "".join(
        f'<div class="exo-list-item" style="animation-delay:{i*0.1}s">'
        f'<span class="exo-list-icon">✦</span><span>{s}</span></div>'
        for i, s in enumerate(strengths)
    )
    warning_items = "".join(
        f'<div class="exo-list-item" style="animation-delay:{i*0.1}s">'
        f'<span class="exo-list-icon">⚠</span><span>{w}</span></div>'
        for i, w in enumerate(warnings)
    )
    return f"""
<div class="exo-panel exo-section">
  <div class="exo-panel-header">AI Scientific Analysis</div>
  <div class="exo-briefing">
    <div class="exo-briefing-title">MISSION BRIEFING</div>
    {summary or "Awaiting analysis data from stellar signal processing pipeline."}
  </div>
  {"<div style='margin-top:20px'><div class='exo-panel-header'>Signal Strengths</div>" + strength_items + "</div>" if strengths else ""}
  {"<div style='margin-top:20px'><div class='exo-panel-header'>Anomaly Warnings</div>" + warning_items + "</div>" if warnings else ""}
</div>
"""


def render_loading(message: str = "Initializing Deep Space Analysis...") -> str:
    return f"""
<div class="exo-loading exo-panel">
  <div class="exo-loading-text">{message}</div>
  <div class="exo-progress-track">
    <div class="exo-progress-fill"></div>
  </div>
</div>
"""


def render_metrics_row() -> str:
    metrics = [
        ("97.4%", "Accuracy"),
        ("96.1%", "Precision"),
        ("94.8%", "Recall"),
        ("95.4%", "F1 Score"),
        ("0.982", "ROC-AUC"),
    ]
    cards = "".join(
        f"""<div class="exo-metric-card">
  <div class="exo-metric-value">{val}</div>
  <div class="exo-metric-name">{name}</div>
</div>"""
        for val, name in metrics
    )
    return f'<div class="exo-metrics-row">{cards}</div>'


def render_panel_header(title: str) -> str:
    return f'<div class="exo-panel-header">{title}</div>'


def render_empty_state(message: str) -> str:
    return f"""
<div class="exo-panel exo-section" style="text-align:center;padding:64px 32px;">
  <div style="font-size:3rem;margin-bottom:16px;opacity:0.5;">🪐</div>
  <div style="font-family:Orbitron,sans-serif;font-size:0.85rem;letter-spacing:3px;color:#00D4FF;">
    AWAITING STELLAR DATA
  </div>
  <div style="color:#94A3B8;margin-top:12px;font-size:1rem;">{message}</div>
</div>
"""


def render_multiclass_panel(class_probs: dict, primary_class: str) -> str:
    """Multi-class probability bars for SIH classification display."""
    if not class_probs:
        return '<div class="exo-panel"><p style="color:#94A3B8;">Classification unavailable</p></div>'
    sorted_items = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    bars = ""
    for i, (name, prob) in enumerate(sorted_items):
        width = min(prob * 100, 100)
        highlight = "border:1px solid #00D4FF;" if name == primary_class else ""
        bars += f"""
<div class="exo-feature-bar" style="animation-delay:{i*0.06}s">
  <div class="exo-feature-name">{name}</div>
  <div class="exo-feature-track" style="{highlight}">
    <div class="exo-feature-fill positive" style="width:{width}%"></div>
  </div>
  <div class="exo-feature-val" style="color:#00FFAA">{prob*100:.1f}%</div>
</div>"""
    return f'<div class="exo-panel exo-section"><div class="exo-panel-header">Multi-Class Classification</div>{bars}</div>'


def render_significance_panel(sig: dict) -> str:
    """SNR, detection significance, FAP, confidence gauges."""
    snr = sig.get("snr", 0)
    sigma = sig.get("detection_significance_sigma", 0)
    fap = sig.get("false_alarm_probability", 1)
    conf = sig.get("confidence_score", 0)
    quality = sig.get("detection_quality", "Low")
    return f"""
<div class="exo-gauge-grid">
  <div class="exo-panel exo-gauge">
    <div class="exo-gauge-label">Signal-to-Noise Ratio</div>
    <div class="exo-gauge-value high">{snr:.1f}</div>
  </div>
  <div class="exo-panel exo-gauge">
    <div class="exo-gauge-label">Detection Significance</div>
    <div class="exo-gauge-value high">{sigma:.1f} σ</div>
  </div>
  <div class="exo-panel exo-gauge">
    <div class="exo-gauge-label">False Alarm Probability</div>
    <div class="exo-gauge-value mid">{fap:.4f}</div>
  </div>
  <div class="exo-panel exo-gauge">
    <div class="exo-gauge-label">Confidence Score</div>
    <div class="exo-gauge-value high">{conf*100:.1f}%</div>
  </div>
</div>
<div class="exo-panel" style="text-align:center;margin-top:12px;">
  <div class="exo-gauge-label">Detection Quality</div>
  <div style="font-family:Orbitron,sans-serif;font-size:1.4rem;color:#22D3EE;">{quality}</div>
</div>"""


def render_transit_params_panel(transit: dict, features: dict) -> str:
    """Transit parameters with uncertainties."""
    period = features.get("period", transit.get("period", 0))
    p_err = features.get("period_uncertainty", transit.get("period_uncertainty", 0))
    depth = features.get("depth", transit.get("depth", 0))
    d_err = features.get("depth_uncertainty", transit.get("depth_uncertainty", 0))
    dur_h = features.get("duration_hours", (features.get("duration", 0) or 0) * 24)
    dur_err = features.get("duration_hours_uncertainty", 0)
    epoch = transit.get("transit_epoch", features.get("transit_epoch", 0))
    n_trans = transit.get("num_transits", features.get("num_transits", 0))
    freq = transit.get("transit_frequency", features.get("transit_frequency", 0))
    rp = transit.get("planet_radius_ratio", features.get("planet_radius_ratio", 0))

    rows = [
        ("Orbital Period", f"{period:.3f} ± {p_err:.3f} days"),
        ("Transit Duration", f"{dur_h:.2f} ± {dur_err:.2f} hours"),
        ("Transit Depth", f"{depth:.5f} ± {d_err:.5f}"),
        ("Planet Radius Ratio", f"{rp:.4f}"),
        ("Transit Epoch", f"{epoch:.4f} days"),
        ("Transit Midpoint", f"{transit.get('transit_midpoint', epoch):.4f} days"),
        ("Transit Frequency", f"{freq:.4f} day⁻¹"),
        ("Observed Transits", f"{n_trans}"),
    ]
    items = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(0,212,255,0.1);">'
        f'<span style="color:#94A3B8;font-family:JetBrains Mono,monospace;font-size:0.8rem;">{k}</span>'
        f'<span style="color:#E2E8F0;font-family:JetBrains Mono,monospace;font-size:0.85rem;">{v}</span></div>'
        for k, v in rows
    )
    return f'<div class="exo-panel exo-section"><div class="exo-panel-header">Transit Parameter Estimation</div>{items}</div>'


def render_preprocessing_panel(prep: dict) -> str:
    """Preprocessing summary for SIH."""
    steps = prep.get("steps", [])
    step_list = "".join(f"<li>{s}</li>" for s in steps)
    return f"""
<div class="exo-panel exo-section">
  <div class="exo-panel-header">Preprocessing Summary</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px;">
    <div class="exo-metric-card"><div class="exo-metric-value">{prep.get('original_points',0)}</div><div class="exo-metric-name">Original Points</div></div>
    <div class="exo-metric-card"><div class="exo-metric-value">{prep.get('outliers_removed',0)}</div><div class="exo-metric-name">Outliers Removed</div></div>
    <div class="exo-metric-card"><div class="exo-metric-value">{prep.get('background_noise',0):.5f}</div><div class="exo-metric-name">Background Noise</div></div>
  </div>
  <p style="color:#94A3B8;font-size:0.9rem;">Pipeline steps applied:</p>
  <ul style="color:#E2E8F0;line-height:1.8;">{step_list}</ul>
</div>"""
