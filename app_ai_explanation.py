"""
AI Exoplanet Detection & Explainability Platform
Futuristic NASA-grade HUD dashboard — UI layer only; backend logic unchanged.
"""

from __future__ import annotations

import os
import sys

# Path bootstrap MUST run before streamlit and src imports.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Streamlit hot-reload keeps old `src.*` modules in sys.modules; always reload from disk.
for _key in list(sys.modules):
    if _key == "src" or _key.startswith("src."):
        del sys.modules[_key]

import streamlit as st

try:
    from src.dashboard_components import (
        predict_and_explain,
        generate_report,
        get_shap_figure,
        friendly_message,
    )
    from src.report_formats import (
        FORMAT_EXTENSIONS,
        FORMAT_MIME_TYPES,
        normalize_format,
    )
    from src.ui.theme import inject_theme, render_starfield
    from src.ui.components import (
        render_status_bar,
        render_hero,
        render_prediction_card,
        render_confidence_gauges,
        render_feature_bars,
        render_briefing,
        render_loading,
        render_metrics_row,
        render_panel_header,
        render_empty_state,
        render_multiclass_panel,
        render_significance_panel,
        render_transit_params_panel,
        render_preprocessing_panel,
    )
    from src.ui.charts import (
        style_shap_waterfall,
        create_orbital_viz,
        create_feature_ranking_chart,
    )
    from src.lightcurve_viz import build_all_figures
except ImportError as e:
    raise RuntimeError(
        f"Import failed from project root '{PROJECT_ROOT}'. "
        f"Original error: {e}. "
        "Run: streamlit run app_ai_explanation.py from D:\\BHA, "
        "or use: streamlit run D:\\BHA\\app_ai_explanation.py"
    ) from e

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EXOAI · Exoplanet Detection Platform",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()
render_starfield()

NAV_ITEMS = [
    ("dashboard", "🏠 Dashboard"),
    ("detection", "🪐 Planet Detection"),
    ("explanation", "🧠 AI Explanation"),
    ("explorer", "📊 Data Explorer"),
    ("science", "🔬 Scientific Analysis"),
    ("reports", "📄 Reports"),
    ("settings", "⚙ Settings"),
]

LOADING_MESSAGE = "Running Deep Space Analysis…"


@st.cache_resource(show_spinner=False)
def _warmup_models() -> bool:
    """Pre-load ML artifacts once per server process."""
    from src.model_cache import get_model_and_calibrator
    model_path = os.path.join(PROJECT_ROOT, "models", "random_forest.pkl")
    cal_path = os.path.join(PROJECT_ROOT, "models", "calibrator.pkl")
    if os.path.isfile(model_path) and os.path.isfile(cal_path):
        get_model_and_calibrator(model_path, cal_path)
        return True
    return False


_warmup_models()


@st.cache_data(show_spinner=False)
def _run_inference(lc_data: bytes, source_label: str, target_id: str):
    return predict_and_explain(lc_data, source_label=source_label, target_id=target_id or None)


@st.cache_data(show_spinner="Generating report…")
def _cached_report(result_sig: str, fmt_key: str, payload_json: str) -> bytes:
    import json
    result = json.loads(payload_json)
    return generate_report(result, fmt_key)


def _result_signature(result: dict) -> str:
    import hashlib
    key = (
        f"{result.get('prediction')}_{result.get('calibrated_probability')}_"
        f"{result.get('features', {}).get('period')}"
    )
    return hashlib.md5(key.encode()).hexdigest()


def _serializable_result(result: dict) -> str:
    import json
    keys = (
        "prediction", "probability", "calibrated_probability", "confidence_label",
        "explanation_confidence", "reliability_score", "raw_probability", "tqi",
        "source_label", "features", "shap", "top_positive_features",
        "top_negative_features", "scientific_reasoning", "explanation",
        "strengths", "warnings",
    )
    return json.dumps({k: result[k] for k in keys if k in result}, default=str)


if "nav_page" not in st.session_state:
    st.session_state.nav_page = "dashboard"
if "explain_result" not in st.session_state:
    st.session_state.explain_result = None
if "result_sig" not in st.session_state:
    st.session_state.result_sig = ""
if "source_label" not in st.session_state:
    st.session_state.source_label = ""


def _nav_button(key: str, label: str) -> None:
    active = st.session_state.nav_page == key
    container_class = "nav-active" if active else ""
    st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
    if st.button(label, key=f"nav_{key}", use_container_width=True):
        st.session_state.nav_page = key
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_nav() -> None:
    st.markdown(render_status_bar(), unsafe_allow_html=True)
    cols = st.columns(len(NAV_ITEMS))
    for col, (key, label) in zip(cols, NAV_ITEMS):
        with col:
            _nav_button(key, label)


def _load_lightcurve_bytes(uploaded_file, demo_option: str) -> tuple[bytes | None, str, str]:
    """Return (bytes, source_label, target_id)."""
    if uploaded_file:
        name = uploaded_file.name or "uploaded.fits"
        return uploaded_file.read(), name, name.replace(".fits", "").replace(".fit", "")
    if demo_option != "None":
        fname = demo_option.replace(" ", "_").lower() + ".fits"
        demo_path = os.path.join(PROJECT_ROOT, "demo", fname)
        if not os.path.isfile(demo_path):
            raise FileNotFoundError(f"Demo file missing: {demo_path}")
        with open(demo_path, "rb") as f:
            return f.read(), demo_option, demo_option.replace(" ", "-")
    return None, "", ""


def _run_detection_pipeline(lc_bytes: bytes, source_label: str, target_id: str) -> None:
    with st.spinner(LOADING_MESSAGE):
        try:
            result = _run_inference(lc_bytes, source_label, target_id)
        except Exception as exc:
            st.error(friendly_message(exc))
            return
    st.session_state.explain_result = result
    st.session_state.source_label = source_label
    st.session_state.result_sig = _result_signature(result)
    st.session_state.nav_page = "explanation"
    st.rerun()


def _get_result() -> dict | None:
    return st.session_state.get("explain_result")


def _target_id(result: dict) -> str:
    features = result.get("features", {})
    return str(features.get("id", "TOI-UNKNOWN"))


# ── Page: Dashboard ─────────────────────────────────────────────────────────
def page_dashboard() -> None:
    st.markdown('<div class="exo-content">', unsafe_allow_html=True)
    st.markdown(render_hero(), unsafe_allow_html=True)
    st.markdown(render_metrics_row(), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="exo-panel exo-section">{render_panel_header("Mission Overview")}'
            f'<p style="color:#94A3B8;line-height:1.8;font-size:1.05rem;">'
            f'The EXOAI platform combines transit photometry analysis with '
            f'machine learning and SHAP-based explainability to classify exoplanet '
            f'candidates from TESS light curves. Navigate to <strong style="color:#00D4FF">'
            f'Planet Detection</strong> to begin analysis.</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        result = _get_result()
        if result:
            cal = result.get("calibrated_probability") or 0
            st.markdown(
                render_prediction_card(
                    _target_id(result),
                    result.get("prediction") == 1,
                    cal,
                    result.get("confidence_label", "Unknown"),
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                render_empty_state(
                    "Upload a FITS light curve to begin exoplanet discovery."
                ),
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Page: Planet Detection ──────────────────────────────────────────────────
def page_detection() -> None:
    st.markdown('<div class="exo-content">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="exo-panel exo-section">{render_panel_header("Stellar Signal Acquisition")}'
        f'<p style="color:#94A3B8;margin-bottom:20px;">Upload a TESS FITS light curve or '
        f'select a pre-loaded demonstration target.</p></div>',
        unsafe_allow_html=True,
    )

    col_upload, col_orbit = st.columns([1, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Transmit FITS Data",
            type=["fits", "fit"],
            help="TESS light curve FITS file",
        )
        demo_option = st.selectbox(
            "Pre-loaded Targets",
            ["None", "Demo Light Curve 1", "Demo Light Curve 2"],
        )
        run_button = st.button("🚀 Initiate Deep Space Scan", type="primary", use_container_width=True)

        if run_button:
            try:
                lc_bytes, source_label, target_id = _load_lightcurve_bytes(uploaded_file, demo_option)
            except FileNotFoundError as exc:
                st.error(friendly_message(exc))
                lc_bytes = None
                source_label = ""
                target_id = ""
            if lc_bytes is None:
                if uploaded_file is None and demo_option == "None":
                    st.warning("Provide a FITS file or select a demo target.")
            else:
                _run_detection_pipeline(lc_bytes, source_label, target_id)

    with col_orbit:
        result = _get_result()
        period = None
        depth = None
        if result:
            feats = result.get("features", {})
            period = feats.get("period")
            depth = feats.get("depth")
        fig = create_orbital_viz(period=period, depth=depth)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    result = _get_result()
    if result:
        cal = result.get("calibrated_probability") or 0
        mc = result.get("multiclass", {})
        st.markdown(
            render_prediction_card(
                _target_id(result),
                result.get("prediction") == 1,
                cal,
                mc.get("primary_class", result.get("confidence_label", "Unknown")),
            ),
            unsafe_allow_html=True,
        )
        if result.get("preprocessing"):
            st.markdown(render_preprocessing_panel(result["preprocessing"]), unsafe_allow_html=True)
        if mc.get("class_probabilities"):
            st.markdown(
                render_multiclass_panel(mc["class_probabilities"], mc.get("primary_class", "")),
                unsafe_allow_html=True,
            )
        if result.get("significance"):
            st.markdown(render_significance_panel(result["significance"]), unsafe_allow_html=True)
        if result.get("transit"):
            st.markdown(
                render_transit_params_panel(result["transit"], result.get("features", {})),
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Page: AI Explanation ──────────────────────────────────────────────────────
def page_explanation() -> None:
    st.markdown('<div class="exo-content">', unsafe_allow_html=True)
    result = _get_result()

    if not result:
        st.markdown(
            render_empty_state("Run planet detection first to generate AI explanations."),
            unsafe_allow_html=True,
        )
        if st.button("Go to Planet Detection", use_container_width=True):
            st.session_state.nav_page = "detection"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cal = result.get("calibrated_probability") or 0
    raw = result.get("raw_probability") or 0

    # 1 — Prediction
    st.markdown(
        render_prediction_card(
            _target_id(result),
            result.get("prediction") == 1,
            cal,
            result.get("confidence_label", "Unknown"),
        ),
        unsafe_allow_html=True,
    )

    # 2 — Multi-class + Significance
    mc = result.get("multiclass", {})
    if mc.get("class_probabilities"):
        st.markdown(
            render_multiclass_panel(mc["class_probabilities"], mc.get("primary_class", "")),
            unsafe_allow_html=True,
        )
    if result.get("significance"):
        st.markdown(render_significance_panel(result["significance"]), unsafe_allow_html=True)

    # 3 — Confidence gauges
    st.markdown(
        render_confidence_gauges(
            result.get("explanation_confidence", 0),
            result.get("reliability_score", 0),
            raw,
            cal,
        ),
        unsafe_allow_html=True,
    )

    # 3 — SHAP Waterfall
    st.markdown(
        f'<div class="exo-panel exo-section">{render_panel_header("SHAP Explainability Engine")}</div>',
        unsafe_allow_html=True,
    )
    shap_fig = get_shap_figure(result)
    if shap_fig:
        styled = style_shap_waterfall(shap_fig)
        st.plotly_chart(styled, use_container_width=True, config={"displayModeBar": False})

    # 4 — Feature importance ranking
    shap_dict = result.get("shap", {}).get("feature_importance_local", {})
    if shap_dict:
        rank_fig = create_feature_ranking_chart(shap_dict)
        st.plotly_chart(rank_fig, use_container_width=True, config={"displayModeBar": False})

    # 5 — Positive / Negative factors
    top_pos = result.get("top_positive_features", [])
    top_neg = result.get("top_negative_features", [])
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown(
            f'<div class="exo-panel exo-section">{render_panel_header("Positive Factors")}'
            f'{render_feature_bars(top_pos, positive=True)}</div>',
            unsafe_allow_html=True,
        )
    with col_neg:
        st.markdown(
            f'<div class="exo-panel exo-section">{render_panel_header("Negative Factors")}'
            f'{render_feature_bars(top_neg, positive=False)}</div>',
            unsafe_allow_html=True,
        )

    # 6 — Scientific reasoning
    reasoning = result.get("scientific_reasoning", {})
    st.markdown(
        render_briefing(
            reasoning.get("summary", ""),
            reasoning.get("strengths", []),
            reasoning.get("warnings", []),
        ),
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Page: Data Explorer ─────────────────────────────────────────────────────
def page_explorer() -> None:
    st.markdown('<div class="exo-content">', unsafe_allow_html=True)
    result = _get_result()

    if not result:
        st.markdown(
            render_empty_state("No feature data available. Run detection first."),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    features = result.get("features", {})
    viz = result.get("viz", {})

    if viz:
        st.markdown(
            f'<div class="exo-panel exo-section">{render_panel_header("Light Curve Analysis")}</div>',
            unsafe_allow_html=True,
        )
        figures = build_all_figures(viz)
        for key in ("overview", "bls", "transit", "folded"):
            if key in figures:
                st.plotly_chart(figures[key], use_container_width=True, config={"displayModeBar": False})

        if result.get("preprocessing"):
            st.markdown(render_preprocessing_panel(result["preprocessing"]), unsafe_allow_html=True)

    st.markdown(
        f'<div class="exo-panel exo-section">{render_panel_header("Extracted Astrophysical Features")}</div>',
        unsafe_allow_html=True,
    )

    import pandas as pd

    df = pd.DataFrame(
        [{"Feature": k, "Value": f"{v:.6f}" if isinstance(v, float) else str(v)} for k, v in features.items()]
    )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="medium"),
        },
    )

    feats = result.get("features", {})
    fig = create_orbital_viz(
        period=feats.get("period"),
        depth=feats.get("depth"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ── Page: Scientific Analysis ───────────────────────────────────────────────
def page_science() -> None:
    st.markdown('<div class="exo-content">', unsafe_allow_html=True)
    result = _get_result()

    if not result:
        st.markdown(
            render_empty_state("Run detection to access scientific analysis."),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if result.get("transit"):
        st.markdown(
            render_transit_params_panel(result["transit"], result.get("features", {})),
            unsafe_allow_html=True,
        )
    if result.get("significance"):
        st.markdown(render_significance_panel(result["significance"]), unsafe_allow_html=True)

    reasoning = result.get("scientific_reasoning", {})
    st.markdown(
        render_briefing(
            reasoning.get("summary", ""),
            reasoning.get("strengths", []),
            reasoning.get("warnings", []),
        ),
        unsafe_allow_html=True,
    )

    cal = result.get("calibrated_probability") or 0
    raw = result.get("raw_probability") or 0
    st.markdown(
        render_confidence_gauges(
            result.get("explanation_confidence", 0),
            result.get("reliability_score", 0),
            raw,
            cal,
        ),
        unsafe_allow_html=True,
    )

    tqi = result.get("tqi", result.get("features", {}).get("tqi", 0))
    st.markdown(
        f'<div class="exo-panel exo-section">'
        f'{render_panel_header("Transit Quality Index")}'
        f'<div style="text-align:center;padding:24px;">'
        f'<div style="font-family:Orbitron,sans-serif;font-size:3rem;color:#22D3EE;">{tqi:.3f}</div>'
        f'<div style="color:#94A3B8;letter-spacing:2px;font-size:0.75rem;margin-top:8px;">'
        f'TRANSIT QUALITY INDEX</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Page: Reports ─────────────────────────────────────────────────────────────
# Report export options — fmt_key values are normalized via candidate_report.normalize_format()
REPORT_FORMATS = [
    ("PDF", "pdf", "📄", "Export PDF"),
    ("HTML", "html", "📊", "Export Analysis"),
    ("Markdown", "md", "🧠", "Export AI Explanation"),
    ("TXT", "txt", "📝", "Export Text"),
]


def page_reports() -> None:
    st.markdown('<div class="exo-content">', unsafe_allow_html=True)
    result = _get_result()

    st.markdown(
        f'<div class="exo-panel exo-section">{render_panel_header("Mission Report Export")}'
        f'<p style="color:#94A3B8;">Generate and download comprehensive analysis reports.</p></div>',
        unsafe_allow_html=True,
    )

    if not result:
        st.markdown(
            render_empty_state("Complete a detection run before exporting reports."),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col1, col2, col3, col4 = st.columns(4)
    payload = _serializable_result(result)
    sig = st.session_state.get("result_sig", _result_signature(result))
    for col, (fmt_label, fmt_key, icon, btn_label) in zip(
        [col1, col2, col3, col4], REPORT_FORMATS
    ):
        with col:
            st.markdown(
                f'<div class="exo-export-btn"><div class="exo-export-icon">{icon}</div>'
                f'<div class="exo-export-label">{btn_label}</div></div>',
                unsafe_allow_html=True,
            )
            try:
                canonical = normalize_format(fmt_key)
                report_bytes = _cached_report(sig, fmt_key, payload)
                extension = FORMAT_EXTENSIONS[canonical]
                mime_type = FORMAT_MIME_TYPES[canonical]
            except Exception as exc:
                st.error(friendly_message(exc))
                continue
            st.download_button(
                label=f"Download {fmt_label}",
                data=report_bytes,
                file_name=f"exoplanet_report.{extension}",
                mime=mime_type,
                use_container_width=True,
                key=f"dl_{fmt_key}",
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Page: Settings ────────────────────────────────────────────────────────────
def page_settings() -> None:
    st.markdown('<div class="exo-content">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="exo-panel exo-section">{render_panel_header("System Configuration")}</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="exo-panel">'
            '<div class="exo-panel-header">Model Configuration</div>'
            '<p style="color:#94A3B8;line-height:1.8;">'
            '<strong style="color:#00D4FF">Classifier:</strong> Random Forest<br>'
            '<strong style="color:#00D4FF">Calibrator:</strong> Isotonic Regression<br>'
            '<strong style="color:#00D4FF">Explainer:</strong> SHAP TreeExplainer<br>'
            '<strong style="color:#00D4FF">Features:</strong> 13 astrophysical indicators'
            '</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="exo-panel">'
            '<div class="exo-panel-header">Data Pipeline</div>'
            '<p style="color:#94A3B8;line-height:1.8;">'
            '<strong style="color:#22D3EE">Input:</strong> TESS FITS light curves<br>'
            '<strong style="color:#22D3EE">Preprocessing:</strong> Detrend + Sigma Clip<br>'
            '<strong style="color:#22D3EE">Period Search:</strong> BLS + Lomb-Scargle<br>'
            '<strong style="color:#22D3EE">Output:</strong> Classification + SHAP'
            '</p></div>',
            unsafe_allow_html=True,
        )

    if st.button("Clear Session Data", use_container_width=True):
        st.session_state.explain_result = None
        st.session_state.result_sig = ""
        st.session_state.source_label = ""
        st.session_state.nav_page = "dashboard"
        _run_inference.clear()
        _cached_report.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────
PAGES = {
    "dashboard": page_dashboard,
    "detection": page_detection,
    "explanation": page_explanation,
    "explorer": page_explorer,
    "science": page_science,
    "reports": page_reports,
    "settings": page_settings,
}

_render_nav()
PAGES[st.session_state.nav_page]()
