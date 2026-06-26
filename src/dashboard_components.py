import os
import tempfile
from typing import Any, Dict, Optional

import lightkurve as lk
import numpy as np

from .candidate_report import generate_report as backend_generate_report
from .errors import ExoAIError, friendly_message
from .explainability import plot_shap_waterfall
from .lightcurve_viz import preprocess_lightcurve
from .multiclass_classifier import classify_multiclass
from .predictor import predict_lightcurve
from .preprocessing_summary import build_preprocessing_summary
from .report_context import build_report_context, compute_reliability
from .report_formats import normalize_format
from .transit_analysis import analyze_transit
from .utils import get_project_root


def predict_and_explain(
    lc_data: bytes,
    source_label: str = "Uploaded FITS",
    target_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full SIH-compliant inference pipeline and return dashboard payload."""
    if not lc_data:
        raise ExoAIError("Empty file uploaded.", "Zero-byte FITS payload")

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp.write(lc_data)
        tmp_path = tmp.name

    try:
        try:
            lc = lk.read(tmp_path)
        except Exception as exc:
            raise ExoAIError(
                "Invalid or corrupted FITS file. Please upload a valid TESS light curve.",
                str(exc),
            ) from exc

        if len(lc) == 0:
            raise ExoAIError("The FITS file contains no usable light-curve data.")

        if not hasattr(lc, "flux") or lc.flux is None:
            raise ExoAIError(
                "FITS file missing flux column. Upload a standard TESS light curve.",
                "No flux column",
            )

        project_root = get_project_root()
        model_path = os.path.join(project_root, "models", "random_forest.pkl")
        calibrator_path = os.path.join(project_root, "models", "calibrator.pkl")

        if not os.path.isfile(model_path):
            raise ExoAIError(
                "ML model not found. Run: python scripts/ensure_assets.py",
                f"Missing {model_path}",
            )
        if not os.path.isfile(calibrator_path):
            raise ExoAIError(
                "Probability calibrator not found. Run: python scripts/ensure_assets.py",
                f"Missing {calibrator_path}",
            )

        n_original = len(lc)
        time_raw, flux_raw, flux_detrended, time_clean, flux_clean = preprocess_lightcurve(lc)

        # Extended BLS transit analysis
        try:
            transit = analyze_transit(time_clean, flux_clean)
        except ValueError:
            from .transit_analysis import _empty_transit_result
            transit = _empty_transit_result()

        result = predict_lightcurve(lc, model_path=model_path, calibrator_path=calibrator_path)

        # Merge transit parameters into features
        feats = result.setdefault("features", {})
        for key, val in transit.items():
            if key.startswith("bls_periodogram"):
                continue
            if isinstance(val, float) and np.isnan(val):
                continue
            if val is not None:
                feats[key] = val

        # Multi-class classification
        ml_prob = float(result.get("calibrated_probability") or result.get("probability") or 0.0)
        multiclass = classify_multiclass(feats, ml_prob)
        result["multiclass"] = multiclass

        # Significance block (SIH)
        snr = float(feats.get("snr", 0.0))
        result["significance"] = {
            "snr": snr,
            "detection_significance_sigma": float(transit.get("detection_significance_sigma", 0.0)),
            "false_alarm_probability": float(transit.get("false_alarm_probability", 1.0)),
            "confidence_score": float(result.get("calibrated_probability") or 0.0),
            "detection_quality": transit.get("detection_quality", "Low"),
        }

        # Preprocessing summary
        result["preprocessing"] = build_preprocessing_summary(
            n_original=n_original,
            n_after_nan=len(flux_raw),
            n_after_clip=len(flux_clean),
            flux_raw=flux_raw,
            flux_detrended=flux_detrended,
            flux_clean=flux_clean,
        )

        # Metadata
        meta_id = target_id or getattr(lc, "targetid", None) or source_label
        feats["id"] = str(meta_id)
        feats["mission"] = getattr(lc, "mission", "TESS") or "TESS"
        sector = getattr(lc, "sector", None)
        if sector is not None:
            feats["sector"] = str(sector)

        shap_dict = result["shap"]["feature_importance_local"]
        result["raw_probability"] = result.get("probability", 0.0)
        result["reliability_score"] = compute_reliability(result)
        result["source_label"] = source_label

        result["top_positive_features"] = [
            (name, float(shap_dict[name])) for name in result["shap"]["top_positive_features"]
        ]
        result["top_negative_features"] = [
            (name, float(shap_dict[name])) for name in result["shap"]["top_negative_features"]
        ]
        result["scientific_reasoning"] = {
            "summary": _natural_language_summary(result, multiclass, transit),
            "strengths": result.get("strengths", []),
            "warnings": result.get("warnings", []),
        }

        result["viz"] = {
            "time_raw": time_raw.tolist(),
            "flux_raw": flux_raw.tolist(),
            "flux_detrended": flux_detrended.tolist(),
            "time_clean": time_clean.tolist(),
            "flux_clean": flux_clean.tolist(),
            "period": feats.get("period"),
            "duration": feats.get("duration"),
            "transit_epoch": transit.get("transit_epoch"),
            "bls_periods": transit.get("bls_periodogram_periods", []),
            "bls_powers": transit.get("bls_periodogram_powers", []),
        }
        result["shap_plot_data"] = shap_dict
        result["transit"] = transit

        return result

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _natural_language_summary(
    result: Dict[str, Any],
    multiclass: Dict[str, Any],
    transit: Dict[str, Any],
) -> str:
    """Compose SIH natural-language explanation."""
    primary = multiclass.get("primary_class", "Unknown")
    prob = multiclass.get("primary_probability", 0.0) * 100
    sigma = transit.get("detection_significance_sigma", 0.0)
    period = transit.get("period", 0.0)
    parts = [
        f"The signal is classified as {primary} ({prob:.1f}% probability).",
        f"BLS transit search yields period {period:.3f} days with {sigma:.1f}σ significance.",
    ]
    if result.get("explanation"):
        parts.append(result["explanation"])
    return " ".join(parts)


def get_shap_figure(result: Dict[str, Any]):
    """Build SHAP waterfall figure lazily from cached dict."""
    shap_dict = result.get("shap_plot_data") or result.get("shap", {}).get("feature_importance_local", {})
    if not shap_dict:
        return None
    return plot_shap_waterfall(shap_dict)


def generate_report(result: Dict[str, Any], format_str: str) -> bytes:
    """Generate professional 3-page report from inference result."""
    ctx = build_report_context(result, source_label=result.get("source_label", "Uploaded FITS"))
    candidate = result.get("features", {}).copy()
    canonical_format = normalize_format(format_str)

    return backend_generate_report(
        candidate=candidate,
        interpretation=ctx["interpretation"],
        top_positive=ctx["top_positive"],
        top_negative=ctx["top_negative"],
        verdict=ctx["verdict"],
        format=canonical_format,
        context=ctx,
    )


__all__ = ["predict_and_explain", "generate_report", "get_shap_figure", "friendly_message"]
