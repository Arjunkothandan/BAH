"""User-friendly error messages for the EXOAI dashboard."""

from __future__ import annotations


class ExoAIError(Exception):
    """Base exception with a user-facing message."""

    def __init__(self, user_message: str, detail: str | None = None):
        self.user_message = user_message
        self.detail = detail or user_message
        super().__init__(self.detail)


def friendly_message(exc: Exception) -> str:
    """Map internal exceptions to user-friendly dashboard messages."""
    text = str(exc).lower()

    if isinstance(exc, ExoAIError):
        return exc.user_message
    if isinstance(exc, FileNotFoundError):
        if "random_forest" in text or "model" in text:
            return "ML model not found. Run `python scripts/ensure_assets.py` to train models."
        if "calibrator" in text:
            return "Probability calibrator not found. Retrain models with scripts/ensure_assets.py."
        if ".fits" in text or "demo" in text:
            return "Demo FITS file missing. Upload your own FITS file or regenerate demo data."
        return f"Required file not found: {exc}"
    if isinstance(exc, ValueError):
        if "empty lightcurve" in text or "empty" in text and "lightcurve" in text:
            return "The FITS file contains no usable light-curve data."
        if "unsupported report format" in text:
            return str(exc)
        if "no transit" in text or "no peaks" in text:
            return "No clear transit signal detected in this light curve."
        return f"Invalid input: {exc}"
    if "fits" in text or "lightkurve" in text or "corrupt" in text:
        return "Invalid or corrupted FITS file. Please upload a valid TESS light curve."
    if "shap" in text:
        return "SHAP explainability engine unavailable for this prediction."
    if "keyerror" in type(exc).__name__.lower() or isinstance(exc, KeyError):
        return "Feature extraction incomplete — prediction unavailable."
    return "An unexpected error occurred during analysis. Please try another file."
