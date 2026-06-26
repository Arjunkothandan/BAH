"""Preprocessing summary for SIH dashboard display."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def build_preprocessing_summary(
    n_original: int,
    n_after_nan: int,
    n_after_clip: int,
    flux_raw: np.ndarray,
    flux_detrended: np.ndarray,
    flux_clean: np.ndarray,
) -> Dict[str, Any]:
    """Summarize preprocessing steps and noise estimates."""
    background_noise = float(np.std(flux_clean)) if len(flux_clean) else 0.0
    raw_std = float(np.std(flux_raw)) if len(flux_raw) else 0.0
    outliers_removed = max(0, n_after_nan - n_after_clip)

    return {
        "original_points": n_original,
        "after_nan_removal": n_after_nan,
        "after_outlier_clip": n_after_clip,
        "outliers_removed": outliers_removed,
        "nan_removed": max(0, n_original - n_after_nan),
        "normalized": True,
        "detrended": True,
        "background_noise": background_noise,
        "raw_flux_std": raw_std,
        "noise_reduction_pct": round(
            (1.0 - background_noise / raw_std) * 100 if raw_std > 0 else 0.0, 1
        ),
        "steps": [
            "NaN removal",
            "Flux normalization",
            "Savitzky-Golay / rolling-median detrending",
            "Sigma-clipping (outlier removal)",
            "Background noise estimation",
        ],
    }
