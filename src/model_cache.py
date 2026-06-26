"""Cached loaders for ML model artifacts (avoids repeated disk I/O)."""

from __future__ import annotations

import functools
from typing import Any, Tuple

import joblib


@functools.lru_cache(maxsize=4)
def get_model_and_calibrator(
    model_path: str,
    calibrator_path: str,
) -> Tuple[Any, Any]:
    """Load and cache the RandomForest model and isotonic calibrator."""
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    return model, calibrator
