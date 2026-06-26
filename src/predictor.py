import os
import logging
import joblib
from typing import Any, Dict

import pandas as pd

from .confidence_calibration import calibrate_probability
from .feature_extraction import extract_features
from .explainability import compute_shap
from .model_cache import get_model_and_calibrator
from .scientific_reasoning import generate_reasoning

# Configure module‑level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Helper: load a pickled sklearn model (and optionally a calibrator)
# ---------------------------------------------------------------------------

def load_model(model_path: str = "models/random_forest.pkl") -> Any:
    """Load a trained RandomForest (or CalibratedClassifierCV) model from a pickle file.

    Parameters
    ----------
    model_path : str, optional
        Path to the pickled model. Defaults to ``"models/random_forest.pkl"``.
    """
    if not os.path.isfile(model_path):
        logger.error(f"Model file not found at '{model_path}'.")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as exc:
        logger.exception(f"Failed to load model from {model_path}: {exc}")
        raise

# ---------------------------------------------------------------------------
# Feature column whitelist – must match training columns
# ---------------------------------------------------------------------------

TRAINING_FEATURES = [
    "period",
    "duration",
    "depth",
    "snr",
    "significance",
    "ls_period",
    "ac_period",
    "agreement_score",
    "shape_score",
    "flat_bottom_score",
    "bottom_std",
    "bottom_slope",
    "tqi",
]


def _prepare_feature_dataframe(feature_dict: Dict[str, Any]) -> pd.DataFrame:
    """Create a one‑row DataFrame containing only the training‑feature columns.

    Missing or NaN values are replaced with ``0.0`` and a warning is logged.
    """
    missing = [k for k in TRAINING_FEATURES if k not in feature_dict]
    if missing:
        logger.error(f"Feature extraction missing keys: {missing}")
        raise KeyError(f"Missing feature keys: {missing}")

    row = []
    for key in TRAINING_FEATURES:
        val = feature_dict.get(key, 0.0)
        if pd.isna(val):
            logger.warning(f"Feature '{key}' is NaN; replacing with 0.0 for inference.")
            val = 0.0
        row.append(float(val))
    return pd.DataFrame([row], columns=TRAINING_FEATURES)


def predict_lightcurve(
    lc: Any,
    model_path: str = "models/random_forest.pkl",
    calibrator_path: str = "models/calibrator.pkl",
) -> Dict[str, Any]:
    """Predict the class of a single LightCurve using the trained RandomForest.

    Returns a rich dictionary containing raw and calibrated probabilities, SHAP
    explanations, scientific reasoning, and a candidate report payload.
    """
    # ---------------------------------------------------------------
    # 1. Feature extraction
    # ---------------------------------------------------------------
    try:
        feature_dict = extract_features(lc)
    except Exception as exc:
        logger.exception(f"Feature extraction failed: {exc}")
        raise

    # ---------------------------------------------------------------
    # 2. Prepare DataFrame for the classifier
    # ---------------------------------------------------------------
    try:
        X = _prepare_feature_dataframe(feature_dict)
    except Exception as exc:
        logger.exception(f"Failed to prepare feature DataFrame: {exc}")
        raise

    # ---------------------------------------------------------------
    # 3. Load model and calibrator (cached)
    # ---------------------------------------------------------------
    model, calibrator = get_model_and_calibrator(model_path, calibrator_path)

    # ---------------------------------------------------------------
    # 4. Raw inference
    # ---------------------------------------------------------------
    try:
        raw_pred = model.predict(X)[0]
    except Exception as exc:
        logger.exception(f"Model prediction failed: {exc}")
        raise

    raw_prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob_arr = model.predict_proba(X)[0]
            if prob_arr.shape[0] == 2:
                raw_prob = float(prob_arr[1])
            else:
                raw_prob = float(prob_arr.max())
        except Exception as exc:
            logger.warning(f"Failed to obtain raw probability: {exc}")

    # ---------------------------------------------------------------
    # 5. Probability calibration
    # ---------------------------------------------------------------
    calibrated_prob = None
    if raw_prob is not None:
        calibrated_prob = calibrate_probability(raw_prob, calibrator)

    # ---------------------------------------------------------------
    # 6. Confidence label based on calibrated probability
    # ---------------------------------------------------------------
    confidence_label = "Unknown"
    if calibrated_prob is not None:
        if calibrated_prob >= 0.90:
            confidence_label = "Very High Confidence"
        elif calibrated_prob >= 0.75:
            confidence_label = "High Confidence"
        elif calibrated_prob >= 0.50:
            confidence_label = "Moderate Confidence"
        else:
            confidence_label = "Low Confidence"

    # ---------------------------------------------------------------
    # 7. SHAP local explanation
    # ---------------------------------------------------------------
    shap_info = compute_shap(model, X)

    # ---------------------------------------------------------------
    # 8. Scientific reasoning
    # ---------------------------------------------------------------
    reasoning = generate_reasoning(feature_dict)

    # ---------------------------------------------------------------
    # 9. Explanation confidence (simple heuristic)
    # ---------------------------------------------------------------
    # Use calibrated probability, TQI and optionally Bayesian score if present.
    tqi_val = float(feature_dict.get("tqi", 0.0))
    bayes_score = float(feature_dict.get("bayesian_score", 0.0))
    explanation_confidence = (
        (calibrated_prob or 0.0) * 0.6 + tqi_val * 0.3 + bayes_score * 0.1
    )

    # ---------------------------------------------------------------
    # 10. Assemble final payload
    # ---------------------------------------------------------------
    result = {
        "prediction": int(raw_pred),
        "probability": raw_prob,
        "calibrated_probability": calibrated_prob,
        "confidence_label": confidence_label,
        "tqi": tqi_val,
        "bayesian_score": bayes_score,
        "explanation": reasoning["summary"],
        "strengths": reasoning["strengths"],
        "warnings": reasoning["warnings"],
        "explanation_confidence": round(explanation_confidence, 3),
        "features": feature_dict,
        "shap": shap_info,
    }

    logger.info(
        f"Prediction completed – pred={raw_pred}, calibrated_prob={calibrated_prob}, label={confidence_label}"
    )
    return result
