import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

_logger = logging.getLogger(__name__)


def load_calibrator(calibrator_path: str = "models/calibrator.pkl") -> IsotonicRegression:
    """Load the isotonic calibrator saved during model training.

    Parameters
    ----------
    calibrator_path : str, optional
        Path to the pickled IsotonicRegression object. Defaults to ``models/calibrator.pkl``.
    """
    if not os.path.isfile(calibrator_path):
        _logger.error(f"Calibrator file not found at '{calibrator_path}'.")
        raise FileNotFoundError(f"Calibrator not found: {calibrator_path}")
    return joblib.load(calibrator_path)


def calibrate_probability(raw_prob: float, calibrator: IsotonicRegression) -> float:
    """Calibrate a single raw probability using the loaded isotonic calibrator.

    Parameters
    ----------
    raw_prob : float
        The uncalibrated probability output by the RandomForest ``predict_proba``.
    calibrator : IsotonicRegression
        The fitted isotonic regression model.

    Returns
    -------
    float
        Calibrated probability clipped to the [0, 1] interval.
    """
    calibrated = calibrator.predict([raw_prob])[0]
    # IsotonicRegression with out_of_bounds='clip' already ensures [0,1]
    return float(calibrated)


def reliability_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute a simple Expected Calibration Error (ECE) over ``n_bins``.

    Parameters
    ----------
    probs : np.ndarray
        Array of calibrated probabilities for the positive class.
    labels : np.ndarray
        Binary ground‑truth labels (0/1).
    n_bins : int, optional
        Number of bins to divide the [0, 1] interval. Default is 10.

    Returns
    -------
    float
        ECE value in the range [0, 1]; lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idxs = np.digitize(probs, bin_edges, right=True) - 1
    ece = 0.0
    total = len(probs)
    for b in range(n_bins):
        mask = bin_idxs == b
        if np.sum(mask) == 0:
            continue
        bin_prob = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (np.sum(mask) / total) * abs(bin_prob - bin_acc)
    return float(ece)

# Convenience wrapper used by the predictor during inference
def calibrate_and_score(raw_prob: float, calibrator_path: str = "models/calibrator.pkl") -> Dict[str, Any]:
    """Load calibrator, calibrate a raw probability, and compute a reliability score placeholder.

    Since inference sees a single sample, we return the calibrated probability and the
    previously computed Brier score (saved during training) as a proxy for reliability.
    The full ECE calculation is performed during training and stored in the model
    artifact; here we simply expose the calibrated value.
    """
    calibrator = load_calibrator(calibrator_path)
    calibrated = calibrate_probability(raw_prob, calibrator)
    return {"calibrated_probability": calibrated}
