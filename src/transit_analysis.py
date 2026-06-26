"""Extended BLS transit analysis with significance and parameter uncertainties."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy import stats

from .phase_folding import phase_fold

_logger = logging.getLogger(__name__)


def _period_uncertainty(periods: np.ndarray, powers: np.ndarray, best_idx: int) -> float:
    """Estimate period uncertainty from half-maximum width of BLS peak."""
    peak = float(powers[best_idx])
    half = peak * 0.5
    above = powers >= half
    if not np.any(above):
        return float(periods[best_idx]) * 0.01
    idxs = np.where(above)[0]
    width = float(periods[idxs[-1]] - periods[idxs[0]]) if len(idxs) > 1 else periods[best_idx] * 0.01
    return max(width / 2.0, periods[best_idx] * 0.005)


def _depth_uncertainty(flux: np.ndarray, depth: float) -> float:
    """Bootstrap-style depth uncertainty from flux scatter."""
    noise = float(np.std(flux))
    return max(noise / max(np.sqrt(len(flux)), 1.0), depth * 0.05)


def analyze_transit(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 30.0,
) -> Dict[str, Any]:
    """Run BLS transit search and compute SIH-required transit parameters."""
    if len(time) < 20:
        raise ValueError("Insufficient points for transit analysis")

    durations = np.linspace(0.05, 0.30, 12)
    bls = BoxLeastSquares(time, flux)
    results = bls.autopower(durations)

    if len(results.period) == 0:
        return _empty_transit_result()

    # Restrict to requested period window
    mask = (results.period >= min_period) & (results.period <= max_period)
    if not np.any(mask):
        return _empty_transit_result()

    periods = results.period[mask]
    powers = results.power[mask]
    durations_m = results.duration[mask]
    transit_times = results.transit_time[mask]
    best_idx = int(np.argmax(powers))
    period = float(periods[best_idx])
    duration = float(durations_m[best_idx])
    power = float(powers[best_idx])
    epoch = float(transit_times[best_idx])

    # Significance (SDE-style)
    median_p = float(np.median(powers))
    std_p = float(np.std(powers)) or 1e-9
    detection_sigma = float((power - median_p) / std_p)
    false_alarm_prob = float(stats.norm.sf(detection_sigma))

    # Transit counting
    span = float(time.max() - time.min())
    n_transits = max(1, int(span / period)) if period > 0 else 0
    frequency = 1.0 / period if period > 0 else 0.0

    # Depth from folded minimum
    phase, folded = phase_fold(time, flux, period)
    depth = float(np.max(folded) - np.min(folded)) if len(folded) else 0.0
    transit_midpoint = float(epoch)

    # Planet radius ratio: (depth)^0.5 for small planets (approximate)
    radius_ratio = float(np.sqrt(max(depth, 0.0)))

    period_err = _period_uncertainty(periods, powers, best_idx)
    depth_err = _depth_uncertainty(flux, depth)
    duration_err = max(duration * 0.08, 0.001)

    if len(periods) > 500:
        idx = np.linspace(0, len(periods) - 1, 500, dtype=int)
        bls_periods = periods[idx].tolist()
        bls_powers = powers[idx].tolist()
    else:
        bls_periods = periods.tolist()
        bls_powers = powers.tolist()

    return {
        "period": period,
        "period_uncertainty": period_err,
        "duration": duration,
        "duration_uncertainty": duration_err,
        "duration_hours": duration * 24.0,
        "duration_hours_uncertainty": duration_err * 24.0,
        "depth": depth,
        "depth_uncertainty": depth_err,
        "transit_epoch": epoch,
        "transit_midpoint": transit_midpoint,
        "transit_frequency": frequency,
        "num_transits": n_transits,
        "bls_power": power,
        "detection_significance_sigma": detection_sigma,
        "false_alarm_probability": false_alarm_prob,
        "planet_radius_ratio": radius_ratio,
        "bls_periodogram_periods": bls_periods,
        "bls_periodogram_powers": bls_powers,
        "detection_quality": _quality_label(detection_sigma, false_alarm_prob),
    }


def _quality_label(sigma: float, fap: float) -> str:
    if sigma >= 8 and fap < 0.001:
        return "Excellent"
    if sigma >= 5 and fap < 0.01:
        return "Good"
    if sigma >= 3:
        return "Moderate"
    return "Low"


def _empty_transit_result() -> Dict[str, Any]:
    return {
        "period": float("nan"),
        "period_uncertainty": float("nan"),
        "duration": float("nan"),
        "duration_uncertainty": float("nan"),
        "duration_hours": float("nan"),
        "duration_hours_uncertainty": float("nan"),
        "depth": float("nan"),
        "depth_uncertainty": float("nan"),
        "transit_epoch": float("nan"),
        "transit_midpoint": float("nan"),
        "transit_frequency": float("nan"),
        "num_transits": 0,
        "bls_power": 0.0,
        "detection_significance_sigma": 0.0,
        "false_alarm_probability": 1.0,
        "planet_radius_ratio": float("nan"),
        "bls_periodogram_periods": [],
        "bls_periodogram_powers": [],
        "detection_quality": "Low",
    }
