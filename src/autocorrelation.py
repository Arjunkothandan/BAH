import numpy as np
import logging
from scipy.signal import find_peaks
from typing import Dict, List


def autocorrelation_period(
    lc,
    min_period: float = 0.5,
    max_period: float = 30.0,
    top_n: int = 3,
    tolerance: float = 0.1,
) -> Dict:
    """Calculate autocorrelation based period candidates for a LightCurve.

    Parameters
    ----------
    lc : LightCurve
        Lightkurve LightCurve object (already cleaned and normalised).
    min_period : float, optional
        Minimum period (days) to consider for peaks. Default 0.5.
    max_period : float, optional
        Maximum period (days) to consider for peaks. Default 30.
    top_n : int, optional
        Number of top candidate periods to return. Default 3.
    tolerance : float, optional
        Relative tolerance for harmonic detection (e.g., 0.1 = 10%).

    Returns
    -------
    dict
        ``{"best_period": float,
        "correlation": float,
        "candidate_periods": List[float],
        "candidate_correlations": List[float],
        "is_harmonic": bool,
        "harmonic_ratio": int or None}``
    """
    # ------------------------------------------------------------------
    # 1. Extract time and flux, ensure they are NumPy arrays
    # ------------------------------------------------------------------
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    if time.size == 0 or flux.size == 0:
        raise ValueError("Empty LightCurve supplied to autocorrelation_period")

    # ------------------------------------------------------------------
    # 2. Compute median sampling interval – used to map lag index to time
    # ------------------------------------------------------------------
    if time.size > 1:
        dt = np.median(np.diff(time))
    else:
        dt = 1.0  # fallback, though a single point cannot produce a period

    # ------------------------------------------------------------------
    # 3. Downsample for speed on long cadences (cap at 8000 points)
    # ------------------------------------------------------------------
    max_points = 8000
    if flux.size > max_points:
        idx = np.linspace(0, flux.size - 1, max_points, dtype=int)
        time = time[idx]
        flux = flux[idx]

    # ------------------------------------------------------------------
    # 4. Normalise flux (zero‑mean) and compute FFT autocorrelation
    # ------------------------------------------------------------------
    flux_centered = flux - np.mean(flux)
    n = flux_centered.size
    ac = np.fft.ifft(np.fft.fft(flux_centered) * np.conj(np.fft.fft(flux_centered))).real
    ac = ac[: n // 2 + 1]
    ac = ac / (ac[0] if ac[0] != 0 else 1.0)

    # ------------------------------------------------------------------
    # 5. Build lag (in days) array
    # ------------------------------------------------------------------
    lags = np.arange(ac.size) * dt

    # ------------------------------------------------------------------
    # 6. Exclude zero‑lag point before peak finding
    # ------------------------------------------------------------------
    ac_nonzero = ac[1:]
    lags_nonzero = lags[1:]

    # ------------------------------------------------------------------
    # 6. Detect local peaks in the autocorrelation function
    # ------------------------------------------------------------------
    peak_indices, _ = find_peaks(ac_nonzero)
    if peak_indices.size == 0:
        # No peaks – fall back to the global maximum (which will be at zero lag)
        logging.warning("No autocorrelation peaks detected; returning NaNs")
        return {
            "best_period": float('nan'),
            "correlation": float('nan'),
            "candidate_periods": [],
            "candidate_correlations": [],
            "is_harmonic": False,
            "harmonic_ratio": None,
        }

    # Convert to original array indices (account for the removed zero‑lag)
    peak_indices += 1

    # ------------------------------------------------------------------
    # 7. Filter peaks that lie inside the requested period window
    # ------------------------------------------------------------------
    valid_mask = (lags >= min_period) & (lags <= max_period)
    valid_peaks = peak_indices[valid_mask[peak_indices]]

    if valid_peaks.size == 0:
        logging.warning(
            f"No autocorrelation peaks found in period range [{min_period}, {max_period}]"
        )
        # Return the highest peak overall (still outside the range) for safety
        best_idx = np.argmax(ac[1:]) + 1
        best_period = lags[best_idx]
        best_corr = ac[best_idx]
        return {
            "best_period": float(best_period),
            "correlation": float(best_corr),
            "candidate_periods": [],
            "candidate_correlations": [],
            "is_harmonic": False,
            "harmonic_ratio": None,
        }

    # ------------------------------------------------------------------
    # 8. Rank the valid peaks by autocorrelation height (descending)
    # ------------------------------------------------------------------
    peak_corrs = ac[valid_peaks]
    sorted_idx = np.argsort(peak_corrs)[::-1]
    top_peaks = valid_peaks[sorted_idx][:top_n]

    candidate_periods = lags[top_peaks].tolist()
    candidate_correlations = ac[top_peaks].tolist()

    # Best candidate is the first in the sorted list
    best_period = candidate_periods[0]
    best_corr = candidate_correlations[0]

    # ------------------------------------------------------------------
    # 9. Simple harmonic detection between the two strongest peaks
    # ------------------------------------------------------------------
    is_harmonic = False
    harmonic_ratio: int | None = None
    if len(candidate_periods) >= 2:
        ratio = candidate_periods[0] / candidate_periods[1]
        nearest_int = round(ratio)
        if nearest_int > 1 and abs(ratio - nearest_int) / nearest_int < tolerance:
            is_harmonic = True
            harmonic_ratio = nearest_int

    # ------------------------------------------------------------------
    # 10. Logging
    # ------------------------------------------------------------------
    logging.info(f"AC best={best_period:.4f}, corr={best_corr:.3f}")
    logging.debug(f"AC candidates={candidate_periods}")

    return {
        "best_period": float(best_period),
        "correlation": float(best_corr),
        "candidate_periods": candidate_periods,
        "candidate_correlations": candidate_correlations,
        "is_harmonic": is_harmonic,
        "harmonic_ratio": harmonic_ratio,
    }
