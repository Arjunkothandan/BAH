import numpy as np
import logging
from typing import Dict


def flat_bottom_score(
    phase: np.ndarray,
    flux: np.ndarray,
    transit_center: float,
    transit_duration: float,
) -> Dict:
    """Measure flatness of the bottom of a folded transit.

    The function extracts the central portion of the transit (approximately the
    middle 30‑40 % of the duration), computes the standard deviation and the
    linear slope of flux within that region, and combines them into a score that
    ranges from 0 (very V‑shaped) to 1 (perfectly flat).

    Parameters
    ----------
    phase : np.ndarray
        Folded phase values (0‑1 range).
    flux : np.ndarray
        Corresponding flux values.
    transit_center : float
        Phase value at the centre of the transit.
    transit_duration : float
        Full transit duration expressed in phase units.

    Returns
    -------
    dict
        ``{"flat_bottom_score": float,
            "bottom_std": float,
            "bottom_slope": float}``
        * ``flat_bottom_score`` – composite flatness metric (0‑1).
        * ``bottom_std`` – standard deviation of flux in the bottom region.
        * ``bottom_slope`` – linear slope (flux per phase) of the bottom region.
    """
    # Convert inputs to NumPy arrays for safety
    phase = np.asarray(phase)
    flux = np.asarray(flux)

    if transit_duration <= 0:
        logging.warning("Transit duration must be positive for flat_bottom_score.")
        return {"flat_bottom_score": float('nan'), "bottom_std": float('nan'), "bottom_slope": float('nan')}

    # Define the central window: take ~35 % of the duration centred on transit_center
    # This yields a width of 0.35 * transit_duration (within the 30‑40 % target range).
    half_width = 0.35 * transit_duration / 2.0
    lower = transit_center - half_width
    upper = transit_center + half_width

    # Account for phase wrapping (phase is 0‑1 cyclic)
    if lower < 0 or upper > 1:
        # Wrap values into 0‑1 interval
        phase_wrapped = np.mod(phase - lower, 1.0) + lower
        mask = (phase_wrapped >= lower) & (phase_wrapped <= upper)
    else:
        mask = (phase >= lower) & (phase <= upper)

    bottom_phase = phase[mask]
    bottom_flux = flux[mask]

    if bottom_flux.size == 0:
        logging.warning("No points found in the central transit region for flat_bottom_score.")
        return {"flat_bottom_score": float('nan'), "bottom_std": float('nan'), "bottom_slope": float('nan')}

    if bottom_flux.size < 3:
        # Not enough points for a reliable slope estimate
        logging.warning("Too few points (<3) in bottom region; slope will be set to 0.")
        slope = 0.0
    else:
        # Linear fit: flux = a * phase + b ; slope = a
        coeffs = np.polyfit(bottom_phase, bottom_flux, deg=1)
        slope = coeffs[0]

    # Standard deviation of the bottom region (measure of flatness)
    std = np.std(bottom_flux)

    # Composite score: larger std or larger |slope| reduces the score.
    # Adding 1 to denominator guarantees the score stays in (0, 1].
    # The formulation is simple yet effective for relative comparisons.
    flat_score = 1.0 / (1.0 + std + abs(slope))

    logging.debug(
        f"Flat‑bottom region: std={std:.5f}, slope={slope:.5e}, score={flat_score:.4f}"
    )

    return {
        "flat_bottom_score": float(flat_score),
        "bottom_std": float(std),
        "bottom_slope": float(slope),
    }
