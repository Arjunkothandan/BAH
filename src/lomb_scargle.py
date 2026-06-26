import numpy as np
import logging
from astropy.timeseries import LombScargle


def lomb_scargle_period(
    lc,
    min_period: float = 0.5,
    max_period: float = 30.0,
) -> dict:
    """Compute Lomb‑Scargle periodogram for a LightCurve.

    Parameters
    ----------
    lc : LightCurve
        Lightkurve LightCurve object (cleaned & normalised).
    min_period, max_period : float
        Period search window in days.

    Returns
    -------
    dict
        {
            "best_period": float,   # period (days) with maximum power
            "power": float,         # corresponding Lomb‑Scargle power
        }
    """
    # Extract arrays
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    if time.size == 0 or flux.size == 0:
        raise ValueError("Empty LightCurve supplied to lomb_scargle_period")

    # Frequency limits corresponding to the period window
    f_min = 1.0 / max_period
    f_max = 1.0 / min_period

    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(minimum_frequency=f_min, maximum_frequency=f_max)

    # Guard against zero frequency (should not happen with the limits above)
    positive_mask = frequency > 0
    if not np.any(positive_mask):
        logging.warning("Lomb‑Scargle returned no positive frequencies")
        return {"best_period": float('nan'), "power": float('nan')}

    frequency = frequency[positive_mask]
    power = power[positive_mask]

    best_idx = np.argmax(power)
    best_freq = frequency[best_idx]
    best_period = 1.0 / best_freq
    best_power = power[best_idx]

    logging.info(f"LS best={best_period:.4f}, power={best_power:.3f}")
    return {"best_period": float(best_period), "power": float(best_power)}
