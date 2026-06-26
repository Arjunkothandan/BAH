import logging
import numpy as np
from typing import Dict

# ---------------------------------------------------------------------------
# Normalization helpers (astronomy‑friendly scaling)
# ---------------------------------------------------------------------------

def normalize_snr(snr: float, max_snr: float = 20.0) -> float:
    """Normalize SNR to the interval [0, 1].

    The raw SNR can be > 1 for strong detections; scaling by a typical
    high‑SNR value (20) retains ranking power for very strong candidates while
    still capping at 1.0.
    """
    if np.isnan(snr) or snr < 0:
        return 0.0
    return float(min(snr / max_snr, 1.0))


def normalize_significance(signif: float, max_signif: float = 15.0) -> float:
    """Normalize BLS significance (power) to [0, 1].

    A typical strong BLS power is around 15; values above that are capped.
    """
    if np.isnan(signif) or signif < 0:
        return 0.0
    return float(min(signif / max_signif, 1.0))


def normalize_shape(shape_score: float) -> float:
    """Map shape symmetry from [-1, 1] → [0, 1].

    ``shape_score`` comes from Pearson correlation and can be negative for
    strongly asymmetric transits.  The linear mapping keeps the full dynamic
    range while ensuring the final value is bounded.
    """
    if np.isnan(shape_score):
        return 0.0
    # Clip to the theoretical range before mapping to avoid out‑of‑bounds
    clipped = max(-1.0, min(1.0, shape_score))
    return float((clipped + 1.0) / 2.0)


def transit_quality_index(
    snr: float,
    significance: float,
    agreement_score: float,
    shape_score: float,
    flat_bottom_score: float,
) -> Dict:
    """Compute the Transit Quality Index (TQI) with astrophysics‑friendly scaling.

    The TQI is a weighted sum of five quality indicators.  Normalisation
    strategies are chosen to preserve ranking power for strong detections:

    * **SNR** – scaled by a typical high value of 20 (``snr / 20``).
    * **Significance** – scaled by a typical high BLS power of 15
      (``significance / 15``).
    * **Agreement** – already in ``[0, 1]``.
    * **Shape** – Pearson correlation may be ``[-1, 1]``; linearly map to
      ``[0, 1]``.
    * **Flat Bottom** – already in ``[0, 1]``.

    After weighting, a confidence *label* is attached to the final score.

    Parameters
    ----------
    snr : float
        Signal‑to‑Noise Ratio.
    significance : float
        BLS power or equivalent significance metric.
    agreement_score : float
        Period‑agreement score (0‑1).
    shape_score : float
        Symmetry correlation (may be ``[-1, 1]``).
    flat_bottom_score : float
        Flat‑bottom metric (0‑1).

    Returns
    -------
    dict
        {
            "tqi": float,
            "label": str,
            "components": {
                "snr": float,
                "significance": float,
                "agreement": float,
                "shape": float,
                "flat_bottom": float
            }
        }
    """
    # Normalise each component according to the new strategy
    snr_norm = normalize_snr(snr)
    signif_norm = normalize_significance(significance)
    agreement_norm = 0.0 if np.isnan(agreement_score) else float(agreement_score)
    shape_norm = normalize_shape(shape_score)
    flat_norm = 0.0 if np.isnan(flat_bottom_score) else float(flat_bottom_score)

    # Weighted sum (weights are fixed per specification)
    tqi = (
        0.30 * snr_norm
        + 0.25 * signif_norm
        + 0.20 * agreement_norm
        + 0.15 * shape_norm
        + 0.10 * flat_norm
    )
    # Clamp to [0, 1] to guard against tiny numerical overshoot
    tqi = float(min(max(tqi, 0.0), 1.0))

    # Confidence label
    if tqi >= 0.80:
        label = "Strong Candidate"
    elif tqi >= 0.60:
        label = "Moderate Candidate"
    elif tqi >= 0.40:
        label = "Borderline"
    else:
        label = "Likely False Positive"

    logging.debug(
        "TQI components – snr: %.3f, significance: %.3f, agreement: %.3f, "
        "shape: %.3f, flat_bottom: %.3f, combined: %.3f, label: %s",
        snr_norm,
        signif_norm,
        agreement_norm,
        shape_norm,
        flat_norm,
        tqi,
        label,
    )
    logging.info(f"Transit Quality Index (TQI) = {tqi:.3f} – {label}")

    return {
        "tqi": tqi,
        "label": label,
        "components": {
            "snr": snr_norm,
            "significance": signif_norm,
            "agreement": agreement_norm,
            "shape": shape_norm,
            "flat_bottom": flat_norm,
        },
    }
