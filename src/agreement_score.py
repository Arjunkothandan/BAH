import numpy as np
import logging

def agreement_score(bls_period: float, ls_period: float, ac_period: float) -> float:
    """Calculate a simple agreement score among three period estimates.

    The score is defined as ``1 - (relative standard deviation)`` of the three
    periods, clipped to the range ``[0, 1]``.  A score of 1 means perfect
    agreement, while values near 0 indicate large disagreement.
    """
    periods = np.array([bls_period, ls_period, ac_period], dtype=float)
    mean = np.mean(periods)
    if mean == 0:
        logging.warning("Mean period is zero while computing agreement score.")
        return 0.0
    rel_std = np.std(periods) / mean
    score = max(0.0, 1.0 - rel_std)
    return float(score)
