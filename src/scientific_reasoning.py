import os
import logging
import yaml
from typing import Dict, List, Any

# Path to the YAML configuration with thresholds
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "reasoning_thresholds.yaml")

_logger = logging.getLogger(__name__)


def _load_thresholds() -> Dict[str, Any]:
    """Load the reasoning thresholds from the YAML configuration file.

    Returns
    -------
    dict
        Nested dictionary containing low/moderate/high (or weak/moderate/strong) cutoffs.
    """
    if not os.path.isfile(THRESHOLDS_PATH):
        _logger.error(f"Threshold configuration file not found at {THRESHOLDS_PATH}")
        raise FileNotFoundError(f"Threshold config missing: {THRESHOLDS_PATH}")
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _categorize(value: float, levels: Dict[str, float]) -> str:
    """Return a textual category based on the provided levels.

    Parameters
    ----------
    value : float
        Numeric metric to categorize.
    levels : dict
        Mapping like {"low": 5, "moderate": 10, "high": 15} or {"weak": 0.4, "moderate": 0.6, "strong": 0.8}.

    Returns
    -------
    str
        The highest matching category (e.g., "high" or "strong").
    """
    high_cutoff = levels.get("high", levels.get("strong", float('inf')))
    if value >= high_cutoff:
        return "strong" if "strong" in levels else "high"
    moderate_cutoff = levels.get("moderate", float('inf'))
    if value >= moderate_cutoff:
        return "moderate"
    return "weak" if "weak" in levels else "low"


def generate_reasoning(features: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a human‑readable scientific explanation for a candidate.

    The function uses the configurable thresholds from ``config/reasoning_thresholds.yaml``
    to turn quantitative feature values into qualitative statements.

    Parameters
    ----------
    features : dict
        Dictionary returned by ``extract_features`` (must contain at least the keys
        used in the thresholds configuration).

    Returns
    -------
    dict
        ``{"summary": str, "strengths": List[str], "warnings": List[str]}``
    """
    thresholds = _load_thresholds()
    strengths: List[str] = []
    warnings: List[str] = []

    # Example categories – expand as needed
    snr_cat = _categorize(float(features.get("snr", 0.0)), thresholds.get("snr", {}))
    if snr_cat in ("moderate", "high"):
        strengths.append(f"High signal‑to‑noise ratio ({features.get('snr'):.2f})")
    else:
        warnings.append(f"Low SNR ({features.get('snr'):.2f})")

    cons_cat = _categorize(float(features.get("agreement_score", 0.0)), thresholds.get("consistency_score", {}))
    if cons_cat in ("moderate", "high"):
        strengths.append(f"Strong transit consistency ({features.get('agreement_score'):.2f})")
    else:
        warnings.append(f"Weak transit consistency ({features.get('agreement_score'):.2f})")

    tqi_cat = _categorize(float(features.get("tqi", 0.0)), thresholds.get("tqi", {}))
    if tqi_cat == "strong":
        strengths.append(f"Excellent Transit Quality Index ({features.get('tqi'):.2f})")
    elif tqi_cat == "moderate":
        strengths.append(f"Good Transit Quality Index ({features.get('tqi'):.2f})")
    else:
        warnings.append(f"Low TQI ({features.get('tqi'):.2f})")

    # Add optional checks when data is available
    if features.get("secondary_eclipse_detected") is True:
        warnings.append("Secondary eclipse detected — possible false positive.")
    elif features.get("secondary_eclipse_detected") is False:
        strengths.append("No secondary eclipse detected.")

    if "odd_even_ratio" in features:
        ratio = float(features["odd_even_ratio"])
        if 0.9 <= ratio <= 1.1:
            strengths.append(f"Odd‑even depth ratio close to unity ({ratio:.2f})")
        else:
            warnings.append(f"Odd‑even depth ratio deviates from unity ({ratio:.2f})")

    # Assemble a concise summary paragraph
    summary_parts = strengths + warnings
    summary = " • ".join(summary_parts)

    return {"summary": summary, "strengths": strengths, "warnings": warnings}
