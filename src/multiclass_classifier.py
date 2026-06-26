"""Multi-class astrophysical classification engine (SIH requirement)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

ASTRO_CLASSES: List[str] = [
    "Planetary Transit",
    "Eclipsing Binary",
    "Stellar Blend",
    "Variable Star",
    "Starspot Modulation",
    "Noise",
    "False Positive",
    "Unknown",
]


def classify_multiclass(
    features: Dict[str, Any],
    ml_planet_probability: float,
) -> Dict[str, Any]:
    """
    Derive multi-class probabilities from morphology features and ML output.

    Uses a physics-informed heuristic ensemble; binary RF output anchors
    the Planetary Transit class while morphology drives other classes.
    """
    p = float(ml_planet_probability or 0.0)
    snr = float(features.get("snr", 0.0))
    tqi = float(features.get("tqi", 0.0))
    shape = float(features.get("shape_score", 0.0))
    flat = float(features.get("flat_bottom_score", 0.0))
    agreement = float(features.get("agreement_score", 0.0))
    significance = float(features.get("significance", 0.0))
    depth = float(features.get("depth", 0.0))

    raw = {
        "Planetary Transit": p * tqi * (0.5 + 0.5 * shape) * (0.5 + 0.5 * flat),
        "Eclipsing Binary": (1.0 - shape) * (1.0 - flat) * min(snr / 10.0, 1.0) * 0.4,
        "Stellar Blend": (1.0 - agreement) * 0.35 * min(depth * 100, 1.0),
        "Variable Star": (1.0 - flat) * agreement * 0.25 * (1.0 - p),
        "Starspot Modulation": (1.0 - flat) * 0.2 * min(significance, 1.0),
        "Noise": max(0.0, 1.0 - snr / 5.0) * (1.0 - tqi) * 0.5,
        "False Positive": (1.0 - agreement) * (1.0 - tqi) * significance * 0.3,
        "Unknown": 0.05,
    }

    total = sum(raw.values()) or 1.0
    probs = {k: v / total for k, v in raw.items()}

    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    primary_class, primary_prob = ranked[0]

    return {
        "class_probabilities": probs,
        "primary_class": primary_class,
        "primary_probability": primary_prob,
        "ranked_classes": ranked,
    }
