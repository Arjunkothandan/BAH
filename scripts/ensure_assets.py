#!/usr/bin/env python3
"""Ensure models and demo FITS assets exist for EXOAI."""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo")
TRAIN_CSV = os.path.join(
    PROJECT_ROOT,
    "AI_Exoplanet_Detection",
    "data",
    "processed",
    "training_dataset.csv",
)


def _train_models() -> None:
    import pandas as pd
    from src.model_training import train_random_forest

    if not os.path.isfile(TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    cal_path = os.path.join(MODEL_DIR, "calibrator.pkl")
    metrics = train_random_forest(df, model_path=model_path, calibrator_path=cal_path)
    print(f"Models trained — accuracy={metrics.get('accuracy', 0):.3f}")


def _create_demo_fits() -> None:
    from astropy.io import fits

    os.makedirs(DEMO_DIR, exist_ok=True)
    source = os.path.join(PROJECT_ROOT, "test_mock.fits")
    for name in ("demo_light_curve_1.fits", "demo_light_curve_2.fits"):
        dest = os.path.join(DEMO_DIR, name)
        if os.path.isfile(dest):
            continue
        if os.path.isfile(source):
            shutil.copy2(source, dest)
            continue
        # Synthetic fallback
        n = 2000
        t = np.linspace(0, 27, n)
        flux = np.ones(n) + np.random.normal(0, 0.0005, n)
        period = 3.5 if "1" in name else 8.2
        phase = (t % period) / period
        transit = (phase > 0.45) & (phase < 0.55)
        flux[transit] -= 0.003 if "1" in name else 0.0015
        fits.writeto(dest, flux, overwrite=True)
    print("Demo FITS files ready.")


def main() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    if not os.path.isfile(model_path):
        print("Training models...")
        _train_models()
    else:
        print("Models already exist.")

    _create_demo_fits()
    print("Asset check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
