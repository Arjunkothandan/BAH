#!/usr/bin/env python3
"""Validate project layout, imports, and assets before launching the Streamlit app."""

from __future__ import annotations

import importlib
import os
import sys
import traceback

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

REQUIRED_PATHS = [
    "src/__init__.py",
    "src/dashboard_components.py",
    "src/predictor.py",
    "src/feature_extraction.py",
    "src/explainability.py",
    "src/scientific_reasoning.py",
    "src/confidence_calibration.py",
    "src/candidate_report.py",
    "config/reasoning_thresholds.yaml",
    "models/random_forest.pkl",
    "models/calibrator.pkl",
    "models/metrics.json",
    "assets/css/futuristic_theme.css",
    "demo/demo_light_curve_1.fits",
    "demo/demo_light_curve_2.fits",
    "app_ai_explanation.py",
]

SRC_MODULES = [
    "src",
    "src.utils",
    "src.detrending",
    "src.sigma_clipping",
    "src.lomb_scargle",
    "src.autocorrelation",
    "src.agreement_score",
    "src.phase_folding",
    "src.shape_symmetry",
    "src.flat_bottom",
    "src.transit_quality_index",
    "src.feature_extraction",
    "src.explainability",
    "src.scientific_reasoning",
    "src.confidence_calibration",
    "src.candidate_report",
    "src.predictor",
    "src.prediction",
    "src.dashboard_components",
]

THIRD_PARTY = [
    "streamlit",
    "lightkurve",
    "astropy",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "shap",
    "plotly",
    "yaml",
    "jinja2",
    "reportlab",
]


def check_paths() -> list[str]:
    errors: list[str] = []
    for rel_path in REQUIRED_PATHS:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            errors.append(f"Missing required path: {rel_path}")
    return errors


def check_third_party() -> list[str]:
    errors: list[str] = []
    for module_name in THIRD_PARTY:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"Missing dependency '{module_name}': {exc}")
    return errors


def check_src_modules() -> list[str]:
    errors: list[str] = []
    for module_name in SRC_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"Failed to import {module_name}: {exc}")
            traceback.print_exc()
    return errors


def check_dashboard_api() -> list[str]:
    errors: list[str] = []
    try:
        from src.dashboard_components import generate_report, predict_and_explain

        if not callable(predict_and_explain):
            errors.append("predict_and_explain is not callable")
        if not callable(generate_report):
            errors.append("generate_report is not callable")
    except Exception as exc:
        errors.append(f"Dashboard API import failed: {exc}")
        traceback.print_exc()
    return errors


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print("Running startup validation...\n")

    all_errors: list[str] = []
    for label, checker in (
        ("Required paths", check_paths),
        ("Third-party dependencies", check_third_party),
        ("src package modules", check_src_modules),
        ("Dashboard API", check_dashboard_api),
    ):
        print(f"[{label}]")
        errors = checker()
        if errors:
            for err in errors:
                print(f"  FAIL: {err}")
            all_errors.extend(errors)
        else:
            print("  OK")
        print()

    if all_errors:
        print(f"Validation failed with {len(all_errors)} error(s).")
        print("Install dependencies: pip install -r requirements.txt")
        return 1

    print("All checks passed. Launch with:")
    print("  streamlit run app_ai_explanation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
