#!/usr/bin/env python3
"""Benchmark EXOAI inference and report generation."""

from __future__ import annotations

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.dashboard_components import generate_report, predict_and_explain

DEMO = os.path.join(PROJECT_ROOT, "demo", "demo_light_curve_1.fits")
with open(DEMO, "rb") as f:
    data = f.read()

t0 = time.perf_counter()
result = predict_and_explain(data, source_label="Demo 1", target_id="DEMO-1")
t_pred = time.perf_counter() - t0
print(f"Prediction: {t_pred:.2f}s | class={result['prediction']} | conf={result.get('calibrated_probability', 0):.3f}")

t1 = time.perf_counter()
pdf = generate_report(result, "pdf")
t_rep = time.perf_counter() - t1
print(f"Report PDF: {t_rep:.2f}s | {len(pdf)} bytes")

for fmt in ("html", "md", "txt"):
    b = generate_report(result, fmt)
    print(f"Report {fmt}: {len(b)} bytes OK")

# Second run — model should be cached
t0 = time.perf_counter()
predict_and_explain(data, source_label="Demo 1", target_id="DEMO-1")
t2 = time.perf_counter() - t0
print(f"Prediction (cached model): {t2:.2f}s")
