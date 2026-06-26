"""Generate models/metrics.json from defaults if missing."""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.report_context import DEFAULT_METRICS

path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
if not os.path.isfile(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_METRICS, f, indent=2)
    print(f"Created {path}")
else:
    print(f"Exists: {path}")
