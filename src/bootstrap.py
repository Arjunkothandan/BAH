"""Ensure D:\\BHA\\src is imported — not a shadowed copy from AI_Exoplanet_Detection."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPECTED_SRC_DIR = os.path.join(PROJECT_ROOT, "src")


def setup_project_path() -> str:
    """Put project root first on sys.path and evict a wrongly cached src package."""
    if PROJECT_ROOT in sys.path:
        sys.path.remove(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    src_mod = sys.modules.get("src")
    src_file = getattr(src_mod, "__file__", None)
    if src_file and os.path.abspath(os.path.dirname(src_file)) != os.path.abspath(EXPECTED_SRC_DIR):
        for key in list(sys.modules):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]

    return PROJECT_ROOT
