"""Theme injection and ambient background effects."""

from __future__ import annotations

import os
import random

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSS_PATH = os.path.join(PROJECT_ROOT, "assets", "css", "futuristic_theme.css")


def inject_theme() -> None:
    """Load and inject the futuristic CSS theme (once per session)."""
    if st.session_state.get("_theme_injected"):
        return
    with open(CSS_PATH, encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state._theme_injected = True


def render_starfield(count: int = 80) -> None:
    """Render an animated twinkling starfield layer (once per session)."""
    if st.session_state.get("_starfield_rendered"):
        return
    stars = []
    for _ in range(count):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        size = random.choice([1, 1, 2, 2, 3])
        dur = random.uniform(2, 6)
        op = random.uniform(0.2, 0.9)
        stars.append(
            f'<div class="star" style="left:{x}%;top:{y}%;width:{size}px;height:{size}px;'
            f'--dur:{dur}s;--op:{op}"></div>'
        )
    st.markdown(
        f'<div class="stars-layer">{"".join(stars)}</div>',
        unsafe_allow_html=True,
    )
    st.session_state._starfield_rendered = True
