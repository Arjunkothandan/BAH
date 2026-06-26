import logging
from functools import lru_cache
from typing import Dict, Any

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go

_logger = logging.getLogger(__name__)
TOP_FEATURES = 5
_explainer_cache: Dict[int, shap.TreeExplainer] = {}


def _get_explainer(model) -> shap.TreeExplainer:
    """Reuse TreeExplainer for the same model instance."""
    key = id(model)
    if key not in _explainer_cache:
        _explainer_cache[key] = shap.TreeExplainer(model)
    return _explainer_cache[key]


def compute_shap(model, sample_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute SHAP values for a single sample using a pre‑loaded model.

    Parameters
    ----------
    model : Any
        A tree‑based sklearn estimator (e.g., RandomForestClassifier).
    sample_df : pd.DataFrame
        DataFrame containing **exactly one** row with the same feature columns used during training.

    Returns
    -------
    dict
        A dictionary with keys:
        * ``top_positive_features`` – list of top contributing feature names (positive).
        * ``top_negative_features`` – list of top contributing feature names (negative).
        * ``feature_importance_local`` – mapping ``feature_name → shap_value``.
    """
    _logger.info("Computing SHAP values for a sample (model pre‑loaded)")
    explainer = _get_explainer(model)
    shap_vals = explainer.shap_values(sample_df)
    
    # Extract SHAP values for the positive class (class 1)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] > 1 else shap_vals[:, :, 0]
        
    # Since we have exactly one sample, take the first row (1D array of features)
    if hasattr(shap_vals, "ndim") and shap_vals.ndim == 2:
        shap_vals = shap_vals[0]
        
    feature_names = sample_df.columns.tolist()
    importance_dict = dict(zip(feature_names, shap_vals))

    # Sort by absolute contribution to find top contributors.
    sorted_features = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
    top_pos = [name for name, val in sorted_features if val > 0][:TOP_FEATURES]
    top_neg = [name for name, val in sorted_features if val < 0][:TOP_FEATURES]

    _logger.debug(
        "Top positive features: %s; Top negative features: %s", top_pos, top_neg
    )
    return {
        "top_positive_features": top_pos,
        "top_negative_features": top_neg,
        "feature_importance_local": importance_dict,
    }


def plot_shap_waterfall(shap_dict: Dict[str, float]) -> go.Figure:
    """Create a Plotly waterfall chart from a SHAP dictionary.

    Parameters
    ----------
    shap_dict : dict
        Mapping ``feature_name → shap_value`` for a single observation.

    Returns
    -------
    go.Figure
        Interactive waterfall figure suitable for embedding in Streamlit.
    """
    # Order features by absolute SHAP magnitude for visual clarity.
    sorted_items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
    features, values = zip(*sorted_items)

    fig = go.Figure(
        go.Waterfall(
            x=list(features),
            y=list(values),
            measure=["relative" for _ in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="SHAP Contribution Waterfall",
        yaxis_title="SHAP value",
        xaxis_tickangle=-45,
        autosize=True,
    )
    return fig
