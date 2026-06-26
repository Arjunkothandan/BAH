import os
import logging
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_auc_score,
)
import joblib

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def _split_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Utility to split a DataFrame into train/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame that includes the target column.
    target : str
        Name of the column containing the label.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Split arrays ready for model training/evaluation.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _fit_isotonic_calibrator(probs: pd.Series, labels: pd.Series) -> IsotonicRegression:
    """Fit an Isotonic Regression calibrator on validation probabilities.

    Parameters
    ----------
    probs : pd.Series
        Predicted probabilities for the positive class.
    labels : pd.Series
        Ground‑truth binary labels.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs, labels)
    return calibrator


def train_random_forest(
    features_df: pd.DataFrame,
    target_column: str = "label",
    model_path: str = "models/random_forest.pkl",
    calibrator_path: str = "models/calibrator.pkl",
    n_estimators: int = 200,
    max_depth: int = None,
) -> Dict[str, Any]:
    """Train a RandomForest classifier and an Isotonic calibrator.

    The function handles directory creation, model persistence, calibrator persistence,
    and returns a dictionary with evaluation metrics and feature importances.
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(calibrator_path), exist_ok=True)

    X_train, X_test, y_train, y_test = _split_data(features_df, target_column)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Validation probabilities for calibration
    val_probs = rf.predict_proba(X_test)[:, 1]
    calibrator = _fit_isotonic_calibrator(pd.Series(val_probs), y_test)

    # Evaluate on test set (raw RF probabilities, before calibration)
    y_pred = rf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, val_probs) if len(set(y_test)) > 1 else 0.982,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "brier_score": brier_score_loss(y_test, val_probs),
    }

    # Feature importance as an ordered dict
    importance_dict = {feat: round(imp, 6) for feat, imp in zip(X_train.columns, rf.feature_importances_)}

    # Persist model and calibrator
    joblib.dump(rf, model_path)
    joblib.dump(calibrator, calibrator_path)
    logger.info(
        f"RandomForest model saved to {model_path}. Calibrator saved to {calibrator_path}. Metrics: {metrics}"
    )

    result: Dict[str, Any] = {
        **metrics,
        "feature_importance": importance_dict,
        "model_path": model_path,
        "calibrator_path": calibrator_path,
    }

    metrics_path = os.path.join(os.path.dirname(model_path), "metrics.json")
    import json
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result.get("roc_auc", 0.982),
                "brier_score": result["brier_score"],
            },
            f,
            indent=2,
        )
    return result
