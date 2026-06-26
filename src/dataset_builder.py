import os
import logging
from typing import List, Any
import pandas as pd

# Import the feature extraction function from the existing module
from .feature_extraction import extract_features


def build_dataset(
    lightcurves: List[Any],
    labels: List[Any],
    output_csv: str = "data/features.csv",
) -> pd.DataFrame:
    """Build a feature dataset from LightCurve objects.

    Parameters
    ----------
    lightcurves : List[Any]
        List of `lightkurve.LightCurve` objects.
    labels : List[Any]
        Corresponding list of target labels (e.g., 0/1 or class names).
    output_csv : str, optional
        Path where the resulting CSV will be stored. Default is ``data/features.csv``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the extracted features and a ``label`` column.
    """
    if len(lightcurves) != len(labels):
        raise ValueError("The number of lightcurves must match the number of labels.")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    records = []
    for idx, (lc, label) in enumerate(zip(lightcurves, labels)):
        try:
            features = extract_features(lc)
            # Append the target label
            features["label"] = label
            records.append(features)
        except Exception as exc:
            logging.warning(
                f"Feature extraction failed for sample index {idx}: {exc}. Sample will be skipped."
            )
            continue

    if not records:
        raise RuntimeError("No features were successfully extracted; dataset is empty.")

    df = pd.DataFrame.from_records(records)
    # Save to CSV (index=False for a clean file)
    df.to_csv(output_csv, index=False)
    logging.info(f"Feature dataset saved to {output_csv} with {len(df)} samples.")
    return df
