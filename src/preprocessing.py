import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def clean_lightcurve(lc):
    """
    Input: lightkurve LightCurve object
    Output: cleaned LightCurve
    """

    df = lc.to_pandas()

    # pick flux column
    flux_col = None
    for col in ["pdcsap_flux", "sap_flux", "flux"]:
        if col in df.columns:
            flux_col = col
            break

    if flux_col is None:
        raise ValueError("No flux column found")

    df = df.dropna(subset=[flux_col])

    time = np.arange(len(df))  # simple index-based time
    flux = df[flux_col].values

    # -------------------------
    # 1. DETRENDING
    # -------------------------
    trend = savgol_filter(flux, 51, 2)
    flux_detrended = flux - trend

    # -------------------------
    # 2. SIGMA CLIPPING
    # -------------------------
    mean = np.mean(flux_detrended)
    std = np.std(flux_detrended)

    mask = np.abs(flux_detrended - mean) < 3 * std

    time_clean = time[mask]
    flux_clean = flux_detrended[mask]

    # rebuild pseudo-lightcurve dataframe
    clean_df = pd.DataFrame({
        "time": time_clean,
        "flux": flux_clean
    })

    return clean_df