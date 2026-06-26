import pandas as pd
from scipy.signal import savgol_filter

def detrend(flux, method="rolling", window=51):
    if method == "rolling":
        flux_series = pd.Series(flux)
        trend = flux_series.rolling(window=window, center=True).median()
        detrended = flux_series - trend
        return detrended.bfill().ffill().values

    elif method == "savgol":
        trend = savgol_filter(flux, window, 2)
        return flux - trend