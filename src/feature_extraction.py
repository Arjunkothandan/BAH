import numpy as np
from astropy.timeseries import BoxLeastSquares

def extract_features(lc):

    # Remove NaNs
    lc = lc.remove_nans()

    # Normalize
    lc = lc.normalize()

    time = lc.time.value
    flux = lc.flux.value

    mean_flux = np.mean(flux)
    std_flux = np.std(flux)

    depth = np.max(flux) - np.min(flux)
    snr = depth / std_flux if std_flux != 0 else 0

    durations = np.linspace(0.05, 0.30, 10)

    bls = BoxLeastSquares(time, flux)
    results = bls.autopower(durations)

    idx = np.argmax(results.power)

    return {
        "period": float(results.period[idx]),
        "duration": float(results.duration[idx]),
        "depth": float(depth),
        "snr": float(snr),
        "significance": float(results.power[idx])
    }