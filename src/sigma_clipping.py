import numpy as np

def sigma_clip(time, flux, sigma=3):
    mean = np.mean(flux)
    std = np.std(flux)

    mask = np.abs(flux - mean) < sigma * std

    return time[mask], flux[mask]