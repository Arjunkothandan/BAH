import numpy as np

def phase_fold(time, flux, period):
    phase = (time % period) / period
    idx = np.argsort(phase)

    return phase[idx], flux[idx]