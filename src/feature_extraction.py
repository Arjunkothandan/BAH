import logging
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

from .agreement_score import agreement_score
from .autocorrelation import autocorrelation_period
from .detrending import detrend
from .flat_bottom import flat_bottom_score
from .lomb_scargle import lomb_scargle_period
from .phase_folding import phase_fold
from .shape_symmetry import shape_symmetry_score
from .sigma_clipping import sigma_clip
from .transit_quality_index import transit_quality_index

def extract_features(lc):
    """Extract transit features from a LightCurve with full preprocessing.

    Steps:
        1. Remove NaNs
        2. Normalize
        3. Detrend (Savitzky‑Golay / rolling median)
        4. Sigma‑clip outliers
        5. Box‑Least‑Squares period search
        6. Lomb‑Scargle periodogram
        7. Autocorrelation period analysis
        8. Agreement scoring
        9. Phase folding & morphology metrics (shape symmetry, flat bottom)
    """
    # ---------------------------------------------------------------
    # 1. Basic cleaning
    # ---------------------------------------------------------------
    original_points = len(lc.time)
    lc = lc.remove_nans()
    lc = lc.normalize()

    # ---------------------------------------------------------------
    # 2. Extract arrays for custom preprocessing
    # ---------------------------------------------------------------
    time = lc.time.value
    flux = lc.flux.value

    # ---------------------------------------------------------------
    # 3. Detrending (operates on flux array only)
    # ---------------------------------------------------------------
    flux_detrended = detrend(flux)

    # ---------------------------------------------------------------
    # 4. Sigma‑clipping (returns masked time & flux)
    # ---------------------------------------------------------------
    time_clean, flux_clean = sigma_clip(time, flux_detrended)
    cleaned_points = len(time_clean)

    # ---------------------------------------------------------------
    # 5. Re‑wrap into a LightCurve for downstream astronomy tools
    # ---------------------------------------------------------------
    lc_clean = lk.LightCurve(time=time_clean, flux=flux_clean)

    # ---------------------------------------------------------------
    # 6. BLS period search
    # ---------------------------------------------------------------
    durations = np.linspace(0.05, 0.30, 10)
    bls = BoxLeastSquares(lc_clean.time.value, lc_clean.flux.value)
    results = bls.autopower(durations)
    bls_idx = np.argmax(results.power)
    bls_period = float(results.period[bls_idx])
    bls_duration = float(results.duration[bls_idx])
    bls_power = float(results.power[bls_idx])

    # ---------------------------------------------------------------
    # 7. Lomb‑Scargle period
    # ---------------------------------------------------------------
    ls_res = lomb_scargle_period(lc_clean)
    ls_period = ls_res["best_period"]
    ls_power = ls_res["power"]

    # ---------------------------------------------------------------
    # 8. Autocorrelation period
    # ---------------------------------------------------------------
    ac_res = autocorrelation_period(lc_clean)
    ac_period = ac_res["best_period"]
    ac_corr = ac_res["correlation"]

    # ---------------------------------------------------------------
    # 9. Agreement score
    # ---------------------------------------------------------------
    agreement = agreement_score(bls_period, ls_period, ac_period)

    # ---------------------------------------------------------------
    # 10. Phase folding and morphology metrics
    # ---------------------------------------------------------------
    # Fold the cleaned light curve on the BLS period
    phase, folded_flux = phase_fold(lc_clean.time.value, lc_clean.flux.value, bls_period)

    if phase.size == 0:
        logging.warning("Phase array empty after folding; morphology scores set to defaults.")
        shape_score = 0.0
        flat_score = 0.0
        bottom_std = 0.0
        bottom_slope = 0.0
    else:
        transit_center = phase[np.argmin(folded_flux)]

        # Shape symmetry score
        try:
            shape_res = shape_symmetry_score(phase, folded_flux, transit_center)
            shape_score = float(shape_res.get("shape_score", 0.0))
            if np.isnan(shape_score):
                shape_score = 0.0
        except Exception as e:
            logging.error(f"Shape symmetry calculation failed: {e}")
            shape_score = 0.0

        # Flat‑bottom score (uses BLS duration as transit duration)
        try:
            flat_res = flat_bottom_score(
                phase, folded_flux, transit_center, bls_duration
            )
            flat_score = float(flat_res.get("flat_bottom_score", 0.0))
            bottom_std = float(flat_res.get("bottom_std", 0.0))
            bottom_slope = float(flat_res.get("bottom_slope", 0.0))
            # Guard against NaNs
            if np.isnan(flat_score):
                flat_score = 0.0
            if np.isnan(bottom_std):
                bottom_std = 0.0
            if np.isnan(bottom_slope):
                bottom_slope = 0.0
        except Exception as e:
            logging.error(f"Flat‑bottom calculation failed: {e}")
            flat_score = 0.0
            bottom_std = 0.0
            bottom_slope = 0.0

    # ---------------------------------------------------------------
    # 11. Logging summary
    # ---------------------------------------------------------------
    logging.info(
        f"BLS={bls_period:.4f}, LS={ls_period:.4f}, AC={ac_period:.4f}, Agreement={agreement:.3f}"
    )
    logging.info(
        f"Points: original={original_points}, after_sigma_clip={cleaned_points}"
    )
    logging.info(f"Shape score: {shape_score:.3f}")
    logging.info(f"Flat‑bottom score: {flat_score:.3f}")

    # ---------------------------------------------------------------
    # 12. Depth & SNR based on cleaned flux
    # ---------------------------------------------------------------
    depth = np.max(flux_clean) - np.min(flux_clean)
    std_flux = np.std(flux_clean)
    snr = depth / std_flux if std_flux != 0 else 0.0

    # ---------------------------------------------------------------
    # 13. Transit Quality Index (TQI)
    # ---------------------------------------------------------------
    tqi_res = transit_quality_index(
        snr=snr,
        significance=bls_power,
        agreement_score=agreement,
        shape_score=shape_score,
        flat_bottom_score=flat_score,
    )
    tqi = tqi_res["tqi"]
    candidate_label = tqi_res["label"]
    comps = tqi_res["components"]
    logging.info(f"TQI={tqi:.3f} | Label={candidate_label}")

    # ---------------------------------------------------------------
    # 14. Return dictionary with all Phase 2 keys, morphology metrics, and TQI
    # ---------------------------------------------------------------
    return {
        "period": bls_period,
        "duration": bls_duration,
        "depth": float(depth),
        "snr": float(snr),
        "significance": bls_power,
        "ls_period": float(ls_period),
        "ac_period": float(ac_period),
        "agreement_score": float(agreement),
        "shape_score": float(shape_score),
        "flat_bottom_score": float(flat_score),
        "bottom_std": float(bottom_std),
        "bottom_slope": float(bottom_slope),
        "tqi": float(tqi),
        "candidate_label": candidate_label,
        "tqi_snr": float(comps.get("snr", 0.0)),
        "tqi_significance": float(comps.get("significance", 0.0)),
        "tqi_agreement": float(comps.get("agreement", 0.0)),
        "tqi_shape": float(comps.get("shape", 0.0)),
        "tqi_flat_bottom": float(comps.get("flat_bottom", 0.0)),
    }