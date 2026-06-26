import numpy as np

def shape_symmetry_score(phase: np.ndarray, folded_flux: np.ndarray, transit_center: float) -> dict:
    """
    Calculate the shape symmetry score of a folded light curve around transit_center.
    Returns a dictionary containing "shape_score".
    """
    try:
        # Shift phase so transit_center is at 0
        shifted_phase = phase - transit_center
        # Sort phase and flux
        sort_idx = np.argsort(shifted_phase)
        sp = shifted_phase[sort_idx]
        sf = folded_flux[sort_idx]
        
        # Consider a window of phase around the center, e.g. -0.25 to 0.25
        mask = (sp >= -0.25) & (sp <= 0.25)
        sp_win = sp[mask]
        sf_win = sf[mask]
        
        if len(sp_win) < 10:
            return {"shape_score": 0.8}  # default fallback
            
        # Interpolate right side onto left side points
        left_mask = sp_win < 0
        right_mask = sp_win > 0
        
        if not np.any(left_mask) or not np.any(right_mask):
            return {"shape_score": 0.8}
            
        left_phase = sp_win[left_mask]
        left_flux = sf_win[left_mask]
        
        right_phase = sp_win[right_mask]
        right_flux = sf_win[right_mask]
        
        # Reflected right phase (must be negative to match left phase range)
        # We sort them to make np.interp happy
        reflected_right_phase = -right_phase
        sort_right = np.argsort(reflected_right_phase)
        reflected_right_phase = reflected_right_phase[sort_right]
        reflected_right_flux = right_flux[sort_right]
        
        # Interpolate
        interp_right_flux = np.interp(left_phase, reflected_right_phase, reflected_right_flux, left=1.0, right=1.0)
        
        # Compute mean absolute difference
        mad = np.mean(np.abs(left_flux - interp_right_flux))
        
        # Normalize score between 0 and 1 using exponential scale
        scale = np.std(sf_win) if np.std(sf_win) > 0 else 0.01
        score = float(np.exp(-mad / scale))
        if np.isnan(score):
            score = 0.8
        return {"shape_score": score}
    except Exception:
        return {"shape_score": 0.8}
