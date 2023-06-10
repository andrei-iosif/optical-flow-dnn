import numpy as np


def compute_sparsification_oracle(flow_epe_img: np.ndarray, flow_valid_mask=None):
    """ Compute oracle sparsification values.
    Rank pixels by endpoint error, then continuously remove pixels with highest error and 
    compute EPE of remaining pixels.

    Args:
        flow_epe_img (np.ndarray): 2D array with EPE value for each pixel; shape [H, W]
        flow_valid_mask (np.ndarray): 2D with valid flag for each pixel; shape [H, W]
    Returns:
        Mean EPE for different fractions of pixels removed; values normalized to [0, 1] interval
    """
    removed_pixels_fraction = np.arange(0, 1, 0.1)

    # If valid mask is given, keep only EPE values for valid pixels
    if flow_valid_mask is not None:
        valid_mask = (flow_valid_mask >= 0.5)
        flow_epe_arr = flow_epe_img[valid_mask]
    else:
        flow_epe_arr = flow_epe_img.flatten()

    # Sort pixels based on EPE, decreasing order
    flow_epe_sorted = np.sort(flow_epe_arr)[::-1]

    epe_vals = []
    for f in removed_pixels_fraction:
        idx = round(f * len(flow_epe_sorted))
        mean_epe = np.mean(flow_epe_sorted[idx:])
        epe_vals.append(mean_epe)

    epe_vals = np.array(epe_vals)
    epe_vals_norm = epe_vals / np.max(epe_vals)
    return epe_vals_norm


def compute_sparsification(flow_epe_img: np.ndarray, flow_uncertainty: np.ndarray, flow_valid_mask=None):
    """ Compute sparsification values.
    Rank pixels by uncertainty, then continuously remove pixels with highest uncertainty and 
    compute EPE of remaining pixels.

    Args:
        flow_epe_img (np.ndarray): 2D array with EPE value for each pixel; shape [H, W]
        flow_uncertainty (np.ndarray): 2D array with flow uncertainty for each pixel; shape [H, W]
        flow_valid_mask (np.ndarray): 2D with valid flag for each pixel; shape [H, W]
    Returns:
        Mean EPE for different fractions of pixels removed; values normalized to [0, 1] interval
    """
    removed_pixels_fraction = np.arange(0, 1, 0.1)

    # If valid mask is given, keep only EPE values for valid pixels
    if flow_valid_mask is not None:
        valid_mask = (flow_valid_mask >= 0.5)
        flow_epe_flat = flow_epe_img[valid_mask]
        flow_uncertainty_flat = flow_uncertainty[valid_mask]
    else:
        flow_epe_flat = flow_epe_img.flatten()
        flow_uncertainty_flat = flow_uncertainty.flatten()

    # Sort pixels based on uncertainty, decreasing order => flattened array 
    sorted_idx = np.argsort(flow_uncertainty_flat, axis=None)[::-1]
    flow_epe_sorted = flow_epe_flat[sorted_idx]

    epe_vals = []
    for f in removed_pixels_fraction:
        idx = round(f * len(flow_epe_sorted))
        mean_epe = np.mean(flow_epe_sorted[idx:])
        epe_vals.append(mean_epe)

    epe_vals = np.array(epe_vals)
    epe_vals_norm = epe_vals / np.max(epe_vals)
    return epe_vals_norm


def compute_ause_metric(epe_vals, epe_vals_oracle):
    """ Compute AUSE metric, defined as area under the sparsification error curve. 
    Sparsification error is the difference between sparsification and its oracle.

    Args:
        epe_vals (np.ndarray): Sparsification curve (EPE values when pixels with highest uncertainty are removed)
        epe_vals_oracle (np.ndarray): Oracle sparsification curve (EPE values when pixels with highest EPE are removed)
    Returns:
        Float value
    """
    x = np.arange(0, 1, 0.1)
    sparsification_error = np.abs(epe_vals - epe_vals_oracle)
    ause = np.trapz(sparsification_error, x=x)
    return ause
