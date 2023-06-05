import numpy as np
import torch

from core.raft import RAFT
from core.evaluation.uncertainty.sparsification_metrics import compute_sparsification, compute_sparsification_oracle


def load_model(checkpoint_path, args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(checkpoint_path))

    model = model.module
    model.to('cuda')
    model.eval()
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    return model


def endpoint_error_numpy(flow_pred, flow_gt, valid_mask=None):
    """ Compute EPE metric between predicted flow and GT flow.

    Args:
        flow_pred (np.ndarray): Predicted flow, shape [2, H, W]
        flow_gt (np.ndarray): GT flow, shape [2, H, W]
        valid_mask (np.ndarray, optional): Flow validity mask, shape [H, W]. Defaults to None.

    Returns:
        EPE for each pixel, flattened to shape [H*W]
    """
    epe = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=0))

    if valid_mask is not None:
        return epe * valid_mask
    return epe


def compute_metrics(pred_flow, pred_flow_var, gt_flow, flow_valid_mask=None):
    flow_epe = endpoint_error_numpy(pred_flow, gt_flow, valid_mask=flow_valid_mask)

    u_flow_var, v_flow_var = pred_flow_var[0, :, :], pred_flow_var[1, :, :]
    flow_uncertainty =  (u_flow_var + v_flow_var) / 2

    epe_vals_oracle = compute_sparsification_oracle(flow_epe, flow_valid_mask)
    epe_vals = compute_sparsification(flow_epe, flow_uncertainty, flow_valid_mask)

    return epe_vals, epe_vals_oracle


def compute_flow_variance_single_pass(pred_flow_list):
    num_pred = len(pred_flow_list)
    sum = np.zeros_like(pred_flow_list[0])
    sum_sq = np.zeros_like(pred_flow_list[0])
    for pred_flow in pred_flow_list:
        sum += pred_flow
        sum_sq += pred_flow ** 2
    
    pred_flow_mean = sum / num_pred
    pred_flow_var = (sum_sq - sum ** 2 / num_pred) / (num_pred - 1)

    return pred_flow_mean, pred_flow_var


def compute_flow_variance_two_pass(pred_flow_list):
    num_pred = len(pred_flow_list)
    sum = np.zeros_like(pred_flow_list[0])
    sum_sq = np.zeros_like(pred_flow_list[0])

    for pred_flow in pred_flow_list:
        sum += pred_flow
    
    pred_flow_mean = sum / num_pred

    for pred_flow in pred_flow_list:
        sum_sq += (pred_flow_mean - pred_flow) ** 2
    
    pred_flow_var = sum_sq / (num_pred - 1)

    return pred_flow_mean, pred_flow_var


def get_flow_confidence(flow_var):
    # Compute total variance
    u_flow_var, v_flow_var = flow_var[0, :, :], flow_var[1, :, :]
    var = u_flow_var + v_flow_var

    # Normalize
    max_var = np.percentile(var, 99.9)
    epsilon = 1e-5
    var = var / (max_var + epsilon)

    return var


def get_flow_confidence_exp(flow_var):
    # Compute total variance
    u_flow_var, v_flow_var = np.exp(flow_var[0, :, :]), np.exp(flow_var[1, :, :])
    var = u_flow_var + v_flow_var

    # Normalize
    max_var = np.percentile(var, 99.9)
    epsilon = 1e-5
    var = var / (max_var + epsilon)

    return var
