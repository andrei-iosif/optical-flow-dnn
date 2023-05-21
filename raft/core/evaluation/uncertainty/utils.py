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
    return epe


def compute_metrics(pred_flow, pred_flow_var, gt_flow):
    flow_epe = endpoint_error_numpy(pred_flow, gt_flow)

    u_flow_var, v_flow_var = pred_flow_var[0, :, :], pred_flow_var[1, :, :]
    flow_uncertainty =  (u_flow_var + v_flow_var) / 2

    epe_vals_oracle = compute_sparsification_oracle(flow_epe)
    epe_vals = compute_sparsification(flow_epe, flow_uncertainty)

    return epe_vals, epe_vals_oracle
