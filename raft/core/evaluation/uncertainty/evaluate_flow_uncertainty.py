import argparse
import numpy as np
import torch

import core.datasets as datasets
from core.raft import RAFT
from core.utils.utils import InputPadder
from core.evaluation.uncertainty.sparsification_metrics import compute_sparsification, compute_sparsification_oracle
from core.evaluation.uncertainty.sparsification_plots import sparsification_plot
from core.evaluation.metrics import Metrics


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


def run(args):
    iters = 24

    # Load model checkpoint
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()
    print(f"Loaded model from checkpoint: {args.model}")

    # Load dataset
    # dataset = datasets.FlyingChairs(split='validation', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/FlyingChairs')
    dataset = datasets.MpiSintel(split='training', dstype='clean', root=r'/home/mnegru/repos/optical-flow-dnn/raft/datasets/Sintel')
    print(f"Loaded dataset, size = {len(dataset)}")

    # Run inference
    results = Metrics()

    with torch.no_grad():
        for sample_id in range(len(dataset)):
            image_1, image_2, gt_flow, _ = dataset[sample_id]
            image_1 = image_1[None].cuda()
            image_2 = image_2[None].cuda()

            padder = InputPadder(image_1.shape, mode='kitti')
            image_1, image_2 = padder.pad(image_1, image_2)

            pred_flow_var, pred_flow = model(image_1, image_2, iters=iters, test_mode=True)

            pred_flow = padder.unpad(pred_flow[0]).cpu().numpy()
            pred_flow_var = padder.unpad(pred_flow_var[0]).cpu().numpy()
            image_1 = image_1[0].cpu().numpy()
            gt_flow = gt_flow.cpu().numpy()

            epe_vals, epe_vals_oracle = compute_metrics(pred_flow, pred_flow_var, gt_flow)
            results.add(sample_id, "epe_vals", epe_vals)
            results.add(sample_id, "epe_vals_oracle", epe_vals_oracle)
            print(f"Processed sample id={sample_id}")
    
    results.save_pickle(args.out)

    epe_vals_mean = results.reduce_mean("epe_vals")
    epe_vals_oracle_mean = results.reduce_mean("epe_vals_oracle")
    sparsification_plot(epe_vals_mean, epe_vals_oracle_mean, output_path=args.out, label="RAFT-Uncertainty-V2")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="Model checkpoint path")
    parser.add_argument('--small', action="store_true", default=False, help="Use small version of RAFT")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="Use mixed precision")
    parser.add_argument('--alternate_corr', action='store_true', default=False, help="Use efficent correlation implementation")
    parser.add_argument('--uncertainty', action='store_true', default=True, help="Enable flow uncertainty estimation")
    parser.add_argument('--residual_variance', action='store_true', help='If True, predict variance for residual flow instead of flow variance')
    parser.add_argument('--log_variance', action='store_true', help='If true, variance is predicted in log space')
    parser.add_argument('--out', help="Output path")
    args = parser.parse_args()

    if "uncertainty" not in args:
        args.uncertainty = False
    if "residual_variance" not in args:
        args.residual_variance = False
    if "log_variance" not in args:
        args.log_variance = False

    run(args)
