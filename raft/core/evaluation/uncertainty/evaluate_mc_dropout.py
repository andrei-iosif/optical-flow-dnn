import argparse
import os
import numpy as np
import torch

import core.datasets as datasets
from core.utils.utils import InputPadder
from core.evaluation.uncertainty.sparsification_plots import sparsification_plot
from core.evaluation.metrics import Metrics
from core.evaluation.uncertainty.utils import load_model, compute_metrics, compute_flow_variance_two_pass, get_flow_confidence_v1
from core.utils.visu.visu import predictions_visu


def enable_dropout(model):
  for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()


def dropout_inference(model, num_inferences, image_1, image_2, gt_flow, args, sample_id=-1):
    """ Run inference with dropout (equivalent to ensemble of models). Compute empirical mean and variance for optical flow.
    Optionally, create predictions visu (every 10th frame).

    Args:
        models (torch.Module): RAFT model trained with dropout.
        num_inferences (int): Number of inferences
        image_1 (torch.Tensor): First input image
        image_2 (torch.Tensor): Second input image
        gt_flow (np.ndarray): GT flow (used for visu)

    Returns:
        Estimated flow mean and variance, both with shape [2, H, W]
    """
    image_1 = image_1[None].cuda()
    image_2 = image_2[None].cuda()
    padder = InputPadder(image_1.shape, mode='kitti')
    image_1, image_2 = padder.pad(image_1, image_2)

    # Run inference multiple times, with dropout activated
    # Keep predictions in a list (may be needed later)
    pred_flow_list = []
    for _ in range(num_inferences):
        _, pred_flow = model(image_1, image_2, iters=args.iters, test_mode=True)
        pred_flow = padder.unpad(pred_flow[0]).cpu().numpy()
        pred_flow_list.append(pred_flow)

    # Estimate mean and variance from list of predictions
    pred_flow_mean, pred_flow_var = compute_flow_variance_two_pass(pred_flow_list)

    if args.create_visu and sample_id % 10 == 0:
        image_1 = image_1[0].cpu().numpy()
        flow_confidence = get_flow_confidence_v1(pred_flow_var)
        predictions_visu(image_1, gt_flow, pred_flow, sample_id, os.path.join(args.out, "visu"), pred_flow_var=flow_confidence)

    return pred_flow_mean, pred_flow_var


def run(args):
    # Load model checkpoint
    model = load_model(args.model, args)
    enable_dropout(model)    

    # Load dataset
    # dataset = datasets.FlyingChairs(split='validation', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/FlyingChairs')
    dataset = datasets.MpiSintel(split='training', dstype='clean', root=r'/home/mnegru/repos/optical-flow-dnn/raft/datasets/Sintel')
    print(f"Loaded dataset, size = {len(dataset)}")

    # Run inference
    metrics = Metrics()

    with torch.no_grad():
        for sample_id in range(len(dataset)):
            image_1, image_2, gt_flow, _ = dataset[sample_id]
            gt_flow = gt_flow.cpu().numpy()
           
            pred_flow, pred_flow_var = dropout_inference(model, args.num_inferences, image_1, image_2, gt_flow, args, sample_id=sample_id)

            epe_vals, epe_vals_oracle = compute_metrics(pred_flow, pred_flow_var, gt_flow)
            metrics.add(sample_id, "epe_vals", epe_vals)
            metrics.add(sample_id, "epe_vals_oracle", epe_vals_oracle)
            print(f"Processed sample id={sample_id}")
    
    metrics.save_pickle(args.out)

    epe_vals_mean = metrics.reduce_mean("epe_vals")
    epe_vals_oracle_mean = metrics.reduce_mean("epe_vals_oracle")
    sparsification_plot(epe_vals_mean, epe_vals_oracle_mean, output_path=args.out, label=f"RAFT-Dropout-{args.num_inferences}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action="store_true", default=False, help="Use small version of RAFT")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="Use mixed precision")
    parser.add_argument('--alternate_corr', action='store_true', default=False, help="Use efficent correlation implementation")
    parser.add_argument('--out', help="Output path")
    args = parser.parse_args()

    args.num_inferences = 5
    args.dropout = 0.2
    args.iters = 24
    args.uncertainty = False
    args.residual_variance = False
    args.log_variance = False
    args.model = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_chairs_seed_42_dropout_encoder_only/raft-chairs.pth'
    args.out = r'/home/mnegru/repos/optical-flow-dnn/dump/uncertainty_evaluation/Sintel/mc_dropout_NEW'
    args.create_visu = True

    run(args)
