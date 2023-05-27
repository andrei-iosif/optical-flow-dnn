import argparse
import os
import torch

import core.datasets as datasets
from core.utils.utils import InputPadder
from core.evaluation.uncertainty.sparsification_plots import sparsification_plot
from core.evaluation.uncertainty.utils import load_model, compute_metrics, get_flow_confidence_v1
from core.evaluation.metrics import Metrics
from core.utils.visu.visu import predictions_visu


def run(args):
    # Load model checkpoint
    model = load_model(args.model, args)

    # Load dataset
    # dataset = datasets.FlyingChairs(split='validation', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/FlyingChairs')
    dataset = datasets.MpiSintel(split='training', dstype='clean', root=r'/home/mnegru/repos/optical-flow-dnn/raft/datasets/Sintel')
    print(f"Loaded dataset, size = {len(dataset)}")

    # Run inference
    metrics = Metrics()

    with torch.no_grad():
        for sample_id in range(len(dataset)):
            image_1, image_2, gt_flow, _ = dataset[sample_id]
            image_1 = image_1[None].cuda()
            image_2 = image_2[None].cuda()

            padder = InputPadder(image_1.shape, mode='kitti')
            image_1, image_2 = padder.pad(image_1, image_2)

            pred_flow_var, pred_flow = model(image_1, image_2, iters=args.iters, test_mode=True)

            pred_flow = padder.unpad(pred_flow[0]).cpu().numpy()
            pred_flow_var = padder.unpad(pred_flow_var[0]).cpu().numpy()
            gt_flow = gt_flow.cpu().numpy()

            if args.create_visu and sample_id % 10 == 0:
                image_1 = image_1[0].cpu().numpy()
                flow_confidence = get_flow_confidence_v1(pred_flow_var)
                predictions_visu(image_1, gt_flow, pred_flow, sample_id, os.path.join(args.out, "visu"), pred_flow_var=flow_confidence)

            epe_vals, epe_vals_oracle = compute_metrics(pred_flow, pred_flow_var, gt_flow)
            metrics.add(sample_id, "epe_vals", epe_vals)
            metrics.add(sample_id, "epe_vals_oracle", epe_vals_oracle)
            print(f"Processed sample id={sample_id}")
    
    metrics.save_pickle(args.out)

    epe_vals_mean = metrics.reduce_mean("epe_vals")
    epe_vals_oracle_mean = metrics.reduce_mean("epe_vals_oracle")
    sparsification_plot(epe_vals_mean, epe_vals_oracle_mean, output_path=args.out, label="RAFT-Uncertainty-V2")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action="store_true", default=False, help="Use small version of RAFT")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="Use mixed precision")
    parser.add_argument('--alternate_corr', action='store_true', default=False, help="Use efficent correlation implementation")
    args = parser.parse_args()

    args.iters = 24
    args.uncertainty = False
    args.residual_variance = False
    args.log_variance = False
    # args.model = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_uncertainty/raft_chairs_seed_0_nll_loss_v1_log_variance/raft-chairs.pth'
    args.model = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_uncertainty/raft_chairs_seed_0_nll_loss_v2_residual_variance/raft-chairs.pth'
    args.out = r'/home/mnegru/repos/optical-flow-dnn/dump/uncertainty_evaluation_FINAL/Sintel/raft_uncertainty_v2'
    args.create_visu = True

    run(args)
