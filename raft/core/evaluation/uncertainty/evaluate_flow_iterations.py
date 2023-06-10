import argparse
import os
import torch

import core.datasets as datasets
from core.utils.utils import InputPadder
from core.evaluation.metrics import Metrics
from core.evaluation.uncertainty.utils import load_model, compute_metrics, compute_flow_variance_two_pass, get_flow_confidence, endpoint_error_numpy, reduce_metrics
from core.utils.visu.visu import predictions_visu_uncertainty


def inference(model, image_1, image_2, gt_flow, flow_valid_mask, args, sample_id=-1):
    """ Run inference with single RAFT model. Compute empirical mean and variance for intermediate iterations of optical flow.
    Optionally, create predictions visu (every 10th frame).

    Args:
        models (torch.Module): RAFT model trained with dropout.
        image_1 (torch.Tensor): First input image
        image_2 (torch.Tensor): Second input image
        gt_flow (np.ndarray): GT flow (used for visu)
        flow_valid_mask (np.ndarray): Flow valid mask (used for EPE visu)

    Returns:
        Estimated flow mean and variance, both with shape [2, H, W]
    """
    image_1 = image_1[None].cuda()
    image_2 = image_2[None].cuda()
    padder = InputPadder(image_1.shape, mode='kitti')
    image_1, image_2 = padder.pad(image_1, image_2)

    # Run inference with test_mode=False => get all intermediate predictions
    pred_list = model(image_1, image_2, iters=args.iters, test_mode=False)

    pred_flow_list = []
    for pred_flow in pred_list:
        pred_flow = padder.unpad(pred_flow[0]).cpu().numpy()
        pred_flow_list.append(pred_flow)

    # Estimate mean and variance from list of predictions
    pred_flow_mean, pred_flow_var = compute_flow_variance_two_pass(pred_flow_list)

    if args.create_visu and sample_id % 10 == 0:
        image_1 = image_1[0].cpu().numpy()
        flow_confidence = get_flow_confidence(pred_flow_var)
        flow_epe = endpoint_error_numpy(pred_flow_mean, gt_flow, flow_valid_mask)

        predictions_visu_uncertainty(image_1, gt_flow, pred_flow, flow_epe, flow_confidence, sample_id, 
                                     os.path.join(args.out, "visu"), save_subplots=args.save_subplots)

    return pred_flow_mean, pred_flow_var


def run(args):
    # Load model checkpoint
    model = load_model(args.model, args)

    # Load dataset
    # dataset = datasets.FlyingChairs(split='validation', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/FlyingChairs')
    dataset = datasets.MpiSintel(split='training', dstype='clean', root=r'/home/mnegru/repos/optical-flow-dnn/raft/datasets/Sintel')
    # dataset = datasets.KITTI(split='training', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/KITTI')
    print(f"Loaded dataset, size = {len(dataset)}")

    # Run inference
    metrics = Metrics()

    with torch.no_grad():
        for sample_id in range(len(dataset)):
            image_1, image_2, gt_flow, flow_valid_mask = dataset[sample_id]
            gt_flow = gt_flow.cpu().numpy()
            flow_valid_mask = flow_valid_mask.cpu().numpy()
           
            pred_flow, pred_flow_var = inference(model, image_1, image_2, gt_flow, flow_valid_mask, args, sample_id=sample_id)

            compute_metrics(pred_flow, pred_flow_var, gt_flow, metrics, sample_id, flow_valid_mask=flow_valid_mask)
            print(f"Processed sample id={sample_id}")
    
    metrics.save_pickle(args.out)
    reduce_metrics(metrics, args.out, args.label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False

    args.iters = 32
    args.label = "RAFT-FlowIterations"
    args.uncertainty = False
    args.residual_variance = False
    args.log_variance = False
    args.model = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_chairs_seed_42/raft-chairs.pth'
    args.out = r'/home/mnegru/repos/optical-flow-dnn/dump/uncertainty_evaluation_FINAL/Sintel_UPDATED/flow_iterations'
    args.create_visu = True
    args.save_subplots = True

    run(args)
