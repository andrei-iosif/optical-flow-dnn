import argparse
import os
import torch

import core.datasets as datasets
from core.raft import RAFT
from core.utils.utils import InputPadder, endpoint_error_numpy
from core.utils.visu.visu import predictions_visu, predictions_visu_uncertainty


def load_model(args, model_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model


def run_visu(model, dataset, output_path, args, step):
    with torch.no_grad():
        for sample_id in range(0, len(dataset), step):
            img_1, img_2, gt_flow, valid_mask = dataset[sample_id]
            img_1 = img_1[None].cuda()
            img_2 = img_2[None].cuda()

            padder = InputPadder(img_1.shape, mode='kitti')
            img_1, img_2 = padder.pad(img_1, img_2)

            _, pred_flow = model(img_1, img_2, iters=args.iters, test_mode=True)
            pred_flow = padder.unpad(pred_flow[0]).cpu()

            img_1 = img_1[0].cpu().numpy()
            gt_flow = gt_flow.cpu().numpy()
            pred_flow = pred_flow.cpu().numpy()
            valid_mask = valid_mask.cpu().numpy()

            epe_img = endpoint_error_numpy(pred_flow, gt_flow, valid_mask=valid_mask)
            predictions_visu(img_1, gt_flow, pred_flow, epe_img, sample_id, output_path, save_subplots=args.save_subplots)


def run(args):
    model = load_model(args, args.model_path)
    print(f"Loaded model: {args.model_path}")

    if args.kitti_visu:
        dataset = datasets.KITTI(split='training', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/KITTI')
        print(f"Loaded dataset, size = {len(dataset)}")

        args.iters = 24
        output_path = os.path.join(args.output_path, "kitti", args.label, "visu")
        run_visu(model, dataset, output_path, args, step=1)

    if args.sintel_visu:
        dataset = datasets.MpiSintel(split='training', dstype='clean', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/Sintel')
        print(f"Loaded dataset, size = {len(dataset)}")

        args.iters = 32
        output_path = os.path.join(args.output_path, "sintel", args.label, "visu")
        run_visu(model, dataset, output_path, args, step=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False

    args.uncertainty = False
    args.residual_variance = False
    args.log_variance = False
    args.flow_variance_last_iter = False

    args.kitti_visu = True
    args.sintel_visu = True
    args.save_subplots = True
    args.output_path = r'/home/mnegru/repos/optical-flow-dnn/dump/visu'

    # Chairs baseline 
    # args.label = "raft_chairs_seed_0"
    # args.model_path = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_chairs_seed_0/raft-chairs.pth'

    # Chairs + Things baseline 
    # args.label = "raft_things_seed_0"
    # args.model_path = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_things_seed_0/raft-things.pth'

    # Chairs + Things + Viper baseline
    args.mixed_precision = True 
    args.label = "raft_viper_seed_0_mixed"
    args.model_path = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_viper_seed_0_mixed/raft-viper.pth'

    # Chairs + Things + Viper Semantic
    # args.mixed_precision = True 
    # args.label = "raft_viper_seed_0_mixed_semantic"
    # args.model_path = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_semantic/train_raft_viper_semantic_loss_seed_0_mixed/raft-viper.pth'

    run(args)
