import sys
sys.path.append('../core')

import argparse
import torch

import datasets
from raft import RAFT
from utils.utils import InputPadder

from visu.visu import predictions_visu


def run_visu(img_1, img_2, gt_flow, pred_flow, sample_id, output_path):
    img_1 = img_1[0].cpu().numpy()
    img_2 = img_2[0].cpu().numpy()

    gt_flow = gt_flow.cpu().numpy()
    pred_flow = pred_flow.cpu().numpy()

    predictions_visu(img_1, img_2, gt_flow, pred_flow, sample_id, output_path)


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
    dataset = datasets.KITTI(split='training', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/KITTI')
    print(f"Loaded dataset, size = {len(dataset)}")

    # Run inference and save visus
    with torch.no_grad():
        for sample_id in range(len(dataset)):
            image1, image2, flow_gt, valid_gt = dataset[sample_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            run_visu(image1, image2, flow_gt, flow, sample_id, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="model checkpoint path")
    parser.add_argument('--small', action="store_true", default=False, help="use small version of RAFT")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', default=False, help='use efficent correlation implementation')
    parser.add_argument('--out', help="output path")
    args = parser.parse_args()

    run(args)
