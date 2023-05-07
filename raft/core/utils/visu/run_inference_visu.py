import argparse
import torch

import core.datasets as datasets
from core.raft import RAFT
from core.utils.utils import InputPadder
from core.utils.visu.visu import predictions_visu
    

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
    # dataset = datasets.KITTI(split='training', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/KITTI')
    dataset = datasets.FlyingChairs(split='validation', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/FlyingChairs')
    print(f"Loaded dataset, size = {len(dataset)}")

    # Run inference and save visus
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

            predictions_visu(image_1, gt_flow, pred_flow, sample_id, args.out, pred_flow_var=pred_flow_var)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="model checkpoint path")
    parser.add_argument('--small', action="store_true", default=False, help="use small version of RAFT")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', default=False, help='use efficent correlation implementation')
    parser.add_argument('--uncertainty', action='store_true', help='Enable flow uncertainty estimation')
    parser.add_argument('--out', help="output path")
    args = parser.parse_args()

    if "uncertainty" not in args:
        args.uncertainty = False

    run(args)
