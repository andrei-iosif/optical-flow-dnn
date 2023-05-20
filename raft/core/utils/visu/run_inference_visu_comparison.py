import argparse
import copy
import torch

import core.datasets as datasets
from core.raft import RAFT
from core.utils.utils import InputPadder
from core.utils.visu.visu import predictions_comparison_visu


def load_model(checkpoint_path, args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(checkpoint_path))

    model = model.module
    model.to('cuda')
    model.eval()
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    return model


def run_visu(img_1, gt_flow, flow_predictions, sample_id, output_path):
    img_1 = img_1[0].cpu().numpy()
    gt_flow = gt_flow.cpu().numpy()

    predictions_comparison_visu(img_1, gt_flow, flow_predictions, sample_id, output_path)


def run(args):
    iters = 24
    args_1, args_2, args_3 = args

    # Load model checkpoints
    model_1 = load_model(args_1.model_1, args_1)    
    model_2 = load_model(args_2.model_2, args_2)
    model_3 = load_model(args_3.model_3, args_3)

    models = [model_1, model_2, model_3]
    model_names = ["baseline", "raft_uncertainty_v1", "raft_uncertainty_v2"]

    # Load dataset
    # dataset = datasets.KITTI(split='training', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/KITTI')
    dataset = datasets.FlyingChairs(split='validation', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/FlyingChairs')
    print(f"Loaded dataset, size = {len(dataset)}")

    # Run inference and save visus
    with torch.no_grad():
        for sample_id in range(len(dataset)):
            image1, image2, flow_gt, _ = dataset[sample_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)

            flow_predictions = []
            for idx, model in enumerate(models):
                _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
                flow = padder.unpad(flow_pr[0]).cpu().numpy()
                flow_predictions.append((model_names[idx], flow))

            run_visu(image1, flow_gt, flow_predictions, sample_id, args_1.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_1', help="First checkpoint path")
    parser.add_argument('--model_2', help="Second checkpoint path")
    parser.add_argument('--model_3', help="Third checkpoint path")
    # parser.add_argument('--model_4', help="Fourth checkpoint path")
    parser.add_argument('--small', action="store_true", default=False, help="use small version of RAFT")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', default=False, help='use efficent correlation implementation')
    parser.add_argument('--out', help="output path")
    
    args_1 = parser.parse_args()
    args_1.uncertainty = False
    args_1.log_variance = False
    args_1.residual_variance = False

    args_2 = copy.deepcopy(args_1)
    args_2.uncertainty = True
    args_2.log_variance = True
    args_2.residual_variance = False

    args_3 = copy.deepcopy(args_1)
    args_3.uncertainty = True
    args_3.log_variance = False
    args_3.residual_variance = True

    run([args_1, args_2, args_3])
