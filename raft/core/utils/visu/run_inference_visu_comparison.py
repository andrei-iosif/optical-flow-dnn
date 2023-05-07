import argparse
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

    # Load model checkpoints
    model_1 = load_model(args.model_1, args)
    model_2 = load_model(args.model_2, args)
    model_3 = load_model(args.model_3, args)
    # model_4 = load_model(args.model_4, args)
    # models = [model_1, model_2, model_3, model_4]
    # model_names = ["things", "viper", "vkitti", "viper_vkitti"]

    models = [model_1, model_2, model_3]
    model_names = ["things", "viper", "viper semantic"]

    # Load dataset
    dataset = datasets.KITTI(split='training', root='/home/mnegru/repos/optical-flow-dnn/raft/datasets/KITTI')
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

            run_visu(image1, flow_gt, flow_predictions, sample_id, args.out)


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
    args = parser.parse_args()

    if "uncertainty" not in args:
        args.uncertainty = False

    run(args)
