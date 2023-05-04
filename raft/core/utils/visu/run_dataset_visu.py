import argparse
import datasets
import os

import core.datasets as datasets
from core.utils.visu.visu import inputs_visu

DATASETS = {
    'HD1K': datasets.HD1K,
    'KITTI': datasets.KITTI,
    'VIPER': datasets.VIPER,
    'VirtualKITTI': datasets.VirtualKITTI,
    'FlyingChairs': datasets.FlyingChairs,
    'FlyingThings3D_subset': datasets.FlyingThingsSubset,
    'Sintel': datasets.MpiSintel
}

def run(args):
    # Load dataset
    dataset_name = os.path.basename(args.dataset_root)
    dataset = DATASETS[dataset_name](root=args.dataset_root)
    print(f"Loaded dataset {dataset_name}, size = {len(dataset)}")

    # Create and save visus
    for sample_id in range(len(dataset)):
        # img_1, _, gt_flow, valid_flow_mask, semseg_1, _ = dataset[sample_id]
        img_1, _, gt_flow, valid_flow_mask = dataset[sample_id]
        img_1 = img_1.cpu().numpy()
        gt_flow = gt_flow.cpu().numpy()
        valid_flow_mask = valid_flow_mask.cpu().numpy()
        # semseg_1 = semseg_1.cpu().numpy()
        inputs_visu(img_1, gt_flow, valid_flow_mask=valid_flow_mask,
            sample_idx=sample_id, output_path=args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True, help="Path to root of dataset")
    parser.add_argument('--out', required=True, help="Output path")
    args = parser.parse_args()

    run(args)
