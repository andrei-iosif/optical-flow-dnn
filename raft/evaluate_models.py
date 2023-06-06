import argparse
import numpy as np
import torch

from clearml import Task

from core.raft import RAFT
from evaluate import validate_sintel, validate_kitti


def load_model(args, model_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model


def compute_mean_std(d):
    for key in d.keys():
        vals = np.array(d[key])
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"Results for '{key}': mean={mean}, std={std}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlaton implementation')
    args = parser.parse_args()

    task = Task.init(project_name='RAFT Evaluation', task_name="raft_viper_evaluate_kitti", task_type=Task.TaskTypes.testing)

    args.uncertainty = False
    args.residual_variance = False
    args.log_variance = False
    args.flow_variance_last_iter = False

    model_path_1 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_viper_seed_0_mixed/raft-viper.pth'
    model_path_2 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_viper_seed_42_mixed/raft-viper.pth'
    model_path_3 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_viper_seed_1234_mixed/raft-viper.pth'
        
    model_paths = [model_path_1, model_path_2, model_path_3]

    # results_sintel_total = {"clean_epe": [], "final_epe": []}
    results_kitti_total = {"epe": [], "fl-all": []}

    with torch.no_grad():
        for path in model_paths:
            model = load_model(args, path)
            print(f"Loaded model: {path}")
            
            # results_sintel = validate_sintel(model.module)
            # results_sintel_total["clean_epe"] += [results_sintel["clean"]]
            # results_sintel_total["final_epe"] += [results_sintel["final"]]

            results_kitti = validate_kitti(model.module)
            results_kitti_total["epe"] += [results_kitti["kitti-epe"]]
            results_kitti_total["fl-all"] += [results_kitti["kitti-f1"]]
        
        # compute_mean_std(results_sintel_total)
        compute_mean_std(results_kitti_total)
