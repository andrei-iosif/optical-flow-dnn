import argparse
import torch

from clearml import Task

from core.raft import RAFT
from evaluate import validate_sintel, validate_kitti_full, validate_chairs
from core.utils.metrics import Metrics


def load_model(args, model_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model


def evaluate(model_paths, args):
    metrics_kitti_total = Metrics()
    metrics_sintel_total = Metrics()
    metrics_chairs_total = Metrics()

    with torch.no_grad():
        for path in model_paths:
            model = load_model(args, path)
            print(f"Loaded model: {path}")
            
            if args.eval_kitti:
                metrics_kitti_total.add_dict(validate_kitti_full(model.module))

            if args.eval_sintel:
                metrics_sintel_total.add_dict(validate_sintel(model.module))

            if args.eval_chairs:
                metrics_chairs_total.add_dict(validate_chairs(model.module))

        if args.eval_kitti:
            metrics_kitti_total = metrics_kitti_total.reduce_mean(compute_std=True)
            print(f"Results KITTI: {metrics_kitti_total}")

        if args.eval_sintel:
            metrics_sintel_total = metrics_sintel_total.reduce_mean(compute_std=True)
            print(f"Results Sintel: {metrics_sintel_total}")
    
        if args.eval_chairs:
            metrics_chairs_total = metrics_chairs_total.reduce_mean(compute_std=True)
            print(f"Results Chairs: {metrics_chairs_total}")


if __name__ == '__main__':
    task = Task.init(project_name='RAFT Evaluation', task_name="raft_chairs_baseline_evaluate_FINAL", task_type=Task.TaskTypes.testing)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False

    args.uncertainty = False
    args.residual_variance = False
    args.log_variance = False
    args.flow_variance_last_iter = False

    args.eval_kitti = True
    args.eval_sintel = True
    args.eval_chairs = True

    # C baseline 
    # model_path_1 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_chairs_seed_0/raft-chairs.pth'
    # model_path_2 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_chairs_seed_42/raft-chairs.pth'
    # model_path_3 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_chairs_seed_1234/raft-chairs.pth'

    # C+T baseline 
    # model_path_1 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_things_seed_0/raft-things.pth'
    # model_path_2 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_things_seed_42/raft-things.pth'
    # model_path_3 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_things_seed_1234/raft-things.pth'

    # C+T+VIPER baseline
    # model_path_1 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_viper_seed_0_mixed/raft-viper.pth'
    # model_path_2 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_viper_seed_42_mixed/raft-viper.pth'
    # model_path_3 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_baseline/raft_viper_seed_1234_mixed/raft-viper.pth'
        
    # C+T+VIPER semantic
    # model_path_1 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_semantic/train_raft_viper_semantic_loss_seed_0_mixed/raft-viper.pth'
    # model_path_2 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_semantic/train_raft_viper_semantic_loss_seed_42_mixed/raft-viper.pth'

    # RAFT-Uncertainty-V1
    args.uncertainty = True
    args.log_variance = True
    model_path_1 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_uncertainty/raft_chairs_seed_0_nll_loss_v1_log_variance/raft-chairs.pth'
    model_path_2 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_uncertainty/raft_chairs_seed_42_nll_loss_v1_log_variance/raft-chairs.pth'
    model_path_3 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_uncertainty/raft_chairs_seed_1234_nll_loss_v1_log_variance/raft-chairs.pth'

    # RAFT-Uncertainty-V2
    # args.uncertainty = True
    # args.residual_variance = True
    # model_path_1 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_uncertainty/raft_chairs_seed_0_nll_loss_v2_residual_variance/raft-chairs.pth'
    # model_path_2 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_uncertainty/raft_chairs_seed_42_nll_loss_v2_residual_variance/raft-chairs.pth'
    # model_path_3 = r'/home/mnegru/repos/optical-flow-dnn/checkpoints/raft_uncertainty/raft_chairs_seed_1234_nll_loss_v2_residual_variance/raft-chairs.pth'

    model_paths = [model_path_1, model_path_2, model_path_3]

    evaluate(model_paths, args)
