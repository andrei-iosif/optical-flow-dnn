from __future__ import print_function, division

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import core.datasets as datasets
import core.losses as losses
from core.raft import RAFT
from core.utils.logger import Logger
from core.utils.random import set_random_seed
from core.utils.training_utils import EarlyStopper

import evaluate

from clearml import Task

try:
    from torch.cuda.amp import GradScaler
except:
    # Dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def fetch_loss_func(args):
    """ Create loss function. """
    if args.semantic_loss:
        print("Training using semantic RAFT loss")
        return losses.RaftSemanticLoss(gamma=args.gamma, w_smooth=args.semantic_loss_weight, debug=args.debug_iter)
    elif args.uncertainty:
        print("Training using RAFT uncertainty loss")
        return losses.RaftUncertaintyLoss(gamma=args.gamma, debug=args.debug_iter)
    else:
        print("Training using base RAFT loss")
        return losses.RaftLoss(gamma=args.gamma, debug=args.debug_iter)


def train(args):
    # Keep this or replace with DistributedDataParallel?
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    # Why freeze batch norm layers?
    # Unanswered question: https://github.com/princeton-vl/RAFT/issues/141
    # Possible answer: https://stackoverflow.com/questions/63016740/why-its-necessary-to-frozen-all-inner-state-of-a-batch-normalization-layer-when
    # When fine tuning on a new dataset, we do not want to re-learned all the weights from scratch.
    # If batch norm is not frozen, its parameters will be re-learned causing the other network parameters to be re-learned.
    if args.stage != 'chairs':
        model.module.freeze_bn()

    # Prepare dataset, optimizer and loss function
    use_semseg = args.semantic_loss
    train_loader = datasets.fetch_dataloader(args, num_overfit_samples=args.num_overfit_samples, use_semseg=use_semseg)
    optimizer, scheduler = fetch_optimizer(args, model)
    loss_func = fetch_loss_func(args)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(scheduler, val_freq=args.val_freq, debug=args.debug)

    should_keep_training = True
    while should_keep_training:

        for _, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            # Unpack data sample
            if use_semseg:
                image_1, image_2, flow, valid_mask, semseg_1, semseg_2 = [x.cuda() for x in data_blob]
            else:
                image_1, image_2, flow, valid_mask = [x.cuda() for x in data_blob]

            # TODO: move data augmentations to separate class
            # Gaussian noise augmentation
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image_1 = (image_1 + stdv * torch.randn(*image_1.shape).cuda()).clamp(0.0, 255.0)
                image_2 = (image_2 + stdv * torch.randn(*image_2.shape).cuda()).clamp(0.0, 255.0)

            # Forward pass (default: 12 iterations)
            flow_predictions = model(image_1, image_2, iters=args.iters)

            logger.debug_log(flow_predictions, flow)

            # Loss computation
            if use_semseg:
                loss, metrics = loss_func(flow_predictions, flow, valid_mask, semseg_1, semseg_2)
            else:
                loss, metrics = loss_func(flow_predictions, flow, valid_mask)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            # Validation
            if total_steps % logger.val_freq == logger.val_freq - 1:
                if args.checkpoint_out is not None:
                    PATH = os.path.join(args.checkpoint_out, f"{total_steps + 1}_raft-{args.stage}.pth")
                    torch.save(model.state_dict(), PATH)
                    print(f"Saved model: {PATH}")

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))
                    elif val_dataset == 'viper':
                        results.update(evaluate.validate_viper(model.module, subset_size=args.validation_set_size))
                        results.update(evaluate.validate_kitti(model.module))
                    else:
                        raise AttributeError(f"Invalid validation dataset: {val_dataset}")

                logger.write_dict(results)

                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()

    if args.checkpoint_out is not None:
        PATH = os.path.join(args.checkpoint_out, f"raft-{args.stage}.pth")
        torch.save(model.state_dict(), PATH)
        print(f"Saved model: {PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--checkpoint_out', help="Output folder for checkpoints")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--val_freq', type=int, default=5000, help="Validation frequency (iterations)")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--seed', type=int, default=1234, help="Seed used for all RNGs")
    parser.add_argument('--num_overfit_samples', type=int, default=-1, help="Number of samples to overfit on (if positive)")
    parser.add_argument('--validation_set_size', type=int, default=-1, 
        help="Number of samples used to compute validation metrics. By default, the entire validation dataset is used.")
    parser.add_argument('--semantic_loss', type=bool, default=False, help="Use semantic correction for training.")
    parser.add_argument('--semantic_loss_weight', type=float, default=0.5, help="Weight for semantic loss term")
    parser.add_argument('--uncertainty', action='store_true', help='Enable flow uncertainty estimation')
    parser.add_argument('--debug', action='store_true', help="In debug mode, additional plots are generated and uploaded to ClearML")
    parser.add_argument('--debug_iter', action='store_true', help="Save metrics for all refinement iterations")
    args = parser.parse_args()

    # Initialize ClearML task
    task = Task.init(project_name='RAFT Uncertainty Debug', task_name=args.name)

    # Initial seed for RNGs; influences the initial model weights
    set_random_seed(args.seed)

    if args.checkpoint_out is not None and not os.path.isdir(args.checkpoint_out):
        os.makedirs(args.checkpoint_out, exist_ok=True)

    if "uncertainty" not in args:
        args.uncertainty = False
    if "debug" not in args:
        args.debug = False
    if "debug_iter" not in args:
        args.debug_iter = False

    train(args)
