#!/bin/bash


# Baseline experiments
python -u evaluate.py --model ../checkpoints/raft_viper/train_raft_viper_seed_1234_mixed/raft-viper.pth --dataset kitti --experiment_name eval_raft_viper_seed_1234_mixed --mixed_precision
python -u evaluate.py --model ../checkpoints/raft_vkitti/train_raft_vkitti_seed_42_mixed/raft-virtual_kitti.pth --dataset kitti --experiment_name eval_raft_vkitti_seed_42_mixed --mixed_precision
python -u evaluate.py --model ../checkpoints/raft_viper_vkitti/train_raft_viper_vkitti_seed_1234_mixed/raft-virtual_kitti.pth --dataset kitti --experiment_name eval_raft_viper_vkitti_seed_1234_mixed --mixed_precision
