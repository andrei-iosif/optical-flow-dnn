#!/bin/bash

python -u evaluate.py --model ../checkpoints/raft_semantic/raft_viper_baseline_seed_0/raft-viper.pth --dataset viper --experiment_name eval_raft_viper_baseline_seed_0 --mixed_precision
# python -u evaluate.py --model ../checkpoints/raft_semantic/raft_viper_baseline_seed_42/raft-viper.pth --dataset viper --experiment_name eval_raft_viper_baseline_seed_42 --mixed_precision
# python -u evaluate.py --model ../checkpoints/raft_semantic/raft_viper_baseline_seed_1234/raft-viper.pth --dataset viper --experiment_name eval_raft_viper_baseline_seed_1234 --mixed_precision