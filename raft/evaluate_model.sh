#!/bin/bash

python -u evaluate.py --model checkpoints/raft-chairs.pth --dataset chairs --small --experiment_name raft_small_train_chairs_eval_chairs
python -u evaluate.py --model checkpoints/raft-chairs.pth --dataset sintel --small --experiment_name raft_small_train_chairs_eval_sintel
python -u evaluate.py --model checkpoints/raft-chairs.pth --dataset kitti --small --experiment_name raft_small_train_chairs_eval_kitti