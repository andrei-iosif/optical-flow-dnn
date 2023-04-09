#!/bin/bash

python -u train.py --name overfit_raft_viper_100_random_samples --num_overfit_samples 100 --validation_set_size 10 --stage viper --validation viper --checkpoint_out ../checkpoints/raft_semantic/overfit_raft_viper_100_random_samples --restore_ckpt ../models/raft_original_models/raft-things.pth --gpus 0 --num_steps 5000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --seed 1234
