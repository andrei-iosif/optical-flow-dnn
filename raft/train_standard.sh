#!/bin/bash
# mkdir -p checkpoints/raft_uncertainty

python -u train.py --name train_raft-chairs-uncertainty --checkpoint_out=../checkpoints/raft_uncertainty --stage chairs --validation chairs --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 --uncertainty true
python -u train.py --name train_raft-things-uncertainty --checkpoint_out=../checkpoints/raft_uncertainty --stage things --validation sintel --restore_ckpt ../checkpoints/raft_uncertainty/raft-chairs.pth --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --uncertainty true
# python -u train.py --small --name train_raft_small_sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85
# python -u train.py --small --name train_raft_small_kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85
