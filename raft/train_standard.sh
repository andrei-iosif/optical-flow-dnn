#!/bin/bash

##  Uncertainty experiments
# python -u train.py --name overfit_raft_chairs_100_sample_seed_0_baseline \
#     --num_overfit_samples 100 --val_freq 1000 \
#     --stage chairs --validation chairs --gpus 0 --num_steps 5000 \
#     --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 --seed 0 --debug_iter

# python -u train.py --name overfit_raft_chairs_100_sample_seed_0_nll_loss_v2_UPDATE \
#     --num_overfit_samples 100 --val_freq 1000 --stage chairs --validation chairs \
#     --gpus 0 --num_steps 5000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter

# python -u train.py --name overfit_raft_chairs_100_sample_seed_0_nll_loss_v1_residual_var \
#     --checkpoint_out=../checkpoints/raft_uncertainty/overfit_raft_chairs_100_sample_seed_0_nll_loss_v1_residual_var \
#     --num_overfit_samples 100 --val_freq 1000 --stage chairs --validation chairs \
#     --gpus 0 --num_steps 5000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter

python -u train.py --name overfit_raft_chairs_100_sample_seed_0_nll_loss_v2_residual_var_no_log \
    --checkpoint_out=../checkpoints/raft_uncertainty/overfit_raft_chairs_100_sample_seed_0_nll_loss_v2_residual_var_no_log \
    --num_overfit_samples 100 --val_freq 1000 --stage chairs --validation chairs \
    --gpus 0 --num_steps 5000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
    --seed 0 --uncertainty --debug_iter


## Baseline experiments
# python -u train.py --name train_raft_vkitti_seed_0_mixed --restore_ckpt=../models/raft_original_models/raft-things.pth --checkpoint_out=../checkpoints/raft_vkitti/train_raft_vkitti_seed_0_mixed --stage virtual_kitti --validation kitti --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --seed 0 --mixed_precision
# python -u train.py --name train_raft_vkitti_seed_42_mixed --restore_ckpt=../models/raft_original_models/raft-things.pth --checkpoint_out=../checkpoints/raft_vkitti/train_raft_vkitti_seed_42_mixed --stage virtual_kitti --validation kitti --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --seed 42 --mixed_precision
#python -u train.py --name train_raft_virtual_kitti_seed_1234_mixed --restore_ckpt=../models/raft_original_models/raft-things.pth --checkpoint_out=../checkpoints/raft_virtual_kitti/train_raft_virtual_kitti_seed_1234_mixed --stage virtual_kitti --validation kitti --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --seed 1234 --mixed_precision

# python -u train.py --name train_raft_viper_vkitti_seed_1234_mixed --restore_ckpt=../checkpoints/raft_viper/train_raft_viper_seed_1234_mixed/raft-viper.pth --checkpoint_out=../checkpoints/raft_viper_vkitti/train_raft_viper_vkitti_seed_1234_mixed --stage virtual_kitti --validation kitti --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --seed 1234 --mixed_precision


## DEBUG
# python -u train.py \
#     --name overfit_raft_viper_100_sample_seed_0_all_iters_semantic_loss_FIXED \
#     --num_overfit_samples 100 \
#     --val_freq 1000 \
#     --restore_ckpt ../models/raft_original_models/raft-things.pth \
#     --stage viper --validation kitti --gpus 0 \
#     --num_steps 5000 --batch_size 6 --lr 1e-4 \
#     --image_size 288 960 \
#     --wdecay 0.0001 \
#     --seed 0 \
#     --debug_iter --semantic_loss true

# python -u train.py \
#     --name overfit_raft_viper_100_sample_seed_0_all_iters_semantic_loss_weight_5 \
#     --num_overfit_samples 100 \
#     --val_freq 1000 \
#     --restore_ckpt ../models/raft_original_models/raft-things.pth \
#     --stage viper --validation kitti --gpus 0 \
#     --num_steps 5000 --batch_size 6 --lr 1e-4 \
#     --image_size 288 960 \
#     --wdecay 0.0001 \
#     --seed 0 \
#     --debug_iter \
#     --semantic_loss true --semantic_loss_weight 5.0
## DEBUG

# RAFT Semantic
# python -u train.py \
#     --name train_raft_viper_semantic_loss_seed_0_mixed \
#     --restore_ckpt=../models/raft_original_models/raft-things.pth \
#     --checkpoint_out=../checkpoints/raft_semantic/train_raft_viper_semantic_loss_seed_0_mixed \
#     --stage viper --validation viper --validation_set_size 200 --gpus 0 \
#     --num_steps 100000 --batch_size 6 --lr 0.0001 \
#     --image_size 288 960 --wdecay 0.0001 \
#     --seed 0 \
#     --mixed_precision \
#     --semantic_loss true \
#     --debug_iter
