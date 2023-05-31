#!/bin/bash

##  Uncertainty experiments
# python -u train.py --name overfit_raft_chairs_100_sample_seed_0_baseline \
#     --num_overfit_samples 100 --val_freq 1000 \
#     --stage chairs --validation chairs --gpus 0 --num_steps 5000 \
#     --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 --seed 0 --debug_iter

# python -u train.py --name raft_chairs_100_sample_seed_0_nll_loss_v1_log_variance \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_100_sample_seed_0_nll_loss_v1_log_variance \
#     --num_overfit_samples 100 --val_freq 1000 --stage chairs --validation chairs \
#     --gpus 0 --num_steps 5000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter --log_variance

# python -u train.py --name raft_chairs_100_sample_seed_0_nll_loss_v2_variance_ELU \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_100_sample_seed_0_nll_loss_v2_variance_ELU \
#     --num_overfit_samples 100 --val_freq 1000 --stage chairs --validation chairs \
#     --gpus 0 --num_steps 5000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter

# python -u train.py --name raft_chairs_100_sample_seed_0_nll_loss_v1_log_variance_residual \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_100_sample_seed_0_nll_loss_v2_variance \
#     --num_overfit_samples 100 --val_freq 1000 --stage chairs --validation chairs \
#     --gpus 0 --num_steps 5000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter --log_variance --residual_variance

# python -u train.py --name raft_chairs_100_sample_seed_0_nll_loss_v2_variance_residual_ELU \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_100_sample_seed_0_nll_loss_v2_variance_residual_ELU \
#     --num_overfit_samples 100 --val_freq 1000 --stage chairs --validation chairs \
#     --gpus 0 --num_steps 5000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter --residual_variance


# python -u train.py --name raft_chairs_seed_0_nll_loss_v1_log_variance \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_0_nll_loss_v1_log_variance \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter --log_variance

# python -u train.py --name raft_chairs_seed_0_nll_loss_v2_residual_variance \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_0_nll_loss_v2_residual_variance \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter --residual_variance

# python -u train.py --name raft_chairs_seed_42_nll_loss_v1_log_variance \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_42_nll_loss_v1_log_variance \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 42 --uncertainty --debug_iter --log_variance

# python -u train.py --name raft_chairs_seed_42_nll_loss_v2_residual_variance \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_42_nll_loss_v2_residual_variance \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 42 --uncertainty --debug_iter --residual_variance

# python -u train.py --name raft_chairs_seed_1234_nll_loss_v1_log_variance \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_1234_nll_loss_v1_log_variance \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 1234 --uncertainty --debug_iter --log_variance

# python -u train.py --name raft_chairs_seed_1234_nll_loss_v2_residual_variance \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_1234_nll_loss_v2_residual_variance \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 1234 --uncertainty --debug_iter --residual_variance

## Baseline experiments
# python -u train.py --name train_raft_vkitti_seed_0_mixed --restore_ckpt=../models/raft_original_models/raft-things.pth --checkpoint_out=../checkpoints/raft_vkitti/train_raft_vkitti_seed_0_mixed --stage virtual_kitti --validation kitti --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --seed 0 --mixed_precision
# python -u train.py --name train_raft_vkitti_seed_42_mixed --restore_ckpt=../models/raft_original_models/raft-things.pth --checkpoint_out=../checkpoints/raft_vkitti/train_raft_vkitti_seed_42_mixed --stage virtual_kitti --validation kitti --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --seed 42 --mixed_precision
# python -u train.py --name train_raft_virtual_kitti_seed_1234_mixed --restore_ckpt=../models/raft_original_models/raft-things.pth --checkpoint_out=../checkpoints/raft_virtual_kitti/train_raft_virtual_kitti_seed_1234_mixed --stage virtual_kitti --validation kitti --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --seed 1234 --mixed_precision

# python -u train.py --name train_raft_viper_vkitti_seed_1234_mixed --restore_ckpt=../checkpoints/raft_viper/train_raft_viper_seed_1234_mixed/raft-viper.pth --checkpoint_out=../checkpoints/raft_viper_vkitti/train_raft_viper_vkitti_seed_1234_mixed --stage virtual_kitti --validation kitti --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --seed 1234 --mixed_precision


# python -u train.py --name raft_chairs_seed_1234 \
#     --checkpoint_out=../checkpoints/raft_baseline/raft_chairs_seed_1234 \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 1234 --debug_iter

# python -u train.py --name raft_chairs_seed_0 \
#     --checkpoint_out=../checkpoints/raft_baseline/raft_chairs_seed_0 \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --debug_iter

# python -u train.py --name raft_chairs_seed_42 \
#     --checkpoint_out=../checkpoints/raft_baseline/raft_chairs_seed_42 \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 42 --debug_iter

# python -u train.py --name raft_chairs_seed_42_dropout \
#     --checkpoint_out=../checkpoints/raft_baseline/raft_chairs_seed_42_dropout \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 42 --debug_iter --dropout 0.2

# python -u train.py --name raft_chairs_seed_42_dropout_encoder_only \
#     --checkpoint_out=../checkpoints/raft_baseline/raft_chairs_seed_42_dropout_encoder_only \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 42 --debug_iter --dropout 0.2

# python -u train.py --name raft_chairs_seed_0_uncertainty_v3 \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_0_uncertainty_v3 \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 0 --uncertainty --debug_iter --log_variance --flow_variance_last_iter

# python -u train.py --name raft_chairs_seed_42_uncertainty_v3 \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_42_uncertainty_v3 \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 42 --uncertainty --debug_iter --log_variance --flow_variance_last_iter

# python -u train.py --name raft_chairs_seed_1234_uncertainty_v3 \
#     --checkpoint_out=../checkpoints/raft_uncertainty/raft_chairs_seed_1234_uncertainty_v3 \
#     --stage chairs --validation chairs \
#     --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
#     --seed 1234 --uncertainty --debug_iter --log_variance --flow_variance_last_iter

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
