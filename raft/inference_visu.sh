#!/bin/bash

# Baseline experiments
# python -m core.utils.visu.run_inference_visu \
#     --model ../checkpoints/raft_viper/train_raft_viper_seed_1234_mixed/raft-viper.pth \ 
#     --mixed_precision \
#     --out ../dump/inference/raft_viper_seed_1234_mixed_eval_kitti
# python -m core.utils.visu.run_inference_visu \
#     --model ../checkpoints/raft_vkitti/train_raft_vkitti_seed_42_mixed/raft-virtual_kitti.pth \
#     --mixed_precision \
#     --out ../dump/inference/raft_vkitti_seed_42_mixed_eval_kitti
# python -m core.utils.visu.run_inference_visu \
#     --model ../checkpoints/raft_viper_vkitti/train_raft_viper_vkitti_seed_1234_mixed/raft-virtual_kitti.pth \
#     --mixed_precision \
#     --out ../dump/inference/raft_viper_vkitti_seed_1234_mixed_eval_kitti

# python -m core.utils.visu.run_inference_visu_comparison \
#     --model_1 ../models/raft_original_models/raft-things.pth \
#     --model_2 ../checkpoints/raft_viper/train_raft_viper_seed_1234_mixed/raft-viper.pth \
#     --model_3 ../checkpoints/raft_vkitti/train_raft_vkitti_seed_42_mixed/raft-virtual_kitti.pth \
#     --model_4 ../checkpoints/raft_viper_vkitti/train_raft_viper_vkitti_seed_1234_mixed/raft-virtual_kitti.pth \
#     --mixed_precision \
#     --out ../dump/inference/raft_viper_vkitti_eval_kitti_comparison


# python -m core.utils.visu.run_inference_visu_comparison \
#     --model_1 ../models/raft_original_models/raft-things.pth \
#     --model_2 ../checkpoints/raft_viper/train_raft_viper_seed_1234_mixed/raft-viper.pth \
#     --model_3 ../checkpoints/raft_semantic/train_raft_viper_semantic_loss_seed_0_mixed/raft-viper.pth \
#     --mixed_precision \
#     --out ../dump/inference/raft_viper_semantic_eval_kitti_comparison


## Uncertainty
python -m core.utils.visu.run_inference_visu \
    --model ../checkpoints/raft_uncertainty/overfit_raft_chairs_100_sample_seed_0_nll_loss_v1_UPDATE/raft-chairs.pth \
    --out ../dump/inference/uncertainty \
    --uncertainty