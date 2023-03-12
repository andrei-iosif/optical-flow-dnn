#!/bin/bash
BASE_PATH="/home/mnegru/repos/optical-flow-dnn"

python -u run_inference_visu.py --model $BASE_PATH/checkpoints/raft/checkpoints/raft-sintel.pth --out $BASE_PATH/dump/raft_sintel
