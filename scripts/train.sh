#!/bin/bash

# Script to train the RDAUP model
# Usage: ./train.sh [source_domain] [target_domain]

# Default parameters
SOURCE=${1:-amazon}
TARGET=${2:-webcam}
BATCH_SIZE=32
EPOCHS=30
MODEL=resnet50
LR=1e-3

# Ensure the output directories exist
mkdir -p experiments/checkpoints experiments/logs experiments/visualizations

# Training command
python train_rdaup.py \
    --source_root "data/Office31/${SOURCE}" \
    --target_root "data/Office31/${TARGET}" \
    --model_name ${MODEL} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --lambda_adv 1.0 \
    --lambda_ent 1.0 \
    --lambda_upl 1.0 \
    --freeze_until 1 \
    --use_amp true \
    --save_name "rdaup_${SOURCE}2${TARGET}.pth"
