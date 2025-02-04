#!/bin/bash

# Script to evaluate the RDAUP model
# Usage: ./evaluate.sh [source_domain] [target_domain]

# Default parameters
SOURCE=${1:-amazon}
TARGET=${2:-webcam}
BATCH_SIZE=32
MODEL=resnet50

# Model path
MODEL_PATH="experiments/checkpoints/rdaup_${SOURCE}2${TARGET}.pth"

# Evaluation command
python test_rdaup.py \
    --model_path ${MODEL_PATH} \
    --test_root "data/Office31/${TARGET}" \
    --model_name ${MODEL} \
    --batch_size ${BATCH_SIZE}
