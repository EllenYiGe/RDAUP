#!/bin/bash

# Office31 dataset training script

# Set data paths
SOURCE_ROOT="./data/Office31/amazon"
TARGET_ROOT="./data/Office31/webcam"
SOURCE_DOMAIN="amazon"
TARGET_DOMAIN="webcam"

# Create necessary directories
mkdir -p experiments/logs
mkdir -p experiments/checkpoints

# Train the model
python train_rdaup.py \
    --source_root $SOURCE_ROOT \
    --target_root $TARGET_ROOT \
    --source_domain $SOURCE_DOMAIN \
    --target_domain $TARGET_DOMAIN \
    --model_name "resnet50" \
    --num_classes 31 \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.001 \
    --lambda_adv 1.0 \
    --seed 42 \
    --log_interval 10 \
    --checkpoint_dir "experiments/checkpoints" \
    --log_dir "experiments/logs" \
    --use_wandb
