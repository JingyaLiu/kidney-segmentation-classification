#!/bin/bash
# Inference script for dual-path medical classification model

python inference/inference.py \
    --data_dir ./processed_data_with_smooth_masks_augmented \
    --model_weights_dir ./results/training \
    --output_dir ./results/inference \
    --folds 1 2 3 4 5 \
    --num_classes 5 \
    --num_slices 4 \
    --gin_channels 4 \
    --lin_channels 4 \
    --use_masks \
    --clear_cell_threshold 0.0 \
    --batch_size 16 \
    --num_workers 2 \
    --device auto
