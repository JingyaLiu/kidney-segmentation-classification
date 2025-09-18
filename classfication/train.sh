#!/bin/bash

python train/train.py \
    --data_dir ./processed_data_with_smooth_masks_augmented \
    --output_dir ./results/training \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --dropout_rate 0.3 \
    --pretrained_type imagenet \
    --pool_type attn \
    --use_masks \
    --input_channels 4 \
    --num_slices 4 \
    --test_size 0.2 \
    --num_folds 5 \
    --scheduler cosine \
    --grad_clip 1.0 \
    --splits_file data_splits.json \
    --imbalance_method both \
    --loss_type cross_entropy \
    --num_workers 2 \
    --seed 42