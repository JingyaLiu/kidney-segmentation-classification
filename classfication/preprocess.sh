#!/bin/bash
conda activate kidney

echo "Starting data preprocessing with smooth tumor masks..."

python preprocessing/data_preprocessing_with_masks.py

echo "Preprocessing complete!"
echo "Output directory: ./processed_data_with_smooth_masks_augmented"
