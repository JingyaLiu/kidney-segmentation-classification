#!/bin/bash

python preprocessing/data_preprocessing_with_masks.py \
    --ct_data_root path/to/ct/data \
    --seg_data_root path/to/segmentation/data \
    --kits_json_path path/to/kits.json \
    --output_dir path/to/output

echo "Preprocessing complete!"