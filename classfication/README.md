# Kidney AI Classification

Dual-path kidney tumor classification 


## Repository Structure

```
classfication/
├── train/                          # Training scripts
│   └── train.py                    # Main training script
├── inference/                      # Inference scripts
│   ├── inference.py                # Ensemble inference
│   └── inference_3_slices.py       # 3-slice inference
├── preprocessing/                  # Data preprocessing
│   └── data_preprocessing_with_masks.py
├── model/                          # Model definitions
│   ├── models.py                   # MedicalDualPathNetwork
│   └── model_utils.py              # Model utilities
├── utils/                          # Utility functions
│   ├── augmentation.py             # Data augmentation
│   ├── losses.py                   # Loss functions
│   ├── training.py                 # Training utilities
│   ├── dataset.py                  # Dataset classes
│   ├── data_split.py               # Data splitting
│   └── visualize_result.ipynb      # Visualization notebook
├── train.sh                        # Training script
├── inference.sh                    # Inference script
├── preprocess.sh                   # Preprocessing script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Activate conda environment (if using)
conda activate kidney
```

### 2. Data Preprocessing

```bash
# Run preprocessing with default paths
./preprocess.sh

# Or specify custom paths
./preprocess.sh \
    --ct-data-root /path/to/ct/data \
    --seg-data-root /path/to/segmentation/data \
    --kits-json-path /path/to/kits.json \
    --output-dir /path/to/output

# Or use environment variables
export CT_DATA_ROOT="/path/to/ct/data"
export SEG_DATA_ROOT="/path/to/segmentation/data"
export KITS_JSON_PATH="/path/to/kits.json"
export OUTPUT_DIR="/path/to/output"
./preprocess.sh
```

### 3. Training

```bash
# Train model
./train.sh
```

### 4. Inference

```bash
# Run inference
./inference.sh
```
### Model Weights ###
Insert link



## Data Preprocessing

### Required Parameters:
- `--ct_data_root`: Path to CT image data directory
- `--seg_data_root`: Path to segmentation data directory  
- `--kits_json_path`: Path to JSON metadata file with tumor type
- `--output_dir`: Output directory for processed data

### Usage Examples:
```bash
# Direct Python usage
python preprocessing/data_preprocessing_with_masks.py \
    --ct_data_root /path/to/ct \
    --seg_data_root /path/to/seg \
    --kits_json_path /path/to/kits.json \
    --output_dir /path/to/output

# Using the shell script
./preprocess.sh --help  # Show all options
```

## Model Architecture

The `MedicalDualPathNetwork` combines:
- **Global path**: ResNet50 backbone for global context
- **Local path**: ResNet50 backbone for local tumor features
- **Attention pooling**: Adaptive slice aggregation
- **Fusion**: Weighted combination of global and local features