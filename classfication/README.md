# Kidney AI Classification

Dual-path kidney tumor classification

## Repository Structure

```
classfication/
├── train/                          # Training scripts
│   └── train.py
├── inference/                      # Inference scripts
│   └── inference.py
├── preprocessing/                  # Data preprocessing
│   ├── data_preprocessing_with_masks.py
│   └── general_data_preprocessing.py
├── model/                          # Model definitions
│   ├── models.py
│   └── model_utils.py
├── utils/                          # Utility functions
│   ├── augmentation.py
│   ├── losses.py
│   ├── training.py
│   ├── dataset.py
│   └── data_split.py
├── analysis/                       # Analysis scripts
│   ├── all_metrics_table.py
│   ├── class_breakdown.py
│   └── compare_all_models.py
├── train.sh                        # Training script
├── inference.sh                    # Inference script
├── preprocess_with_masks.sh        # Preprocessing script
├── results/                        # Results directory
└── requirements.txt
```


