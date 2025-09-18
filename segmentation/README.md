# Kidney Segmentation with nnUNet



## 📦 **Installation**

### **nnUNet v1 Installation**
```bash
# Install nnUNet v1 (recommended for this project)
pip install nnunet

```

**Official nnUNet Repository:** [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

### **Additional Dependencies**
```bash
# Install additional dependencies
pip install -r requirements.txt
```

## 🔧 **Environment Setup**

### **Set Environment Variables**
```bash
# Set these environment variables before running any nnUNet commands
export nnUNet_raw_data_base="./dataset"
export nnUNet_preprocessed="./dataset/preprocessed"
export RESULTS_FOLDER="./nnUNet_trained_models/nnUNet"

# For permanent setup, add to ~/.bashrc
echo 'export nnUNet_raw_data_base="./dataset"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="./dataset/preprocessed"' >> ~/.bashrc
echo 'export RESULTS_FOLDER="./nnUNet_trained_models/nnUNet"' >> ~/.bashrc
source ~/.bashrc
```


### **Dataset JSON Format**
Create a `dataset.json` file in your task folder:
```json
{
    "name": "KiTS_labelsFixed",
    "description": "Kidney Tumor Segmentation",
    "tensorImageSize": "3D",
    "reference": "KiTS Challenge",
    "licence": "CC-BY-SA 4.0",
    "release": "1.0",
    "modality": {
        "0": "CT"
    },
    "labels": {
        "0": "background",
        "1": "kidney",
        "2": "tumor"
    },
    "numTraining": 210,
    "numTest": 90,
    "training": [
        {
            "image": "./imagesTr/case_00000_0000.nii.gz",
            "label": "./labelsTr/case_00000.nii.gz"
        }
    ],
    "test": [
        "./imagesTs/case_00000_0000.nii.gz"
    ]
}
```

## ⚙️ **Preprocessing**

### **Run Preprocessing**
```bash
# Plan and preprocess the dataset
nnUNet_plan_and_preprocess -t 064 --verify_dataset_integrity

# Alternative: Use task name
nnUNet_plan_and_preprocess -t Task064_KiTS_labelsFixed --verify_dataset_integrity
```

**What preprocessing does:**
- Analyzes dataset properties (spacing, intensity distribution, etc.)
- Creates experiment plans for different configurations
- Preprocesses images (resampling, normalization, etc.)
- Generates data splits for cross-validation

## 🏋️ **Training**

### **Train Individual Models**
```bash
# Train 3D full resolution model (recommended for kidney segmentation)
nnUNet_train 3d_fullres nnUNetTrainerV2 064 0  # Fold 0
nnUNet_train 3d_fullres nnUNetTrainerV2 064 1  # Fold 1
nnUNet_train 3d_fullres nnUNetTrainerV2 064 2  # Fold 2
nnUNet_train 3d_fullres nnUNetTrainerV2 064 3  # Fold 3
nnUNet_train 3d_fullres nnUNetTrainerV2 064 4  # Fold 4

```

### **Training Parameters**
- **Configuration**: `2d`, `3d_lowres`, `3d_fullres`, `3d_cascade_fullres`
- **Trainer**: `nnUNetTrainerV2` (default), `nnUNetTrainerV2_ep4000`, etc.
- **Task**: `064` or `Task064_KiTS_labelsFixed`
- **Fold**: `0`, `1`, `2`, `3`, `4` (for 5-fold cross-validation)

```

### **Training Output**
Trained models are saved in:
```
$RESULTS_FOLDER/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/
├── fold_0/
│   ├── model_final_checkpoint.model
│   └── model_final_checkpoint.model.pkl
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
└── plans.pkl
```

## 🚀 **Inference**

```bash
# Run inference using trained model
nnUNet_predict -i test_3_cases_input/NIH/ -o test_3_cases_output/NIH/ -t Task064_KiTS_labelsFixed -m 3d_fullres -f 3 --overwrite_existing
```

**Command Parameters:**
- `-i`: Input folder containing test images
- `-o`: Output folder for predictions
- `-t Task064_KiTS_labelsFixed`: Task name/ID
- `-m 3d_fullres`: Model configuration
- `-f 3`: Fold number to use
- `--overwrite_existing`: Overwrite existing predictions

**Model Path Resolution:**
nnUNet automatically finds the model at:
```
$RESULTS_FOLDER/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/
```

## 📥 **Models Weights**

### **Download Model Weights**
Pre-trained models are available at:
**Model weights:** [Download Link](https://ccnymailcuny-my.sharepoint.com/:f:/g/personal/jliu008_citymail_cuny_edu/EkaFGLQB5XdKvVb1y2G7NHgB6RGk344fS3S2oV9-X4D_9g?e=qo0npb)

### **Model Structure**
Extract the downloaded models to:
```
./nnUNet_trained_models/
└── nnUNet/
    └── nnUNet/
        └── 3d_fullres/
            └── Task064_KiTS_labelsFixed/
                └── nnUNetTrainerV2__nnUNetPlansv2.1/
                    ├── fold_0/
                    │   ├── model_final_checkpoint.model
                    │   └── model_final_checkpoint.model.pkl
                    ├── fold_1/
                    ├── fold_2/
                    ├── fold_3/
                    ├── fold_4/
                    └── plans.pkl
```

### **Verify Model Installation**
```bash
# Check if models exist
ls -la ./nnUNet_trained_models/nnUNet/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/

# Check specific fold
ls -la ./nnUNet_trained_models/nnUNet/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/
```

**Required files in each fold directory:**
- `model_final_checkpoint.model` (trained model weights)
- `model_final_checkpoint.model.pkl` (model metadata)


## 📚 **Additional Resources**

- **nnUNet Documentation**: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- **nnUNet Paper**: [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://www.nature.com/articles/s41592-020-01008-z)
- **KiTS Challenge**: [https://kits-challenge.org/](https://kits-challenge.org/)
- **Medical Image Segmentation**: [https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/)