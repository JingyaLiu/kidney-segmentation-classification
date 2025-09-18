# Kidney Segmentation with nnUNet

This repository provides kidney segmentation using nnUNet for medical image analysis.

## 🚀 **How to Run Inference**

### **Method 1: Robust Script (Recommended)**
```bash
# Run with default settings
python run_kidney_segmentation_robust.py

# Run with custom input/output
python run_kidney_segmentation_robust.py -i your_input_folder -o your_output_folder

# Run with specific folds
python run_kidney_segmentation_robust.py -f 3

# Run with specific test cases
python run_kidney_segmentation_robust.py --test-cases case_00000 case_00001
```

### **Method 2: Direct Command**
```bash
# Step 1: Set environment variables (this tells nnUNet where to find the model)
export nnUNet_raw_data_base="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/"
# export nnUNet_preprocessed="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/"
export RESULTS_FOLDER="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/nnUNet_trained_models/nnUNet"

# Step 2: Run inference (nnUNet finds the model using RESULTS_FOLDER + task name)
nnUNet_predict -i test_3_cases_input/NIH/ -o test_3_cases_output/NIH/ -t Task064_KiTS_labelsFixed -m 3d_fullres -f 3 --overwrite_existing
```

**How it works:**
- `-t Task064_KiTS_labelsFixed` tells nnUNet which task/model to use
- `-m 3d_fullres` specifies the model configuration
- `-f 3` specifies which fold to use
- nnUNet automatically finds the model at: `$RESULTS_FOLDER/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/`

**Model weights:**
https://ccnymailcuny-my.sharepoint.com/:f:/g/personal/jliu008_citymail_cuny_edu/EkaFGLQB5XdKvVb1y2G7NHgB6RGk344fS3S2oV9-X4D_9g?e=qo0npb


## 📋 **Requirements**

```bash
# Install nnUNet v1
pip install nnunet

# Install additional dependencies
pip install -r requirements.txt
```

## 🔗 **Model Setup**

### **1. Download/Prepare Trained Models**
The repository expects trained models in the following structure:
```
nnUNet_trained_models/
└── nnUNet/
    └── nnUNet/
        └── 3d_fullres/
            └── Task064_KiTS_labelsFixed/
                └── nnUNetTrainerV2__nnUNetPlansv2.1/
                    ├── fold_3/
                    │   ├── model_final_checkpoint.model
                    │   └── model_final_checkpoint.model.pkl
                    └── plans.pkl
```

### **2. Set Environment Variables**
```bash
# Set these environment variables before running inference
export nnUNet_raw_data_base="/path/to/your/dataset"
export nnUNet_preprocessed="/path/to/your/preprocessed"
export RESULTS_FOLDER="/path/to/your/nnUNet_trained_models/nnUNet"
```

### **3. For This Repository (Default Paths)**
```bash
# Use the default paths in this repository
export nnUNet_raw_data_base="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/dataset"
export nnUNet_preprocessed="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/dataset/preprocessed"
export RESULTS_FOLDER="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/nnUNet_trained_models/nnUNet"
```

### **4. Verify Model is Available**
```bash
# Check if the model exists
ls -la nnUNet_trained_models/nnUNet/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/
```

### **5. Model Path Resolution**
nnUNet automatically constructs the model path as follows:
```bash
# Environment variable + task name + model config + trainer name + fold
$RESULTS_FOLDER/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/

# Full path in this repository:
/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/nnUNet_trained_models/nnUNet/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/
```

**Required files in the model directory:**
- `model_final_checkpoint.model` (the trained model weights)
- `model_final_checkpoint.model.pkl` (model metadata)
- `plans.pkl` (training configuration)

##  **Visualize 
python visual/visualize_segmentation.py -i input_folder -o output_folder --case case_name --save-dir visualizations

.nii.gz

python visual/visualize_segmentation.py -i test_3_cases_input -o test_3_cases_output/NIH/ --case TCGA-B9-4113_1.3.6.1.4.1.14519.5.2.1.8421.4010.244934789279678299029033011559 --save-dir visualizations

## 🔧 **Troubleshooting**

1. **nnUNet not found**: `pip install nnunet`
2. **Model not found**: Check if `nnUNet_trained_models/nnUNet/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/` exists
3. **CUDA out of memory**: Use CPU inference or reduce batch size
4. **Input format**: Ensure files are named `case_XXXX_0000.nii.gz`
