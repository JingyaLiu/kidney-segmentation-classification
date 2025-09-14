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
# Set environment variables
export nnUNet_raw_data_base="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/dataset"
export nnUNet_preprocessed="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/dataset/preprocessed"
export RESULTS_FOLDER="/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/nnUNet_trained_models/nnUNet"

# Run inference
nnUNet_predict -i test_3_cases_input -o test_3_cases_output -t Task064_KiTS_labelsFixed -m 3d_fullres -f 3 --overwrite_existing
```


## 📋 **Requirements**

```bash
# Install nnUNet v1
pip install nnunet

# Install additional dependencies
pip install -r requirements.txt
```

## 📊 **Results**

- **Segmentation Labels**: 0=background, 1=kidney, 3=tumor
- **Model Performance**: 96%+ Dice score for kidney segmentation
- **Output Format**: NIfTI (.nii.gz) files

## 🔧 **Troubleshooting**

1. **nnUNet not found**: `pip install nnunet`
2. **Model not found**: Check if `nnUNet_trained_models/nnUNet/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/` exists
3. **CUDA out of memory**: Use CPU inference or reduce batch size
4. **Input format**: Ensure files are named `case_XXXX_0000.nii.gz`