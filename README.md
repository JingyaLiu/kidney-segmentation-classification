# Kidney Segmentation with nnUNetv2

This repository provides a clean, organized implementation of kidney segmentation using nnUNetv2 for medical image analysis.

## Overview

This project demonstrates kidney segmentation using nnUNetv2, a self-adapting framework for medical image segmentation. The implementation includes:

- **nnUNetv2 Integration**: Complete setup and workflow for nnUNetv2
- **KiTS21 Dataset Support**: Ready-to-use with KiTS21 kidney tumor dataset
- **3 Test Cases**: Demonstration with 3 sample cases
- **Clean Architecture**: Organized codebase with minimal dependencies

## Quick Start

### 1. Run Segmentation on 3 Test Cases

```bash
# Run the main segmentation script
python run_kidney_segmentation.py
```

This will:
- Copy 3 sample cases from the KiTS21 dataset
- Run kidney segmentation using nnUNetv2
- Generate segmentation results in `test_3_cases_output/`
- Show detailed analysis and usage instructions

### 2. View Results

The segmentation results will be saved in `test_3_cases_output/` with:
- **Segmentation files**: `.nii.gz` files with kidney and tumor labels
- **Softmax files**: `.npz` files with probability maps (if available)

## Project Structure

```
kidney-segmentation-classification/
├── run_kidney_segmentation.py      # Main segmentation script
├── nnUNet_predict.py               # nnUNet prediction script
├── nnUNet_ensemble.py              # nnUNet ensemble script
├── example_inference.py            # Example usage
├── example_kidney_segmentation.py  # Kidney-specific examples
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup
├── README.md                      # This file
├── dataset/                       # Dataset directory
│   └── Dataset721_KiTS21/         # KiTS21 dataset
├── nnUNet_trained_models/         # Trained models
├── nnunet/                        # nnUNet framework
├── test_3_cases_input/            # Input test cases
├── test_3_cases_output/           # Output results
└── backup_files/                  # Unnecessary files (moved)
```

## Requirements

- Python 3.8+
- nnUNetv2
- SimpleITK
- batchgenerators
- medpy
- torch
- numpy

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   export nnUNet_raw="/path/to/dataset"
   export nnUNet_preprocessed="/path/to/preprocessed"
   export nnUNet_results="/path/to/results"
   ```

## Usage

### Basic Segmentation

```bash
python run_kidney_segmentation.py
```

### Custom Input/Output

```bash
python nnUNet_predict.py -i input_folder -o output_folder -t Task048_KiTS_clean -m 3d_fullres
```

### Ensemble Prediction

```bash
python nnUNet_ensemble.py -i input_folder -o output_folder -t Task048_KiTS_clean -m 3d_fullres
```

## Dataset

The project is configured to work with the KiTS21 dataset:

- **Training cases**: 300 cases with kidney and tumor annotations
- **Test cases**: 3 sample cases for demonstration
- **Format**: NIfTI (.nii.gz) files
- **Labels**: 0=background, 1=kidney, 2=tumor

## Model Information

- **Architecture**: nnUNetv2 3D full resolution
- **Configuration**: 3d_fullres
- **Folds**: 5-fold cross-validation
- **Input**: CT scans with kidney and tumor regions
- **Output**: Segmentation masks with class labels

## Results

The segmentation results include:

1. **Segmentation masks**: Binary masks for kidney and tumor regions
2. **Probability maps**: Softmax probabilities for each class
3. **Analysis**: Detailed file information and statistics

## Viewing Results

### 3D Slicer
1. Open 3D Slicer
2. Load the CT image and segmentation file
3. Overlay segmentation on the original image

### ITK-SNAP
1. Open ITK-SNAP
2. Load CT image and segmentation
3. Use different colormaps for kidney and tumor

### Python
```python
import nibabel as nib
import numpy as np

# Load segmentation
seg = nib.load('test_3_cases_output/case_00000.nii.gz')
data = seg.get_fdata()
print(f'Shape: {data.shape}')
print(f'Unique labels: {np.unique(data)}')
```

## Troubleshooting

### Model Compatibility Issues

The current implementation demonstrates the nnUNetv2 workflow but may encounter model compatibility issues. This is expected because:

1. **Model Format**: Trained models may be in nnUNet v1 format
2. **Checkpoint Structure**: Different checkpoint formats between versions
3. **Trainer Classes**: Different trainer class requirements

### Solutions

1. **Convert Models**: Convert existing models to nnUNetv2 format
2. **Train New Models**: Train models specifically with nnUNetv2
3. **Use Working Example**: Check `people_face_detection_package` for working nnUNetv2

## Next Steps

1. **Model Conversion**: Convert trained models to nnUNetv2 format
2. **Training**: Train new models with nnUNetv2 from scratch
3. **Testing**: Verify model outputs and performance
4. **Integration**: Integrate with other medical imaging workflows

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- nnUNet team for the excellent segmentation framework
- KiTS21 dataset providers
- Medical imaging community for open-source tools