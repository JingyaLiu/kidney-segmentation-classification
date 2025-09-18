# AI-Driven Robust Kidney and Renal Mass Segmentation and Classification

This repository provides a comprehensive AI-driven framework for automatic kidney and renal mass segmentation and classification on 3D CT images.

## Overview

This end-to-end AI framework addresses two critical tasks in kidney cancer diagnosis:

1. **Segmentation**: Automatic segmentation of kidney and renal mass regions using 3D deep learning architectures
2. **Classification**: Prediction of histological subtypes of renal cell carcinoma (clear cell, chromophobe, oncocytoma, papillary, and other RCC subtypes)

The framework utilizes a novel dual-path classification network that leverages both local and global features, enhanced with weakly supervised learning to improve robustness across datasets from various institutions.

## Repository Structure

```
kidney-segmentation-classification/
├── segmentation/                    # Kidney and renal mass segmentation
│   ├── README.md                   # Detailed segmentation guide
│   ├── run_kidney_segmentation_robust.py
│   ├── visual/                     # Visualization tools
│   └── ...                        # nnUNet-based segmentation
├── classfication/                  # Renal mass classification
│   ├── README.md                   # Detailed classification guide
│   ├── train/                      # Training scripts
│   ├── inference/                  # Inference scripts
│   ├── preprocessing/              # Data preprocessing
│   ├── model/                      # Model definitions
│   └── ...                        # Dual-path classification network
└── README.md                      # This file
```

## Quick Start

### 1. Segmentation (nnUNet-based)
Navigate to the `segmentation/` folder for kidney and renal mass segmentation:
```bash
cd segmentation/
# Follow the detailed README.md in that folder
```

### 2. Classification (Dual-path Network)
Navigate to the `classfication/` folder for renal mass subtype classification:
```bash
cd classfication/
# Follow the detailed README.md in that folder
```

## Key Features

- **3D Segmentation**: Res-UNet architecture for accurate kidney and renal mass segmentation
- **Dual-path Classification**: Combines global and local features for robust tumor subtype prediction
- **Weakly Supervised Learning**: Leverages domain adaptation for multi-institutional datasets
- **End-to-end Pipeline**: Complete workflow from raw CT images to diagnosis
- **Cross-dataset Validation**: Robust performance across different institutions



## Documentation

- **[Segmentation Guide](segmentation/README.md)**: Complete nnUNet-based segmentation workflow
- **[Classification Guide](classfication/README.md)**: Dual-path network training and inference

## Related Resources

- **KiTS Challenge**: [https://kits-challenge.org/](https://kits-challenge.org/)
- **nnUNet Repository**: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- **Medical Image Segmentation**: [https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/)

## Citation

If you use this repository in your research, please cite our paper:

```bibtex
@article{liu2023ai,
  title={AI-Driven Robust Kidney and Renal Mass Segmentation and Classification on 3D CT Images},
  author={Liu, Jingya and Yildirim, Onur and Akin, Oguz and Tian, Yingli},
  journal={Bioengineering},
  volume={10},
  number={1},
  pages={116},
  year={2023},
  publisher={MDPI},
  doi={10.3390/bioengineering10010116},
  pmid={36671688},
  pmcid={PMC9854669}
}
```

**Paper Link**: [https://pubmed.ncbi.nlm.nih.gov/36671688/](https://pubmed.ncbi.nlm.nih.gov/36671688/)

