#!/usr/bin/env python3
"""
nnUNet Ensemble Script
Main entry point for ensembling multiple nnUNet predictions
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Handle optional dependencies
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("Warning: SimpleITK not found. Please install it for full functionality.")

try:
    from batchgenerators.utilities.file_and_folder_operations import *
    HAS_BATCHGENERATORS = True
except ImportError:
    HAS_BATCHGENERATORS = False
    print("Warning: batchgenerators not found. Please install it for full functionality.")


def ensemble_predictions(prediction_folders: List[str], 
                        output_folder: str,
                        postprocessing_file: Optional[str] = None,
                        overwrite_existing: bool = False) -> None:
    """
    Ensemble multiple prediction folders
    
    Args:
        prediction_folders: List of folders containing predictions
        output_folder: Output folder for ensembled predictions
        postprocessing_file: Path to postprocessing configuration file
        overwrite_existing: Whether to overwrite existing files
    """
    
    print(f"Ensembling predictions from {len(prediction_folders)} folders")
    print(f"Output folder: {output_folder}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all case identifiers
    all_cases = set()
    for folder in prediction_folders:
        if os.path.exists(folder):
            cases = get_identifiers_from_splitted_dataset_folder(folder)
            all_cases.update(cases)
    
    if not all_cases:
        print("No cases found in prediction folders")
        return
    
    print(f"Found {len(all_cases)} cases to ensemble")
    
    # Process each case
    for case_id in all_cases:
        print(f"Processing case {case_id}")
        
        # Find all prediction files for this case
        prediction_files = []
        for folder in prediction_folders:
            pred_file = os.path.join(folder, f"{case_id}.nii.gz")
            if os.path.exists(pred_file):
                prediction_files.append(pred_file)
        
        if not prediction_files:
            print(f"No prediction files found for case {case_id}")
            continue
        
        # Output file
        output_file = os.path.join(output_folder, f"{case_id}.nii.gz")
        
        if not overwrite_existing and os.path.exists(output_file):
            print(f"Output file {output_file} already exists, skipping...")
            continue
        
        try:
            # Load and ensemble predictions
            ensemble_prediction = load_and_ensemble_predictions(prediction_files)
            
            # Save ensembled prediction
            save_ensemble_prediction(ensemble_prediction, output_file)
            
            print(f"Successfully ensembled case {case_id}")
        except Exception as e:
            print(f"Error ensembling case {case_id}: {str(e)}")
            continue
    
    print("Ensemble completed!")


def load_and_ensemble_predictions(prediction_files: List[str]) -> np.ndarray:
    """Load and ensemble multiple prediction files"""
    
    if not HAS_SITK:
        raise ImportError("SimpleITK is required for loading images. Please install it.")
    
    predictions = []
    
    for file in prediction_files:
        # Load prediction
        image = sitk.ReadImage(file)
        prediction = sitk.GetArrayFromImage(image)
        predictions.append(prediction)
    
    # Simple majority voting ensemble
    predictions = np.stack(predictions, axis=0)
    ensemble = np.argmax(np.sum(predictions, axis=0), axis=0)
    
    return ensemble


def save_ensemble_prediction(prediction: np.ndarray, output_file: str) -> None:
    """Save ensembled prediction"""
    
    if not HAS_SITK:
        raise ImportError("SimpleITK is required for saving images. Please install it.")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as NIfTI
    image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    sitk.WriteImage(image, output_file)


def main():
    parser = argparse.ArgumentParser(description="nnUNet Ensemble")
    parser.add_argument("-f", "--folders", nargs="+", required=True,
                        help="Folders containing predictions to ensemble")
    parser.add_argument("-o", "--output_folder", required=True,
                        help="Output folder for ensembled predictions")
    parser.add_argument("-pp", "--postprocessing_file", type=str, default=None,
                        help="Path to postprocessing configuration file")
    parser.add_argument("--overwrite_existing", action="store_true",
                        help="Overwrite existing predictions")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate input folders
    for folder in args.folders:
        if not os.path.exists(folder):
            print(f"Error: Prediction folder {folder} does not exist")
            sys.exit(1)
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    try:
        ensemble_predictions(
            prediction_folders=args.folders,
            output_folder=args.output_folder,
            postprocessing_file=args.postprocessing_file,
            overwrite_existing=args.overwrite_existing
        )
        print("Ensemble completed successfully!")
    except Exception as e:
        print(f"Error during ensemble: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
