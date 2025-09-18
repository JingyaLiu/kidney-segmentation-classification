#!/usr/bin/env python3
"""
nnUNet Prediction Functions
Core prediction functionality for nnUNet inference
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Union, Tuple

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

# Import nnUNet modules
try:
    from nnunet.training.model_restore import restore_model
    from nnunet.preprocessing.preprocessing import resample_data_or_seg
    from nnunet.utilities.file_path_utilities import get_identifiers_from_splitted_dataset_folder
    from nnunet.utilities.one_hot_encoding import to_one_hot
    from nnunet.network_architecture.neural_network import SegmentationNetwork
    from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
    HAS_NNUNET = True
except ImportError as e:
    HAS_NNUNET = False
    print(f"Warning: nnUNet modules not found: {e}")
    print("Please ensure nnUNet is properly installed.")
    # Define dummy classes for type hints
    SegmentationNetwork = object
    nnUNetTrainer = object


def predict_case(model: SegmentationNetwork, 
                input_files: List[str], 
                output_file: str,
                do_tta: bool = True,
                mixed_precision: bool = True,
                all_in_gpu: bool = False,
                step_size: float = 0.5,
                checkpoint_name: str = "model_final_checkpoint",
                save_npz: bool = False,
                use_gaussian: bool = True,
                overwrite_existing: bool = False) -> None:
    """
    Predict segmentation for a single case
    
    Args:
        model: Loaded nnUNet model
        input_files: List of input file paths
        output_file: Output file path for segmentation
        do_tta: Whether to use test time augmentation
        mixed_precision: Whether to use mixed precision
        all_in_gpu: Whether to use all GPU memory
        step_size: Step size for sliding window inference
        checkpoint_name: Name of the checkpoint to load
        save_npz: Whether to save softmax probabilities
        use_gaussian: Whether to use Gaussian weighting
        overwrite_existing: Whether to overwrite existing files
    """
    
    if not overwrite_existing and isfile(output_file):
        print(f"Output file {output_file} already exists, skipping...")
        return
    
    # Load input data
    data, properties = load_case(input_files)
    
    # Get model properties
    trainer = model.trainer
    plans = trainer.plans
    configuration = trainer.configuration
    dataset_json = trainer.dataset_json
    
    # Preprocess data
    data = preprocess_data(data, properties, configuration, plans)
    
    # Run prediction
    if do_tta:
        prediction = predict_with_tta(model, data, step_size, mixed_precision, all_in_gpu)
    else:
        prediction = predict_without_tta(model, data, step_size, mixed_precision, all_in_gpu)
    
    # Postprocess prediction
    prediction = postprocess_prediction(prediction, properties, configuration, plans)
    
    # Save results
    save_prediction(prediction, output_file, properties, save_npz)


def load_case(input_files: List[str]) -> Tuple[np.ndarray, dict]:
    """Load and preprocess input case"""
    
    if not HAS_SITK:
        raise ImportError("SimpleITK is required for loading images. Please install it.")
    
    # Load all modalities
    data = []
    properties = []
    
    for file in input_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Input file {file} not found")
        
        # Load image
        image = sitk.ReadImage(file)
        image_array = sitk.GetArrayFromImage(image)
        
        # Get properties
        prop = {
            'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': image.GetDirection(),
            'size': image.GetSize()
        }
        
        data.append(image_array)
        properties.append(prop)
    
    # Stack modalities
    data = np.stack(data, axis=0)
    
    # Use properties from first modality
    properties = properties[0]
    
    return data, properties


def preprocess_data(data: np.ndarray, 
                   properties: dict, 
                   configuration: dict, 
                   plans: dict) -> np.ndarray:
    """Preprocess input data according to configuration"""
    
    # Normalize data
    if configuration['normalization_scheme'] == 'ZScore':
        data = (data - np.mean(data)) / np.std(data)
    elif configuration['normalization_scheme'] == 'MinMax':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Resample if needed
    target_spacing = plans['plans_per_stage'][0]['current_spacing']
    current_spacing = properties['spacing']
    
    if not np.allclose(current_spacing, target_spacing):
        data = resample_data_or_seg(data, target_spacing, is_seg=False)
    
    # Add batch dimension
    data = data[np.newaxis, ...]
    
    return data


def predict_with_tta(model: SegmentationNetwork, 
                    data: np.ndarray, 
                    step_size: float,
                    mixed_precision: bool,
                    all_in_gpu: bool) -> np.ndarray:
    """Predict with test time augmentation"""
    
    # This is a simplified version - full TTA implementation would include
    # multiple augmentations and averaging
    return predict_without_tta(model, data, step_size, mixed_precision, all_in_gpu)


def predict_without_tta(model: SegmentationNetwork, 
                       data: np.ndarray, 
                       step_size: float,
                       mixed_precision: bool,
                       all_in_gpu: bool) -> np.ndarray:
    """Predict without test time augmentation"""
    
    model.eval()
    
    with torch.no_grad():
        # Convert to tensor
        data_tensor = torch.from_numpy(data).float()
        
        if torch.cuda.is_available():
            data_tensor = data_tensor.cuda()
            model = model.cuda()
        
        # Run inference
        if mixed_precision:
            with torch.cuda.amp.autocast():
                prediction = model(data_tensor)
        else:
            prediction = model(data_tensor)
        
        # Convert to numpy
        prediction = prediction.cpu().numpy()
    
    return prediction


def postprocess_prediction(prediction: np.ndarray, 
                          properties: dict, 
                          configuration: dict, 
                          plans: dict) -> np.ndarray:
    """Postprocess prediction results"""
    
    # Convert to segmentation
    segmentation = prediction.argmax(0)
    
    # Resample back to original spacing if needed
    target_spacing = plans['plans_per_stage'][0]['current_spacing']
    current_spacing = properties['spacing']
    
    if not np.allclose(current_spacing, target_spacing):
        segmentation = resample_data_or_seg(segmentation, current_spacing, is_seg=True)
    
    return segmentation


def save_prediction(prediction: np.ndarray, 
                   output_file: str, 
                   properties: dict, 
                   save_npz: bool = False) -> None:
    """Save prediction results"""
    
    if not HAS_SITK:
        raise ImportError("SimpleITK is required for saving images. Please install it.")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as NIfTI
    image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    image.SetSpacing(properties['spacing'])
    image.SetOrigin(properties['origin'])
    image.SetDirection(properties['direction'])
    
    sitk.WriteImage(image, output_file)
    
    # Save softmax probabilities if requested
    if save_npz:
        npz_file = output_file.replace('.nii.gz', '.npz')
        np.savez_compressed(npz_file, softmax=prediction)


def predict_from_folder(input_folder: str,
                       output_folder: str,
                       task_name: str,
                       model: str,
                       folds: Optional[List[int]] = None,
                       checkpoint_name: str = "model_final_checkpoint",
                       trainer_class_name: str = "nnUNetTrainerV2",
                       plans_identifier: str = "nnUNetPlansv2.1",
                       save_npz: bool = False,
                       num_threads_preprocessing: int = 6,
                       num_threads_nifti_save: int = 2,
                       lowres_segmentations: Optional[str] = None,
                       part_id: int = 0,
                       num_parts: int = 1,
                       disable_tta: bool = False,
                       disable_mixed_precision: bool = False,
                       mode: str = "normal",
                       all_in_gpu: bool = False,
                       step_size: float = 0.5,
                       interp_order: int = 3,
                       interp_order_z: int = 0,
                       force_separate_z: bool = False,
                       overwrite_existing: bool = False,
                       disable_postprocessing: bool = False,
                       verbose: bool = False) -> None:
    """
    Main function to predict from a folder of images
    """
    
    if not HAS_NNUNET:
        raise ImportError("nnUNet modules are required. Please install nnUNet properly.")
    
    if not HAS_SITK:
        raise ImportError("SimpleITK is required. Please install it.")
    
    print(f"Starting prediction for task {task_name} with model {model}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get input files
    input_files = get_identifiers_from_splitted_dataset_folder(input_folder)
    
    if not input_files:
        print(f"No input files found in {input_folder}")
        return
    
    # Load model
    model_path = os.path.join(os.environ['RESULTS_FOLDER'], 'nnUNet', model, task_name)
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist")
        return
    
    # Load trainer and model
    trainer = restore_model(os.path.join(model_path, 'nnUNetTrainerV2__nnUNetPlansv2.1'), 
                          checkpoint_name, train=False)
    
    model = trainer.network
    
    # Process each case
    for case_id in input_files:
        print(f"Processing case {case_id}")
        
        # Find input files for this case
        case_files = []
        for file in os.listdir(input_folder):
            if file.startswith(case_id) and file.endswith('.nii.gz'):
                case_files.append(os.path.join(input_folder, file))
        
        if not case_files:
            print(f"No input files found for case {case_id}")
            continue
        
        # Sort files by modality
        case_files.sort()
        
        # Output file
        output_file = os.path.join(output_folder, f"{case_id}.nii.gz")
        
        # Predict
        try:
            predict_case(
                model=model,
                input_files=case_files,
                output_file=output_file,
                do_tta=not disable_tta,
                mixed_precision=not disable_mixed_precision,
                all_in_gpu=all_in_gpu,
                step_size=step_size,
                checkpoint_name=checkpoint_name,
                save_npz=save_npz,
                overwrite_existing=overwrite_existing
            )
            print(f"Successfully processed case {case_id}")
        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            continue
    
    print("Prediction completed!")


if __name__ == "__main__":
    # This file can be imported or run directly
    pass
