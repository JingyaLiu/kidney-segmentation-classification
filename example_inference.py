#!/usr/bin/env python3
"""
Example Inference Script
Demonstrates how to use nnUNet for inference programmatically
"""

import os
import sys
import argparse
from pathlib import Path

# Add the nnunet module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nnunet'))

from nnunet.inference.predict import predict_from_folder


def example_inference():
    """Example of running nnUNet inference"""
    
    # Example parameters
    input_folder = "/path/to/your/input/images"
    output_folder = "/path/to/your/output/predictions"
    task_name = "Task001_MyDataset"
    model = "3d_fullres"
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist")
        print("Please update the input_folder path in this script")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up environment variables
    os.environ['nnUNet_raw_data_base'] = os.path.join(os.path.dirname(__file__), 'dataset')
    os.environ['nnUNet_preprocessed'] = os.path.join(os.path.dirname(__file__), 'dataset', 'preprocessed')
    os.environ['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'nnUNet_trained_models')
    
    print("Running nnUNet inference...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Task: {task_name}")
    print(f"Model: {model}")
    
    try:
        # Run prediction
        predict_from_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            task_name=task_name,
            model=model,
            save_npz=True,  # Save softmax probabilities
            overwrite_existing=True
        )
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        print("Please check your input data format and model availability")


def main():
    parser = argparse.ArgumentParser(description="Example nnUNet Inference")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to input folder containing images")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to output folder for predictions")
    parser.add_argument("--task_name", type=str, default="Task001_MyDataset",
                        help="Task name or ID")
    parser.add_argument("--model", type=str, default="3d_fullres",
                        help="Model configuration")
    
    args = parser.parse_args()
    
    # Update paths
    input_folder = args.input_folder
    output_folder = args.output_folder
    task_name = args.task_name
    model = args.model
    
    # Set up environment variables
    os.environ['nnUNet_raw_data_base'] = os.path.join(os.path.dirname(__file__), 'dataset')
    os.environ['nnUNet_preprocessed'] = os.path.join(os.path.dirname(__file__), 'dataset', 'preprocessed')
    os.environ['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'nnUNet_trained_models')
    
    print("Running nnUNet inference...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Task: {task_name}")
    print(f"Model: {model}")
    
    try:
        # Run prediction
        predict_from_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            task_name=task_name,
            model=model,
            save_npz=True,
            overwrite_existing=True
        )
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        print("Please check your input data format and model availability")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run example
        example_inference()
    else:
        # Arguments provided, run with arguments
        main()
