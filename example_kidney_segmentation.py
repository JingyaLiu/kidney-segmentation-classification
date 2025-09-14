#!/usr/bin/env python3
"""
Example Kidney Segmentation Script
Demonstrates how to use the available kidney segmentation models
"""

import os
import sys
import argparse
from pathlib import Path

# Add the nnunet module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nnunet'))

from nnunet.inference.predict import predict_from_folder


def list_available_models():
    """List all available models in the trained models directory"""
    
    results_folder = os.path.join(os.path.dirname(__file__), 'nnUNet_trained_models')
    
    if not os.path.exists(results_folder):
        print("No trained models found!")
        return
    
    print("Available Models:")
    print("=" * 50)
    
    nnunet_path = os.path.join(results_folder, 'nnUNet')
    if os.path.exists(nnunet_path):
        for item in os.listdir(nnunet_path):
            item_path = os.path.join(nnunet_path, item)
            if os.path.isdir(item_path):
                print(f"\nConfiguration: {item}")
                print("-" * 30)
                
                for task in os.listdir(item_path):
                    task_path = os.path.join(item_path, task)
                    if os.path.isdir(task_path):
                        print(f"  Task: {task}")
                        
                        # List trainer directories
                        trainer_dirs = []
                        for trainer_dir in os.listdir(task_path):
                            trainer_path = os.path.join(task_path, trainer_dir)
                            if os.path.isdir(trainer_path) and 'nnUNetTrainer' in trainer_dir:
                                trainer_dirs.append(trainer_dir)
                        
                        if trainer_dirs:
                            print(f"    Available trainers:")
                            for trainer in trainer_dirs:
                                print(f"      - {trainer}")


def run_kidney_segmentation(input_folder, output_folder, task_name="Task048_KiTS_clean", model="3d_fullres"):
    """Run kidney segmentation using the available models"""
    
    print(f"Running kidney segmentation...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Task: {task_name}")
    print(f"Model: {model}")
    
    # Set up environment variables
    os.environ['nnUNet_raw_data_base'] = os.path.join(os.path.dirname(__file__), 'dataset')
    os.environ['nnUNet_preprocessed'] = os.path.join(os.path.dirname(__file__), 'dataset', 'preprocessed')
    os.environ['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'nnUNet_trained_models')
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return False
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
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
        print("✓ Kidney segmentation completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during kidney segmentation: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Kidney Segmentation Example")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models")
    parser.add_argument("--input_folder", type=str,
                        help="Path to input folder containing kidney images")
    parser.add_argument("--output_folder", type=str,
                        help="Path to output folder for segmentation results")
    parser.add_argument("--task_name", type=str, default="Task048_KiTS_clean",
                        help="Task name (default: Task048_KiTS_clean)")
    parser.add_argument("--model", type=str, default="3d_fullres",
                        help="Model configuration (default: 3d_fullres)")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    if not args.input_folder or not args.output_folder:
        print("Error: Both --input_folder and --output_folder are required")
        print("Use --list-models to see available models")
        return
    
    # Run kidney segmentation
    success = run_kidney_segmentation(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        task_name=args.task_name,
        model=args.model
    )
    
    if success:
        print("\n🎉 Kidney segmentation completed successfully!")
        print(f"Results saved to: {args.output_folder}")
    else:
        print("\n❌ Kidney segmentation failed!")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help and list models
        print("Kidney Segmentation & Classification with nnUNet")
        print("=" * 50)
        print()
        print("Available commands:")
        print("  --list-models              List all available models")
        print("  --input_folder PATH        Input folder with kidney images")
        print("  --output_folder PATH       Output folder for results")
        print("  --task_name NAME           Task name (default: Task048_KiTS_clean)")
        print("  --model CONFIG             Model config (default: 3d_fullres)")
        print()
        print("Example usage:")
        print("  python example_kidney_segmentation.py --list-models")
        print("  python example_kidney_segmentation.py --input_folder /path/to/images --output_folder /path/to/results")
        print()
        list_available_models()
    else:
        main()
