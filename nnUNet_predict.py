#!/usr/bin/env python3
"""
nnUNet Prediction Script
Main entry point for running inference with nnUNet models
"""

import argparse
import os
import sys
from pathlib import Path

# Add the nnunet module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nnunet'))

from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, default_cascade_trainer, default_trainer


def main():
    parser = argparse.ArgumentParser(description="nnUNet Prediction")
    parser.add_argument("-i", "--input_folder", required=True, type=str,
                        help="Input folder containing the images to segment")
    parser.add_argument("-o", "--output_folder", required=True, type=str,
                        help="Output folder where the segmentations will be saved")
    parser.add_argument("-t", "--task_name", required=True, type=str,
                        help="Task name or task ID")
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Model configuration (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres)")
    parser.add_argument("-f", "--folds", nargs="+", type=int, default=None,
                        help="Folds to use for prediction. Default: all available folds")
    parser.add_argument("-chk", "--checkpoint_name", type=str, default="model_final_checkpoint",
                        help="Checkpoint name to use")
    parser.add_argument("-tr", "--trainer_class_name", type=str, default=default_trainer,
                        help="Trainer class name")
    parser.add_argument("-p", "--plans_identifier", type=str, default=default_plans_identifier,
                        help="Plans identifier")
    parser.add_argument("--save_npz", action="store_true",
                        help="Save softmax probabilities as .npz files")
    parser.add_argument("--num_threads_preprocessing", type=int, default=6,
                        help="Number of threads for preprocessing")
    parser.add_argument("--num_threads_nifti_save", type=int, default=2,
                        help="Number of threads for saving nifti files")
    parser.add_argument("--lowres_segmentations", type=str, default=None,
                        help="Path to low resolution segmentations for cascade")
    parser.add_argument("--part_id", type=int, default=0,
                        help="Part ID for multi-GPU inference")
    parser.add_argument("--num_parts", type=int, default=1,
                        help="Number of parts for multi-GPU inference")
    parser.add_argument("--disable_tta", action="store_true",
                        help="Disable test time augmentation")
    parser.add_argument("--disable_mixed_precision", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--mode", type=str, default="normal",
                        help="Mode for prediction (normal, fast, accurate)")
    parser.add_argument("--all_in_gpu", action="store_true",
                        help="Use all available GPU memory")
    parser.add_argument("--step_size", type=float, default=0.5,
                        help="Step size for sliding window inference")
    parser.add_argument("--interp_order", type=int, default=3,
                        help="Interpolation order for resampling")
    parser.add_argument("--interp_order_z", type=int, default=0,
                        help="Interpolation order for z-axis resampling")
    parser.add_argument("--force_separate_z", action="store_true",
                        help="Force separate z-axis interpolation")
    parser.add_argument("--overwrite_existing", action="store_true",
                        help="Overwrite existing predictions")
    parser.add_argument("--disable_postprocessing", action="store_true",
                        help="Disable postprocessing")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist")
        sys.exit(1)

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Set up environment variables
    os.environ['nnUNet_raw_data_base'] = os.path.join(os.path.dirname(__file__), 'dataset')
    os.environ['nnUNet_preprocessed'] = os.path.join(os.path.dirname(__file__), 'dataset', 'preprocessed')
    os.environ['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'nnUNet_trained_models')

    try:
        predict_from_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            task_name=args.task_name,
            model=args.model,
            folds=args.folds,
            checkpoint_name=args.checkpoint_name,
            trainer_class_name=args.trainer_class_name,
            plans_identifier=args.plans_identifier,
            save_npz=args.save_npz,
            num_threads_preprocessing=args.num_threads_preprocessing,
            num_threads_nifti_save=args.num_threads_nifti_save,
            lowres_segmentations=args.lowres_segmentations,
            part_id=args.part_id,
            num_parts=args.num_parts,
            disable_tta=args.disable_tta,
            disable_mixed_precision=args.disable_mixed_precision,
            mode=args.mode,
            all_in_gpu=args.all_in_gpu,
            step_size=args.step_size,
            interp_order=args.interp_order,
            interp_order_z=args.interp_order_z,
            force_separate_z=args.force_separate_z,
            overwrite_existing=args.overwrite_existing,
            disable_postprocessing=args.disable_postprocessing,
            verbose=args.verbose
        )
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
