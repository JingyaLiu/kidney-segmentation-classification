"""
nnUNet Paths Configuration
Default paths and identifiers for nnUNet
"""

import os

# Default identifiers
default_plans_identifier = "nnUNetPlansv2.1"
default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"
default_trainer = "nnUNetTrainerV2"

# Default paths (can be overridden by environment variables)
default_raw_data_base = os.path.join(os.path.dirname(__file__), "..", "dataset")
default_preprocessed = os.path.join(default_raw_data_base, "preprocessed")
default_results_folder = os.path.join(os.path.dirname(__file__), "..", "nnUNet_trained_models")

# Get paths from environment variables or use defaults
nnUNet_raw_data_base = os.environ.get('nnUNet_raw_data_base', default_raw_data_base)
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed', default_preprocessed)
RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', default_results_folder)

# Ensure paths exist
os.makedirs(nnUNet_raw_data_base, exist_ok=True)
os.makedirs(nnUNet_preprocessed, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
