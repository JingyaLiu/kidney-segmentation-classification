"""
nnUNet Configuration
Default configuration settings for nnUNet
"""

import os

# Default number of threads for various operations
default_num_threads = 6

# Default settings
default_plans_identifier = "nnUNetPlansv2.1"
default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"
default_trainer = "nnUNetTrainerV2"

# Paths
default_raw_data_base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
default_preprocessed = os.path.join(default_raw_data_base, "preprocessed")
default_results_folder = os.path.join(os.path.dirname(__file__), "..", "..", "nnUNet_trained_models")

# Get paths from environment variables or use defaults
nnUNet_raw_data_base = os.environ.get('nnUNet_raw_data_base', default_raw_data_base)
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed', default_preprocessed)
RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', default_results_folder)

# Ensure paths exist
os.makedirs(nnUNet_raw_data_base, exist_ok=True)
os.makedirs(nnUNet_preprocessed, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
