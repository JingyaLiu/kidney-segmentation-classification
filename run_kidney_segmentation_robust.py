#!/usr/bin/env python3
"""
Robust Kidney Segmentation Inference using nnUNet v1
This script provides a robust implementation for kidney segmentation with comprehensive error handling.
"""

import os
import sys
import shutil
import time
import subprocess
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KidneySegmentationInference:
    """Robust kidney segmentation inference class"""
    
    def __init__(self, base_dir=None):
        """Initialize the inference class"""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.setup_environment_variables()
        
    def setup_environment_variables(self):
        """Set up nnUNet v1 environment variables"""
        logger.info("🔧 Setting up nnUNet v1 environment variables...")
        
        # Set nnUNet v1 environment variables
        os.environ['nnUNet_raw_data_base'] = str(self.base_dir / "dataset")
        os.environ['nnUNet_preprocessed'] = str(self.base_dir / "dataset" / "preprocessed")
        os.environ['RESULTS_FOLDER'] = str(self.base_dir / "nnUNet_trained_models" / "nnUNet")
        
        logger.info(f"   nnUNet_raw_data_base: {os.environ['nnUNet_raw_data_base']}")
        logger.info(f"   nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
        logger.info(f"   RESULTS_FOLDER: {os.environ['RESULTS_FOLDER']}")
        
    def check_dependencies(self):
        """Check if required dependencies are available"""
        logger.info("🔍 Checking dependencies...")
        
        # Check if nnUNet is installed
        try:
            result = subprocess.run(['nnUNet_predict', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("nnUNet_predict command failed")
            logger.info("   ✓ nnUNet v1 is installed")
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
            logger.error(f"   ❌ nnUNet v1 not found: {e}")
            logger.error("   Please install nnUNet v1: pip install nnunet")
            return False
            
        return True
    
    def check_model_availability(self, task_name="Task064_KiTS_labelsFixed", model="3d_fullres"):
        """Check if the trained model is available"""
        logger.info(f"🔍 Checking model availability...")
        
        model_path = self.base_dir / "nnUNet_trained_models" / "nnUNet" / "nnUNet" / model / task_name
        
        if not model_path.exists():
            logger.error(f"   ❌ Model path not found: {model_path}")
            return False
            
        # Check for available folds
        available_folds = []
        for fold_dir in model_path.glob("fold_*"):
            if fold_dir.is_dir() and (fold_dir / "model_final_checkpoint.model").exists():
                fold_num = fold_dir.name.split("_")[1]
                available_folds.append(int(fold_num))
        
        if not available_folds:
            logger.error(f"   ❌ No trained folds found in {model_path}")
            return False
            
        logger.info(f"   ✓ Model found: {task_name}")
        logger.info(f"   ✓ Available folds: {sorted(available_folds)}")
        return available_folds
    
    def prepare_input_data(self, input_dir, test_cases=None):
        """Prepare input data for inference"""
        logger.info("📋 Preparing input data...")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"   ❌ Input directory not found: {input_path}")
            return False
            
        # Check for test cases
        if test_cases is None:
            test_cases = ["case_00000", "case_00001", "case_00002"]
            
        found_cases = []
        for case in test_cases:
            case_file = input_path / f"{case}_0000.nii.gz"
            if case_file.exists():
                size_mb = case_file.stat().st_size / (1024 * 1024)
                logger.info(f"   ✓ {case}: {size_mb:.1f} MB")
                found_cases.append(case)
            else:
                logger.warning(f"   ⚠️  {case}: File not found")
                
        if not found_cases:
            logger.error("   ❌ No valid test cases found")
            return False
            
        logger.info(f"   ✓ Found {len(found_cases)} valid test cases")
        return found_cases
    
    def run_inference(self, input_dir, output_dir, task_name="Task064_KiTS_labelsFixed", 
                     model="3d_fullres", folds=None, overwrite=True):
        """Run nnUNet v1 inference"""
        logger.info("🤖 Running nnUNet v1 Inference...")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get available folds if not specified
        if folds is None:
            available_folds = self.check_model_availability(task_name, model)
            if not available_folds:
                return False
            folds = [available_folds[0]]  # Use first available fold
            
        logger.info(f"   Input folder: {input_path.resolve()}")
        logger.info(f"   Output folder: {output_path.resolve()}")
        logger.info(f"   Task: {task_name}")
        logger.info(f"   Model: {model}")
        logger.info(f"   Folds: {folds}")
        
        # Build command
        cmd = [
            "nnUNet_predict",
            "-i", str(input_path),
            "-o", str(output_path),
            "-t", task_name,
            "-m", model,
            "-f"] + [str(f) for f in folds]
            
        if overwrite:
            cmd.append("--overwrite_existing")
            
        logger.info(f"   Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
            duration = time.time() - start_time
            
            logger.info(f"✅ nnUNet v1 inference completed successfully!")
            logger.info(f"   Duration: {duration:.1f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ nnUNet v1 inference failed!")
            logger.error(f"   Return code: {e.returncode}")
            logger.error(f"   Error: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"❌ nnUNet v1 inference timed out!")
            logger.error(f"   Inference took longer than 1 hour")
            return False
        except FileNotFoundError:
            logger.error(f"❌ nnUNet_predict command not found!")
            logger.error(f"   Make sure nnUNet v1 is properly installed")
            return False
    
    def analyze_results(self, output_dir):
        """Analyze the segmentation results"""
        logger.info("📊 Analyzing results...")
        
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.error("   ❌ Output directory not found")
            return
            
        # Count files
        nii_files = list(output_path.glob("*.nii.gz"))
        logger.info(f"   📁 Output Directory: {output_path.resolve()}")
        logger.info(f"   📄 Segmentation files (.nii.gz): {len(nii_files)}")
        
        total_size = 0
        for file in nii_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            logger.info(f"   - {file.name}: {size_mb:.1f} MB")
            
        logger.info(f"   📋 Total size: {total_size:.1f} MB")
        
        if nii_files:
            logger.info(f"\n👀 How to View Results:")
            logger.info(f"   1. Using 3D Slicer: Load {nii_files[0].name}")
            logger.info(f"   2. Using ITK-SNAP: Load CT image and segmentation")
            logger.info(f"   3. Using Python: import nibabel as nib; seg = nib.load('{nii_files[0].name}')")
    
    def run_complete_inference(self, input_dir, output_dir, task_name="Task064_KiTS_labelsFixed", 
                             model="3d_fullres", test_cases=None, folds=None):
        """Run complete inference pipeline"""
        logger.info("🚀 Starting Robust Kidney Segmentation Inference")
        logger.info("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
            
        # Check model availability
        available_folds = self.check_model_availability(task_name, model)
        if not available_folds:
            return False
            
        # Prepare input data
        if not self.prepare_input_data(input_dir, test_cases):
            return False
            
        # Run inference
        if not self.run_inference(input_dir, output_dir, task_name, model, folds):
            return False
            
        # Analyze results
        self.analyze_results(output_dir)
        
        logger.info(f"\n🎉 Inference completed successfully!")
        logger.info(f"   Check results in: {output_dir}")
        return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Robust Kidney Segmentation Inference")
    parser.add_argument("-i", "--input", default="test_3_cases_input", 
                       help="Input directory containing CT scans")
    parser.add_argument("-o", "--output", default="test_3_cases_output", 
                       help="Output directory for segmentation results")
    parser.add_argument("-t", "--task", default="Task064_KiTS_labelsFixed", 
                       help="Task name")
    parser.add_argument("-m", "--model", default="3d_fullres", 
                       help="Model configuration")
    parser.add_argument("-f", "--folds", nargs="+", type=int, 
                       help="Folds to use (default: first available)")
    parser.add_argument("--test-cases", nargs="+", 
                       help="Test cases to process (default: case_00000, case_00001, case_00002)")
    
    args = parser.parse_args()
    
    # Create inference instance
    inference = KidneySegmentationInference()
    
    # Run inference
    success = inference.run_complete_inference(
        input_dir=args.input,
        output_dir=args.output,
        task_name=args.task,
        model=args.model,
        test_cases=args.test_cases,
        folds=args.folds
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
