#!/usr/bin/env python3
"""
Kidney Segmentation Inference using nnUNet v1
This script uses the original nnUNet v1 to run inference with the trained models.
"""

import os
import sys
import shutil
import time
from pathlib import Path
import subprocess

def setup_environment_variables():
    """Set up nnUNet v1 environment variables"""
    print("🔧 Setting up nnUNet v1 environment variables...")
    
    # Set nnUNet v1 environment variables
    os.environ['nnUNet_raw_data_base'] = str(Path("/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/dataset").resolve())
    os.environ['nnUNet_preprocessed'] = str(Path("/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/dataset/preprocessed").resolve())
    os.environ['RESULTS_FOLDER'] = str(Path("/home/ciml/Documents/code/kidneyAI/kidney-segmentation-classification/nnUNet_trained_models/nnUNet").resolve())
    
    print(f"   nnUNet_raw_data_base: {os.environ['nnUNet_raw_data_base']}")
    print(f"   nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"   RESULTS_FOLDER: {os.environ['RESULTS_FOLDER']}")

def copy_test_cases():
    """Check if test cases are already available"""
    print("\n📋 Checking test cases...")
    
    test_input_dir = Path("test_3_cases_input")
    test_output_dir = Path("test_3_cases_output")
    
    # Create output directory
    test_output_dir.mkdir(exist_ok=True)
    
    # Check if test cases exist
    test_cases = ["case_00000", "case_00001", "case_00002"]
    
    for case in test_cases:
        test_file = test_input_dir / f"{case}_0000.nii.gz"
        
        if test_file.exists():
            size_mb = test_file.stat().st_size / (1024 * 1024)
            print(f"   ✓ {case}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {case}: File not found")

def run_nnunet_v1_inference():
    """Run nnUNet v1 inference"""
    print("\n🤖 Running nnUNet v1 Inference...")
    
    input_folder = Path("test_3_cases_input")
    output_folder = Path("test_3_cases_output")
    task_name = "Task064_KiTS_labelsFixed"
    model = "3d_fullres"
    
    print(f"   Input folder: {input_folder.resolve()}")
    print(f"   Output folder: {output_folder.resolve()}")
    print(f"   Task: {task_name}")
    print(f"   Model: {model}")
    
    # Use nnUNet v1 predict command
    cmd = [
        "nnUNet_predict",
        "-i", str(input_folder),
        "-o", str(output_folder),
        "-t", task_name,
        "-m", model,
        "-f", "0", "1", "2", "3", "4"
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        
        print(f"\n✅ nnUNet v1 inference completed successfully!")
        print(f"   Duration: {duration:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ nnUNet v1 inference failed!")
        print(f"   Return code: {e.returncode}")
        print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"\n❌ nnUNet_predict command not found!")
        print(f"   Make sure nnUNet v1 is properly installed")
        return False

def analyze_results():
    """Analyze the segmentation results"""
    print("\n📊 Results Analysis:")
    
    output_dir = Path("test_3_cases_output")
    
    if not output_dir.exists():
        print("   ❌ Output directory not found")
        return
    
    # Count files
    nii_files = list(output_dir.glob("*.nii.gz"))
    print(f"   📁 Output Directory: {output_dir.resolve()}")
    print(f"   📄 Segmentation files (.nii.gz): {len(nii_files)}")
    
    total_size = 0
    for file in nii_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   - {file.name}: {size_mb:.1f} MB")
    
    print(f"\n   📋 Total size: {total_size:.1f} MB")
    
    if nii_files:
        print(f"\n👀 How to View Results:")
        print(f"   1. Using 3D Slicer: Load {nii_files[0].name}")
        print(f"   2. Using ITK-SNAP: Load CT image and segmentation")
        print(f"   3. Using Python: import nibabel as nib; seg = nib.load('{nii_files[0].name}')")

def main():
    """Main function"""
    print("🚀 Starting nnUNet v1 Kidney Segmentation Inference")
    print("=" * 60)
    
    # Setup
    setup_environment_variables()
    copy_test_cases()
    
    # Run inference
    success = run_nnunet_v1_inference()
    
    if success:
        analyze_results()
        print(f"\n🎉 Inference completed successfully!")
        print(f"   Check results in: test_3_cases_output/")
    else:
        print(f"\n⚠️  Inference failed. Check the error messages above.")
        print(f"   This might be due to model compatibility or missing dependencies.")

if __name__ == "__main__":
    main()
