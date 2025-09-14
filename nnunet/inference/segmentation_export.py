"""
Segmentation Export Functions
Functions for exporting segmentation results
"""

import os
import numpy as np
from typing import Union, Tuple, List
import SimpleITK as sitk


def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray],
                                        out_fname: str,
                                        properties_dict: dict,
                                        order: int = 1,
                                        region_class_order: List[Union[int, Tuple[int, ...]]] = None,
                                        seg_postprogess_fn: callable = None,
                                        seg_postprocess_args: tuple = None,
                                        resampled_npz_fname: str = None,
                                        non_postprocessed_fname: str = None,
                                        force_separate_z: bool = None,
                                        interpolation_order_z: int = 0,
                                        verbose: bool = True) -> None:
    """
    Save segmentation from softmax probabilities
    
    Args:
        segmentation_softmax: Softmax probabilities or path to .npz file
        out_fname: Output filename
        properties_dict: Properties dictionary
        order: Interpolation order
        region_class_order: Order of regions
        seg_postprogess_fn: Postprocessing function
        seg_postprocess_args: Postprocessing arguments
        resampled_npz_fname: Resampled npz filename
        non_postprocessed_fname: Non-postprocessed filename
        force_separate_z: Force separate z interpolation
        interpolation_order_z: Z interpolation order
        verbose: Verbose output
    """
    
    if isinstance(segmentation_softmax, str):
        # Load from file
        if segmentation_softmax.endswith('.npz'):
            data = np.load(segmentation_softmax)
            segmentation_softmax = data['softmax']
        else:
            # Assume it's a nifti file
            image = sitk.ReadImage(segmentation_softmax)
            segmentation_softmax = sitk.GetArrayFromImage(image)
    
    # Convert softmax to segmentation
    segmentation = segmentation_softmax.argmax(0)
    
    # Create output directory
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    
    # Save as NIfTI
    image = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    
    # Set properties if available
    if 'spacing' in properties_dict:
        image.SetSpacing(properties_dict['spacing'])
    if 'origin' in properties_dict:
        image.SetOrigin(properties_dict['origin'])
    if 'direction' in properties_dict:
        image.SetDirection(properties_dict['direction'])
    
    sitk.WriteImage(image, out_fname)
    
    if verbose:
        print(f"Segmentation saved to: {out_fname}")


def save_segmentation_nifti(segmentation: Union[str, np.ndarray],
                           out_fname: str,
                           properties_dict: dict,
                           order: int = 1,
                           region_class_order: List[Union[int, Tuple[int, ...]]] = None,
                           seg_postprogess_fn: callable = None,
                           seg_postprocess_args: tuple = None,
                           resampled_npz_fname: str = None,
                           non_postprocessed_fname: str = None,
                           force_separate_z: bool = None,
                           interpolation_order_z: int = 0,
                           verbose: bool = True) -> None:
    """
    Save segmentation directly
    
    Args:
        segmentation: Segmentation array or path to file
        out_fname: Output filename
        properties_dict: Properties dictionary
        order: Interpolation order
        region_class_order: Order of regions
        seg_postprogess_fn: Postprocessing function
        seg_postprocess_args: Postprocessing arguments
        resampled_npz_fname: Resampled npz filename
        non_postprocessed_fname: Non-postprocessed filename
        force_separate_z: Force separate z interpolation
        interpolation_order_z: Z interpolation order
        verbose: Verbose output
    """
    
    if isinstance(segmentation, str):
        # Load from file
        image = sitk.ReadImage(segmentation)
        segmentation = sitk.GetArrayFromImage(image)
    
    # Create output directory
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    
    # Save as NIfTI
    image = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    
    # Set properties if available
    if 'spacing' in properties_dict:
        image.SetSpacing(properties_dict['spacing'])
    if 'origin' in properties_dict:
        image.SetOrigin(properties_dict['origin'])
    if 'direction' in properties_dict:
        image.SetDirection(properties_dict['direction'])
    
    sitk.WriteImage(image, out_fname)
    
    if verbose:
        print(f"Segmentation saved to: {out_fname}")
