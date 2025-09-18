import os
import json
import logging
import numpy as np
import cv2
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional  
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
from skimage import morphology, filters, segmentation
from skimage.measure import label, regionprops
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataPreprocessor:
    
    def __init__(self, ct_data_root: str, seg_data_root: str, kits_json_path: str):
        self.ct_data_root = Path(ct_data_root)
        self.seg_data_root = Path(seg_data_root)
        self.kits_json_path = Path(kits_json_path)
        
        with open(self.kits_json_path, 'r') as f:
            self.kits_data = json.load(f)
        
        self.class_mapping = {
            'clear_cell_rcc': 0,
            'papillary': 1, 
            'chromophobe': 2,
            'oncocytoma': 3,
            'clear_cell_papillary_rcc': 0,
            'angiomyolipoma': 4,
            'urothelial': 4,
            'mest': 4,
            'rcc_unclassified': 4,
            'wilms': 4,
            'spindle_cell_neoplasm': 4,
            'other': 4,
            'multilocular_cystic_rcc': 4,
            'collecting_duct_undefined': 4
        }
        self.class_names = [
            'Clear Cell', 'Papillary', 'Chromophobe', 'Oncocytoma', 'Others'
        ]
    
    def get_case_labels(self) -> Dict[str, int]:
        case_labels = {}
        
        for case_data in self.kits_data:
            case_id = case_data['case_id']
            if 'tumor_histologic_subtype' in case_data:
                subtype = case_data['tumor_histologic_subtype'].lower()
                if subtype in self.class_mapping:
                    case_labels[case_id] = self.class_mapping[subtype]
                else:
                    case_labels[case_id] = 4
            else:
                case_labels[case_id] = 4
                
        return case_labels
    
    def load_ct_image(self, case_id: str) -> np.ndarray:
        image_path = self.ct_data_root / case_id / "imaging.nii.gz"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for case {case_id}")
            
        sitk_image = sitk.ReadImage(str(image_path))
        image_array = sitk.GetArrayFromImage(sitk_image)
        
        return image_array
    
    def load_segmentation(self, case_id: str) -> np.ndarray:
        seg_path = self.seg_data_root / case_id / "segmentation.nii.gz"
        
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation not found for case {case_id}")
            
        sitk_seg = sitk.ReadImage(str(seg_path))
        seg_array = sitk.GetArrayFromImage(sitk_seg)
        
        return seg_array
    
    
    def extract_slice_rois(self, ct_slice: np.ndarray, seg_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kidney_mask = (seg_slice == 1) | (seg_slice == 2)
        tumor_mask = (seg_slice == 2)
        
        if not np.any(kidney_mask) or not np.any(tumor_mask):
            return np.zeros((64, 64)), np.zeros((64, 64))
        
        kidney_coords = np.where(kidney_mask)
        kidney_y_min, kidney_y_max = kidney_coords[0].min(), kidney_coords[0].max()
        kidney_x_min, kidney_x_max = kidney_coords[1].min(), kidney_coords[1].max()
        
        kidney_padding = 15
        kidney_y_min = max(0, kidney_y_min - kidney_padding)
        kidney_y_max = min(ct_slice.shape[0], kidney_y_max + kidney_padding)
        kidney_x_min = max(0, kidney_x_min - kidney_padding)
        kidney_x_max = min(ct_slice.shape[1], kidney_x_max + kidney_padding)
        
        kidney_tumor_roi = ct_slice[kidney_y_min:kidney_y_max, kidney_x_min:kidney_x_max]
        
        tumor_coords = np.where(tumor_mask)
        tumor_y_min, tumor_y_max = tumor_coords[0].min(), tumor_coords[0].max()
        tumor_x_min, tumor_x_max = tumor_coords[1].min(), tumor_coords[1].max()
        
        tumor_padding = 5
        tumor_y_min = max(0, tumor_y_min - tumor_padding)
        tumor_y_max = min(ct_slice.shape[0], tumor_y_max + tumor_padding)
        tumor_x_min = max(0, tumor_x_min - tumor_padding)
        tumor_x_max = min(ct_slice.shape[1], tumor_x_max + tumor_padding)
        
        tumor_roi = ct_slice[tumor_y_min:tumor_y_max, tumor_x_min:tumor_x_max]
        
        kidney_tumor_roi = cv2.resize(kidney_tumor_roi, (64, 64), interpolation=cv2.INTER_CUBIC)
        tumor_roi = cv2.resize(tumor_roi, (64, 64), interpolation=cv2.INTER_CUBIC)
        
        return kidney_tumor_roi, tumor_roi
    
    def get_tumor_regions(self, image: np.ndarray, segmentation: np.ndarray) -> List[Dict]:
        unique_labels = np.unique(segmentation)
        unique_labels = unique_labels[unique_labels > 0]
        
        tumor_label = None
        if 2 in unique_labels:
            tumor_label = 2
        elif 3 in unique_labels:
            tumor_label = 3
        
        if tumor_label is None:
            return []
        
        tumor_mask_3d = (segmentation == tumor_label)
        labeled_tumors, num_tumors = ndimage.label(tumor_mask_3d)
        
        tumor_regions = []
        
        for tumor_id in range(1, num_tumors + 1):
            tumor_slice_data = []
            for z in range(segmentation.shape[0]):
                seg_slice = segmentation[z]
                tumor_mask = (labeled_tumors[z] == tumor_id)
                tumor_voxels = np.sum(tumor_mask)
                if tumor_voxels > 0:
                    tumor_slice_data.append((z, tumor_voxels))
            
            if len(tumor_slice_data) < 3:
                continue
            
            tumor_slice_data.sort(key=lambda x: x[1], reverse=True)
            
            for order in range(1, len(tumor_slice_data) + 1, 2):
                max_tumor_slice = tumor_slice_data[order-1][0]
                start_slice = max_tumor_slice - 1
                end_slice = max_tumor_slice + 2
                start_slice = max(0, start_slice)
                end_slice = min(segmentation.shape[0], end_slice)
                
                if end_slice - start_slice < 3:
                    continue
                
                slice_indices = list(range(start_slice, start_slice + 3))
                
                tumor_regions.append({
                    'slice_indices': slice_indices,
                    'tumor_id': tumor_id,
                    'max_tumor_slice': max_tumor_slice
                })
            
        return tumor_regions
    

    def extract_slice_rois_with_masks(self, ct_slice: np.ndarray, seg_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kidney_mask = (seg_slice == 1) | (seg_slice == 2)
        tumor_mask = (seg_slice == 2)
        
        if not np.any(kidney_mask) or not np.any(tumor_mask):
            return np.zeros((64, 64)), np.zeros((64, 64)), np.zeros((64, 64)), np.zeros((64, 64))
        
        kidney_coords = np.where(kidney_mask)
        kidney_y_min, kidney_y_max = kidney_coords[0].min(), kidney_coords[0].max()
        kidney_x_min, kidney_x_max = kidney_coords[1].min(), kidney_coords[1].max()
        
        kidney_padding = 15
        kidney_y_min = max(0, kidney_y_min - kidney_padding)
        kidney_y_max = min(ct_slice.shape[0], kidney_y_max + kidney_padding)
        kidney_x_min = max(0, kidney_x_min - kidney_padding)
        kidney_x_max = min(ct_slice.shape[1], kidney_x_max + kidney_padding)
        
        kidney_tumor_roi = ct_slice[kidney_y_min:kidney_y_max, kidney_x_min:kidney_x_max]
        kidney_tumor_seg = seg_slice[kidney_y_min:kidney_y_max, kidney_x_min:kidney_x_max]
        
        tumor_coords = np.where(tumor_mask)
        tumor_y_min, tumor_y_max = tumor_coords[0].min(), tumor_coords[0].max()
        tumor_x_min, tumor_x_max = tumor_coords[1].min(), tumor_coords[1].max()
        
        tumor_padding = 5
        tumor_y_min = max(0, tumor_y_min - tumor_padding)
        tumor_y_max = min(ct_slice.shape[0], tumor_y_max + tumor_padding)
        tumor_x_min = max(0, tumor_x_min - tumor_padding)
        tumor_x_max = min(ct_slice.shape[1], tumor_x_max + tumor_padding)
        
        tumor_roi = ct_slice[tumor_y_min:tumor_y_max, tumor_x_min:tumor_x_max]
        tumor_seg = seg_slice[tumor_y_min:tumor_y_max, tumor_x_min:tumor_x_max]
        
        kidney_tumor_roi = cv2.resize(kidney_tumor_roi, (64, 64), interpolation=cv2.INTER_CUBIC)
        tumor_roi = cv2.resize(tumor_roi, (64, 64), interpolation=cv2.INTER_CUBIC)
        kidney_tumor_seg = cv2.resize(kidney_tumor_seg, (64, 64), interpolation=cv2.INTER_NEAREST)
        tumor_seg = cv2.resize(tumor_seg, (64, 64), interpolation=cv2.INTER_NEAREST)
        
        kidney_tumor_mask = (kidney_tumor_seg == 2).astype(np.float32)
        tumor_mask = (tumor_seg == 2).astype(np.float32)
        
        kidney_tumor_mask = cv2.GaussianBlur(kidney_tumor_mask, (3, 3), 0)
        tumor_mask = cv2.GaussianBlur(tumor_mask, (3, 3), 0)
        
        kidney_tumor_mask = (kidney_tumor_mask > 0.1).astype(np.float32)
        tumor_mask = (tumor_mask > 0.1).astype(np.float32)
        
        return kidney_tumor_roi, tumor_roi, kidney_tumor_mask, tumor_mask
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)
            
        return image.astype(np.float32)
    
    def process_single_case(self, case_id: str) -> List[Dict]:
        try:
            image = self.load_ct_image(case_id)
            segmentation = self.load_segmentation(case_id)
            tumor_regions = self.get_tumor_regions(image, segmentation)
            case_labels = self.get_case_labels()
            label = case_labels[case_id]
            
            processed_tumors = []
            
            for tumor_region in tumor_regions:
                tumor_slice_indices = tumor_region['slice_indices']
                tumor_id = tumor_region['tumor_id']
                
                kidney_tumor_slices = []
                tumor_slices = []
                
                for slice_idx in tumor_slice_indices:
                    ct_slice = image[slice_idx]
                    seg_slice = segmentation[slice_idx]
                    kidney_tumor_slice, tumor_slice = self.extract_slice_rois(ct_slice, seg_slice)
                    
                    if np.all(kidney_tumor_slice == 0) and np.all(tumor_slice == 0):
                        continue
                    
                    kidney_tumor_slice = self.normalize_image(kidney_tumor_slice)
                    tumor_slice = self.normalize_image(tumor_slice)
                    
                    kidney_tumor_slices.append(kidney_tumor_slice)
                    tumor_slices.append(tumor_slice)
                
                if len(kidney_tumor_slices) != 3:
                    continue
                
                middle_slice_idx = tumor_slice_indices[1]
                middle_ct_slice = image[middle_slice_idx]
                middle_seg_slice = segmentation[middle_slice_idx]
                
                kt_middle, t_middle, kidney_tumor_mask, tumor_mask = self.extract_slice_rois_with_masks(middle_ct_slice, middle_seg_slice)
                
                kidney_tumor_3d = np.stack(kidney_tumor_slices, axis=0)
                tumor_3d = np.stack(tumor_slices, axis=0)
                
                kidney_tumor_4slice = np.concatenate([kidney_tumor_3d, kidney_tumor_mask[np.newaxis, :, :]], axis=0)
                tumor_4slice = np.concatenate([tumor_3d, tumor_mask[np.newaxis, :, :]], axis=0)
                
                tumor_case = {
                    'case_id': f"{case_id}_tumor_{tumor_id}_{tumor_region['max_tumor_slice']}",
                    'original_case_id': case_id,
                    'tumor_id': tumor_id,
                    'kidney_tumor_roi': kidney_tumor_4slice,
                    'tumor_roi': tumor_4slice,
                    'label': label,
                    'class_name': self.class_names[label],
                    'num_slices': 4,
                    'max_tumor_slice': tumor_region['max_tumor_slice']
                }
                
                processed_tumors.append(tumor_case)
            
            return processed_tumors
            
        except Exception as e:
            logger.error(f"Error processing case {case_id}: {str(e)}")
            return []
    
    def process_all_cases(self, output_dir: str) -> List[Dict]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        case_labels = self.get_case_labels()
        processed_cases = []
        slice_counts = []
        
        logger.info(f"Processing {len(case_labels)} cases with enhanced preprocessing...")
        
        for i, case_id in enumerate(case_labels.keys()):
            logger.info(f"Processing case {i+1}/{len(case_labels)}: {case_id}")
            
            tumor_cases = self.process_single_case(case_id)
            if tumor_cases:
                for tumor_case in tumor_cases:
                    processed_cases.append(tumor_case)
                    slice_counts.append(tumor_case['num_slices'])
                    
                    case_output_dir = output_dir / tumor_case['case_id']
                    case_output_dir.mkdir(exist_ok=True)
                    
                    np.save(case_output_dir / "kidney_tumor_roi.npy", tumor_case['kidney_tumor_roi'])
                    np.save(case_output_dir / "tumor_roi.npy", tumor_case['tumor_roi'])
                    
                    metadata = {
                        'case_id': tumor_case['case_id'],
                        'original_case_id': tumor_case['original_case_id'],
                        'tumor_id': tumor_case['tumor_id'],
                        'label': tumor_case['label'],
                        'class_name': tumor_case['class_name'],
                        'num_slices': tumor_case['num_slices'],
                        'max_tumor_slice': tumor_case['max_tumor_slice'],
                        'has_4slice_data': True,
                        'slices': 4,
                        'mask_slice': 3
                    }
                    with open(case_output_dir / "metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                logger.info(f"  Found {len(tumor_cases)} slice(s) with tumor in case {case_id}")
        
        summary = {
            'total_cases': len(processed_cases),
            'class_distribution': self.get_class_distribution(),
            'class_names': self.class_names,
            'processed_cases': [case['case_id'] for case in processed_cases],
            'slice_statistics': {
                'mean_slices': float(np.mean(slice_counts)),
                'std_slices': float(np.std(slice_counts)),
                'min_slices': int(np.min(slice_counts)),
                'max_slices': int(np.max(slice_counts)),
                'slice_counts': [int(x) for x in slice_counts]
            },
            'features': {
                'has_4slice_data': True,
                'slices': 4,
                'mask_slice': 3,
                'mask_method': 'smooth_morphological'
            }
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processed {len(processed_cases)} cases successfully")
        logger.info(f"Slice statistics: mean={np.mean(slice_counts):.1f}, std={np.std(slice_counts):.1f}, min={np.min(slice_counts)}, max={np.max(slice_counts)}")
        logger.info(f"Class distribution: {self.get_class_distribution()}")
        
        return processed_cases
    
    def get_class_distribution(self) -> Dict[str, int]:
        case_labels = self.get_case_labels()
        distribution = Counter()
        
        for case_id, label in case_labels.items():
            class_name = self.class_names[label]
            distribution[class_name] += 1
            
        return dict(distribution)


if __name__ == "__main__":
    preprocessor = EnhancedDataPreprocessor(
        ct_data_root="../Path/to/ct_data",
        seg_data_root="../Path/to/seg_data",
        kits_json_path="../Path/to/kits.json"
    )
    
    processed_cases = preprocessor.process_all_cases(
        output_dir="./processed_data_with_smooth_masks_augmented"
    )
    
    print(f"Processed {len(processed_cases)} cases with enhanced preprocessing")
