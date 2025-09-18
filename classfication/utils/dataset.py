import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiSliceDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, split='train', num_slices: int = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.num_slices = num_slices
        self.cases_data = self._load_cases_data()
        self.class_names = ['Clear Cell', 'Papillary', 'Chromophobe', 'Oncocytoma', 'Others']
    
    def _load_cases_data(self) -> List[Dict]:
        cases_data = []
        for case_dir in self.data_dir.iterdir():
            if case_dir.is_dir() and (case_dir / "metadata.json").exists():
                with open(case_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                cases_data.append(metadata)
        return cases_data
    
    def __len__(self):
        return len(self.cases_data)
    
    def __getitem__(self, idx):
        case_data = self.cases_data[idx]
        case_id = case_data['case_id']
        
        kidney_tumor_roi = np.load(self.data_dir / case_id / "kidney_tumor_roi.npy")
        tumor_roi = np.load(self.data_dir / case_id / "tumor_roi.npy")
        
        actual_num_slices = kidney_tumor_roi.shape[0]
        target_slices = self.num_slices if self.num_slices is not None else actual_num_slices
        
        if actual_num_slices < target_slices:
            padding_needed = target_slices - actual_num_slices
            kidney_tumor_roi = np.concatenate([
                kidney_tumor_roi,
                np.zeros((padding_needed, kidney_tumor_roi.shape[1], kidney_tumor_roi.shape[2]))
            ], axis=0)
            tumor_roi = np.concatenate([
                tumor_roi,
                np.zeros((padding_needed, tumor_roi.shape[1], tumor_roi.shape[2]))
            ], axis=0)
        elif actual_num_slices > target_slices:
            kidney_tumor_roi = kidney_tumor_roi[:target_slices]
            tumor_roi = tumor_roi[:target_slices]
        
        kidney_tumor_roi = torch.from_numpy(kidney_tumor_roi).unsqueeze(1)
        tumor_roi = torch.from_numpy(tumor_roi).unsqueeze(1)
        
        if self.transform:
            kidney_tumor_slices = []
            tumor_slices = []
            num_slices_to_process = kidney_tumor_roi.shape[0]
            
            for i in range(num_slices_to_process):
                kt_slice = kidney_tumor_roi[i].squeeze(0).numpy()
                t_slice = tumor_roi[i].squeeze(0).numpy()
                
                kt_slice = (kt_slice * 255).astype(np.uint8)
                t_slice = (t_slice * 255).astype(np.uint8)
                
                transformed = self.transform(image=kt_slice, mask=t_slice)
                kt_transformed = transformed['image']
                t_transformed = transformed['mask']
                
                kidney_tumor_slices.append(kt_transformed)
                tumor_slices.append(t_transformed)
            
            kidney_tumor_roi = torch.stack(kidney_tumor_slices, dim=0).float()
            tumor_roi = torch.stack(tumor_slices, dim=0).unsqueeze(1).float()
        
        return {
            'kidney_tumor_roi': kidney_tumor_roi,
            'tumor_roi': tumor_roi,
            'label': case_data['label'],
            'case_id': case_id,
            'num_slices': actual_num_slices
        }
