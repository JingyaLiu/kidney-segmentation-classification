import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_train_test_split(labels, test_size=0.2, random_state=42, splits_file='data_splits.json'):
    train_indices, test_indices = train_test_split(
        range(len(labels)), 
        test_size=test_size, 
        stratify=labels, 
        random_state=random_state
    )
    
    splits = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'test_size': test_size,
        'random_state': random_state,
        'total_samples': len(labels)
    }
    
    utils_dir = Path(__file__).parent
    splits_path = utils_dir / splits_file
    
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    return train_indices, test_indices


def load_train_test_split(splits_file='data_splits.json'):
    utils_dir = Path(__file__).parent
    splits_path = utils_dir / splits_file
    
    if not splits_path.exists():
        return None, None
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    return splits['train_indices'], splits['test_indices']
