"""
Random Utility Functions
Utility functions for random operations
"""

import torch
import numpy as np


def no_op(x):
    """
    No operation function - returns input as is
    
    Args:
        x: Input tensor
        
    Returns:
        Input tensor unchanged
    """
    return x


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_random_state():
    """
    Get current random state
    
    Returns:
        Dictionary containing random states
    """
    return {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state()
    }


def set_random_state(state):
    """
    Set random state from dictionary
    
    Args:
        state: Dictionary containing random states
    """
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
