"""
N-dimensional Softmax Functions
Utility functions for softmax operations
"""

import torch
import torch.nn.functional as F
import numpy as np


def softmax_helper(x):
    """
    Apply softmax to the last dimension of x
    
    Args:
        x: Input tensor
        
    Returns:
        Softmax applied tensor
    """
    return F.softmax(x, dim=1)


def softmax_helper_dim(x, dim):
    """
    Apply softmax to the specified dimension of x
    
    Args:
        x: Input tensor
        dim: Dimension to apply softmax
        
    Returns:
        Softmax applied tensor
    """
    return F.softmax(x, dim=dim)


def softmax_helper_np(x, axis=-1):
    """
    Apply softmax to numpy array
    
    Args:
        x: Input numpy array
        axis: Axis to apply softmax
        
    Returns:
        Softmax applied numpy array
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
