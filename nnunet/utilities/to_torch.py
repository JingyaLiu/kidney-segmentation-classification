"""
Torch Conversion Utilities
Utility functions for converting data to PyTorch tensors
"""

import torch
import numpy as np
from typing import Union, Any


def to_cuda(data: Any, non_blocking: bool = True) -> Any:
    """
    Move data to CUDA if available
    
    Args:
        data: Data to move to CUDA
        non_blocking: Whether to use non-blocking transfer
        
    Returns:
        Data moved to CUDA
    """
    if torch.cuda.is_available():
        if isinstance(data, (list, tuple)):
            return [to_cuda(item, non_blocking) for item in data]
        elif isinstance(data, dict):
            return {key: to_cuda(value, non_blocking) for key, value in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.cuda(non_blocking=non_blocking)
        else:
            return data
    else:
        return data


def maybe_to_torch(data: Any) -> Any:
    """
    Convert data to torch tensor if it's a numpy array
    
    Args:
        data: Data to convert
        
    Returns:
        Data converted to torch tensor if applicable
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return [maybe_to_torch(item) for item in data]
    elif isinstance(data, dict):
        return {key: maybe_to_torch(value) for key, value in data.items()}
    else:
        return data


def to_numpy(data: Any) -> Any:
    """
    Convert data to numpy array if it's a torch tensor
    
    Args:
        data: Data to convert
        
    Returns:
        Data converted to numpy array if applicable
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, (list, tuple)):
        return [to_numpy(item) for item in data]
    elif isinstance(data, dict):
        return {key: to_numpy(value) for key, value in data.items()}
    else:
        return data


def to_tensor(data: Any, dtype: torch.dtype = None) -> Any:
    """
    Convert data to torch tensor
    
    Args:
        data: Data to convert
        dtype: Desired data type
        
    Returns:
        Data converted to torch tensor
    """
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor
    elif isinstance(data, (list, tuple)):
        return [to_tensor(item, dtype) for item in data]
    elif isinstance(data, dict):
        return {key: to_tensor(value, dtype) for key, value in data.items()}
    else:
        return data
