import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from focal_loss import FocalLoss as FocalLossPackage
from sklearn.utils.class_weight import compute_class_weight


class FocalLossWrapper(nn.Module):
    def __init__(self, gamma=2.0, weights=None, reduction='mean'):
        super(FocalLossWrapper, self).__init__()
        self.gamma = gamma
        self.weights = weights
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        focal_loss = FocalLossPackage(gamma=self.gamma, weights=self.weights, reduction=self.reduction)
        return focal_loss(probs, targets)


def calculate_class_weights(labels, method='balanced'):
    
    classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight=method,
        classes=classes,
        y=labels
    )
    
    return torch.FloatTensor(class_weights)


def get_loss_function(loss_type='cross_entropy', class_weights=None, 
                     focal_alpha=0.25, focal_gamma=2.0):
    if loss_type == 'focal':
        if class_weights is not None:
            weights = class_weights * focal_alpha
            return FocalLossWrapper(gamma=focal_gamma, weights=weights, reduction='mean')
        else:
            return FocalLossWrapper(gamma=focal_gamma, reduction='mean')
    elif loss_type == 'cross_entropy':
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
