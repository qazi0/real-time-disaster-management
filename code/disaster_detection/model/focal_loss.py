import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    This loss function is particularly useful for imbalanced datasets as it down-weights
    easy examples and focuses training on hard examples.
    
    Args:
        alpha (Optional[torch.Tensor]): Weighting factor for each class. If None, no class weights are used.
        gamma (float): Focusing parameter. Higher values increase the effect of hard examples.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    
    References:
        - Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
          Focal loss for dense object detection. In Proceedings of the IEEE international
          conference on computer vision (pp. 2980-2988).
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) where C is the number of classes.
            targets (torch.Tensor): Ground truth labels of shape (N,) where each value is 0 <= targets[i] <= C-1.
            
        Returns:
            torch.Tensor: The computed focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    """Calculate class weights based on inverse frequency.
    
    Args:
        class_counts (torch.Tensor): Tensor containing the count of samples for each class.
        
    Returns:
        torch.Tensor: Normalized class weights.
    """
    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts)
    return weights / weights.sum()  # Normalize weights 