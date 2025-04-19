import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def linear_combination(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Combine two tensors with a weighted average using epsilon.

    Args:
        x: First tensor
        y: Second tensor
        epsilon: Weight factor between 0 and 1

    Returns:
        Linear combination of x and y: epsilon * x + (1 - epsilon) * y
    """
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Reduce loss tensor according to the specified reduction method.

    Args:
        loss: Loss tensor
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Reduced loss tensor
    """
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    This implements label smoothing as described in the paper
    "Rethinking the Inception Architecture for Computer Vision"
    (https://arxiv.org/abs/1512.00567).

    Label smoothing is a regularization technique that prevents the model from
    becoming too confident in its predictions. Instead of using one-hot encoded
    target vectors, it smooths the target distribution by assigning some probability
    mass to non-target classes.

    Args:
        epsilon: Smoothing factor between 0 and 1 (default: 0.1)
        reduction: Specifies the reduction to apply to the output:
                   'none' | 'mean' | 'sum' (default: 'mean')
        ignore_index: Target value to ignore (default: -100)

    Shapes:
        - Input: (N, C) where N is the batch size and C is the number of classes
        - Target: (N) where each value is 0 ≤ target[i] ≤ C-1
        - Output: scalar if reduction is 'mean' or 'sum', (N) if reduction is 'none'
    """

    def __init__(
        self, 
        epsilon: float = 0.1, 
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

        # Validate epsilon
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"Epsilon must be between 0 and 1, got {epsilon}")

        # Validate reduction
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Reduction must be one of 'none', 'mean', or 'sum', got {reduction}")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the label smoothing cross entropy loss.

        Args:
            preds: Predictions tensor (N, C) where C is the number of classes
            target: Target tensor (N) with class indices

        Returns:
            Label smoothed cross entropy loss
        """
        # Get number of classes from predictions
        n_classes = preds.size(-1)
        
        # Create mask for ignored indices if needed
        mask = None
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index)
            target = target * mask.long()
        
        # Apply log softmax to get log probabilities
        log_preds = F.log_softmax(preds, dim=-1)
        
        # For smoothed loss, we take uniformly distributed probability mass
        # with epsilon / (n_classes - 1) for non-target classes
        # and 1 - epsilon + epsilon / (n_classes - 1) for the target class
        
        # Negative log likelihood loss (NLL) - standard cross entropy
        nll_loss = F.nll_loss(
            log_preds, 
            target, 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        # Smoothed loss - use uniform distribution across all classes
        smooth_loss = -log_preds.sum(dim=-1)
        
        # If we have a mask, apply it
        if mask is not None:
            smooth_loss = smooth_loss * mask.float()
            # Adjust the denominator for proper average
            if self.reduction == 'mean':
                smooth_loss = smooth_loss.sum() / mask.sum().float().clamp(min=1.0)
        
        # Normalize the smooth loss to account for the number of classes
        smooth_loss = reduce_loss(smooth_loss, self.reduction) / n_classes
        
        # Reduce the NLL loss according to the specified reduction
        if self.reduction != 'none':
            nll_loss = reduce_loss(nll_loss, self.reduction)
        
        # Combine the losses: epsilon * smooth_loss + (1 - epsilon) * nll_loss
        return linear_combination(smooth_loss, nll_loss, self.epsilon)
