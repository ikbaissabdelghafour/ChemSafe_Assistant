"""
loss.py

Implements custom loss functions to handle class imbalance and missing/masked labels
in multi-task learning on the Tox21 dataset.

Missing labels are encoded as NaN in the target tensor. The loss function
creates a binary mask to exclude these from gradient computation.
"""

import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy loss with logits that masks out NaN (missing) labels.

    In the Tox21 dataset, not every compound is screened against every assay,
    resulting in missing labels. This loss ensures those missing entries do not
    contribute to the gradient or the loss value.

    Args:
        reduction: How to reduce the valid losses ('mean' or 'sum').
    """

    def __init__(self, reduction: str = "mean", pos_weight: torch.Tensor = None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute masked BCE loss.

        Args:
            logits:  Raw model predictions of shape (batch_size, num_tasks).
            targets: Ground truth labels of shape (batch_size, num_tasks).
                     NaN values indicate missing labels.

        Returns:
            Scalar loss tensor.
        """
        # Build mask: True where label exists, False where NaN
        mask = ~torch.isnan(targets)

        # If no valid labels exist in this batch, return zero loss
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Replace NaNs with 0.0 so BCEWithLogitsLoss doesn't produce NaN gradients
        safe_targets = targets.clone()
        safe_targets[~mask] = 0.0

        # Compute element-wise BCE loss (no reduction yet)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, safe_targets, reduction="none", pos_weight=self.pos_weight
        )

        # Zero out loss for missing labels
        bce = bce * mask.float()

        # Reduce
        if self.reduction == "mean":
            return bce.sum() / mask.float().sum()
        elif self.reduction == "sum":
            return bce.sum()
        else:
            return bce


class MaskedFocalLoss(nn.Module):
    """
    Focal Loss with logits that masks out NaN (missing) labels.
    Down-weights easy examples and focuses on hard ones.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"):
        super(MaskedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        safe_targets = targets.clone()
        safe_targets[~mask] = 0.0

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, safe_targets, reduction="none"
        )
        
        probs = torch.sigmoid(logits)
        pt = torch.where(safe_targets == 1.0, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(safe_targets == 1.0, self.alpha, 1.0)
            focal_weight = focal_weight * alpha_t
            
        focal_loss = focal_weight * bce
        focal_loss = focal_loss * mask.float()
        
        if self.reduction == "mean":
            return focal_loss.sum() / mask.float().sum()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
