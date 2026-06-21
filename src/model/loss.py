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

    def __init__(self, reduction: str = "mean"):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

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
            logits, safe_targets, reduction="none"
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
