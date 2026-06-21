"""
gnn.py

Implements Graph Neural Network architecture for multi-label toxicity prediction
on the Tox21 dataset (12 endpoints).

Architecture:
    - 3 GCNConv message passing layers with BatchNorm and Dropout
    - Global pooling: mean + max concatenation
    - MLP head: 256 → 128 → 12 (raw logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class Tox21GNN(nn.Module):
    """
    Graph Neural Network for multi-label toxicity prediction.
    
    Args:
        num_node_features: Dimension of input node feature vectors.
        hidden_dim: Hidden dimension for GCN layers (default: 128).
        num_tasks: Number of output toxicity tasks (default: 12).
        dropout: Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_tasks: int = 12,
        dropout: float = 0.3,
    ):
        super(Tox21GNN, self).__init__()

        self.num_tasks = num_tasks
        self.dropout = dropout

        # ── Message Passing Layers ──────────────────────────────────────
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # ── MLP Prediction Head ─────────────────────────────────────────
        # Input dim = hidden_dim * 2 because we concatenate mean + max pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_tasks),
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyTorch Geometric Data/Batch object with attributes:
                  x (node features), edge_index, batch.

        Returns:
            logits: Raw (pre-sigmoid) predictions of shape (batch_size, num_tasks).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ── Layer 1 ────────────────────────────────────────────────────
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layer 2 ────────────────────────────────────────────────────
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layer 3 ────────────────────────────────────────────────────
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # ── Global Pooling (mean + max concatenation) ──────────────────
        x_mean = global_mean_pool(x, batch)   # (batch_size, hidden_dim)
        x_max = global_max_pool(x, batch)     # (batch_size, hidden_dim)
        x = torch.cat([x_mean, x_max], dim=1) # (batch_size, hidden_dim * 2)

        # ── MLP Head ──────────────────────────────────────────────────
        logits = self.mlp(x)  # (batch_size, num_tasks)

        return logits

    def predict_proba(self, data):
        """
        Returns sigmoid probabilities instead of raw logits.
        Used during inference only.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            return torch.sigmoid(logits)
