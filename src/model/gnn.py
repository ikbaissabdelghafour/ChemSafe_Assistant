"""
gnn.py

Implements Graph Neural Network architecture for multi-label toxicity prediction
on the Tox21 dataset (12 endpoints).

Architecture:
    - Node encoder & Edge encoder
    - 5 GINEConv message passing layers with edge features, BatchNorm and Dropout
    - Jumping Knowledge: Concatenation of all layers
    - Global pooling: mean + max concatenation per layer
    - MLP head: 1280 -> 512 → 256 → 12 (raw logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm, GlobalAttention


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
        dropout: float = 0.4,
    ):
        super(Tox21GNN, self).__init__()

        self.num_tasks = num_tasks
        self.dropout = dropout

        # ── Encoders ────────────────────────────────────────────────────
        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        # graph_builder extracts 6 bond features
        self.edge_encoder = nn.Linear(6, hidden_dim)

        # ── Message Passing Layers (GINEConv) ───────────────────────────
        def create_gine_nn():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

        self.conv1 = GINEConv(create_gine_nn(), edge_dim=hidden_dim)
        self.gn1 = GraphNorm(hidden_dim)

        self.conv2 = GINEConv(create_gine_nn(), edge_dim=hidden_dim)
        self.gn2 = GraphNorm(hidden_dim)

        self.conv3 = GINEConv(create_gine_nn(), edge_dim=hidden_dim)
        self.gn3 = GraphNorm(hidden_dim)

        # ── Global Attention Pooling ────────────────────────────────────
        self.pool1 = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)))
        self.pool2 = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)))
        self.pool3 = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)))

        # ── MLP Prediction Head ─────────────────────────────────────────
        # Jumping Knowledge: we concatenate the attention-pooled representations of all 3 layers
        # Input dim = hidden_dim * 3 layers
        jk_dim = hidden_dim * 3
        
        self.mlp = nn.Sequential(
            nn.Linear(jk_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_tasks),
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyTorch Geometric Data/Batch object with attributes:
                  x, edge_index, edge_attr, batch.

        Returns:
            logits: Raw predictions of shape (batch_size, num_tasks).
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        layer_outputs = []

        # ── Layer 1 ────────────────────────────────────────────────────
        x = self.conv1(x, edge_index, edge_attr)
        x = self.gn1(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_outputs.append(x)

        # ── Layer 2 ────────────────────────────────────────────────────
        x = self.conv2(x, edge_index, edge_attr)
        x = self.gn2(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_outputs.append(x)

        # ── Layer 3 ────────────────────────────────────────────────────
        x = self.conv3(x, edge_index, edge_attr)
        x = self.gn3(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_outputs.append(x)

        # ── Jumping Knowledge + Global Attention Pooling ───────────────
        pooled_outputs = [
            self.pool1(layer_outputs[0], batch),
            self.pool2(layer_outputs[1], batch),
            self.pool3(layer_outputs[2], batch)
        ]

        # Concatenate all layer poolings
        x_jk = torch.cat(pooled_outputs, dim=1)

        # ── MLP Head ──────────────────────────────────────────────────
        logits = self.mlp(x_jk)  

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
