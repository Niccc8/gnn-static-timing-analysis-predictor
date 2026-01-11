"""
Heterogeneous Graph Neural Network for Timing Prediction

Implements a 3-layer Graph Attention Network (GAT) for binary classification
of timing endpoints (violating vs. safe).

Architecture:
    - Layer 1: Input (10 dim) -> Hidden (128 dim, 4 heads)
    - Layer 2: Hidden -> Hidden (128 dim, 4 heads)
    - Layer 3: Hidden -> Output (2 classes, 1 head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional


class HeterogeneousTimingGNN(nn.Module):
    """
    Heterogeneous GAT for endpoint timing classification.
    
    Features:
        - Multi-head attention to learn timing-critical connections
        - Dropout for regularization
        - ReLU activations
        - Log-softmax output for classification
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: Optional[int] = 3
    ):
        """
        Initialize GNN model.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            num_classes: Number of output classes (2 for binary)
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
            edge_dim: Edge feature dimension (optional)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # Layer 1: Input -> Hidden
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim
            )
        )
        
        # Middle layers: Hidden -> Hidden
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
        
        # Final layer: Hidden -> Output
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                num_classes,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim
            )
        )
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr
        
        Returns:
            Logits (num_nodes, num_classes)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply GAT layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        
        return x
    
    def predict_proba(self, data):
        """
        Predict class probabilities.
        
        Args:
            data: PyG Data object
        
        Returns:
            Softmax probabilities (num_nodes, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probs = F.softmax(logits, dim=1)
        return probs
