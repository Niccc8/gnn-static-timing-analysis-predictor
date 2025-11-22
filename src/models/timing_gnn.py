"""
Heterogeneous Graph Neural Network for Timing Prediction

Implements a 3-layer Graph Attention Network (GAT) for binary classification
of timing endpoints (violating vs. safe).

Architecture inspired by TimingPredict (DAC 2022) and ASPDAC 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from typing import Optional


class HeterogeneousTimingGNN(nn.Module):
    """
    Heterogeneous GAT for endpoint timing classification.
    
    Architecture:
        Layer 1: 10 features → 128 hidden (4 attention heads)
        Layer 2: 512 → 128 hidden (4 attention heads)
        Layer 3: 512 → 2 classes (1 attention head)
    
    Features:
        - Multi-head attention learns timing-critical connections
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
        
        # Layer 1: in_channels → hidden_channels * heads
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim
            )
        )
        
        # Middle layers
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
        
        # Final layer: hidden_channels * heads → num_classes
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
            Log-softmax predictions (num_nodes, num_classes)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        
        return F.log_softmax(x, dim=1)
    
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
            log_probs = self.forward(data)
            probs = torch.exp(log_probs)
        return probs


class TimingGAT(nn.Module):
    """
    Alternative GAT implementation (simplified for comparison).
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * 4, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)


class TimingGCN(nn.Module):
    """
    GCN baseline for comparison (simpler than GAT).
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    # Test model
    from torch_geometric.data import Data
    
    # Create dummy data
    x = torch.randn(100, 10)  # 100 nodes, 10 features
    edge_index = torch.randint(0, 100, (2, 300))  # 300 edges
    edge_attr = torch.randn(300, 3)  # Edge features
    y = torch.randint(0, 2, (100,))  # Binary labels
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Test model forward pass
    model = HeterogeneousTimingGNN()
    output = model(data)
    
    print(f"\nModel Test:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Predicted classes: {output.argmax(dim=1)[:10]}")
