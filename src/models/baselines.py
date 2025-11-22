"""
Baseline Models for Comparison

Implements XGBoost and MLP baselines to demonstrate the advantage of GNN's 
graph structure modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Optional


class XGBoostBaseline:
    """
    XGBoost classifier on hand-crafted node features.
    
    Uses the same 10-dimensional features as GNN but without graph structure.
    """
    
    def __init__(
        self,
        max_depth: int = 8,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 9.0,  # For class imbalance
        random_state: int = 42
    ):
        """
        Initialize XGBoost classifier.
        
        Args:
            max_depth: Maximum tree depth
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            scale_pos_weight: Weight for positive class (violating)
            random_state: Random seed
        """
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='auc'
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=20):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features (num_nodes, num_features)
            y_train: Training labels (num_nodes,)
            X_val: Validation features
            y_val: Validation labels
            early_stopping_rounds: Early stopping patience
        """
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=False
        )
    
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Compute ROC-AUC score."""
        y_pred_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_pred_proba)


class MLPBaseline(nn.Module):
    """
    Multi-layer Perceptron baseline.
    
    Tests whether graph structure is necessary - if MLP performs poorly,
    it shows that graph connectivity encodes important timing information.
    """
    
    def __init__(
        self,
        in_features: int = 10,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        """
        Initialize MLP.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, in_features)
        
        Returns:
            Log-softmax predictions (num_nodes, num_classes)
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)


class RandomForestBaseline:
    """Random Forest baseline (alternative to XGBoost)."""
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: str = "balanced",
        random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X_train, y_train):
        """Train Random Forest."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Compute ROC-AUC score."""
        y_pred_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_pred_proba)


if __name__ == "__main__":
    # Test baselines
    from sklearn.datasets import make_classification
    
    # Generate dummy data
    X_train, y_train = make_classification(
        n_samples=1000, n_features=10, n_classes=2,
        weights=[0.9, 0.1], random_state=42
    )
    X_test, y_test = make_classification(
        n_samples=200, n_features=10, n_classes=2,
        weights=[0.9, 0.1], random_state=43
    )
    
    # Test XGBoost
    print("Testing XGBoost Baseline:")
    xgb_model = XGBoostBaseline()
    xgb_model.fit(X_train, y_train)
    xgb_auc = xgb_model.score(X_test, y_test)
    print(f"  XGBoost AUC: {xgb_auc:.4f}")
    
    # Test MLP
    print("\nTesting MLP Baseline:")
    mlp_model = MLPBaseline()
    X_train_torch = torch.FloatTensor(X_train)
    y_train_torch = torch.LongTensor(y_train)
    
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Quick training
    mlp_model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = mlp_model(X_train_torch)
        loss = criterion(output, y_train_torch)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    mlp_model.eval()
    with torch.no_grad():
        X_test_torch = torch.FloatTensor(X_test)
        output = mlp_model(X_test_torch)
        probs = torch.exp(output)[:, 1].numpy()
        mlp_auc = roc_auc_score(y_test, probs)
    
    print(f"  MLP AUC: {mlp_auc:.4f}")
