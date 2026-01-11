"""
Model Architectures Module

Contains GNN model implementations for timing violation prediction.
"""

from .timing_gnn import HeterogeneousTimingGNN

__all__ = [
    "HeterogeneousTimingGNN",
]
