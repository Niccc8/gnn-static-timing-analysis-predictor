"""
Training Utilities

Helper functions for training, checkpointing, and reproducibility.
"""

import torch
import random
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug(f"Set random seed to {seed}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    logger.debug(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: str, 
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logger.info(f"Loaded checkpoint from {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
