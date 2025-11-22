"""
Training script for GNN timing predictor.

Trains HeterogeneousTimingGNN on circuit timing datasets with early stopping,
checkpointing, and TensorBoard logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import argparse
import yaml
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timing_gnn import HeterogeneousTimingGNN
from src.data.dataset import TimingDataset
from src.training.utils import set_seed, save_checkpoint, EarlyStopping

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    SummaryWriter = None
    logger.warning("TensorBoard not available")


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: GNN model
        loader: DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: torch.device
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(data)
        
        # Compute loss (only on labeled nodes)
        mask = data.y >= 0  # Valid labels (≥0)
        if mask.sum() == 0:
            continue
        
        loss = criterion(logits[mask], data.y[mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: GNN model
        loader: DataLoader
        criterion: Loss function
        device: torch.device
    
    Returns:
        Dictionary with loss, AUC, F1 metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = data.to(device)
        
        # Forward pass
        logits = model(data)
        
        # Get valid labels
        mask = data.y >= 0
        if mask.sum() == 0:
            continue
        
        # Compute loss
        loss = criterion(logits[mask], data.y[mask])
        total_loss += loss.item()
        num_batches += 1
        
        # Get predictions
        probs = F.softmax(logits[mask], dim=1)[:, 1]  # P(class=1)
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(data.y[mask].cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
        logger.warning("Insufficient labels for metric computation")
        return {"loss": avg_loss, "auc": 0.0, "f1": 0.0}
    
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    
    return {
        "loss": avg_loss,
        "auc": auc,
        "f1": f1
    }


def main(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = TimingDataset(root=args.data_dir, split='train')
    val_dataset = TimingDataset(root=args.data_dir, split='val')
    
    logger.info(f"Train: {len(train_dataset)} graphs, Val: {len(val_dataset)} graphs")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Creating model...")
    model = HeterogeneousTimingGNN(
        in_channels=model_config['in_channels'],
        hidden_channels=model_config['hidden_channels'],
        num_classes=model_config['num_classes'],
        num_layers=model_config['num_layers'],
        heads=model_config['attention_heads'],
        dropout=model_config['dropout'],
        edge_dim=3  # Edge feature dimension
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Loss function (with class weights)
    class_weights = torch.tensor(training_config['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=training_config['early_stop_patience'],
        mode='max'  # Maximize AUC
    )
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir) if SummaryWriter else None
    
    # Training loop
    logger.info("Starting training...")
    best_val_auc = 0
    
    for epoch in range(training_config['max_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_auc = val_metrics['auc']
        val_f1 = val_metrics['f1']
        
        # Log
        logger.info(
            f"Epoch {epoch+1}/{training_config['max_epochs']}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Val AUC={val_auc:.4f}, "
            f"Val F1={val_f1:.4f}"
        )
        
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('AUC/val', val_auc, epoch)
            writer.add_scalar('F1/val', val_f1, epoch)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_auc": val_auc, "val_f1": val_f1},
                f"{args.checkpoint_dir}/best_model.pth"
            )
            logger.info(f"✓ New best model saved (AUC={val_auc:.4f})")
        
        # Early stopping
        if early_stopping(val_auc):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"Training complete! Best Val AUC: {best_val_auc:.4f}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN timing predictor")
    
    parser.add_argument("--config", type=str, default="experiments/configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Root directory containing processed datasets")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="experiments/logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader workers")
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    main(args)
