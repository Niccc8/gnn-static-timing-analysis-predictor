"""
Training Script

Trains HeterogeneousTimingGNN on circuit timing datasets.
Features:
    - Early stopping
    - Checkpointing (best model saving)
    - TensorBoard logging
    - Class weighting for imbalance handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import argparse
import yaml
import sys
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timing_gnn import HeterogeneousTimingGNN
from src.data.dataset import TimingDataset
from src.training.utils import set_seed, save_checkpoint, EarlyStopping

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    logger.warning("TensorBoard not available")


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        
        logits = model(data)
        
        # Only compute loss on labeled nodes (label >= 0)
        mask = data.y >= 0
        if mask.sum() == 0:
            continue
        
        loss = criterion(logits[mask], data.y[mask])
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = data.to(device)
        
        logits = model(data)
        
        mask = data.y >= 0
        if mask.sum() == 0:
            continue
        
        loss = criterion(logits[mask], data.y[mask])
        total_loss += loss.item()
        num_batches += 1
        
        # Get probabilities for class 1 (violation)
        probs = F.softmax(logits[mask], dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(data.y[mask].cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
        return {"loss": avg_loss, "auc": 0.0, "f1": 0.0}
    
    auc = roc_auc_score(all_labels, all_preds)
    # F1 at 0.5 threshold (can be tuned later)
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    
    return {
        "loss": avg_loss,
        "auc": auc,
        "f1": f1
    }


def main():
    parser = argparse.ArgumentParser(description="Train GNN Timing Predictor")
    parser.add_argument("--config", type=str, default="experiments/configs/default.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/processed/timing_predict", help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="experiments/logs", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logger.info(f"Device: {device}")
    
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Load Config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    # Load Data
    logger.info("Loading datasets...")
    train_dataset = TimingDataset(root=args.data_dir, split='train')
    val_dataset = TimingDataset(root=args.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=args.num_workers)
    
    logger.info(f"Train: {len(train_dataset)} graphs | Val: {len(val_dataset)} graphs")
    
    # Initialize Model
    model = HeterogeneousTimingGNN(
        in_channels=model_cfg['in_channels'],
        hidden_channels=model_cfg['hidden_channels'],
        num_classes=model_cfg['num_classes'],
        num_layers=model_cfg['num_layers'],
        heads=model_cfg['attention_heads'],
        dropout=model_cfg['dropout'],
        edge_dim=model_cfg.get('edge_dim', 3)
    ).to(device)
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )
    
    class_weights = torch.tensor(train_cfg['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training Loop
    early_stopping = EarlyStopping(patience=train_cfg['early_stop_patience'], mode='max')
    writer = SummaryWriter(args.log_dir) if SummaryWriter else None
    
    best_auc = 0.0
    
    logger.info("Starting training...")
    
    for epoch in range(train_cfg['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        val_loss = val_metrics['loss']
        val_auc = val_metrics['auc']
        val_f1 = val_metrics['f1']
        
        logger.info(
            f"Epoch {epoch+1:03d}: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val AUC={val_auc:.4f} | "
            f"Val F1={val_f1:.4f}"
        )
        
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('AUC/val', val_auc, epoch)
            writer.add_scalar('F1/val', val_f1, epoch)
        
        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                f"{args.checkpoint_dir}/best_model.pth"
            )
            logger.success(f"✓ New best model (AUC={best_auc:.4f})")
        
        # Early Stopping
        if early_stopping(val_auc):
            logger.info("Early stopping triggered")
            break
            
    if writer:
        writer.close()
    
    logger.success(f"Training Complete. Best Validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
