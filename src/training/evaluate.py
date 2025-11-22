"""
Comprehensive evaluation script for GNN timing predictor.

Evaluates trained models on test set with detailed metrics including:
- ROC-AUC, PR-AUC, F1, Precision, Recall
- Cross-design generalization statistics
- Per-design breakdown
- Runtime analysis
- Visualization (ROC/PR curves)
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.timing_gnn import HeterogeneousTimingGNN
from data.dataset import TimingDataset
from training.utils import load_checkpoint


class Evaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, model, device, output_dir):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @torch.no_grad()
    def evaluate_dataset(self, loader):
        """
        Evaluate model on a dataset.
        
        Args:
            loader: DataLoader
        
        Returns:
            Dictionary with predictions, labels, and metrics
        """
        self.model.eval()
        
        all_preds_prob = []
        all_labels = []
        all_design_names = []
        inference_times = []
        
        for data in loader:
            data = data.to(self.device)
            
            # Measure inference time
            start_time = time.time()
            logits = self.model(data)
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)
            
            # Get predictions
            probs = F.softmax(logits, dim=1)[:, 1]
            
            # Extract valid labels
            mask = data.y >= 0
            all_preds_prob.extend(probs[mask].cpu().numpy())
            all_labels.extend(data.y[mask].cpu().numpy())
            
            # Track design names for per-design breakdown
            if hasattr(data, 'design_name'):
                all_design_names.extend([data.design_name] * mask.sum().item())
        
        all_preds_prob = np.array(all_preds_prob)
        all_labels = np.array(all_labels)
        all_preds_hard = (all_preds_prob > 0.5).astype(int)
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds_prob, all_preds_hard)
        metrics['avg_inference_time_ms'] = np.mean(inference_times)
        metrics['std_inference_time_ms'] = np.std(inference_times)
        
        return {
            'predictions_prob': all_preds_prob,
            'predictions_hard': all_preds_hard,
            'labels': all_labels,
            'design_names': all_design_names,
            'metrics': metrics,
            'inference_times': inference_times
        }
    
    def _compute_metrics(self, y_true, y_prob, y_pred):
        """Compute all classification metrics."""
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present, some metrics will be invalid")
            return {}
        
        # ROC-AUC and PR-AUC
        roc_auc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': prec,
            'recall': rec,
            'confusion_matrix': cm.tolist()
        }
    
    def per_design_breakdown(self, results):
        """Compute per-design metrics."""
        design_names = results['design_names']
        if not design_names:
            logger.warning("No design names available for breakdown")
            return pd.DataFrame()
        
        # Group by design
        unique_designs = np.unique(design_names)
        per_design_metrics = []
        
        for design in unique_designs:
            mask = np.array(design_names) == design
            y_true = results['labels'][mask]
            y_prob = results['predictions_prob'][mask]
            y_pred = results['predictions_hard'][mask]
            
            if len(np.unique(y_true)) < 2:
                continue
            
            metrics = self._compute_metrics(y_true, y_prob, y_pred)
            metrics['design'] = design
            metrics['num_endpoints'] = len(y_true)
            metrics['num_violating'] = y_true.sum()
            metrics['violation_rate'] = y_true.mean()
            
            per_design_metrics.append(metrics)
        
        df = pd.DataFrame(per_design_metrics)
        return df
    
    def plot_roc_curve(self, results, save_path=None):
        """Plot ROC curve."""
        y_true = results['labels']
        y_prob = results['predictions_prob']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = results['metrics']['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'GNN (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Timing Violation Prediction', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved ROC curve to {save_path}")
        plt.close()
    
    def plot_pr_curve(self, results, save_path=None):
        """Plot Precision-Recall curve."""
        y_true = results['labels']
        y_prob = results['predictions_prob']
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = results['metrics']['pr_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'GNN (AUC = {pr_auc:.3f})', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Timing Violation Prediction', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")
        plt.close()
    
    def save_results(self, results, prefix="test"):
        """Save evaluation results to disk."""
        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_path = self.output_dir / f"{prefix}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'label': results['labels'],
            'prediction_prob': results['predictions_prob'],
            'prediction_class': results['predictions_hard']
        })
        if results['design_names']:
            predictions_df['design'] = results['design_names']
        
        pred_path = self.output_dir / f"{prefix}_predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions to {pred_path}")


def main(args):
    """Main evaluation function."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = TimingDataset(root=args.data_dir, split='test')
    logger.info(f"Test: {len(test_dataset)} graphs")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Loading model...")
    model = HeterogeneousTimingGNN(
        in_channels=10,
        hidden_channels=128,
        num_classes=2,
        num_layers=3,
        heads=4,
        dropout=0.2,
        edge_dim=3
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model)
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Create evaluator
    evaluator = Evaluator(model, device, args.output_dir)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    results = evaluator.evaluate_dataset(test_loader)
    
    # Print metrics
    logger.info("\n=== Test Set Metrics ===")
    for key, value in results['metrics'].items():
        if key != 'confusion_matrix':
            logger.info(f"  {key}: {value:.4f}")
    
    # Per-design breakdown
    if args.per_design:
        logger.info("\nComputing per-design breakdown...")
        per_design_df = evaluator.per_design_breakdown(results)
        per_design_path = Path(args.output_dir) / "per_design_metrics.csv"
        per_design_df.to_csv(per_design_path, index=False)
        logger.info(f"Saved per-design metrics to {per_design_path}")
        
        logger.info(f"\nCross-design statistics:")
        logger.info(f"  Mean AUC: {per_design_df['roc_auc'].mean():.4f}")
        logger.info(f"  Std AUC: {per_design_df['roc_auc'].std():.4f}")
        logger.info(f"  Min AUC: {per_design_df['roc_auc'].min():.4f}")
        logger.info(f"  Max AUC: {per_design_df['roc_auc'].max():.4f}")
    
    # Plot curves
    if args.plot:
        logger.info("\nGenerating plots...")
        evaluator.plot_roc_curve(results, Path(args.output_dir) / "roc_curve.png")
        evaluator.plot_pr_curve(results, Path(args.output_dir) / "pr_curve.png")
    
    # Save results
    evaluator.save_results(results, prefix="test")
    
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GNN timing predictor")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Root directory containing processed datasets")
    parser.add_argument("--output_dir", type=str, default="experiments/results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--per_design", action="store_true",
                        help="Compute per-design breakdown")
    parser.add_argument("--plot", action="store_true",
                        help="Generate ROC/PR plots")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader workers")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    main(args)
