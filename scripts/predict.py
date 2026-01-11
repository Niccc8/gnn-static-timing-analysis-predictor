"""
GNN Timing Predictor - Inference Script

Supports three modes:
1. Threshold-based: Flag nodes above risk threshold
2. Rank-based: Return top-K riskiest nodes (recommended)
3. Adaptive: Dynamically determine K based on sensitivity
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from torch_geometric.data import Data

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.timing_gnn import HeterogeneousTimingGNN
from src.data.dataset import TimingDataset


class TimingPredictor:
    """
    Trained GNN model for timing violation prediction.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize predictor with trained model.
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.pth file)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract model config from checkpoint
        model_config = checkpoint.get('config', {}).get('model', {})
        
        # Initialize model
        self.model = HeterogeneousTimingGNN(
            in_channels=model_config.get('in_channels', 10),
            hidden_channels=model_config.get('hidden_channels', 128),
            num_classes=model_config.get('num_classes', 2),
            num_layers=model_config.get('num_layers', 3),
            heads=model_config.get('attention_heads', 4),
            dropout=model_config.get('dropout', 0.2),
            edge_dim=model_config.get('edge_dim', 3)
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    @torch.no_grad()
    def predict(self, data: Data) -> np.ndarray:
        """
        Get violation probabilities for all nodes.
        
        Returns:
            Array of violation probabilities (0.0 to 1.0) for each node
        """
        data = data.to(self.device)
        logits = self.model(data)
        probs = F.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (violation)
        return probs.cpu().numpy()
    
    def predict_threshold(
        self, 
        data: Data, 
        threshold: float = 0.58
    ) -> Dict[str, Any]:
        """Threshold-based prediction."""
        probs = self.predict(data)
        predictions = (probs >= threshold).astype(int)
        
        num_flagged = predictions.sum()
        flagged_indices = np.where(predictions == 1)[0]
        
        return {
            'probabilities': probs,
            'predictions': predictions,
            'num_nodes': len(probs),
            'num_flagged': int(num_flagged),
            'flagged_ratio': float(num_flagged / len(probs)),
            'flagged_indices': flagged_indices.tolist(),
            'threshold': threshold
        }
    
    def predict_top_k(
        self, 
        data: Data, 
        k: int = None,
        top_percent: float = 5.0
    ) -> Dict[str, Any]:
        """Rank-based prediction."""
        probs = self.predict(data)
        
        if k is None:
            k = max(1, int(len(probs) * top_percent / 100))
        
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_probs = probs[top_k_indices]
        
        return {
            'probabilities': probs,
            'num_nodes': len(probs),
            'k': k,
            'top_percent': float(k / len(probs) * 100),
            'top_k_indices': top_k_indices.tolist(),
            'top_k_probs': top_k_probs.tolist(),
            'min_top_k_prob': float(top_k_probs.min()),
            'max_top_k_prob': float(top_k_probs.max())
        }

    def predict_adaptive(
        self,
        data: Data,
        sensitivity_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Adaptive Ranking: Determine K based on sensitivity threshold.
        Captures the 'tail' of the probability distribution.
        """
        probs = self.predict(data)
        
        # Determine K dynamically
        k = (probs > sensitivity_threshold).sum()
        k = max(k, 100)  # Ensure minimum inspection budget
        
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_probs = probs[top_k_indices]
        
        return {
            'probabilities': probs,
            'num_nodes': len(probs),
            'sensitivity_threshold': sensitivity_threshold,
            'k': int(k),
            'top_percent': float(k / len(probs) * 100),
            'top_k_indices': top_k_indices.tolist(),
            'top_k_probs': top_k_probs.tolist(),
            'min_top_k_prob': float(top_k_probs.min()),
            'max_top_k_prob': float(top_k_probs.max())
        }


def main():
    parser = argparse.ArgumentParser(description='GNN Timing Violation Prediction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to preprocessed data directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--mode', type=str, default='adaptive', choices=['threshold', 'ranking', 'adaptive'], help='Prediction mode')
    parser.add_argument('--threshold', type=float, default=0.58, help='Risk threshold')
    parser.add_argument('--top_percent', type=float, default=5.0, help='Top percentage for ranking')
    parser.add_argument('--sensitivity', type=float, default=0.01, help='Sensitivity for adaptive ranking')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    predictor = TimingPredictor(args.checkpoint, device=args.device)
    
    print(f"\nLoading {args.split} dataset from {args.data_dir}...")
    dataset = TimingDataset(root=args.data_dir, split=args.split)
    print(f"✅ Loaded {len(dataset)} graphs")
    
    # Process first graph as example
    data = dataset[0]
    print(f"\nExample Prediction on Graph 0 ({data.num_nodes} nodes)")
    
    if args.mode == 'threshold':
        res = predictor.predict_threshold(data, threshold=args.threshold)
        print(f"Threshold ({args.threshold}): Flagged {res['num_flagged']} nodes ({res['flagged_ratio']*100:.2f}%)")
    
    elif args.mode == 'ranking':
        res = predictor.predict_top_k(data, top_percent=args.top_percent)
        print(f"Ranking (Top {args.top_percent}%): Inspecting {res['k']} nodes")
        print(f"Risk Range: {res['min_top_k_prob']:.4f} - {res['max_top_k_prob']:.4f}")
        
    elif args.mode == 'adaptive':
        res = predictor.predict_adaptive(data, sensitivity_threshold=args.sensitivity)
        print(f"Adaptive (Sensitivity {args.sensitivity}): Inspecting {res['k']} nodes ({res['top_percent']:.2f}%)")
        print(f"Risk Range: {res['min_top_k_prob']:.4f} - {res['max_top_k_prob']:.4f}")


if __name__ == '__main__':
    main()
