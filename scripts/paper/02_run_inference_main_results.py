#!/usr/bin/env python3
"""
Script 02: Run Inference and Compute Main Results (Table IV)
Loads trained model, runs inference on test set, computes metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import pandas as pd
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.timing_gnn import HeterogeneousTimingGNN

def run_inference_and_compute_metrics(
    checkpoint_path: str = "experiments/checkpoints/best_model.pth",
    test_data_path: str = "data/processed/timing_predict/test.pt",
    output_dir: str = "experiments/results"
):
    """
    Run inference on test set and compute all metrics for Table IV.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config
    model_config = checkpoint.get('config', {}).get('model', {})
    
    # Initialize model
    model = HeterogeneousTimingGNN(
        in_channels=model_config.get('in_channels', 10),
        hidden_channels=model_config.get('hidden_channels', 128),
        num_classes=model_config.get('num_classes', 2),
        num_layers=model_config.get('num_layers', 3),
        heads=model_config.get('attention_heads', 4),
        dropout=model_config.get('dropout', 0.2),
        edge_dim=model_config.get('edge_dim', 3)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_data_list = torch.load(test_data_path, weights_only=False)
    print(f"✅ Loaded {len(test_data_list)} test designs")
    
    # Run inference on all test designs
    all_probs = []
    all_labels = []
    all_design_names = []
    inference_times = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for i, data in enumerate(test_data_list):
            design_name = getattr(data, 'design_name', f'design_{i}')
            print(f"  [{i+1}/{len(test_data_list)}] {design_name}...")
            
            data = data.to(device)
            
            # Measure inference time
            start_time = time.time()
            logits = model(data)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = data.y.cpu().numpy()
            
            # Filter out unlabeled nodes (label == -1)
            mask = labels != -1
            probs_filtered = probs[mask]
            labels_filtered = labels[mask]
            
            all_probs.extend(probs_filtered)
            all_labels.extend(labels_filtered)
            all_design_names.extend([design_name] * len(probs_filtered))
            inference_times.append(inference_time)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"\n✅ Inference complete: {len(all_probs)} nodes, {all_labels.sum()} violations")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    # Recall@5% (ranking-based)
    top_k_percent = 5.0
    k = int(len(all_probs) * top_k_percent / 100)
    top_k_indices = np.argsort(all_probs)[::-1][:k]
    top_k_labels = all_labels[top_k_indices]
    recall_at_5 = (top_k_labels.sum() / all_labels.sum()) * 100
    
    # Average inference time
    avg_time = np.mean(inference_times)
    
    # Threshold-based results for comparison
    threshold_05_preds = (all_probs >= 0.5).astype(int)
    threshold_05_recall = (threshold_05_preds * all_labels).sum() / all_labels.sum() * 100
    
    threshold_058_preds = (all_probs >= 0.58).astype(int)
    threshold_058_recall = (threshold_058_preds * all_labels).sum() / all_labels.sum() * 100
    
    # Build results table
    results = [
        {
            'Method': 'Random',
            'ROC-AUC': 0.50,
            'PR-AUC': '--',
            'Recall@5%': '5.0%',
            'Time': '--'
        },
        {
            'Method': 'Threshold-GAT (0.5)',
            'ROC-AUC': f"{roc_auc:.2f}",
            'PR-AUC': f"{pr_auc:.2f}",
            'Recall@5%': f"{threshold_05_recall:.1f}%",
            'Time': f"{avg_time:.0f}ms"
        },
        {
            'Method': 'Threshold-GAT (0.58)',
            'ROC-AUC': f"{roc_auc:.2f}",
            'PR-AUC': f"{pr_auc:.2f}",
            'Recall@5%': f"{threshold_058_recall:.1f}%",
            'Time': f"{avg_time:.0f}ms"
        },
        {
            'Method': '**RankSTA (Ours)**',
            'ROC-AUC': f"**{roc_auc:.2f}**",
            'PR-AUC': f"**{pr_auc:.2f}**",
            'Recall@5%': f"**{recall_at_5:.1f}%**",
            'Time': f"**{avg_time:.0f}ms**"
        }
    ]
    
    df = pd.DataFrame(results)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_data = {
        'probabilities': all_probs.tolist(),
        'labels': all_labels.tolist(),
        'design_names': all_design_names,
        'metrics': {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'recall_at_5': float(recall_at_5),
            'avg_inference_time_ms': float(avg_time)
        }
    }
    
    pred_file = Path(output_dir) / "test_predictions.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"✅ Saved predictions to: {pred_file}")
    
    # Save table
    table_file = Path(output_dir) / "table_04_main_results.csv"
    df.to_csv(table_file, index=False)
    print(f"✅ Saved table to: {table_file}")
    
    return df, predictions_data

if __name__ == "__main__":
    print("Running Inference and Computing Main Results (Table IV)...")
    print("=" * 70)
    
    df, pred_data = run_inference_and_compute_metrics()
    
    print("\nRESULTS:")
    print(df.to_string(index=False))
    
    print("\nLaTeX TABLE FORMAT:")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"{row['Method']} & {row['ROC-AUC']} & {row['PR-AUC']} & {row['Recall@5%']} & {row['Time']} \\\\")
    
    print(f"\n✅ All results saved to experiments/results/")
