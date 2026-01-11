#!/usr/bin/env python3
"""
COMPLETE DATA GENERATOR - All Tables from Existing Data
This script generates all computable tables without modifying any existing code.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.timing_gnn import HeterogeneousTimingGNN

OUTPUT_DIR = Path("experiments/results/paper_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def table_03_dataset_stats():
    """Table III: Dataset Statistics - from actual .pt files."""
    print("\n" + "="*70)
    print("TABLE III: Dataset Statistics")
    print("="*70)
    
    results = []
    for split in ['train', 'val', 'test']:
        pt_file = Path(f"data/processed/timing_predict/{split}.pt")
        if not pt_file.exists():
            continue
            
        print(f"Loading {split}.pt...")
        data_list = torch.load(pt_file, weights_only=False)
        
        total_nodes = 0
        total_viol = 0
        sizes = []
        
        for data in data_list:
            nn = data.num_nodes
            viol = (data.y == 1).sum().item()
            total_nodes += nn
            total_viol += viol
            sizes.append(nn)
        
        rate = (total_viol/total_nodes*100) if total_nodes > 0 else 0
        size_range = f"{min(sizes)//1000}K–{max(sizes)//1000}K"
        
        results.append({
            'Split': split.capitalize(),
            '# Designs': len(data_list),
            'Nodes': f"{total_nodes:,}",
            'Viol.': f"{total_viol:,}",
            'Rate': f"{rate:.2f}%",
            'Size': size_range
        })
    
    # Total row
    total = {
        'Split': 'Total',
        '# Designs': sum(r['# Designs'] for r in results),
        'Nodes': sum(int(r['Nodes'].replace(',','')) for r in results),
        'Viol.': sum(int(r['Viol.'].replace(',','')) for r in results),
    }
    total['Rate'] = f"{(int(total['Viol.'].replace(',',''))/int(total['Nodes'].replace(',',''))*100):.2f}%"
    
    all_sizes = []
    for split in ['train','val','test']:
        pt = Path(f"data/processed/timing_predict/{split}.pt")
        if pt.exists():
            dl = torch.load(pt, weights_only=False)
            all_sizes.extend([d.num_nodes for d in dl])
    total['Size'] = f"{min(all_sizes)//1000}K–{max(all_sizes)//1000}K"
    total['Nodes'] = f"{total['Nodes']:,}"
    total['Viol.'] = f"{total['Viol.']:,}"
    
    results.append(total)
    df = pd.DataFrame(results)
    
    print("\n" + df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "table_03_dataset_stats.csv", index=False)
    print(f"\n✅ Saved: {OUTPUT_DIR}/table_03_dataset_stats.csv")
    return df

def table_04_and_predictions():
    """Table IV: Main Results + generate predictions for other tables."""
    print("\n" + "="*70)
    print("TABLE IV: Main Results (Running Inference)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    ckpt = torch.load("experiments/checkpoints/best_model.pth", map_location=device, weights_only=False)
    cfg = ckpt.get('config', {}).get('model', {})
    
    model = HeterogeneousTimingGNN(
        in_channels=cfg.get('in_channels', 10),
        hidden_channels=cfg.get('hidden_channels', 128),
        num_classes=2,
        num_layers=cfg.get('num_layers', 3),
        heads=cfg.get('attention_heads', 4),
        dropout=cfg.get('dropout', 0.2),
        edge_dim=cfg.get('edge_dim', 3)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    
    # Load test data
    test_data = torch.load("data/processed/timing_predict/test.pt", weights_only=False)
    print(f"✅ Loaded {len(test_data)} test designs")
    
    # Run inference
    all_probs, all_labels, all_designs = [], [], []
    times = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for i, data in enumerate(test_data):
            data = data.to(device)
            
            t0 = time.time()
            logits = model(data)
            t_ms = (time.time() - t0) * 1000
            
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = data.y.cpu().numpy()
            
            mask = labels != -1
            all_probs.extend(probs[mask])
            all_labels.extend(labels[mask])
            all_designs.extend([i] * mask.sum())
            times.append(t_ms)
            
            print(f"  [{i+1}/{len(test_data)}] {mask.sum()} nodes, {t_ms:.1f}ms")
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_designs = np.array(all_designs)
    
    print(f"\n✅ Total: {len(all_probs)} nodes, {all_labels.sum()} violations")
    
    # Compute metrics
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_probs, all_labels)
    
    # Ranking @ 5%
    k5 = int(len(all_probs) * 0.05)
    top_k = np.argsort(all_probs)[::-1][:k5]
    recall_rank = (all_labels[top_k].sum() / all_labels.sum()) * 100
    
    # Thresholds
    r50 = ((all_probs >= 0.5) * all_labels).sum() / all_labels.sum() * 100
    r58 = ((all_probs >= 0.58) * all_labels).sum() / all_labels.sum() * 100
    
    avg_time = np.mean(times)
    
    # Table
    results = [
        {'Method': 'Random', 'ROC-AUC': '0.50', 'PR-AUC': '--', 'Recall@5%': '5.0%', 'Time': '--'},
        {'Method': 'Threshold-GAT (0.5)', 'ROC-AUC': f'{roc_auc:.2f}', 'PR-AUC': f'{pr_auc:.2f}', 
         'Recall@5%': f'{r50:.1f}%', 'Time': f'{avg_time:.0f}ms'},
        {'Method': 'Threshold-GAT (0.58)', 'ROC-AUC': f'{roc_auc:.2f}', 'PR-AUC': f'{pr_auc:.2f}',
         'Recall@5%': f'{r58:.1f}%', 'Time': f'{avg_time:.0f}ms'},
        {'Method': 'RankSTA (Ours)', 'ROC-AUC': f'{roc_auc:.2f}', 'PR-AUC': f'{pr_auc:.2f}',
         'Recall@5%': f'{recall_rank:.1f}%', 'Time': f'{avg_time:.0f}ms'},
    ]
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "table_04_main_results.csv", index=False)
    
    # Save predictions
    pred_data = {
        'probs': all_probs.tolist(),
        'labels': all_labels.tolist(),
        'designs': all_designs.tolist(),
        'metrics': {'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc)}
    }
    with open(OUTPUT_DIR / "predictions.json", 'w') as f:
        json.dump(pred_data, f)
    
    print(f"\n✅ Saved: {OUTPUT_DIR}/table_04_main_results.csv")
    print(f"✅ Saved: {OUTPUT_DIR}/predictions.json")
    
    return df, pred_data

def table_06_recall_at_k(pred_data):
    """Table VI: Recall at Different K%."""
    print("\n" + "="*70)
    print("TABLE VI: Recall at Different K%")
    print("="*70)
    
    probs = np.array(pred_data['probs'])
    labels = np.array(pred_data['labels'])
    
    results = []
    for k_pct in [1, 3, 5, 10, 20]:
        k = int(len(probs) * k_pct / 100)
        top_k = np.argsort(probs)[::-1][:k]
        
        tp = labels[top_k].sum()
        recall = (tp / labels.sum()) * 100
        precision = (tp / k) * 100
        gain = int(100 / k_pct)
        
        results.append({
            'Top K%': f"{k_pct}%",
            'Nodes': f"~{k//1000}K" if k > 1000 else f"~{k}",
            'Recall': f"{recall:.1f}%",
            'Precision': f"{precision:.1f}%",
            'Gain': f"{gain}×"
        })
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "table_06_recall_at_k.csv", index=False)
    print(f"\n✅ Saved: {OUTPUT_DIR}/table_06_recall_at_k.csv")
    return df

def main():
    """Run all quick computations."""
    print("\n" + "="*70)
    print(" GENERATING ALL PAPER TABLES")
    print("="*70)
    
    # Table III (no model needed)
    table_03_dataset_stats()
    
    # Table IV + predictions
    table_04, pred_data = table_04_and_predictions()
    
    # Table VI
    table_06_recall_at_k(pred_data)
    
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETE!")
    print("="*70)
    print(f"\nAll tables saved to: {OUTPUT_DIR}")
    print("\n Generated:")
    print("  ✅ Table III: Dataset Statistics")
    print("  ✅ Table IV: Main Results")
    print("  ✅ Table VI: Recall@K")
    print("  ✅ Predictions (JSON)")
    
    print("\n📝 Still TODO (mark in paper):")
    print("  ⏭️  Table V: Statistical Validation (5 runs - expensive)")
    print("  ⏭️  Table VII-IX: Ablation Studies (if models exist)")
    print("  ⏭️  Table X: Per-Design (needs design labels)")
    print("  ⏭️  Table XI: Adaptive K")

if __name__ == "__main__":
    main()
