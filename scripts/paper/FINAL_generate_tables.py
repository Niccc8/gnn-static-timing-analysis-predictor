#!/usr/bin/env python3
"""
FINAL WORKING VERSION - Generate Paper Tables
Handles actual data format (tuple of train_data, val_data lists).
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

def load_split_data(split_name):
    """Load data handling tuple format."""
    pt_file = Path(f"data/processed/timing_predict/{split_name}.pt")
    data = torch.load(pt_file, weights_only=False)
    
    # Handle tuple format (train_data, val_data) or direct list
    if isinstance(data, tuple):
        data_list = data[0] if isinstance(data[0], list) else [data[0]]
    elif isinstance(data, list):
        data_list = data
    else:
        data_list = [data]
    
    return data_list

def table_03_dataset_stats():
    """Table III: Dataset Statistics."""
    print("\n" + "="*70)
    print("TABLE III: Dataset Statistics")
    print("="*70)
    
    results = []
    for split in ['train', 'val', 'test']:
        try:
            data_list = load_split_data(split)
            print(f"✅ Loaded {split}: {len(data_list)} graphs")
            
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
        except Exception as e:
            print(f"⚠️  Error loading {split}: {e}")
    
    # Total
    total_designs = sum(r['# Designs'] for r in results)
    total_nodes = sum(int(r['Nodes'].replace(',','')) for r in results)
    total_viol = sum(int(r['Viol.'].replace(',','')) for r in results)
    total_rate = (total_viol/total_nodes*100) if total_nodes > 0 else 0
    
    all_sizes = []
    for split in ['train','val','test']:
        try:
            dl = load_split_data(split)
            all_sizes.extend([d.num_nodes for d in dl])
        except:
            pass
    
    results.append({
        'Split': 'Total',
        '# Designs': total_designs,
        'Nodes': f"{total_nodes:,}",
        'Viol.': f"{total_viol:,}",
        'Rate': f"{total_rate:.2f}%",
        'Size': f"{min(all_sizes)//1000}K–{max(all_sizes)//1000}K"
    })
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "table_03_dataset_stats.csv", index=False)
    print(f"\n✅ table_03_dataset_stats.csv")
    return df

def table_04_main_results():
    """Table IV: Main Results."""
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
    test_data = load_split_data('test')
    print(f"✅ Test: {len(test_data)} designs")
    
    # Inference
    all_probs, all_labels = [], []
    times = []
    
    print("\nInference:")
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
            times.append(t_ms)
            
            print(f"  [{i+1}/{len(test_data)}] {mask.sum()} nodes, {t_ms:.0f}ms")
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"\n✅ {len(all_probs)} nodes, {all_labels.sum()} violations")
    
    # Metrics
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    k5 = int(len(all_probs) * 0.05)
    top_k = np.argsort(all_probs)[::-1][:k5]
    recall_rank = (all_labels[top_k].sum() / all_labels.sum()) * 100
    
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
    with open(OUTPUT_DIR / "predictions.json", 'w') as f:
        json.dump({'probs': all_probs.tolist(), 'labels': all_labels.tolist()}, f)
    
    print(f"\n✅ table_04_main_results.csv")
    print(f"✅ predictions.json")
    
    return df, all_probs, all_labels

def table_06_recall_k(probs, labels):
    """Table VI: Recall@K."""
    print("\n" + "="*70)
    print("TABLE VI: Recall at K%")
    print("="*70)
    
    results = []
    for k_pct in [1, 3, 5, 10, 20]:
        k = int(len(probs) * k_pct / 100)
        top = np.argsort(probs)[::-1][:k]
        
        tp = labels[top].sum()
        recall = (tp / labels.sum()) * 100
        prec = (tp / k) * 100
        
        results.append({
            'Top K%': f"{k_pct}%",
            'Nodes': f"~{k:,}",
            'Recall': f"{recall:.1f}%",
            'Prec.': f"{prec:.1f}%",
            'Gain': f"{int(100/k_pct)}×"
        })
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "table_06_recall_at_k.csv", index=False)
    print(f"\n✅ table_06_recall_at_k.csv")
    return df

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" GENERATING ALL PAPER TABLES")
    print("="*70)
    
    t03 = table_03_dataset_stats()
    t04, probs, labels = table_04_main_results()
    t06 = table_06_recall_k(probs, labels)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nSaved to: {OUTPUT_DIR}/")
    print("  ✅ Table III, IV, VI")
    print("  ✅ predictions.json")
