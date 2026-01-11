#!/usr/bin/env python3
"""
MASTER SCRIPT: Compute All Paper Tables and Metrics
This script generates all data for the paper tables from actual experiments.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import time
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timing_gnn import HeterogeneousTimingGNN

class PaperDataGenerator:
    """Generate all tables and data for the paper."""
    
    def __init__(self, 
                 checkpoint_path="experiments/checkpoints/best_model.pth",
                 data_dir="data/processed/timing_predict",
                 output_dir="experiments/results/paper_data"):
        self.checkpoint_path = checkpoint_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load trained model."""
        print(f"\n📦 Loading model from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        model_config = checkpoint.get('config', {}).get('model', {})
        
        self.model = HeterogeneousTimingGNN(
            in_channels=model_config.get('in_channels', 10),
            hidden_channels=model_config.get('hidden_channels', 128),
            num_classes=model_config.get('num_classes', 2),
            num_layers=model_config.get('num_layers', 3),
            heads=model_config.get('attention_heads', 4),
            dropout=model_config.get('dropout', 0.2),
            edge_dim=model_config.get('edge_dim', 3)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✅ Model loaded successfully")
        
    def compute_table_03_dataset_stats(self):
        """Table III: Dataset Statistics."""
        print("\n" + "="*70)
        print("TABLE III: Dataset Statistics")
        print("="*70)
        
        results = []
        
        for split in ['train', 'val', 'test']:
            pt_file = self.data_dir / f"{split}.pt"
            
            if not pt_file.exists():
                print(f"⚠️  {split}.pt not found")
                continue
            
            print(f"Processing {split}.pt...")
            data_list = torch.load(pt_file, weights_only=False)
            
            total_nodes = 0
            total_violations = 0
            design_count = len(data_list)
            sizes = []
            
            for data in data_list:
                # Handle both dict and Data objects
                if isinstance(data, dict):
                    num_nodes = data['num_nodes']
                    y = data['y']
                else:
                    num_nodes = data.num_nodes
                    y = data.y
                
                violations = (y == 1).sum().item() if torch.is_tensor(y) else (y == 1).sum()
                
                total_nodes += num_nodes
                total_violations += violations
                sizes.append(num_nodes)
            
            viol_rate = (total_violations / total_nodes * 100) if total_nodes > 0 else 0
            
            min_size = min(sizes) if sizes else 0
            max_size = max(sizes) if sizes else 0
            size_range = f"{int(min_size/1000)}K–{int(max_size/1000)}K"
            
            results.append({
                'Split': split.capitalize(),
                '# Designs': design_count,
                'Nodes': total_nodes,
                'Viol.': total_violations,
                'Rate': f"{viol_rate:.2f}%",
                'Size': size_range
            })
        
        # Total row
        total_designs = sum([r['# Designs'] for r in results])
        total_nodes = sum([r['Nodes'] for r in results])
        total_viol = sum([r['Viol.'] for r in results])
        total_rate = (total_viol / total_nodes * 100) if total_nodes > 0 else 0
        
        all_sizes = []
        for split in ['train', 'val', 'test']:
            pt_file = self.data_dir / f"{split}.pt"
            if pt_file.exists():
                data_list = torch.load(pt_file, weights_only=False)
                for data in data_list:
                    num_nodes = data['num_nodes'] if isinstance(data, dict) else data.num_nodes
                    all_sizes.append(num_nodes)
        
        overall_size = f"{int(min(all_sizes)/1000)}K–{int(max(all_sizes)/1000)}K"
        
        results.append({
            'Split': 'Total',
            '# Designs': total_designs,
            'Nodes': total_nodes,
            'Viol.': total_viol,
            'Rate': f"{total_rate:.2f}%",
            'Size': overall_size
        })
        
        df = pd.DataFrame(results)
        
        # Format for display
        df_display = df.copy()
        df_display['Nodes'] = df_display['Nodes'].apply(lambda x: f"{x:,}" if isinstance(x, int) else x)
        df_display['Viol.'] = df_display['Viol.'].apply(lambda x: f"{x:,}" if isinstance(x, int) else x)
        
        print("\n" + df_display.to_string(index=False))
        
        # Save
        df.to_csv(self.output_dir / "table_03_dataset_stats.csv", index=False)
        print(f"\n✅ Saved to: {self.output_dir / 'table_03_dataset_stats.csv'}")
        
        return df
    
    def compute_table_04_main_results(self):
        """Table IV: Main Results (requires inference)."""
        print("\n" + "="*70)
        print("TABLE IV: Main Results")
        print("="*70)
        
        # Load test data
        test_file = self.data_dir / "test.pt"
        print(f"Loading {test_file}...")
        test_data_list = torch.load(test_file, weights_only=False)
        print(f"Loaded {len(test_data_list)} test designs")
        
        # Run inference
        all_probs = []
        all_labels = []
        inference_times = []
        
        print("\nRunning inference on test set...")
        with torch.no_grad():
            for i, data_item in enumerate(test_data_list):
                # Convert dict to Data object if needed
                if isinstance(data_item, dict):
                    from torch_geometric.data import Data
                    data = Data(
                        x=torch.tensor(data_item['x'], dtype=torch.float32),
                        edge_index=torch.tensor(data_item['edge_index'], dtype=torch.long),
                        edge_attr=torch.tensor(data_item.get('edge_attr', []), dtype=torch.float32) if 'edge_attr' in data_item and data_item['edge_attr'] is not None else None,
                        y=torch.tensor(data_item['y'], dtype=torch.long)
                    )
                else:
                    data = data_item
                
                data = data.to(self.device)
                
                start_time = time.time()
                logits = self.model(data)
                inference_time = (time.time() - start_time) * 1000  # ms
                
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                labels = data.y.cpu().numpy()
                
                # Filter unlabeled (-1)
                mask = labels != -1
                probs_filtered = probs[mask]
                labels_filtered = labels[mask]
                
                all_probs.extend(probs_filtered)
                all_labels.extend(labels_filtered)
                inference_times.append(inference_time)
                
                print(f"  [{i+1}/{len(test_data_list)}] {len(probs_filtered)} nodes, time: {inference_time:.1f}ms")
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        print(f"\n✅ Total: {len(all_probs)} nodes, {all_labels.sum()} violations")
        
        # Compute metrics
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        
        # Ranking-based Recall@5%
        k = int(len(all_probs) * 0.05)
        top_k_indices = np.argsort(all_probs)[::-1][:k]
        recall_ranking = (all_labels[top_k_indices].sum() / all_labels.sum()) * 100
        
        # Threshold-based recalls
        recall_threshold_50 = ((all_probs >= 0.5) * all_labels).sum() / all_labels.sum() * 100
        recall_threshold_58 = ((all_probs >= 0.58) * all_labels).sum() / all_labels.sum() * 100
        
        avg_time = np.mean(inference_times)
        
        # Build table
        results = [
            {'Method': 'Random', 'ROC-AUC': '0.50', 'PR-AUC': '--', 'Recall@5%': '5.0%', 'Time': '--'},
            {'Method': 'Threshold-GAT (0.5)', 'ROC-AUC': f'{roc_auc:.2f}', 'PR-AUC': f'{pr_auc:.2f}', 
             'Recall@5%': f'{recall_threshold_50:.1f}%', 'Time': f'{avg_time:.0f}ms'},
            {'Method': 'Threshold-GAT (0.58)', 'ROC-AUC': f'{roc_auc:.2f}', 'PR-AUC': f'{pr_auc:.2f}',
             'Recall@5%': f'{recall_threshold_58:.1f}%', 'Time': f'{avg_time:.0f}ms'},
            {'Method': 'RankSTA (Ours)', 'ROC-AUC': f'{roc_auc:.2f}', 'PR-AUC': f'{pr_auc:.2f}',
             'Recall@5%': f'{recall_ranking:.1f}%', 'Time': f'{avg_time:.0f}ms'},
        ]
        
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        
        # Save
        df.to_csv(self.output_dir / "table_04_main_results.csv", index=False)
        
        # Save predictions for later use
        pred_data = {
            'probabilities': all_probs.tolist(),
            'labels': all_labels.tolist(),
            'metrics': {
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc),
                'recall_ranking_5': float(recall_ranking),
                'avg_time_ms': float(avg_time)
            }
        }
        with open(self.output_dir / "test_predictions.json", 'w') as f:
            json.dump(pred_data, f, indent=2)
        
        print(f"\n✅ Saved to: {self.output_dir / 'table_04_main_results.csv'}")
        print(f"✅ Saved predictions to: {self.output_dir / 'test_predictions.json'}")
        
        return df, pred_data

def main():
    """Run all computations."""
    print("\n" + "="*70)
    print(" PAPER DATA GENERATION - MASTER SCRIPT")
    print("="*70)
    
    generator = PaperDataGenerator()
    
    # Table III: Dataset Statistics (no model needed)
    print("\n🔢 Generating Table III...")
    table_03 = generator.compute_table_03_dataset_stats()
    
    # Load model
    generator.load_model()
    
    # Table IV: Main Results (needs model)
    print("\n🔢 Generating Table IV...")
    table_04, predictions = generator.compute_table_04_main_results()
    
    print("\n" + "="*70)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {generator.output_dir}")
    print("\nGenerated:")
    print("  ✅ Table III: Dataset Statistics")
    print("  ✅ Table IV: Main Results")
    print("  ✅ Test predictions (JSON)")
    
    print("\n💡 Next: Run additional scripts for:")  
    print("  - Table VI: Recall@K")
    print("  - Table X: Per-design metrics")
    print("  - Table XI: Adaptive K")
    print("  - All figures")

if __name__ == "__main__":
    main()
