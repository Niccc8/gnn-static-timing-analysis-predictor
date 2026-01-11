#!/usr/bin/env python3
"""
Script 01: Compute Dataset Statistics (Table III)
Analyzes train.pt, val.pt, test.pt to extract actual statistics.
"""

import torch
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def compute_dataset_stats(data_dir: str = "data/processed/timing_predict"):
    """
    Compute actual dataset statistics from processed .pt files.
    
    Returns:
        DataFrame with split statistics
    """
    data_path = Path(data_dir)
    
    results = []
    
    for split in ['train', 'val', 'test']:
        pt_file = data_path / f"{split}.pt"
        
        if not pt_file.exists():
            print(f"Warning: {pt_file} not found, skipping...")
            continue
        
        print(f"Loading {split}.pt...")
        data_list = torch.load(pt_file, weights_only=False)
        
        # Aggregate statistics
        total_nodes = 0
        total_violations = 0
        design_count = len(data_list)
        min_size = float('inf')
        max_size = 0
        
        for data in data_list:
            num_nodes = data.num_nodes
            # Count violations (label == 1)
            # Labels might be -1 (unlabeled), 0 (clean), 1 (violation)
            violations = (data.y == 1).sum().item()
            
            total_nodes += num_nodes
            total_violations += violations
            min_size = min(min_size, num_nodes)
            max_size = max(max_size, num_nodes)
        
        # Calculate violation rate
        viol_rate = (total_violations / total_nodes * 100) if total_nodes > 0 else 0
        
        # Format size range
        if min_size < 1000:
            size_range = f"{int(min_size)}–{int(max_size)}"
        else:
            size_range = f"{int(min_size/1000)}K–{int(max_size/1000)}K"
        
        results.append({
            'Split': split.capitalize(),
            '# Designs': design_count,
            'Nodes': f"{total_nodes:,}",
            'Viol.': f"{total_violations:,}",
            'Rate': f"{viol_rate:.2f}%",
            'Size': size_range
        })
    
    # Add Total row
    total_designs = sum([r['# Designs'] for r in results])
    total_nodes_sum = sum([int(r['Nodes'].replace(',', '')) for r in results])
    total_viol_sum = sum([int(r['Viol.'].replace(',', '')) for r in results])
    total_rate = (total_viol_sum / total_nodes_sum * 100) if total_nodes_sum > 0 else 0
    
    # Get overall size range
    all_sizes = []
    for split in ['train', 'val', 'test']:
        pt_file = data_path / f"{split}.pt"
        if pt_file.exists():
            data_list = torch.load(pt_file, weights_only=False)
            all_sizes.extend([data.num_nodes for data in data_list])
    
    overall_min = min(all_sizes)
    overall_max = max(all_sizes)
    overall_size = f"{int(overall_min/1000)}K–{int(overall_max/1000)}K"
    
    results.append({
        'Split': '**Total**',
        '# Designs': f"**{total_designs}**",
        'Nodes': f"**{total_nodes_sum:,}**",
        'Viol.': f"**{total_viol_sum:,}**",
        'Rate': f"**{total_rate:.2f}%**",
        'Size': f"**{overall_size}**"
    })
    
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    print("Computing Dataset Statistics (Table III)...")
    print("=" * 60)
    
    df = compute_dataset_stats()
    
    print("\nRESULTS:")
    print(df.to_string(index=False))
    
    # Save to CSV
    output_file = "experiments/results/table_03_dataset_stats.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    # Print LaTeX table format
    print("\nLaTeX TABLE FORMAT:")
    print("=" * 60)
    for _, row in df.iterrows():
        print(f"{row['Split']} & {row['# Designs']} & {row['Nodes']} & {row['Viol.']} & {row['Rate']} & {row['Size']} \\\\")
