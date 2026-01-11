#!/usr/bin/env python3
"""Quick diagnostic to check .pt file structure"""
import torch
from pathlib import Path

data_file = Path("data/processed/timing_predict/train.pt")
print(f"Loading {data_file}...")
data = torch.load(data_file, weights_only=False)

print(f"\nType: {type(data)}")
print(f"Length: {len(data) if isinstance(data, (list, tuple)) else 'N/A'}")

if isinstance(data, list):
    print(f"\nFirst item type: {type(data[0])}")
    if hasattr(data[0], '__dict__'):
        print(f"Attributes: {dir(data[0])}")
    print(f"\nFirst item: {data[0]}")
