# GNN Timing Predictor - Inference Guide

## Quick Start

### 1. Extract Your Trained Model
After downloading results from Colab, extract the zip to your project root:
```
d:/GNN-Based Static Timing Analysis Predictor/
```

This creates:
```
experiments/checkpoints/best_model.pth  ← Your trained model
experiments/logs/                       ← Training logs
optimal_threshold.txt                   ← Optimal threshold (0.58)
```

### 2. Run Prediction

**Compare both methods:**
```bash
python scripts/predict.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --data_dir data/processed/timing_predict/ \
    --split test \
    --mode both \
    --top_percent 5
```

**Threshold only:**
```bash
python scripts/predict.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --data_dir data/processed/timing_predict/ \
    --mode threshold \
    --threshold 0.58
```

**Ranking only (recommended):**
```bash
python scripts/predict.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --data_dir data/processed/timing_predict/ \
    --mode ranking \
    --top_percent 5
```

**Adaptive Ranking (Advanced):**
Uses a sensitivity threshold (0.01) to dynamically determine K. Best for circuits with unknown violation rates.
```python
# Python API
results = predictor.predict_adaptive(circuit, sensitivity_threshold=0.01)
print(f"Adaptive K: {results['k']} ({results['top_percent']:.2f}%)")
```

---

## Python API Usage

```python
from scripts.predict import TimingPredictor
from src.data.dataset import TimingDataset

# Load model
predictor = TimingPredictor(
    checkpoint_path='experiments/checkpoints/best_model.pth',
    device='cuda'
)

# Load circuit
dataset = TimingDataset(root='data/processed/timing_predict/', split='test')
circuit = dataset[0]

# Method 1: Threshold (flag nodes > 0.58 risk)
results = predictor.predict_threshold(circuit, threshold=0.58)
print(f"Flagged: {results['num_flagged']} nodes")

# Method 2: Ranking (top 5% riskiest) - RECOMMENDED
results = predictor.predict_top_k(circuit, top_percent=5.0)
print(f"Inspect top {results['k']} nodes")
print(f"Top 10: {results['top_k_indices'][:10]}")

# Method 3: Both
results = predictor.predict_both(circuit, threshold=0.58, top_percent=5.0)
```

---

## Expected Output

```
Loading model from experiments/checkpoints/best_model.pth...
✅ Model loaded successfully on cuda
   Parameters: 275,470

Loading test dataset from data/processed/timing_predict/...
✅ Loaded 4 graphs

============================================================
Example Prediction on Graph 0
============================================================
Graph nodes: 137432

📊 Comparison:
   Total nodes: 137432
   Threshold flags: 3251 (2.37%)
   Top 5% = 6872 nodes
   Overlap: 2819/6872

💡 Use ranking method for better coverage
```

---

## Why Ranking > Threshold?

| Method | Coverage | Precision |
|--------|----------|-----------|
| Threshold (0.58) | Flags ~2-3% nodes | 26% precision |
| **Ranking (top 5%)** | **Inspect 5% nodes** | **Catches 70-90% violations!** |

**Recommendation:** Use ranking to inspect top 5-10% of nodes sorted by risk. This catches most violations without overwhelming your manual review capacity.
