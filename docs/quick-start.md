# Quick Start Guide: GNN Timing Predictor

**Current Status:** Production v1.0  
**Timeline:** Project complete and ready for use

---

## Overview

This guide provides a quick workflow to get started with the GNN timing predictor. For comprehensive context and paper writing, see [PAPER_CONTEXT.md](PAPER_CONTEXT.md).

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, CPU works fine)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/GNN-STA-Predictor.git
cd "GNN-Based Static Timing Analysis Predictor"

# Create environment
conda env create -f environment.yml
conda activate gnn-sta

# Or use pip
pip install -r requirements.txt
```

---

## Complete Workflow

### 1. Dataset Preparation

The project uses the**TimingPredict Dataset** (21 designs):

```
data/raw/timing_predict_data/
└── design_name/
    ├── design_name.synthesis_preroute.v  (Verilog netlist)
    ├── design_name.sdc                    (Timing constraints)
    └── design_name.spef                   (Parasitics, optional)
```

See [dataset_download_guide.md](dataset_download_guide.md) for acquisition.

### 2. Extract Timing Labels

```bash
python scripts/extract_labels.py \
  --data_dir data/raw/timing_predict_data \
  --output_dir data/labels/node_level
```

**What it does:**
- Runs OpenSTA on each design
- Extracts slack values for all endpoints
- Creates CSV files: `design_name_node_labels.csv`

**Output:** `data/labels/node_level/design_name_node_labels.csv`

### 3. Build Graph Dataset

```bash
python scripts/build_dataset.py
```

**What it does:**
- Parses Verilog netlists
- Builds heterogeneous DAGs (net + cell edges)
- Extracts 10-dimensional node features
- Creates PyTorch Geometric datasets

**Output:**
- `data/processed/timing_predict/train.pt`
- `data/processed/timing_predict/val.pt`
- `data/processed/timing_predict/test.pt`

### 4. Train Model

```bash
python src/training/train.py \
  --config experiments/configs/default.yaml \
  --data_dir data/processed/timing_predict \
  --checkpoint_dir experiments/checkpoints \
  --log_dir experiments/logs \
  --gpu  # Optional
```

**Monitor with TensorBoard:**
```bash
tensorboard --logdir experiments/logs
```

**Training takes:** ~2-3 hours on CPU, ~20 minutes on GPU

### 5. Evaluate

```bash
python src/training/evaluate.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data_dir data/processed/timing_predict \
  --output_dir experiments/results \
  --per_design \ # Per-design breakdown
  --plot \        # Generate ROC/PR curves
  --gpu
```

**Output:**
- `experiments/results/test_metrics.csv`
- `experiments/results/test_predictions.csv`
- `experiments/results/roc_curve.png`
- `experiments/results/pr_curve.png`

### 6. Inference

#### Adaptive Ranking (Recommended)

```bash
python scripts/predict.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data_dir data/processed/timing_predict \
  --split test \
  --mode adaptive \
  --sensitivity 0.01
```

#### Top-K Ranking

```bash
python scripts/predict.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data_dir data/processed/timing_predict \
  --split test \
  --mode ranking \
  --top_percent 5.0
```

#### Threshold-Based (Not Recommended)

```bash
python scripts/predict.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data_dir data/processed/timing_predict \
  --split test \
  --mode threshold \
  --threshold 0.58
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

# Load data
dataset = TimingDataset(root='data/processed/timing_predict/', split='test')
circuit = dataset[0]

# Adaptive prediction (recommended)
results = predictor.predict_adaptive(circuit, sensitivity_threshold=0.01)

print(f"Inspecting {results['k']} nodes ({results['top_percent']:.2f}%)")
print(f"Top 10 risky nodes: {results['top_k_indices'][:10]}")
```

See [examples/inference_example.py](../examples/inference_example.py) for comprehensive examples.

---

## Validation & Analysis

### Validate Ranking Performance

```bash
python scripts/validate_ranking.py
```

**Output:** Recall/precision at different top-K percentages

### Example Inference

```bash
python examples/inference_example.py
```

**Output:** Comparison of threshold vs ranking methods

---

## Key Concepts

### Prediction Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Adaptive** | Dynamic K based on sensitivity | **Production (recommended)** |
| **Ranking** | Fixed top-K% | Batch analysis |
| **Threshold** | Probability cutoff | Legacy/comparison only |

### Why Ranking > Threshold?

**Problem:** Fixed thresholds fail across circuits
- Clean circuit: False alarms at P=0.16
- Violating circuit: True violations at P=0.08

**Solution:** Rank by risk, ignore absolute probabilities

See [PAPER_NOTES_RANKING.md](PAPER_NOTES_RANKING.md) for detailed analysis.

---

## File Structure

```
├── scripts/
│   ├── extract_labels.py      # Label extraction from OpenSTA
│   ├── build_dataset.py        # Graph dataset builder
│   ├── predict.py              # Inference script
│   └── validate_ranking.py     # Ranking validation
├── src/
│   ├── data/                   # Data processing
│   ├── models/                 # GNN architecture
│   └── training/               # Training & evaluation
├── examples/
│   └── inference_example.py    # Usage examples
├── experiments/
│   ├── configs/                # YAML configs
│   ├── checkpoints/            # Trained models
│   └── logs/                   # TensorBoard logs
└── docs/
    ├── PAPER_CONTEXT.md        # Comprehensive paper guide
    ├── INFERENCE.md            # API reference
    └── INDUSTRIAL_USE_CASE.md  # Practical applications
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/GNN-Based Static Timing Analysis Predictor"
```

**2. CUDA Out of Memory**
```bash
# Use CPU or smaller batch size
python src/training/train.py --batch_size 1 # No --gpu flag
```

**3. Dataset Not Found**
```bash
# Check directory structure
ls data/processed/timing_predict/
# Should contain: train.pt, val.pt, test.pt, feature_scaler.pkl
```

**4. OpenSTA Not Found (for label extraction)**
```bash
# Install OpenSTA
sudo apt-get install opensta
# Or build from source (see opensta_setup_guide.md)
```

---

## Next Steps

1. **For Inference:** Use pre-trained model with `scripts/predict.py`
2. **For Training:** Follow complete workflow above
3. **For Paper Writing:** See `docs/PAPER_CONTEXT.md`
4. **For Industrial Use:** See `docs/INDUSTRIAL_USE_CASE.md`

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| ROC-AUC | >0.95 |
| PR-AUC | >0.90 |
| Inference Time | <100ms |
| Top-5% Recall | 78% |
| Speedup vs STA | 10-20× |

---

## Additional Resources

- **API Reference:** [INFERENCE.md](INFERENCE.md)
- **Paper Context:** [PAPER_CONTEXT.md](PAPER_CONTEXT.md)
- **Ranking Analysis:** [PAPER_NOTES_RANKING.md](PAPER_NOTES_RANKING.md)
- **Industrial Applications:** [INDUSTRIAL_USE_CASE.md](INDUSTRIAL_USE_CASE.md)

---

**Version:** 1.0.0  
**Last Updated:** November 25, 2025
