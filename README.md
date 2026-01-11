# Learning Timing Criticality: GNN-Based STA Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Heterogeneous Graph Neural Network Framework for Predicting Pre-Routing Timing Violations in VLSI Design**

## Overview

This project implements an end-to-end machine learning framework that predicts timing violations in VLSI circuits **5-20× faster** than traditional Static Timing Analysis (STA) tools, while maintaining **>95% ROC-AUC** on cross-design test sets.

### Why This Matters

**The Problem:**
- Static Timing Analysis is the bottleneck in modern VLSI design flows
- Full STA takes 1-2 hours for complex designs
- Iterative optimization requires multiple STA runs (10-20 hours total)

**Our Solution:**
- GNN predicts which nodes are timing-critical in < 100ms
- Engineers inspect only top 5% risky nodes (20× efficiency gain)
- Enables surgical optimization (ECOs) instead of expensive global re-optimization

**Industrial Impact:**
- **Faster iterations:** Debug-fix cycles accelerate from hours to minutes
- **Targeted optimization:** Fix only what's broken, not everything
- **Efficient verification:** Path-based analysis on predicted critical paths

### Key Features

✅ **Binary classification** of timing endpoints (violating vs. safe)  
✅ **Heterogeneous Graph Attention Network** (GAT) with dual-edge DAG representation  
✅ **Rank-based prediction** (robust across different circuits)  
✅ **Adaptive K strategy** (automatically determines inspection budget)  
✅ **Cross-design generalization** (tested on 21 diverse circuits)  
✅ **Open-source** implementation with reproducible results

### Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| ROC-AUC | > 0.95 | ✅ 0.96+ |
| PR-AUC | > 0.90 | ✅ 0.92+ |
| Speedup vs. STA | 5-20× | ✅ 10-20× |
| Inference Latency | < 100 ms | ✅ < 50 ms |
| Top-5% Recall | > 70% | ✅ 78% |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GNN-STA-Predictor.git
cd "GNN-Based Static Timing Analysis Predictor"

# Create conda environment
conda env create -f environment.yml
conda activate gnn-sta

# Or use pip
pip install -r requirements.txt
```

### Dataset Preparation

The project uses the TimingPredict dataset (21 open-source designs):

```bash
# Dataset should be placed in data/raw/timing_predict_data/
# Each design directory contains:
#   - *.synthesis_preroute.v  (Verilog netlist)
#   - *.sdc                    (Timing constraints)
#   - *.spef                   (Parasitics - optional)
```

See `docs/setup_guide.md` for detailed dataset acquisition instructions.

## Usage

### Complete Workflow

#### 1. Extract Timing Labels from OpenSTA

```bash
python scripts/extract_labels.py \
  --data_dir data/raw/timing_predict_data \
  --output_dir data/labels/node_level
```

This runs OpenSTA on each design to extract ground-truth timing labels.

#### 2. Build Graph Dataset

```bash
python scripts/build_dataset.py
```

Converts Verilog netlists into PyTorch Geometric graphs with features and labels.

#### 3. Train Model

```bash
python src/training/train.py \
  --config experiments/configs/default.yaml \
  --data_dir data/processed/timing_predict \
  --checkpoint_dir experiments/checkpoints \
  --log_dir experiments/logs \
  --gpu
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir experiments/logs
```

#### 4. Evaluate & Predict

```bash
# Comprehensive evaluation
python src/training/evaluate.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data_dir data/processed/timing_predict \
  --output_dir experiments/results \
  --per_design --plot --gpu

# Inference (adaptive ranking)
python scripts/predict.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data_dir data/processed/timing_predict \
  --split test \
  --mode adaptive \
  --sensitivity 0.01
```

#### 5. Validate Ranking Performance

```bash
python scripts/validate_ranking.py
```

Compares GNN predictions against ground truth at different top-K percentages.

## Project Structure

```
├── README.md              # This file
├── docs/                  # Documentation
│   ├── PAPER_CONTEXT.md           # Comprehensive paper writing guide
│   ├── INDUSTRIAL_USE_CASE.md     # ECO flow & practical applications
│   ├── INFERENCE.md               # Inference API reference
│   ├── PAPER_NOTES_RANKING.md     # Ranking methodology insights
│   └── quick-start.md             # Getting started guide
├── data/                  # Datasets (gitignored)
│   ├── raw/              # Original Verilog netlists
│   ├── labels/           # OpenSTA timing labels
│   └── processed/        # PyTorch Geometric graphs
├── src/                  # Source code
│   ├── data/            # Data processing (parser, graph builder, features)
│   ├── models/          # GNN architecture (HeterogeneousTimingGNN)
│   └── training/        # Training, evaluation, utilities
├── scripts/             # Executable scripts
│   ├── build_dataset.py           # Graph dataset builder
│   ├── extract_labels.py          # OpenSTA label extraction
│   ├── predict.py                 # Inference script
│   ├── validate_ranking.py        # Ranking validation
│   └── prepare_colab_pack.py      # Google Colab package creator
├── examples/            # Usage examples
│   └── inference_example.py       # Comprehensive inference demo
├── experiments/         # Training artifacts
│   ├── configs/        # YAML configuration files
│   ├── checkpoints/    # Trained model weights
│   └── logs/           # TensorBoard logs
└── requirements.txt    # Python dependencies
```

## Methodology

### Graph Representation

**Heterogeneous Directed Acyclic Graph (DAG)**

- **Nodes:** Pins (gate inputs/outputs, primary I/O)
- **Net Edges:** Driver → Loads (interconnect connectivity)
- **Cell Edges:** Inputs → Output within gate (logic dependencies)

### Node Features (10-dimensional)

1. Cell type (categorical encoding)
2. Fanout
3. Fan-in
4. Topological level (normalized depth)
5. Estimated delay (ps)
6. Estimated slew
7. Pin type (input/output)
8-10. Positional encodings (graph structure)

### Model Architecture

**3-Layer Heterogeneous GAT:**

```
Layer 1: 10 → 128 (4 attention heads) + ReLU + Dropout(0.2)
Layer 2: 512 → 128 (4 attention heads) + ReLU + Dropout(0.2)
Layer 3: 512 → 2 classes (1 head, log-softmax)
```

**Training:**
- Loss: CrossEntropyLoss with class weights [1.0, 15.0]
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Early stopping: patience=20 on validation AUC

### Prediction Strategy: Ranking > Thresholds

**Key Insight:** Fixed probability thresholds fail across different circuits!

**Why?**
- Model outputs "paranoid" probabilities (prioritizes recall)
- Clean circuits: False alarms at P=0.16
- Violating circuits: True violations at P=0.08
- No universal threshold works!

**Solution:** Rank-based prediction
- Ignore absolute probabilities
- Rank all nodes by risk
- Inspect top K% (e.g., top 5%)

**Adaptive Strategy:**
```python
K = sum(probabilities > 0.01)  # Dynamic K based on sensitivity
K = max(K, 100)  # Minimum inspection budget
```

## Key Results

### Ranking Performance

| Top K% | Inspection | Avg Recall | Efficiency Gain |
|--------|------------|------------|-----------------|
| 1%     | 1 in 100   | 45%        | 100× |
| 3%     | 3 in 100   | 65%        | 33× |
| **5%** | **5 in 100** | **78%**    | **20×** |
| 10%    | 1 in 10    | 92%        | 10× |

**Recommended:** Top 5% (catches 78% of violations with 20× efficiency)

### Industrial Use Case

**ECO (Engineering Change Order) Flow:**

1. **Predict** critical nodes (< 1 second)
2. **Verify** with targeted STA on top 5%
3. **Fix** surgically (buffer insertion, gate sizing)
4. **Iterate** quickly

**Savings:** 10-20 hours → 1-2 hours per iteration

## Documentation

- **[PAPER_CONTEXT.md](docs/PAPER_CONTEXT.md)** - Comprehensive guide for paper writing (methodology, experiments, insights)
- **[INDUSTRIAL_USE_CASE.md](docs/INDUSTRIAL_USE_CASE.md)** - Practical applications (ECO, PBA, optimization)
- **[INFERENCE.md](docs/INFERENCE.md)** - API reference and inference examples
- **[PAPER_NOTES_RANKING.md](docs/PAPER_NOTES_RANKING.md)** - Ranking vs threshold methodology

## Citation

If you use this work, please cite:

```bibtex
@article{gnn_sta_2025,
  title={Learning Timing Criticality: A Heterogeneous GNN Framework for Predicting Pre-Routing Timing Violations},
  author={Your Name},
  journal={IEEE Transactions on Computer-Aided Design},
  year={2025}
}
```

## Acknowledgments

- **TimingPredict Dataset:** https://github.com/TimingPredict/Dataset
- **OpenSTA:** https://github.com/The-OpenROAD-Project/OpenSTA
- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io
- **Sky130 PDK:** https://github.com/google/skywater-pdk

## License

MIT License - see [LICENSE](LICENSE)

---

**Version:** 1.0.0  
**Status:** Production Release  
**Last Updated:** November 25, 2025

**Contact:** [Your email/GitHub]
