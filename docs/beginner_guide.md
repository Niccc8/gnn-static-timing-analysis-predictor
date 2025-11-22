# Complete Beginner's Guide to Running GNN-STA Predictor

**For users with NO prior machine learning experience**

---

## Overview

You'll go through 4 main phases:
1. **Environment Setup** (30-60 min) - Install software
2. **Dataset Download** (1-2 hours) - Get circuit benchmarks  
3. **Test Pipeline** (1-2 hours) - Validate on 1 small circuit
4. **Full Training** (4-8 hours) - Train the GNN model

**Total time: ~1-2 days** (mostly automated, just need to monitor)

---

## Phase 1: Environment Setup (30-60 minutes)

### Step 1.1: Install Python Environment

**Open PowerShell as Administrator** and run:

```powershell
# Check Python version (need 3.10+)
python --version

# If not installed or < 3.10, download from:
# https://www.python.org/downloads/ (Python 3.10.11 recommended)
```

### Step 1.2: Create Project Environment

```powershell
# Navigate to project
cd "d:\GNN-Based Static Timing Analysis Predictor"

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate

# You should see (venv) in your terminal now
```

### Step 1.3: Install Python Packages

```powershell
# Install PyTorch (CPU version first, easier to test)
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install PyG dependencies
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install other requirements
pip install -r requirements.txt
```

**Expected output:** Lots of download messages, should end with "Successfully installed..."

**Test installation:**
```powershell
python -c "import torch; import torch_geometric; print('✓ PyTorch:', torch.__version__); print('✓ PyG:', torch_geometric.__version__)"
```

**Expected output:**
```
✓ PyTorch: 2.0.0+cpu
✓ PyG: 2.3.0
```

### Step 1.4: Install OpenSTA (Optional for now, needed for labeling)

**Note:** OpenSTA requires WSL (Windows Subsystem for Linux) on Windows. For now, **skip this step** - we'll use pre-labeled data from TimingPredict dataset.

If you want to install it later:
1. Install WSL: `wsl --install` (requires Windows 10/11)
2. Follow Linux instructions in `docs/setup_guide.md`

---

## Phase 2: Dataset Download (1-2 hours)

### Step 2.1: Download TimingPredict Dataset (Recommended - Pre-labeled!)

This dataset is **already labeled** so you don't need OpenSTA right away.

**Option A: Direct Download (Easier)**
1. Go to: https://github.com/TimingPredict/Dataset
2. Look for download links (Google Drive or PKU Drive)
3. Download the zip file (~500 MB)
4. Extract to: `d:\GNN-Based Static Timing Analysis Predictor\data\raw\timing_predict\`

**Option B: Git Clone**
```powershell
cd "d:\GNN-Based Static Timing Analysis Predictor\data\raw"
git clone https://github.com/TimingPredict/Dataset.git timing_predict
```

**What you should have now:**
```
data/raw/timing_predict/
  ├── design1/
  │   ├── design1.v
  │   ├── design1.def
  │   └── timing_labels.csv
  ├── design2/
  └── ...
```

### Step 2.2: Download ISCAS Benchmarks (Optional - for more data)

1. Visit: http://sportlab.usc.edu/~msabrishami/benchmarks.html
2. Download ISCAS'85 and ISCAS'89 zip files
3. Extract to:
   - `data/raw/iscas85/`
   - `data/raw/iscas89/`

**These are small test circuits** (gates ranging from 6 to 3,000)

---

## Phase 3: Test Pipeline (1-2 hours)

### Step 3.1: Test Individual Modules

Let's test each component to make sure everything works:

**Test 1: Netlist Parser**
```powershell
# Find a small Verilog file in your downloaded data
python src/data/netlist_parser.py "data\raw\iscas85\c17.v"
```

**Expected output:**
```
Circuit Statistics:
  num_gates: 1234
  num_nets: 1567
  num_primary_inputs: 32
  num_primary_outputs: 8
```

**If you get an error:**
- Check file path is correct
- Make sure pyverilog is installed: `pip install pyverilog`

**Test 2: Feature Extractor**
```powershell
python src/data/feature_extractor.py
```

**Expected output:**
```
Node features shape: (2, 10)
Edge features shape: (1, 3)
```

**Test 3: GNN Model**
```powershell
python src/models/timing_gnn.py
```

**Expected output:**
```
Model Test:
  Input: torch.Size([100, 10])
  Output: torch.Size([100, 2])
  Predicted classes: tensor([0, 1, 0, ...])
```

### Step 3.2: Build a Small Test Dataset

Let's process just **ONE** circuit to test the full pipeline:

**Create a test script** `test_single_circuit.py`:
```python
from pathlib import Path
import pandas as pd
import torch
from src.data.netlist_parser import VerilogParser
from src.data.graph_builder import TimingDAGBuilder
from src.data.feature_extractor import FeatureExtractor
from src.data.dataset import TimingDataset

# Pick one Verilog file
verilog_file = "data/raw/timing_predict/design1/design1.v"  # Adjust path

print("Step 1: Parsing netlist...")
parser = VerilogParser(verilog_file)
if not parser.parse():
    print("❌ Parsing failed!")
    exit(1)

print("✓ Parsed:", parser.get_statistics())

print("\nStep 2: Building DAG...")
builder = TimingDAGBuilder(
    parser.gates, parser.nets, 
    parser.primary_inputs, parser.primary_outputs
)
graph, pin_to_id, levels = builder.build()
print(f"✓ DAG built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

print("\nStep 3: Extracting features...")
extractor = FeatureExtractor(normalize=True)
node_features = extractor.extract_node_features(graph, levels)
edge_features = extractor.extract_edge_features(graph)
print(f"✓ Features extracted: {node_features.shape}, {edge_features.shape}")

print("\nStep 4: Creating PyG Data object...")
edge_list = list(graph.edges())
edge_index = torch.tensor(edge_list, dtype=torch.long).t()
labels = torch.zeros(graph.number_of_nodes(), dtype=torch.long)  # Dummy labels for now

data = TimingDataset.create_pyg_data(
    torch.FloatTensor(node_features),
    edge_index,
    torch.FloatTensor(edge_features),
    labels,
    "test_design"
)
print(f"✓ PyG Data created: {data}")

print("\n🎉 SUCCESS! Pipeline works end-to-end!")
```

**Run it:**
```powershell
python test_single_circuit.py
```

**Expected output:**
```
Step 1: Parsing netlist...
✓ Parsed: {'num_gates': 1234, ...}

Step 2: Building DAG...
✓ DAG built: 2468 nodes, 3456 edges

Step 3: Extracting features...
✓ Features extracted: (2468, 10), (3456, 3)

Step 4: Creating PyG Data object...
✓ PyG Data created: Data(x=[2468, 10], edge_index=[2, 3456], ...)

🎉 SUCCESS! Pipeline works end-to-end!
```

**If this works, you're golden!** 🎉

---

## Phase 4: Full Training (4-8 hours)

### Step 4.1: Prepare Full Dataset

**If you have TimingPredict dataset with labels:**

```powershell
# Build the full PyG dataset from all circuits
python scripts/build_dataset.py \
  --netlists "data/raw/timing_predict/*/*.v" \
  --labels "data/raw/timing_predict/all_labels.csv" \
  --output_dir "data/processed" \
  --train_split 0.6 \
  --val_split 0.2
```

**This will take 10-30 minutes** depending on dataset size.

**Expected output:**
```
Processing netlists: 100%|████████| 21/21
Successfully processed 21/21 netlists
Split: Train=13, Val=4, Test=4
✓ Dataset building complete! Saved to data/processed
```

### Step 4.2: Train the GNN Model

```powershell
# Start training (this will take 2-6 hours)
python src/training/train.py \
  --config experiments/configs/default.yaml \
  --data_dir data \
  --checkpoint_dir experiments/checkpoints \
  --log_dir experiments/logs
```

**What you'll see:**
```
Using device: cpu
Loading datasets...
Train: 13 graphs, Val: 4 graphs
Creating model...
Model parameters: 387,586
Starting training...
Epoch 1/100: Train Loss=0.6234, Val Loss=0.5123, Val AUC=0.7234, Val F1=0.6543
Epoch 2/100: Train Loss=0.5234, Val Loss=0.4876, Val AUC=0.7654, Val F1=0.6987
...
✓ New best model saved (AUC=0.9123)
...
Early stopping triggered at epoch 67
Training complete! Best Val AUC: 0.9456
```

**Monitor training in another terminal:**
```powershell
# In a new PowerShell window
cd "d:\GNN-Based Static Timing Analysis Predictor"
.\venv\Scripts\Activate
tensorboard --logdir experiments/logs
```

Then open browser to: http://localhost:6006

### Step 4.3: Evaluate the Model

```powershell
python src/training/evaluate.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data_dir data \
  --output_dir experiments/results \
  --per_design \
  --plot
```

**Expected output:**
```
Loading test dataset...
Test: 4 graphs
Loading model...
Evaluating on test set...

=== Test Set Metrics ===
  roc_auc: 0.9512
  pr_auc: 0.8976
  accuracy: 0.9234
  f1_score: 0.8543
  precision: 0.8876
  recall: 0.8234

Cross-design statistics:
  Mean AUC: 0.9456
  Std AUC: 0.0123
  Min AUC: 0.9234
  Max AUC: 0.9678

✓ Evaluation complete!
```

**Check results:**
```powershell
# Open results folder
explorer experiments\results

# You'll find:
# - test_metrics.csv
# - test_predictions.csv
# - per_design_metrics.csv
# - roc_curve.png
# - pr_curve.png
```

---

## Troubleshooting Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```powershell
# Make sure venv is activated
.\venv\Scripts\Activate

# Reinstall torch
pip install torch==2.0.0
```

### Issue 2: "pyverilog parsing error"

**Solution:**
```powershell
pip uninstall pyverilog
pip install pyverilog==1.3.0
```

### Issue 3: "CUDA out of memory" (if using GPU)

**Solution:** Use CPU instead:
```powershell
# Don't use --gpu flag
python src/training/train.py --config experiments/configs/default.yaml
```

### Issue 4: Dataset building fails

**Solution:** Test on one file first:
```powershell
python test_single_circuit.py
```

---

## What to Do If Something Doesn't Work

1. **Check the error message** - copy it
2. **Look in the specific module** - error will tell you which file
3. **Run that module standalone** - e.g., `python src/data/netlist_parser.py file.v`
4. **Ask me for help** - share the error message

---

## Summary Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch and PyG installed (test with `python -c "import torch; import torch_geometric"`)
- [ ] Project dependencies installed (`pip install -r requirements.txt`)
- [ ] TimingPredict dataset downloaded to `data/raw/timing_predict/`
- [ ] Single circuit test successful (`python test_single_circuit.py`)
- [ ] Full dataset built (`python scripts/build_dataset.py`)
- [ ] Model training started (`python src/training/train.py`)
- [ ] Results evaluated (`python src/training/evaluate.py`)

---

## Next Steps After Training

Once you have a trained model:

1. **Write the paper** - Use `paper/main.tex` template
2. **Create demo notebook** - Show inference on new circuit
3. **Optimize model** - Try different hyperparameters
4. **Test on more data** - Download more benchmarks

**Good luck! Start with Phase 1 and work through systematically. Each phase should work before moving to the next.** 🚀
