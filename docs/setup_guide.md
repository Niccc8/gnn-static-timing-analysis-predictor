# GNN-Based STA Predictor: Getting Started

## Prerequisites

### System Requirements
- **OS:** Linux, macOS, or Windows (with WSL for OpenSTA)
- **Python:** 3.10 or higher
- **Memory:** 8GB RAM minimum (16GB recommended)
- **Disk:** 5GB free space for datasets + models

### Software Dependencies
- Git
- Conda or pip
- CMake (for OpenSTA compilation)
- C++ compiler (GCC 7+ or Clang)

---

## Installation

### Step 1: Clone Repository

```bash
cd "d:/"
# Repository already created locally
cd "GNN-Based Static Timing Analysis Predictor"
```

### Step 2: Create Python Environment

**Option A: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate gnn-sta
```

**Option B: pip + venv**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Install OpenSTA

OpenSTA is required for generating timing labels. Follow these steps:

**Linux/WSL:**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y cmake clang tcl-dev swig bison flex

# Clone OpenSTA
cd ~
git clone --recursive https://github.com/The-OpenROAD-Project/OpenSTA.git
cd OpenSTA

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Add to PATH
echo 'export PATH="$HOME/OpenSTA/build:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
opensta -version
```

**macOS:**
```bash
# Install dependencies via Homebrew
brew install cmake tcl-tk swig bison flex

# Clone and build (same as Linux)
git clone --recursive https://github.com/The-OpenROAD-Project/OpenSTA.git
cd OpenSTA
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)

# Add to PATH
echo 'export PATH="$HOME/OpenSTA/build:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Windows:**
Use WSL (Windows Subsystem for Linux) and follow Linux instructions above.

---

## Quick Test

### Test Python Environment

```bash
python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}, PyG: {torch_geometric.__version__}')"
```

Expected output:
```
PyTorch: 2.0.0+cpu, PyG: 2.3.0
```

### Test Core Modules

```bash
# Test netlist parser
python src/data/netlist_parser.py --help

# Test feature extractor
python src/data/feature_extractor.py

# Test GNN model
python src/models/timing_gnn.py
```

---

## Dataset Setup

### Download Benchmarks

```bash
# Coming soon: automated download script
# For now, manual downloads:

# 1. ISCAS Benchmarks
mkdir -p data/raw/iscas85 data/raw/iscas89
# Download from: http://sportlab.usc.edu/~msabrishami/benchmarks.html

# 2. IWLS Benchmarks
git clone https://github.com/ispras/hdl-benchmarks.git data/raw/iwls05

# 3. TimingPredict Dataset (Highly Recommended)
# Download from: https://github.com/TimingPredict/Dataset
# Extract to: data/raw/timing_predict/
```

### Standard Cell Library

You'll need a Liberty (.lib) file for timing information:

```bash
# Option 1: Skywater130 PDK (open-source, recommended)
git clone https://github.com/google/skywater-pdk.git
# Library file: skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib

# Option 2: Use library from TimingPredict dataset (included)
```

---

## Usage Workflow

### 1. Label Netlists with STA

```bash
python src/data/sta_labeler.py \
  data/raw/iscas85/c17.v \
  skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib \
  c17
```

### 2. Build Dataset

```python
# Build graphs from netlists (script coming soon)
python scripts/build_dataset.py \
  --netlists "data/raw/iscas85/*.v" \
  --labels data/processed/labels.csv \
  --output data/processed/graphs/
```

### 3. Train Model

```python
# Train GNN (script coming soon)
python src/training/train.py \
  --config experiments/configs/default.yaml
```

### 4. Evaluate

```python
# Evaluate on test set (script coming soon)
python src/training/evaluate.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --test_data data/splits/test/
```

---

## Troubleshooting

### OpenSTA Build Failures

**Error:** `fatal error: tcl.h: No such file or directory`
**Solution:** Install TCL development headers:
```bash
sudo apt-get install tcl-dev
```

**Error:** `SWIG not found`
**Solution:**
```bash
sudo apt-get install swig
```

### PyTorch Geometric Installation Issues

If `torch-geometric` fails to install via conda:
```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### pyverilog Parsing Errors

**Error:** `ImportError: No module named vparser`
**Solution:** Reinstall pyverilog:
```bash
pip uninstall pyverilog
pip install pyverilog==1.3.0
```

---

## Next Steps

1. **Download datasets** (see Dataset Setup above)
2. **Test STA labeling** on c17.v (smallest ISCAS benchmark)
3. **Build graph dataset** for 5 sample circuits
4. **Review training script** (coming in next update)
5. **Run end-to-end pipeline** once datasets are ready

---

## Support

- **Documentation:** See `docs/` folder
- **Issues:** Open GitHub issue 
- **Email:** [your contact]

---

**Updated:** November 19, 2025
