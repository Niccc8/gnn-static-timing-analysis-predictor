#!/bin/bash

# Full Pipeline Script: From Raw Netlists to Trained Model
# Usage: bash scripts/run_full_pipeline.sh

set -e  # Exit on error

echo "========================================="
echo "  GNN-STA Predictor - Full Pipeline"
echo "========================================="

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"
LIBRARY_FILE="skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib"
EXPERIMENTS_DIR="experiments"
CONFIG_FILE="$EXPERIMENTS_DIR/configs/default.yaml"

# Step 1: Check prerequisites
echo ""
echo "[Step 1/6] Checking prerequisites..."
python -c "import torch; import torch_geometric; print('✓ PyTorch and PyG installed')"
which opensta > /dev/null && echo "✓ OpenSTA found" || echo "⚠ OpenSTA not found - skipping STA labeling"

# Step 2: Download datasets (if not already present)
echo ""
echo "[Step 2/6] Checking datasets..."
if [ ! -d "$RAW_DIR/iscas85" ]; then
    echo "⚠ ISCAS benchmarks not found. Please download manually:"
    echo "   http://sportlab.usc.edu/~msabrishami/benchmarks.html"
    echo "   Extract to: $RAW_DIR/iscas85/"
fi

if [ ! -d "$RAW_DIR/timing_predict" ]; then
    echo "⚠ TimingPredict dataset not found. Please download manually:"
    echo "   https://github.com/TimingPredict/Dataset"
    echo "   Extract to: $RAW_DIR/timing_predict/"
fi

# Step 3: Generate STA labels
echo ""
echo "[Step 3/6] Generating timing labels with OpenSTA..."
if which opensta > /dev/null && [ -f "$LIBRARY_FILE" ]; then
    python src/data/sta_labeler.py \
        "$RAW_DIR/iscas85/*.v" \
        "$LIBRARY_FILE" \
        --output "$PROCESSED_DIR/labels"
    echo "✓ Labeling complete"
else
    echo "⚠ Skipping STA labeling (OpenSTA or library file not found)"
    echo "   Using pre-labeled data if available"
fi

# Step 4: Build graph dataset
echo ""
echo "[Step 4/6] Building PyG dataset..."
python scripts/build_dataset.py \
    --netlists "$RAW_DIR/iscas85/*.v" "$RAW_DIR/iscas89/*.v" \
    --labels "$PROCESSED_DIR/labels.csv" \
    --output_dir "$DATA_DIR/processed" \
    --train_split 0.6 \
    --val_split 0.2

echo "✓ Dataset building complete"

# Step 5: Train GNN model
echo ""
echo "[Step 5/6] Training GNN model..."
python src/training/train.py \
    --config "$CONFIG_FILE" \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$EXPERIMENTS_DIR/checkpoints" \
    --log_dir "$EXPERIMENTS_DIR/logs" \
    --gpu

echo "✓ Training complete"

# Step 6: Evaluate on test set
echo ""
echo "[Step 6/6] Evaluating on test set..."
python src/training/evaluate.py \
    --checkpoint "$EXPERIMENTS_DIR/checkpoints/best_model.pth" \
    --data_dir "$DATA_DIR" \
    --output_dir "$EXPERIMENTS_DIR/results" \
    --per_design \
    --plot \
    --gpu

echo ""
echo "========================================="
echo "  ✓ Full pipeline complete!"
echo "========================================="
echo ""
echo "Results saved to: $EXPERIMENTS_DIR/results/"
echo "  - test_metrics.csv"
echo "  - test_predictions.csv"
echo "  - per_design_metrics.csv"
echo "  - roc_curve.png"
echo "  - pr_curve.png"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir $EXPERIMENTS_DIR/logs"
