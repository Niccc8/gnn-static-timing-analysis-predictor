#!/bin/bash

# Full Pipeline Script: From Raw Data to Trained Model
# Usage: bash scripts/run_full_pipeline.sh

set -e  # Exit on error

echo "========================================="
echo "  GNN-STA Predictor - Full Pipeline"
echo "========================================="

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw/timing_predict_data"
PROCESSED_DIR="$DATA_DIR/processed/timing_predict"
LABELS_DIR="$DATA_DIR/labels/node_level"
EXPERIMENTS_DIR="experiments"
CONFIG_FILE="$EXPERIMENTS_DIR/configs/default.yaml"

# Step 1: Check prerequisites
echo ""
echo "[Step 1/4] Checking prerequisites..."
python -c "import torch; import torch_geometric; print('✓ PyTorch and PyG installed')" || {
    echo "❌ PyTorch/PyG not installed. Run: pip install -r requirements.txt"
    exit 1
}

which opensta > /dev/null && echo "✓ OpenSTA found" || echo "⚠ OpenSTA not found - will skip label extraction"

# Step 2: Extract labels (if needed)
echo ""
echo "[Step 2/4] Extracting timing labels..."
if [ ! -d "$LABELS_DIR" ] || [ -z "$(ls -A $LABELS_DIR 2>/dev/null)" ]; then
    if which opensta > /dev/null; then
        python scripts/extract_labels.py \
            --data_dir "$RAW_DIR" \
            --output_dir "$LABELS_DIR"
        echo "✓ Label extraction complete"
    else
        echo "❌ OpenSTA not found and no labels exist. Cannot proceed."
        echo "   Install OpenSTA or provide pre-labeled data"
        exit 1
    fi
else
    echo "✓ Labels already exist, skipping extraction"
fi

# Step 3: Build graph dataset
echo ""
echo "[Step 3/4] Building PyG dataset..."
if [ ! -f "$PROCESSED_DIR/train.pt" ]; then
    python scripts/build_dataset.py
    echo "✓ Dataset building complete"
else
    echo "✓ Dataset already exists, skipping build"
fi

# Step 4: Train GNN model
echo ""
echo "[Step 4/4] Training GNN model..."
python src/training/train.py \
    --config "$CONFIG_FILE" \
    --data_dir "$PROCESSED_DIR" \
    --checkpoint_dir "$EXPERIMENTS_DIR/checkpoints" \
    --log_dir "$EXPERIMENTS_DIR/logs" \
    --gpu

echo ""
echo "========================================="
echo "  ✓ Full pipeline complete!"
echo "========================================="
echo ""
echo "Model saved to: $EXPERIMENTS_DIR/checkpoints/best_model.pth"
echo ""
echo "To evaluate:"
echo "  python src/training/evaluate.py \\"
echo "    --checkpoint $EXPERIMENTS_DIR/checkpoints/best_model.pth \\"
echo "    --data_dir $PROCESSED_DIR \\"
echo "    --output_dir $EXPERIMENTS_DIR/results \\"
echo "    --per_design --plot --gpu"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir $EXPERIMENTS_DIR/logs"
echo ""
