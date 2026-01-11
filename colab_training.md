# GNN Timing Predictor - Google Colab Training

Quick training notebook for Google Colab with GPU acceleration.

## Setup

Upload the dataset package (created with `scripts/prepare_colab_pack.py`):
```python
from google.colab import files
uploaded = files.upload()  # Upload gnn_timing_pack.zip
```

## Install Dependencies

```python
!pip install -q torch torch-geometric loguru pyyaml scikit-learn matplotlib tensorboard
```

## Extract and Setup

```python
!unzip -o -q gnn_timing_pack.zip
!ls -la
```

## Train Model

```python
!python -m src.training.train \
    --config experiments/configs/default.yaml \
    --data_dir data/processed/timing_predict \
    --checkpoint_dir experiments/checkpoints \
    --log_dir experiments/logs \
    --gpu
```

## Monitor Training (Optional)

```python
%load_ext tensorboard
%tensorboard --logdir experiments/logs
```

## Download Results

```python
from google.colab import files
from datetime import datetime
import os

# Create results package
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_zip = f"gnn_timing_results_{timestamp}.zip"

!zip -r {output_zip} experiments/checkpoints/best_model.pth experiments/logs/

# Download
files.download(output_zip)
print(f"✅ Downloaded {output_zip}")
```

## Expected Performance

- **Training time:** ~20-30 minutes on Colab GPU (vs 2-3 hours on CPU)
- **Validation AUC:** ~0.96+
- **Test AUC:** ~0.95+

## Next Steps

After downloading the trained model:
1. Place `best_model.pth` in `experiments/checkpoints/`
2. Run evaluation: `python src/training/evaluate.py ...`
3. Run inference: `python scripts/predict.py ...`
