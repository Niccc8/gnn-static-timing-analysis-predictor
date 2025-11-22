"""
Build TimingPredict Dataset for GNN Training

Converts TimingPredict raw data (.v, .def files) into PyG Data objects.
Since the pre-processed graphs are in a different format, we build from scratch.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.simple_parser import SimpleVerilogParser
from src.data.graph_builder import TimingDAGBuilder
from src.data.feature_extractor import FeatureExtractor
from src.data.dataset import TimingDataset


def process_timingpredict_design(design_dir, feature_extractor):
    """
    Process one TimingPredict design into a PyG Data object.
    
    Args:
        design_dir: Path to design directory (e.g., data/raw/timing_predict_data/aes/)
        feature_extractor: FeatureExtractor instance
    
    Returns:
        PyG Data object or None if processing fails
    """
    design_dir = Path(design_dir)
    design_name = design_dir.name
    
    # Find Verilog file
    verilog_files = list(design_dir.glob("*.v"))
    if not verilog_files:
        logger.warning(f"No .v file found in {design_dir}")
        return None
    
    verilog_file = verilog_files[0]
    
    try:
        # Step 1: Parse netlist
        logger.info(f"Processing {design_name}...")
        parser = SimpleVerilogParser(str(verilog_file))
        
        if not parser.parse():
            logger.error(f"Failed to parse {verilog_file}")
            return None
        
        # Step 2: Build DAG
        builder = TimingDAGBuilder(
            parser.gates,
            parser.nets,
            parser.primary_inputs,
            parser.primary_outputs
        )
        graph, pin_to_id, levels = builder.build()
        
        # Step 3: Extract features
        node_features = feature_extractor.extract_node_features(graph, levels)
        edge_features = feature_extractor.extract_edge_features(graph)
        
        # Step 4: Build edge index
        edge_list = list(graph.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Step 5: Create dummy labels (we'll use real labels if available)
        # For now, create dummy labels - you can load real labels from .sdf or timing reports
        num_nodes = graph.number_of_nodes()
        labels = torch.zeros(num_nodes, dtype=torch.long)
        
        # TODO: Load actual timing labels from TimingPredict data if available
        # This would involve parsing .sdf or timing report files
        
        # Step 6: Create PyG Data object
        data = TimingDataset.create_pyg_data(
            node_features=torch.FloatTensor(node_features),
            edge_index=edge_index,
            edge_features=torch.FloatTensor(edge_features),
            labels=labels,
            design_name=design_name
        )
        
        logger.info(f"✓ {design_name}: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
        return data
        
    except Exception as e:
        logger.error(f"Error processing {design_name}: {e}")
        return None


def main():
    """Build TimingPredict dataset."""
    
    # Configuration
    data_root = Path("data/raw/timing_predict_data")
    output_dir = Path("data/processed/timing_predict").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Building TimingPredict Dataset")
    logger.info("="*60)
    
    # Find all design directories
    design_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name != "techlib"]
    logger.info(f"Found {len(design_dirs)} design directories")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(normalize=True)
    
    # Process all designs
    data_list = []
    for design_dir in tqdm(design_dirs, desc="Processing designs"):
        data = process_timingpredict_design(design_dir, feature_extractor)
        if data is not None:
            data_list.append(data)
    
    logger.info(f"\n✓ Successfully processed {len(data_list)}/{len(design_dirs)} designs")
    
    if len(data_list) == 0:
        logger.error("No designs were successfully processed!")
        return
    
    # Split into train/val/test (60/20/20)
    num_designs = len(data_list)
    num_train = int(num_designs * 0.6)
    num_val = int(num_designs * 0.2)
    
    train_data = data_list[:num_train]
    val_data = data_list[num_train:num_train+num_val]
    test_data = data_list[num_train+num_val:]
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Train: {len(train_data)} designs")
    logger.info(f"  Val:   {len(val_data)} designs")
    logger.info(f"  Test:  {len(test_data)} designs")
    
    # Save datasets with absolute paths
    logger.info("\nSaving datasets...")
    TimingDataset.save_dataset(train_data, str(output_dir / "train.pt"))
    TimingDataset.save_dataset(val_data, str(output_dir / "val.pt"))
    TimingDataset.save_dataset(test_data, str(output_dir / "test.pt"))
    
    # Save feature scaler
    feature_extractor.save_scaler(str(output_dir / "feature_scaler.pkl"))
    
    logger.info("\n" + "="*60)
    logger.info("✅ Dataset building complete!")
    logger.info("="*60)
    logger.info(f"\nDatasets saved to: {output_dir}")
    logger.info("\nNext step: Train the model with:")
    logger.info("  python src/training/train.py --data_dir data/processed/timing_predict")


if __name__ == "__main__":
    main()
