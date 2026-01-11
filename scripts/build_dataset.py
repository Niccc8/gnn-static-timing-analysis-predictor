"""
Build TimingPredict Dataset

This script processes raw Verilog netlists and OpenSTA labels into PyTorch Geometric datasets.
It performs the following steps:
1. Parses Verilog netlists to build heterogeneous DAGs (gates + nets).
2. Extracts node and edge features (logic type, fanout, delay estimates).
3. Maps OpenSTA timing labels (slack/violation) to graph nodes.
4. Splits data into Train/Val/Test sets using a stratified strategy to ensure
   balanced violation distribution across splits.

Usage:
    python scripts/build_dataset.py
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.simple_parser import SimpleVerilogParser
from src.data.graph_builder import TimingDAGBuilder
from src.data.feature_extractor import FeatureExtractor
from src.data.dataset import TimingDataset


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Stratified Split Configuration
# Ensures high-violation designs are present in training to prevent class collapse.
# Total: 21 designs
DESIGN_SPLITS = {
    'train': [
        # High-violation designs (Critical for learning)
        'aes256', 'jpeg_encoder', 'aes128',
        # Low/Zero-violation designs (Regularization)
        'synth_ram', 'blabla', 'cic_decimator', 'spm', 'zipdiv',
        'usb', 'y_huff', 'usb_cdc_core', 'wbqspiflash'
    ],
    'val': [
        'aes192', 'picorv32a', 'xtea', 'genericfir', 'salsa20'
    ],
    'test': [
        'des', 'BM64', 'aes_cipher', 'usbf_device'
    ]
}

PATHS = {
    'raw_data': Path("data/raw/timing_predict_data"),
    'labels': Path("data/labels/node_level"),
    'processed': Path("data/processed/timing_predict")
}


# ==============================================================================
# CORE LOGIC
# ==============================================================================

def process_design(
    design_dir: Path, 
    label_dir: Path, 
    feature_extractor: FeatureExtractor
) -> Optional[Data]:
    """
    Process a single design directory into a PyG Data object.
    
    Args:
        design_dir: Path to design directory containing Verilog file
        label_dir: Path to directory containing label CSVs
        feature_extractor: Initialized FeatureExtractor instance
        
    Returns:
        PyG Data object if successful, None otherwise
    """
    design_name = design_dir.name
    
    # 1. Locate Files
    verilog_files = list(design_dir.glob("*.v"))
    if not verilog_files:
        logger.warning(f"Skipping {design_name}: No Verilog file found")
        return None
    
    verilog_file = verilog_files[0]
    label_file = label_dir / f"{design_name}_node_labels.csv"
    
    if not label_file.exists():
        logger.warning(f"Skipping {design_name}: No label file found at {label_file}")
        return None

    try:
        # 2. Parse Netlist
        logger.info(f"[{design_name}] Parsing netlist...")
        parser = SimpleVerilogParser(str(verilog_file))
        if not parser.parse():
            logger.error(f"[{design_name}] Parsing failed")
            return None

        # 3. Build Graph (DAG)
        logger.info(f"[{design_name}] Building graph...")
        builder = TimingDAGBuilder(
            parser.gates,
            parser.nets,
            parser.primary_inputs,
            parser.primary_outputs
        )
        graph, pin_to_id, levels = builder.build()
        
        if graph.number_of_nodes() == 0:
            logger.error(f"[{design_name}] Graph is empty")
            return None

        # 4. Extract Features
        logger.info(f"[{design_name}] Extracting features...")
        node_features = feature_extractor.extract_node_features(graph, levels)
        edge_features = feature_extractor.extract_edge_features(graph)
        
        # Build edge index
        edge_list = list(graph.edges())
        if not edge_list:
            logger.warning(f"[{design_name}] No edges found")
            return None
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # 5. Map Labels
        logger.info(f"[{design_name}] Mapping labels...")
        df = pd.read_csv(label_file)
        
        # Create label tensor (default -1 for unlabeled)
        num_nodes = graph.number_of_nodes()
        labels = torch.full((num_nodes,), -1, dtype=torch.long)
        
        # Create fast lookup map
        label_map = dict(zip(df['endpoint'], df['label']))
        
        mapped_count = 0
        violation_count = 0
        
        for pin_name, node_id in pin_to_id.items():
            if pin_name in label_map:
                label = int(label_map[pin_name])
                labels[node_id] = label
                mapped_count += 1
                if label == 1:
                    violation_count += 1
        
        # 6. Create Data Object
        data = TimingDataset.create_pyg_data(
            node_features=torch.FloatTensor(node_features),
            edge_index=edge_index,
            edge_features=torch.FloatTensor(edge_features),
            labels=labels,
            design_name=design_name
        )
        
        logger.success(
            f"✓ {design_name}: {data.num_nodes} nodes, "
            f"{mapped_count} labeled ({violation_count} violations)"
        )
        return data

    except Exception as e:
        logger.exception(f"Error processing {design_name}: {e}")
        return None


def main():
    """Execute dataset building pipeline."""
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    
    logger.info("="*60)
    logger.info("BUILDING TIMING PREDICTOR DATASET")
    logger.info("="*60)
    
    # Ensure directories exist
    PATHS['processed'].mkdir(parents=True, exist_ok=True)
    
    if not PATHS['raw_data'].exists():
        logger.error(f"Raw data directory not found: {PATHS['raw_data']}")
        return

    # Initialize Feature Extractor
    feature_extractor = FeatureExtractor(normalize=True)
    
    # Process All Designs
    design_dirs = sorted([d for d in PATHS['raw_data'].iterdir() 
                         if d.is_dir() and d.name != "techlib"])
    
    logger.info(f"Found {len(design_dirs)} design directories")
    
    processed_data: Dict[str, Data] = {}
    
    for design_dir in design_dirs:
        data = process_design(design_dir, PATHS['labels'], feature_extractor)
        if data:
            processed_data[design_dir.name] = data
            
    if not processed_data:
        logger.error("No designs were successfully processed. Exiting.")
        return

    # Split Data
    logger.info("\nSplitting data...")
    splits: Dict[str, List[Data]] = {'train': [], 'val': [], 'test': []}
    
    for split_name, design_names in DESIGN_SPLITS.items():
        for name in design_names:
            if name in processed_data:
                splits[split_name].append(processed_data[name])
            else:
                logger.warning(f"Design '{name}' assigned to {split_name} but was not processed.")

    # Validate and Save
    for split_name, data_list in splits.items():
        if not data_list:
            logger.warning(f"Split '{split_name}' is empty!")
            continue
            
        # Stats
        total_nodes = sum(d.num_nodes for d in data_list)
        total_viols = sum((d.y == 1).sum().item() for d in data_list)
        viol_rate = (total_viols / total_nodes * 100) if total_nodes > 0 else 0
        
        logger.info(f"  {split_name.upper():<5}: {len(data_list):>2} designs | "
                   f"{total_nodes:>7,} nodes | {total_viols:>5,} violations ({viol_rate:.2f}%)")
        
        # Save
        save_path = PATHS['processed'] / f"{split_name}.pt"
        TimingDataset.save_dataset(data_list, str(save_path))
        logger.info(f"    Saved to {save_path}")

    # Save Feature Scaler
    scaler_path = PATHS['processed'] / "feature_scaler.pkl"
    feature_extractor.save_scaler(str(scaler_path))
    logger.info(f"    Saved feature scaler to {scaler_path}")
    
    logger.success("\n✅ Dataset build complete!")


if __name__ == "__main__":
    main()
