"""
Dataset building script.

Processes raw Verilog netlists into PyG Data objects:
1. Parse netlists
2. Build heterogeneous DAGs
3. Extract features
4. Attach labels from STA
5. Save as PyG datasets
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from glob import glob
from tqdm import tqdm
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.netlist_parser import VerilogParser
from data.graph_builder import TimingDAGBuilder
from data.feature_extractor import FeatureExtractor
from data.dataset import TimingDataset


def process_netlist(verilog_file, labels_df, feature_extractor):
    """
    Process a single netlist into PyG Data object.
    
    Args:
        verilog_file: Path to Verilog file
        labels_df: DataFrame with timing labels
        feature_extractor: FeatureExtractor instance
    
    Returns:
        PyG Data object or None if processing fails
    """
    try:
        # Step 1: Parse netlist
        parser = VerilogParser(verilog_file)
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
        
        # Step 5: Attach labels
        num_nodes = graph.number_of_nodes()
        labels = torch.full((num_nodes,), -1, dtype=torch.long)  # -1 = unlabeled
        
        # Match labels to nodes
        design_name = Path(verilog_file).stem
        design_labels = labels_df[labels_df.get('design', '') == design_name]
        
        for endpoint_name, row in design_labels.iterrows():
            if endpoint_name in pin_to_id:
                node_id = pin_to_id[endpoint_name]
                labels[node_id] = int(row['is_violating'])
        
        # Create PyG Data object
        data = TimingDataset.create_pyg_data(
            node_features=torch.FloatTensor(node_features),
            edge_index=edge_index,
            edge_features=torch.FloatTensor(edge_features),
            labels=labels,
            design_name=design_name
        )
        
        logger.info(
            f"✓ Processed {design_name}: "
            f"{data.num_nodes} nodes, {data.edge_index.size(1)} edges"
        )
        return data
        
    except Exception as e:
        logger.error(f"Error processing {verilog_file}: {e}")
        return None


def main(args):
    """Main dataset building function."""
    # Load labels
    logger.info(f"Loading labels from {args.labels}")
    labels_df = pd.read_csv(args.labels, index_col='endpoint_name')
    
    # Get all Verilog files
    verilog_files = []
    for pattern in args.netlists:
        verilog_files.extend(glob(pattern))
    
    logger.info(f"Found {len(verilog_files)} Verilog files")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(normalize=True)
    
    # Process all netlists
    data_list = []
    for verilog_file in tqdm(verilog_files, desc="Processing netlists"):
        data = process_netlist(verilog_file, labels_df, feature_extractor)
        if data is not None:
            data_list.append(data)
    
    logger.info(f"Successfully processed {len(data_list)}/{len(verilog_files)} netlists")
    
    if len(data_list) == 0:
        logger.error("No netlists were successfully processed!")
        return
    
    # Split into train/val/test
    num_graphs = len(data_list)
    num_train = int(num_graphs * args.train_split)
    num_val = int(num_graphs * args.val_split)
    
    train_data = data_list[:num_train]
    val_data = data_list[num_train:num_train+num_val]
    test_data = data_list[num_train+num_val:]
    
    logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    TimingDataset.save_dataset(train_data, str(output_dir / "train.pt"))
    TimingDataset.save_dataset(val_data, str(output_dir / "val.pt"))
    TimingDataset.save_dataset(test_data, str(output_dir / "test.pt"))
    
    # Save feature scaler
    feature_extractor.save_scaler(str(output_dir / "feature_scaler.pkl"))
    
    logger.info(f"✓ Dataset building complete! Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyG dataset from netlists")
    
    parser.add_argument("--netlists", nargs="+", required=True,
                        help="Glob patterns for Verilog files")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to labels CSV file")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed datasets")
    parser.add_argument("--train_split", type=float, default=0.6,
                        help="Fraction of data for training")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data for validation")
    
    args = parser.parse_args()
    
    # Validate splits
    if args.train_split + args.val_split >= 1.0:
        parser.error("train_split + val_split must be < 1.0")
    
    main(args)
