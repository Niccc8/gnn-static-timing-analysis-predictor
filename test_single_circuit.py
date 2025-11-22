"""
Test single circuit through entire pipeline.
Quick validation that all modules work together.
"""

from pathlib import Path
import pandas as pd
import torch
import sys

sys.path.append(str(Path(__file__).parent))

from src.data.netlist_parser import VerilogParser
from src.data.graph_builder import TimingDAGBuilder
from src.data.feature_extractor import FeatureExtractor
from src.data.dataset import TimingDataset


def test_pipeline(verilog_file):
    """Test complete pipeline on one circuit."""
    
    print("="*60)
    print("  Testing GNN-STA Pipeline on Single Circuit")
    print("="*60)
    
    # Step 1: Parse netlist
    print("\n[1/4] Parsing Verilog netlist...")
    
    # Try VerilogParser first, fall back to SimpleVerilogParser
    parser = VerilogParser(verilog_file)
    if not parser.parse():
        print("⚠ pyverilog parser failed, trying simple regex parser...")
        from src.data.simple_parser import SimpleVerilogParser
        parser = SimpleVerilogParser(verilog_file)
        if not parser.parse():
            print("❌ Both parsers failed!")
            return False
        print("✓ Simple parser succeeded!")
    
    stats = parser.get_statistics()
    print(f"✓ Parsed successfully:")
    print(f"    Gates: {stats['num_gates']}")
    print(f"    Nets: {stats['num_nets']}")
    print(f"    Inputs: {stats['num_primary_inputs']}")
    print(f"    Outputs: {stats['num_primary_outputs']}")
    
    # Step 2: Build DAG
    print("\n[2/4] Building heterogeneous DAG...")
    builder = TimingDAGBuilder(
        parser.gates, 
        parser.nets,
        parser.primary_inputs, 
        parser.primary_outputs
    )
    graph, pin_to_id, levels = builder.build()
    
    print(f"✓ DAG built successfully:")
    print(f"    Nodes (pins): {graph.number_of_nodes()}")
    print(f"    Edges: {graph.number_of_edges()}")
    print(f"    Max topological level: {max(levels.values())}")
    
    # Step 3: Extract features
    print("\n[3/4] Extracting node and edge features...")
    extractor = FeatureExtractor(normalize=True)
    node_features = extractor.extract_node_features(graph, levels)
    edge_features = extractor.extract_edge_features(graph)
    
    print(f"✓ Features extracted:")
    print(f"    Node features: {node_features.shape} (nodes × 10 features)")
    print(f"    Edge features: {edge_features.shape} (edges × 3 features)")
    
    # Step 4: Create PyG Data object
    print("\n[4/4] Creating PyTorch Geometric Data object...")
    edge_list = list(graph.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create dummy labels (all safe) since we don't have STA labels yet
    labels = torch.zeros(graph.number_of_nodes(), dtype=torch.long)
    
    data = TimingDataset.create_pyg_data(
        node_features=torch.FloatTensor(node_features),
        edge_index=edge_index,
        edge_features=torch.FloatTensor(edge_features),
        labels=labels,
        design_name=Path(verilog_file).stem
    )
    
    print(f"✓ PyG Data object created:")
    print(f"    {data}")
    
    # Test GNN forward pass
    print("\n[Bonus] Testing GNN model forward pass...")
    from src.models.timing_gnn import HeterogeneousTimingGNN
    
    model = HeterogeneousTimingGNN(
        in_channels=10,
        hidden_channels=128,
        num_classes=2,
        num_layers=3,
        heads=4,
        dropout=0.2,
        edge_dim=3
    )
    
    output = model(data)
    predictions = output.argmax(dim=1)
    
    print(f"✓ Model inference successful:")
    print(f"    Output shape: {output.shape}")
    print(f"    Sample predictions: {predictions[:10].tolist()}")
    
    print("\n" + "="*60)
    print("  🎉 SUCCESS! All pipeline components working!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pipeline on single circuit")
    parser.add_argument("verilog_file", nargs="?", 
                        help="Path to Verilog file (optional, uses example if not provided)")
    
    args = parser.parse_args()
    
    # If no file provided, try to find one
    if args.verilog_file:
        test_file = args.verilog_file
    else:
        # Try to find a file in data/raw
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            verilog_files = list(raw_dir.rglob("*.v"))
            if verilog_files:
                test_file = str(verilog_files[0])
                print(f"Using: {test_file}")
            else:
                print("❌ No .v files found in data/raw/")
                print("Usage: python test_single_circuit.py <path_to_verilog_file>")
                sys.exit(1)
        else:
            print("❌ data/raw/ directory not found")
            print("Usage: python test_single_circuit.py <path_to_verilog_file>")
            sys.exit(1)
    
    # Run test
    success = test_pipeline(test_file)
    
    if not success:
        sys.exit(1)
