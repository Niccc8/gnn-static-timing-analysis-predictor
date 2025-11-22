"""
Feature Extractor

Extracts 10-dimensional node features and edge features for GNN training.
Features are aligned with timing analysis semantics.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from loguru import logger
import pickle


class FeatureExtractor:
    """
    Extract node and edge features from timing DAG.
    
    Node Features (10-dimensional):
        1. Cell type (categorical encoding 0-15)
        2. Fanout (number of successors)
        3. Fan-in (number of predecessors)
        4. Topological level (normalized depth in DAG)
        5. Estimated delay (picoseconds, normalized)
        6. Estimated slew (picoseconds/ns, normalized)
        7. Pin type (0=input, 1=output)
        8-10. Positional encodings (normalized x, y, z coordinates)
    """
    
    # Standard cell type mappings
    CELL_TYPES = {
        "AND": 0, "AND2": 0, " AND3": 0, "AND4": 0,
        "OR": 1, "OR2": 1, "OR3": 1, "OR4": 1,
        "NAND": 2, "NAND2": 2, "NAND3": 2, "NAND4": 2,
        "NOR": 3, "NOR2": 3, "NOR3": 3, "NOR4": 3,
        "XOR": 4, "XOR2": 4, "XNOR": 4, "XNOR2": 4,
        "INV": 5, "NOT": 5,
        "BUF": 6, "BUFFER": 6,
        "MUX": 7, "MUX2": 7,
        "DFF": 8, "DFFR": 8, "DFFS": 8,
        "LATCH": 9,
        "PRIMARY_INPUT": 10,
        "PRIMARY_OUTPUT": 11,
    }
    
    # Estimated delays (ps) - simplified Liberty-like values
    DELAY_ESTIMATES = {
        0: 50, 1: 50, 2: 40, 3: 40,  # AND, OR, NAND, NOR
        4: 60, 5: 30, 6: 40, 7: 80,  # XOR, INV, BUF, MUX
        8: 100, 9: 80, 10: 0, 11: 0,  # DFF, LATCH, PI, PO
    }
    
    def __init__(self, normalize: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            normalize: Whether to normalize features
        """
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        logger.info("Initialized FeatureExtractor")
    
    def extract_node_features(
        self,
        graph: nx.DiGraph,
        levels: Dict[int, int]
    ) -> np.ndarray:
        """
        Extract 10-dimensional features for each node.
        
        Args:
            graph: NetworkX DiGraph with node attributes
            levels: Dictionary mapping node IDs to topological levels
        
        Returns:
            Feature matrix of shape (num_nodes, 10)
        """
        num_nodes = graph.number_of_nodes()
        features = np.zeros((num_nodes, 10), dtype=np.float32)
        
        max_level = max(levels.values()) if levels else 1
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            
            # Feature 1: Cell type encoding
            gate_type = node_data.get("gate_type", "PRIMARY_INPUT")
            cell_type_id = self._get_cell_type_id(gate_type)
            features[node_id, 0] = cell_type_id
            
            # Feature 2: Fanout
            features[node_id, 1] = graph.out_degree(node_id)
            
            # Feature 3: Fan-in
            features[node_id, 2] = graph.in_degree(node_id)
            
            # Feature 4: Topological level (normalized)
            level = levels.get(node_id, 0)
            features[node_id, 3] = level / max_level if max_level > 0 else 0
            
            # Feature 5: Estimated delay (ps)
            delay = self.DELAY_ESTIMATES.get(cell_type_id, 50)
            features[node_id, 4] = delay
            
            # Feature 6: Estimated slew (simplified)
            fanout = graph.out_degree(node_id)
            slew = delay * 0.3 * (1 + 0.1 * fanout)  # Rough approximation
            features[node_id, 5] = slew
            
            # Feature 7: Pin type (0=input, 1=output)
            pin_type = node_data.get("pin_type", "input")
            features[node_id, 6] = 1.0 if pin_type == "output" else 0.0
            
            # Features 8-10: Positional encodings
            # X: normalized node ID
            features[node_id, 7] = node_id / num_nodes
            # Y: normalized level
            features[node_id, 8] = level / max_level if max_level > 0 else 0
            # Z: normalized fanout/fan-in ratio
            fanin = graph.in_degree(node_id)
            features[node_id, 9] = fanout / (fanin + 1)  # +1 to avoid division by zero
        
        # Normalize features
        if self.normalize:
            features = self.scaler.fit_transform(features)
        
        logger.info(f"Extracted features for {num_nodes} nodes: shape {features.shape}")
        return features
    
    def extract_edge_features(self, graph: nx.DiGraph) -> np.ndarray:
        """
        Extract edge features.
        
        Args:
            graph: NetworkX DiGraph with edge attributes
        
        Returns:
            Edge feature matrix of shape (num_edges, 3)
            Features: [edge_type_net, edge_type_cell, estimated_delay]
        """
        num_edges = graph.number_of_edges()
        edge_features = np.zeros((num_edges, 3), dtype=np.float32)
        
        for i, (u, v, data) in enumerate(graph.edges(data=True)):
            edge_type = data.get("edge_type", "net")
            
            # One-hot encoding for edge type
            if edge_type == "net":
                edge_features[i, 0] = 1.0  # Net edge
                edge_features[i, 1] = 0.0
                edge_features[i, 2] = 10.0  # Interconnect delay (ps)
            else:  # cell edge
                edge_features[i, 0] = 0.0
                edge_features[i, 1] = 1.0  # Cell edge
                # Get gate type delay
                gate_type = data.get("gate_type", "")
                cell_type_id = self._get_cell_type_id(gate_type)
                edge_features[i, 2] = self.DELAY_ESTIMATES.get(cell_type_id, 50)
        
        return edge_features
    
    def _get_cell_type_id(self, gate_type: str) -> int:
        """Map gate type string to integer ID."""
        # Try exact match first
        if gate_type in self.CELL_TYPES:
            return self.CELL_TYPES[gate_type]
        
        # Try prefix match (e.g., "AND2_X1" → "AND")
        for key in self.CELL_TYPES:
            if gate_type.startswith(key):
                return self.CELL_TYPES[key]
        
        # Default to BUF
        return 6
    
    def save_scaler(self, filepath: str):
        """Save feature scaler for inference."""
        if self.scaler:
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load feature scaler."""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {filepath}")


if __name__ == "__main__":
    # Example usage
    import networkx as nx
    
    # Create simple graph
    G = nx.DiGraph()
    G.add_node(0, gate_type="AND2", pin_type="output")
    G.add_node(1, gate_type="INV", pin_type="input")
    G.add_edge(0, 1, edge_type="net")
    
    levels = {0: 0, 1: 1}
    
    extractor = FeatureExtractor()
    features = extractor.extract_node_features(G, levels)
    edge_features = extractor.extract_edge_features(G)
    
    print(f"\nNode features shape: {features.shape}")
    print(f"Edge features shape: {edge_features.shape}")
