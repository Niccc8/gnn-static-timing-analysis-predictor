"""
Feature Extractor

Extracts 10-dimensional node features and edge features for GNN training.
Features are aligned with timing analysis semantics (delay, slew, capacitance proxies).
"""

import numpy as np
import networkx as nx
import pickle
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from loguru import logger


class FeatureExtractor:
    """
    Extract node and edge features from timing DAG.
    
    Node Features (10-dimensional):
        0. Cell type (categorical encoding 0-15)
        1. Fanout (number of successors)
        2. Fan-in (number of predecessors)
        3. Topological level (normalized depth in DAG)
        4. Estimated delay (picoseconds, normalized)
        5. Estimated slew (picoseconds/ns, normalized)
        6. Pin type (0=input, 1=output)
        7. Positional encoding X (normalized node ID)
        8. Positional encoding Y (normalized level)
        9. Positional encoding Z (fanout/fanin ratio)
    """
    
    # Standard cell type mappings
    CELL_TYPES = {
        "AND": 0, "AND2": 0, "AND3": 0, "AND4": 0,
        "OR": 1, "OR2": 1, "OR3": 1, "OR4": 1,
        "NAND": 2, "NAND2": 2, "NAND3": 2, "NAND4": 2,
        "NOR": 3, "NOR2": 3, "NOR3": 3, "NOR4": 3,
        "XOR": 4, "XOR2": 4, "XNOR": 4, "XNOR2": 4,
        "INV": 5, "NOT": 5,
        "BUF": 6, "BUFFER": 6, "CLKBUF": 6,
        "MUX": 7, "MUX2": 7, "MUX4": 7,
        "DFF": 8, "DFFR": 8, "DFFS": 8, "SDFF": 8,
        "LATCH": 9,
        "FA": 12, "HA": 12,  # Adders
        "PRIMARY_INPUT": 10,
        "PRIMARY_OUTPUT": 11,
    }
    
    # Estimated delays (ps) - simplified Liberty-like values
    DELAY_ESTIMATES = {
        0: 50, 1: 50, 2: 40, 3: 40,   # AND, OR, NAND, NOR
        4: 60, 5: 30, 6: 40, 7: 80,   # XOR, INV, BUF, MUX
        8: 100, 9: 80, 10: 0, 11: 0,  # DFF, LATCH, PI, PO
        12: 120,                      # Adders
    }
    
    def __init__(self, normalize: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            normalize: Whether to normalize features using StandardScaler
        """
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        logger.debug("Initialized FeatureExtractor")
    
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
            
            # 1. Cell Type
            gate_type = node_data.get("gate_type", "PRIMARY_INPUT")
            cell_type_id = self._get_cell_type_id(gate_type)
            features[node_id, 0] = cell_type_id
            
            # 2. Fanout
            fanout = graph.out_degree(node_id)
            features[node_id, 1] = fanout
            
            # 3. Fan-in
            fanin = graph.in_degree(node_id)
            features[node_id, 2] = fanin
            
            # 4. Topological Level
            level = levels.get(node_id, 0)
            features[node_id, 3] = level / max_level if max_level > 0 else 0
            
            # 5. Estimated Delay
            delay = self.DELAY_ESTIMATES.get(cell_type_id, 50)
            features[node_id, 4] = delay
            
            # 6. Estimated Slew (Heuristic)
            # Slew increases with delay and fanout
            slew = delay * 0.3 * (1 + 0.1 * fanout)
            features[node_id, 5] = slew
            
            # 7. Pin Type (0=input, 1=output)
            pin_type = node_data.get("pin_type", "input")
            features[node_id, 6] = 1.0 if pin_type == "output" else 0.0
            
            # 8-10. Positional Encodings
            features[node_id, 7] = node_id / num_nodes if num_nodes > 0 else 0
            features[node_id, 8] = level / max_level if max_level > 0 else 0
            features[node_id, 9] = fanout / (fanin + 1)  # Ratio
        
        # Normalize
        if self.normalize and self.scaler:
            # We fit_transform on training data, but for inference we should use transform.
            # However, since this is per-graph feature extraction, we typically fit on the 
            # training set globally. Here we are doing per-graph normalization which is 
            # suboptimal but acceptable for this architecture. 
            # Ideally, we should fit on the whole dataset.
            # For now, we keep the existing behavior but note it.
            features = self.scaler.fit_transform(features)
        
        return features
    
    def extract_edge_features(self, graph: nx.DiGraph) -> np.ndarray:
        """
        Extract edge features.
        
        Returns:
            Edge feature matrix of shape (num_edges, 3)
            Features: [is_net_edge, is_cell_edge, estimated_delay]
        """
        num_edges = graph.number_of_edges()
        edge_features = np.zeros((num_edges, 3), dtype=np.float32)
        
        for i, (u, v, data) in enumerate(graph.edges(data=True)):
            edge_type = data.get("edge_type", "net")
            
            if edge_type == "net":
                edge_features[i, 0] = 1.0  # Net edge
                edge_features[i, 1] = 0.0
                edge_features[i, 2] = 10.0  # Interconnect delay proxy (ps)
            else:
                edge_features[i, 0] = 0.0
                edge_features[i, 1] = 1.0  # Cell edge
                
                # Get gate delay
                gate_type = data.get("gate_type", "")
                cell_type_id = self._get_cell_type_id(gate_type)
                edge_features[i, 2] = self.DELAY_ESTIMATES.get(cell_type_id, 50)
        
        return edge_features
    
    def _get_cell_type_id(self, gate_type: str) -> int:
        """Map gate type string to integer ID."""
        gate_type = gate_type.upper().strip()
        
        # Exact match
        if gate_type in self.CELL_TYPES:
            return self.CELL_TYPES[gate_type]
        
        # Prefix match (e.g., "AND2_X1" -> "AND")
        for key in self.CELL_TYPES:
            if gate_type.startswith(key):
                return self.CELL_TYPES[key]
        
        # Default to Buffer if unknown
        return 6
    
    def save_scaler(self, filepath: str):
        """Save feature scaler."""
        if self.scaler:
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load feature scaler."""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {filepath}")
