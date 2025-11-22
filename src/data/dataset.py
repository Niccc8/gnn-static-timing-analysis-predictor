"""
PyTorch Geometric Dataset

Custom dataset class for loading and batching timing DAGs.
"""

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from pathlib import Path
from typing import List, Optional, Callable
import pandas as pd
import pickle
from loguru import logger


class TimingDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for timing prediction.
    
    Loads pre-processed graphs from disk with labels.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory containing processed/ and raw/ subdirectories
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform to apply to each graph
            pre_transform: Optional pre-processing transform
            pre_filter: Optional filter to exclude graphs
        """
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        # weights_only=False is needed because PyG Data objects are custom classes
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        logger.info(f"Loaded {split} dataset: {len(self)} graphs")
    
    @property
    def processed_dir(self) -> str:
        """Override processed directory to be same as root."""
        return self.root

    @property
    def raw_dir(self) -> str:
        """Override raw directory to be same as root to avoid permission issues."""
        return self.root
    
    @property
    def raw_file_names(self) -> List[str]:
        """List of raw file names (not used if graphs are pre-processed)."""
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        """List of processed file names."""
        return [f"{self.split}.pt"]
    
    def download(self):
        """Download raw data (not implemented - use pre-processed data)."""
        pass
    
    def process(self):
        """Process raw data into PyG Data objects (already done offline)."""
        # This is typically called if processed files don't exist
        # In our case, graphs are processed offline by build_dataset.py
        logger.warning("Dataset processing should be done offline using build_dataset.py")
    
    @staticmethod
    def create_pyg_data(
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        labels: torch.Tensor,
        design_name: str
    ) -> Data:
        """
        Create a PyG Data object.
        
        Args:
            node_features: Node feature matrix (num_nodes, 10)
            edge_index: Edge connectivity (2, num_edges)
            edge_features: Edge feature matrix (num_edges, 3)
            labels: Node labels (num_nodes,) - binary classification
            design_name: Circuit design name
        
        Returns:
            PyG Data object
        """
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=labels,
            design_name=design_name,
            num_nodes=node_features.size(0)
        )
        return data
    
    @staticmethod
    def save_dataset(data_list: List[Data], save_path: str):
        """
        Save a list of Data objects to disk.
        
        Args:
            data_list: List of PyG Data objects
            save_path: Path to save the dataset (.pt file)
        """
        # Workaround for Windows file opening issues with torch.save
        # We open the file manually and pass the handle
        with open(save_path, 'wb') as f:
            torch.save(InMemoryDataset.collate(data_list), f)
        logger.info(f"Saved {len(data_list)} graphs to {save_path}")
    
    def get_statistics(self) -> dict:
        """Return dataset statistics."""
        stats = {
            "num_graphs": len(self),
            "avg_num_nodes": sum([d.num_nodes for d in self]) / len(self),
            "avg_num_edges": sum([d.edge_index.size(1) for d in self]) / len(self),
            "num_violating": sum([(d.y == 1).sum().item() for d in self]),
            "total_endpoints": sum([d.y.numel() for d in self]),
        }
        stats["violation_rate"] = stats["num_violating"] / stats["total_endpoints"]
        return stats


def load_labels_from_csv(csv_file: str) -> pd.DataFrame:
    """
    Load timing labels from CSV.
    
    Expected format:
        endpoint_name,slack_ps,is_violating
        out[0],-45.2,1
        out[1],12.8,0
    
    Args:
        csv_file: Path to CSV file
    
    Returns:
        DataFrame with labels
    """
    df = pd.DataFrame(csv_file, index_col="endpoint_name")
    logger.info(f"Loaded {len(df)} labels from {csv_file}")
    return df


if __name__ == "__main__":
    # Example: Load dataset
    dataset = TimingDataset(root="data/", split="train")
    
    print(f"\nDataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Example: Access a graph
    if len(dataset) > 0:
        data = dataset[0]
        print(f"\nFirst graph:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.edge_index.size(1)}")
        print(f"  Features: {data.x.shape}")
        print(f"  Labels: {data.y.shape}")
