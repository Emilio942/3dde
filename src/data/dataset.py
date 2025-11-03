"""Dataset and data loading utilities for diffusion model training."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Callable
import h5py
import numpy as np
from pathlib import Path


class GraphDiffusionDataset(Dataset):
    """Dataset for graph-structured data for diffusion model training."""
    
    def __init__(
        self,
        data: torch.Tensor,
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        """
        Args:
            data: Node states (num_samples, N, d)
            transform: Optional transform to apply
            normalize: Whether to normalize data
        """
        self.data = data
        self.transform = transform
        
        if normalize:
            self.mean = data.mean(dim=0, keepdim=True)
            self.std = data.std(dim=0, keepdim=True) + 1e-8
            self.data = (data - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return {'states': sample}
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale."""
        if self.mean is not None:
            return data * self.std + self.mean
        return data


class HDF5Dataset(Dataset):
    """Dataset for loading data from HDF5 files (memory-efficient for large datasets)."""
    
    def __init__(
        self,
        hdf5_path: str,
        dataset_name: str = "states",
        normalize: bool = True,
        preload: bool = False
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file
            dataset_name: Name of dataset in HDF5 file
            normalize: Whether to normalize data
            preload: If True, load all data into memory
        """
        self.hdf5_path = hdf5_path
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.preload = preload
        
        # Get dataset info
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f[dataset_name])
            self.shape = f[dataset_name].shape[1:]
            
            # Compute normalization stats
            if normalize:
                # Sample-based estimation for large datasets
                if self.length > 10000:
                    indices = np.random.choice(self.length, 10000, replace=False)
                    samples = f[dataset_name][indices]
                else:
                    samples = f[dataset_name][:]
                
                self.mean = torch.tensor(samples.mean(axis=0, keepdims=True), dtype=torch.float32)
                self.std = torch.tensor(samples.std(axis=0, keepdims=True) + 1e-8, dtype=torch.float32)
            else:
                self.mean = None
                self.std = None
            
            # Preload if requested
            if preload:
                self.data = torch.tensor(f[dataset_name][:], dtype=torch.float32)
                if normalize:
                    self.data = (self.data - self.mean) / self.std
        
        self.file = None
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> dict:
        if self.preload:
            sample = self.data[idx]
        else:
            # Open file on first access (thread-safe)
            if self.file is None:
                self.file = h5py.File(self.hdf5_path, 'r')
            
            sample = torch.tensor(self.file[self.dataset_name][idx], dtype=torch.float32)
            
            if self.normalize:
                sample = (sample - self.mean) / self.std
        
        return {'states': sample}
    
    def __del__(self):
        if self.file is not None:
            self.file.close()


class SyntheticDataset(Dataset):
    """Generate synthetic data for testing and development."""
    
    def __init__(
        self,
        num_samples: int,
        num_nodes: int,
        node_dim: int,
        mode: str = "random",
        noise_level: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            num_samples: Number of samples to generate
            num_nodes: Number of nodes in graph
            node_dim: Dimension per node
            mode: "random", "smooth", "wave", "cluster"
            noise_level: Amount of noise to add
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        
        # Generate base patterns
        if mode == "random":
            self.data = torch.randn(num_samples, num_nodes, node_dim)
        
        elif mode == "smooth":
            # Smooth transitions between nodes
            self.data = torch.zeros(num_samples, num_nodes, node_dim)
            for i in range(num_samples):
                for d in range(node_dim):
                    # Low-frequency sinusoid
                    t = torch.linspace(0, 2 * np.pi, num_nodes)
                    self.data[i, :, d] = torch.sin(t * (d + 1))
        
        elif mode == "wave":
            # Wave-like patterns
            self.data = torch.zeros(num_samples, num_nodes, node_dim)
            for i in range(num_samples):
                freq = torch.rand(1).item() * 5 + 1
                phase = torch.rand(1).item() * 2 * np.pi
                t = torch.linspace(0, 2 * np.pi, num_nodes)
                for d in range(node_dim):
                    self.data[i, :, d] = torch.sin(freq * t + phase + d)
        
        elif mode == "cluster":
            # Clustered node states
            num_clusters = max(2, num_nodes // 5)
            cluster_centers = torch.randn(num_clusters, node_dim) * 2
            
            self.data = torch.zeros(num_samples, num_nodes, node_dim)
            for i in range(num_samples):
                cluster_assignments = torch.randint(0, num_clusters, (num_nodes,))
                for j in range(num_nodes):
                    self.data[i, j] = cluster_centers[cluster_assignments[j]]
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Add noise
        if noise_level > 0:
            self.data += torch.randn_like(self.data) * noise_level
        
        # Normalize
        self.mean = self.data.mean(dim=0, keepdim=True)
        self.std = self.data.std(dim=0, keepdim=True) + 1e-8
        self.data = (self.data - self.mean) / self.std
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        return {'states': self.data[idx]}


def create_dataloaders(
    train_data: torch.Tensor,
    val_data: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.1
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation data loaders.
    
    Args:
        train_data: Training data (num_samples, N, d)
        val_data: Optional validation data
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Fraction of training data to use for validation if val_data is None
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader (if val_data provided or val_split > 0)
    """
    # Split training data if no validation data provided
    if val_data is None and val_split > 0:
        num_val = int(len(train_data) * val_split)
        num_train = len(train_data) - num_val
        
        indices = torch.randperm(len(train_data))
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        val_data = train_data[val_indices]
        train_data = train_data[train_indices]
    
    # Create datasets
    train_dataset = GraphDiffusionDataset(train_data, normalize=True)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    if val_data is not None:
        val_dataset = GraphDiffusionDataset(val_data, normalize=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        val_loader = None
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing dataset module...")
    
    # Test 1: GraphDiffusionDataset
    print("\n=== Test 1: GraphDiffusionDataset ===")
    data = torch.randn(100, 25, 3)  # 100 samples, 25 nodes, 3D
    dataset = GraphDiffusionDataset(data, normalize=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0]['states'].shape}")
    print(f"Normalized mean: {dataset.data.mean():.6f}")
    print(f"Normalized std: {dataset.data.std():.6f}")
    
    # Test denormalization
    sample = dataset[0]['states']
    denorm = dataset.denormalize(sample.unsqueeze(0))
    original = data[0]
    error = (denorm.squeeze(0) - original).abs().mean()
    print(f"Denormalization error: {error:.6f}")
    
    # Test 2: SyntheticDataset
    print("\n=== Test 2: SyntheticDataset ===")
    for mode in ["random", "smooth", "wave", "cluster"]:
        syn_dataset = SyntheticDataset(
            num_samples=50,
            num_nodes=20,
            node_dim=3,
            mode=mode,
            seed=42
        )
        print(f"  {mode}: {len(syn_dataset)} samples, shape {syn_dataset[0]['states'].shape}")
    
    # Test 3: DataLoader
    print("\n=== Test 3: DataLoader ===")
    train_loader, val_loader = create_dataloaders(
        data,
        batch_size=8,
        val_split=0.2
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['states'].shape}")
    
    # Test 4: Iteration
    print("\n=== Test 4: Iteration ===")
    total_samples = 0
    for batch in train_loader:
        total_samples += batch['states'].shape[0]
    print(f"Total training samples: {total_samples}")
    
    print("\n✓ Dataset module test passed!")
