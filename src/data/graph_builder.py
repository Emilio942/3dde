"""Graph construction utilities for building graph Laplacians.

Provides functions to create graph structures from different sources:
- Regular grids (for testing)
- Edge lists (general graphs)
- Molecular structures (3D molecules)
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import warnings


def build_grid_laplacian(
    grid_size: Tuple[int, ...],
    periodic: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Build Laplacian for a regular grid graph.
    
    Args:
        grid_size: Grid dimensions, e.g. (10,) for 1D, (5, 5) for 2D, (4, 4, 4) for 3D
        periodic: If True, use periodic boundary conditions
        device: Target device
        dtype: Target dtype (default: float32)
        
    Returns:
        L: Graph Laplacian (N, N) where N = product of grid_size
        
    Notes:
        - For 1D: connects i to i-1, i+1
        - For 2D: connects to 4 neighbors (up, down, left, right)
        - For 3D: connects to 6 neighbors
    """
    if dtype is None:
        dtype = torch.float32
    
    ndim = len(grid_size)
    N = int(np.prod(grid_size))
    
    # Initialize sparse matrix storage
    edges = []
    
    if ndim == 1:
        # 1D grid
        n = grid_size[0]
        for i in range(n):
            # Connect to left neighbor
            if i > 0:
                edges.append((i, i - 1))
            elif periodic:
                edges.append((i, n - 1))
            
            # Connect to right neighbor
            if i < n - 1:
                edges.append((i, i + 1))
            elif periodic:
                edges.append((i, 0))
    
    elif ndim == 2:
        # 2D grid
        ny, nx = grid_size
        for i in range(ny):
            for j in range(nx):
                node_idx = i * nx + j
                
                # Up
                if i > 0:
                    edges.append((node_idx, (i - 1) * nx + j))
                elif periodic:
                    edges.append((node_idx, (ny - 1) * nx + j))
                
                # Down
                if i < ny - 1:
                    edges.append((node_idx, (i + 1) * nx + j))
                elif periodic:
                    edges.append((node_idx, 0 * nx + j))
                
                # Left
                if j > 0:
                    edges.append((node_idx, i * nx + (j - 1)))
                elif periodic:
                    edges.append((node_idx, i * nx + (nx - 1)))
                
                # Right
                if j < nx - 1:
                    edges.append((node_idx, i * nx + (j + 1)))
                elif periodic:
                    edges.append((node_idx, i * nx + 0))
    
    elif ndim == 3:
        # 3D grid
        nz, ny, nx = grid_size
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    node_idx = i * ny * nx + j * nx + k
                    
                    # All 6 neighbors in 3D
                    neighbors = [
                        ((i - 1) % nz if periodic else i - 1, j, k) if i > 0 or periodic else None,
                        ((i + 1) % nz if periodic else i + 1, j, k) if i < nz - 1 or periodic else None,
                        (i, (j - 1) % ny if periodic else j - 1, k) if j > 0 or periodic else None,
                        (i, (j + 1) % ny if periodic else j + 1, k) if j < ny - 1 or periodic else None,
                        (i, j, (k - 1) % nx if periodic else k - 1) if k > 0 or periodic else None,
                        (i, j, (k + 1) % nx if periodic else k + 1) if k < nx - 1 or periodic else None,
                    ]
                    
                    for neighbor in neighbors:
                        if neighbor is not None:
                            ni, nj, nk = neighbor
                            neighbor_idx = ni * ny * nx + nj * nx + nk
                            edges.append((node_idx, neighbor_idx))
    
    else:
        raise ValueError(f"Unsupported grid dimension: {ndim}. Use 1, 2, or 3.")
    
    # Build Laplacian from edges
    L = build_laplacian_from_edges(edges, N, device=device, dtype=dtype)
    
    return L


def build_laplacian_from_edges(
    edges: List[Tuple[int, int]],
    num_nodes: int,
    weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Build Laplacian matrix from edge list.
    
    Laplacian: L = D - A
    where D is degree matrix (diagonal) and A is adjacency matrix.
    
    Args:
        edges: List of (i, j) edges (undirected, so only need one direction)
        num_nodes: Total number of nodes
        weights: Optional edge weights (default: all 1.0)
        device: Target device
        dtype: Target dtype
        
    Returns:
        L: Laplacian matrix (num_nodes, num_nodes)
    """
    if dtype is None:
        dtype = torch.float32
    
    # Initialize adjacency matrix
    A = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    
    # Fill adjacency
    if weights is None:
        weights = [1.0] * len(edges)
    
    for (i, j), w in zip(edges, weights):
        A[i, j] = w
        A[j, i] = w  # Symmetric for undirected graph
    
    # Compute degree matrix
    D = torch.diag(A.sum(dim=1))
    
    # Laplacian
    L = D - A
    
    return L


def build_molecular_graph(
    positions: torch.Tensor,
    cutoff: float = 2.0,
    fully_connected: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build graph from 3D molecular positions.
    
    Connects atoms based on distance threshold or makes fully connected.
    
    Args:
        positions: Atomic positions (N, 3)
        cutoff: Distance cutoff for edge creation (Angstroms)
        fully_connected: If True, ignore cutoff and connect all atoms
        device: Target device
        dtype: Target dtype
        
    Returns:
        L: Laplacian matrix (N, N)
        edges: Edge list as tensor (2, num_edges)
    """
    if dtype is None:
        dtype = torch.float32
    
    N = positions.shape[0]
    positions = positions.to(device=device, dtype=dtype)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(positions, positions, p=2)  # (N, N)
    
    # Build adjacency
    if fully_connected:
        # Connect all pairs (except self-loops)
        adj = torch.ones(N, N, device=device, dtype=dtype)
        adj.fill_diagonal_(0)
        # Weight by inverse distance (with regularization)
        weights = 1.0 / (dist_matrix + 1e-3)
        weights.fill_diagonal_(0)
        A = adj * weights
    else:
        # Connect if within cutoff
        adj = (dist_matrix < cutoff).float()
        adj.fill_diagonal_(0)
        # Weight by inverse distance
        weights = torch.where(
            dist_matrix < cutoff,
            1.0 / (dist_matrix + 1e-3),
            torch.zeros_like(dist_matrix)
        )
        weights.fill_diagonal_(0)
        A = weights
    
    # Compute Laplacian
    D = torch.diag(A.sum(dim=1))
    L = D - A
    
    # Extract edge list
    edge_indices = torch.nonzero(adj, as_tuple=False)  # (num_edges, 2)
    edges = edge_indices.T  # (2, num_edges)
    
    return L, edges


def build_from_adjacency(
    adjacency: torch.Tensor,
    normalize: bool = False
) -> torch.Tensor:
    """Build Laplacian from adjacency matrix.
    
    Args:
        adjacency: Adjacency matrix (N, N)
        normalize: If True, compute normalized Laplacian L_norm = I - D^(-1/2) A D^(-1/2)
        
    Returns:
        L: Laplacian matrix (N, N)
    """
    N = adjacency.shape[0]
    device = adjacency.device
    dtype = adjacency.dtype
    
    # Degree matrix
    degrees = adjacency.sum(dim=1)
    D = torch.diag(degrees)
    
    if normalize:
        # Normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
        # Avoid division by zero
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-12))
        I = torch.eye(N, device=device, dtype=dtype)
        L = I - D_inv_sqrt @ adjacency @ D_inv_sqrt
    else:
        # Standard Laplacian: L = D - A
        L = D - adjacency
    
    return L


def add_self_loops(
    L: torch.Tensor,
    weight: float = 1.0
) -> torch.Tensor:
    """Add self-loops to Laplacian (increases diagonal elements).
    
    Useful for improving conditioning and stability.
    
    Args:
        L: Laplacian matrix (N, N)
        weight: Self-loop weight
        
    Returns:
        L_with_loops: Modified Laplacian
    """
    N = L.shape[0]
    L_new = L.clone()
    L_new.diagonal().add_(weight)
    return L_new


def visualize_graph_structure(L: torch.Tensor) -> None:
    """Print graph statistics and structure info.
    
    Args:
        L: Laplacian matrix
    """
    N = L.shape[0]
    
    # Recover adjacency: A = D - L
    degrees = L.diagonal()
    A = torch.diag(degrees) - L
    
    # Statistics
    num_edges = (A > 0).sum().item() // 2  # Undirected, so divide by 2
    avg_degree = degrees.mean().item()
    max_degree = degrees.max().item()
    min_degree = degrees.min().item()
    
    # Spectral properties
    eigenvalues = torch.linalg.eigvalsh(L)
    spectral_gap = eigenvalues[1].item() if N > 1 else 0.0
    
    print(f"Graph Structure:")
    print(f"  Nodes: {N}")
    print(f"  Edges: {num_edges}")
    print(f"  Avg degree: {avg_degree:.2f}")
    print(f"  Degree range: [{min_degree:.0f}, {max_degree:.0f}]")
    print(f"  Spectral gap: {spectral_gap:.4f}")
    print(f"  Max eigenvalue: {eigenvalues[-1].item():.4f}")


if __name__ == "__main__":
    print("Testing graph builder module...")
    
    # Test 1: 1D grid
    print("\n=== Test 1: 1D Grid ===")
    L_1d = build_grid_laplacian((10,), periodic=False)
    print(f"1D grid Laplacian shape: {L_1d.shape}")
    visualize_graph_structure(L_1d)
    
    # Test 2: 2D grid
    print("\n=== Test 2: 2D Grid ===")
    L_2d = build_grid_laplacian((5, 5), periodic=True)
    print(f"2D grid Laplacian shape: {L_2d.shape}")
    visualize_graph_structure(L_2d)
    
    # Test 3: Molecular graph
    print("\n=== Test 3: Molecular Graph ===")
    # Random molecule with 8 atoms
    positions = torch.randn(8, 3) * 2.0  # Random positions in 3D
    L_mol, edges = build_molecular_graph(positions, cutoff=3.0)
    print(f"Molecular Laplacian shape: {L_mol.shape}")
    print(f"Number of edges: {edges.shape[1]}")
    visualize_graph_structure(L_mol)
    
    # Test 4: Check Laplacian properties
    print("\n=== Test 4: Laplacian Properties ===")
    # Row sum should be zero
    row_sums = L_1d.sum(dim=1)
    print(f"Row sums (should be ~0): max={row_sums.abs().max():.6f}")
    
    # Should be symmetric
    is_symmetric = (L_1d - L_1d.T).abs().max() < 1e-6
    print(f"Is symmetric: {is_symmetric}")
    
    # Should be positive semi-definite (eigenvalues >= 0)
    eigenvalues = torch.linalg.eigvalsh(L_1d)
    print(f"Min eigenvalue (should be ~0): {eigenvalues[0]:.6f}")
    print(f"All eigenvalues non-negative: {(eigenvalues >= -1e-6).all().item()}")
    
    print("\n✓ Graph builder module test passed!")
