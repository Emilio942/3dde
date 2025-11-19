"""
Evaluation metrics for 3D structures.
Focuses on physical plausibility and geometric quality without requiring complex trained models.
"""

import torch
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional

def compute_connectivity_score(
    positions: torch.Tensor, 
    edge_index: torch.Tensor,
    threshold: float = 2.0
) -> float:
    """
    Computes the connectivity score of the graph.
    
    Args:
        positions: (N, 3) tensor of node positions
        edge_index: (2, E) tensor of edges
        threshold: Maximum distance to consider an edge 'valid' (optional geometric check)
        
    Returns:
        score: Ratio of nodes in the largest connected component (0.0 to 1.0).
               1.0 means the object is fully connected (physically plausible).
    """
    if positions.ndim > 2:
        # Handle batch: take first or average? Usually evaluation is per sample.
        # For simplicity, we assume single sample input here or loop outside.
        raise ValueError("Input must be (N, 3). Loop over batch dimension externally.")

    num_nodes = positions.shape[0]
    
    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    edges = edge_index.t().cpu().numpy()
    G.add_edges_from(edges)
    
    # Get connected components
    components = list(nx.connected_components(G))
    
    if not components:
        return 0.0
        
    largest_component_size = max(len(c) for c in components)
    
    return largest_component_size / num_nodes


def compute_smoothness_score(
    positions: torch.Tensor,
    L: torch.Tensor
) -> float:
    """
    Computes the Dirichlet energy (smoothness) of the structure.
    Lower is smoother.
    
    Energy = trace(X^T L X)
    
    Args:
        positions: (N, 3) tensor
        L: (N, N) Laplacian matrix
        
    Returns:
        energy: Scalar smoothness value.
    """
    # E = tr(X^T L X)
    # (3, N) @ (N, N) @ (N, 3) -> (3, 3) -> trace
    
    # Ensure L is on same device
    L = L.to(positions.device)
    
    # X^T L X
    term = torch.matmul(positions.t(), torch.matmul(L, positions))
    energy = torch.trace(term)
    
    # Normalize by number of nodes to make it comparable across sizes
    return energy.item() / positions.shape[0]


def compute_edge_consistency(
    positions: torch.Tensor,
    edge_index: torch.Tensor
) -> Dict[str, float]:
    """
    Computes statistics about edge lengths.
    Physically plausible structures usually have consistent edge lengths (low variance).
    
    Args:
        positions: (N, 3) tensor
        edge_index: (2, E) tensor
        
    Returns:
        metrics: Dict with 'mean_length', 'std_length', 'max_length'
    """
    src, dst = edge_index
    
    # Compute distances
    p_src = positions[src]
    p_dst = positions[dst]
    
    distances = torch.norm(p_src - p_dst, dim=1)
    
    return {
        'edge_len_mean': distances.mean().item(),
        'edge_len_std': distances.std().item(),
        'edge_len_max': distances.max().item(),
        'edge_len_min': distances.min().item()
    }


def compute_chamfer_distance(
    pc1: torch.Tensor,
    pc2: torch.Tensor
) -> float:
    """
    Computes Chamfer Distance between two point clouds.
    Measures how similar two shapes are geometrically.
    
    Args:
        pc1: (N, 3) tensor
        pc2: (M, 3) tensor
        
    Returns:
        dist: Chamfer distance
    """
    # Expand dims for broadcasting: (N, 1, 3) - (1, M, 3)
    diff = pc1.unsqueeze(1) - pc2.unsqueeze(0)
    dist_sq = torch.sum(diff**2, dim=2) # (N, M)
    
    # For each point in pc1, find nearest in pc2
    min_dist_12, _ = torch.min(dist_sq, dim=1)
    
    # For each point in pc2, find nearest in pc1
    min_dist_21, _ = torch.min(dist_sq, dim=0)
    
    # Average distance
    chamfer_dist = torch.mean(min_dist_12) + torch.mean(min_dist_21)
    
    return chamfer_dist.item()

class PhysicalEvaluator:
    """Helper class to run all evaluations on a batch."""
    
    def __init__(self, L_template: torch.Tensor, edge_index_template: torch.Tensor):
        self.L = L_template
        self.edge_index = edge_index_template
        
    def evaluate_batch(self, batch_positions: torch.Tensor) -> Dict[str, float]:
        """
        Args:
            batch_positions: (B, N, 3) tensor
            
        Returns:
            metrics: Averaged metrics over the batch
        """
        B = batch_positions.shape[0]
        metrics_sum = {
            'connectivity': 0.0,
            'smoothness': 0.0,
            'edge_len_std': 0.0
        }
        
        for i in range(B):
            pos = batch_positions[i]
            
            # Connectivity
            metrics_sum['connectivity'] += compute_connectivity_score(pos, self.edge_index)
            
            # Smoothness
            metrics_sum['smoothness'] += compute_smoothness_score(pos, self.L)
            
            # Edge Consistency
            edge_stats = compute_edge_consistency(pos, self.edge_index)
            metrics_sum['edge_len_std'] += edge_stats['edge_len_std']
            
        # Average
        return {k: v / B for k, v in metrics_sum.items()}
