"""Epsilon prediction network for diffusion model.

Predicts the noise ε added during forward diffusion:
    ε_θ(S_t, t, L) ≈ ε
where S_t = Φ(t)S_0 + √Σ(t)ε
"""

import torch
import torch.nn as nn
from typing import Optional
import math

from .gnn_layers import GNNBlock, GraphConvLayer


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps.
    
    Encodes timestep t using sinusoidal functions of different frequencies.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        """
        Args:
            dim: Embedding dimension (must be even)
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps (B,) with values in [0, num_steps]
            
        Returns:
            emb: Time embeddings (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb


class EpsilonNetwork(nn.Module):
    """Neural network for predicting noise ε in diffusion model.
    
    Architecture:
        1. Input projection: S_t (B, N, d) -> (B, N, hidden_dim)
        2. Time embedding: t (B,) -> (B, time_dim) -> broadcasted to all nodes
        3. GNN blocks: Process with graph structure
        4. Output projection: (B, N, hidden_dim) -> (B, N, d)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        time_dim: int = 64,
        dropout: float = 0.1,
        use_attention: bool = True,
        mlp_ratio: float = 2.0
    ):
        """
        Args:
            input_dim: Input feature dimension per node (usually 3 for 3D positions)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN blocks
            num_heads: Number of attention heads
            time_dim: Time embedding dimension
            dropout: Dropout probability
            use_attention: Whether to use attention in GNN blocks
            mlp_ratio: MLP expansion ratio in GNN blocks
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN blocks
        self.gnn_blocks = nn.ModuleList([
            GNNBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_attention=use_attention
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
    
    def forward(
        self,
        St: torch.Tensor,
        t: torch.Tensor,
        L: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict noise ε from noised state S_t and timestep t.
        
        Args:
            St: Noised node states (B, N, input_dim)
            t: Timestep indices (B,), values in [0, num_steps]
            L: Graph Laplacian (N, N)
            adjacency: Optional adjacency matrix (N, N)
            
        Returns:
            eps_pred: Predicted noise (B, N, input_dim)
        """
        B, N, d = St.shape
        
        # Project input to hidden dimension
        h = self.input_proj(St)  # (B, N, hidden_dim)
        
        # Embed time
        t_emb = self.time_embed(t)  # (B, time_dim)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)
        
        # Add time embedding to all nodes
        h = h + t_emb.unsqueeze(1)  # (B, N, hidden_dim) + (B, 1, hidden_dim)
        
        # Process through GNN blocks
        for block in self.gnn_blocks:
            h = block(h, L, adjacency)
        
        # Project to output
        eps_pred = self.output_proj(h)  # (B, N, input_dim)
        
        return eps_pred


class ConditionalEpsilonNetwork(nn.Module):
    """Epsilon network with conditioning on additional features.
    
    Useful for conditional generation (e.g., given atom types, constraints).
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        time_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            condition_dim: Conditioning feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            time_dim: Time embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Combined input projection (state + condition)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN blocks
        self.gnn_blocks = nn.ModuleList([
            GNNBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_attention=True
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(
        self,
        St: torch.Tensor,
        t: torch.Tensor,
        L: torch.Tensor,
        condition: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            St: Noised states (B, N, input_dim)
            t: Timesteps (B,)
            L: Laplacian (N, N)
            condition: Conditioning features (B, N, condition_dim)
            adjacency: Optional adjacency (N, N)
            
        Returns:
            eps_pred: Predicted noise (B, N, input_dim)
        """
        # Concatenate state and condition
        x = torch.cat([St, condition], dim=-1)  # (B, N, input_dim + condition_dim)
        
        # Project to hidden
        h = self.input_proj(x)
        
        # Add time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb.unsqueeze(1)
        
        # GNN processing
        for block in self.gnn_blocks:
            h = block(h, L, adjacency)
        
        # Output
        eps_pred = self.output_proj(h)
        
        return eps_pred


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Epsilon Network...")
    
    # Configuration
    B, N, d = 4, 25, 3  # 4 samples, 25 nodes (5x5 grid), 3 dimensions
    num_steps = 1000
    
    # Create Laplacian (5x5 grid)
    from ..data.graph_builder import build_grid_laplacian
    L = build_grid_laplacian((5, 5))
    
    # Generate test data
    St = torch.randn(B, N, d)
    t = torch.randint(0, num_steps, (B,))
    
    print(f"\nTest configuration:")
    print(f"  Batch: {B}, Nodes: {N}, Dimensions: {d}")
    print(f"  Timesteps sampled: {t.tolist()}")
    
    # Test 1: Basic EpsilonNetwork
    print("\n=== Test 1: EpsilonNetwork ===")
    model = EpsilonNetwork(
        input_dim=d,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        use_attention=True
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    eps_pred = model(St, t, L)
    print(f"Input shape: {St.shape}")
    print(f"Output shape: {eps_pred.shape}")
    print(f"Output mean: {eps_pred.mean():.4f}, std: {eps_pred.std():.4f}")
    
    # Test 2: Time embedding
    print("\n=== Test 2: Time Embedding ===")
    time_embed = TimeEmbedding(dim=64)
    t_emb = time_embed(t)
    print(f"Time embedding shape: {t_emb.shape}")
    print(f"Time 0 vs time {num_steps-1} similarity: {torch.cosine_similarity(t_emb[0:1], time_embed(torch.tensor([num_steps-1]))).item():.4f}")
    
    # Test 3: Conditional network
    print("\n=== Test 3: ConditionalEpsilonNetwork ===")
    condition = torch.randn(B, N, 5)  # 5-dim conditioning (e.g., atom types)
    cond_model = ConditionalEpsilonNetwork(
        input_dim=d,
        condition_dim=5,
        hidden_dim=64,
        num_layers=2
    )
    
    print(f"Conditional model parameters: {count_parameters(cond_model):,}")
    
    eps_pred_cond = cond_model(St, t, L, condition)
    print(f"Conditional output shape: {eps_pred_cond.shape}")
    
    # Test 4: Forward pass timing
    print("\n=== Test 4: Performance ===")
    import time
    
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = model(St, t, L)
        elapsed = time.time() - start
    
    print(f"10 forward passes: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms per pass)")
    
    print("\n✓ Epsilon Network test passed!")
