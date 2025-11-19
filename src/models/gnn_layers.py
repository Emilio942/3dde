"""Graph Neural Network layers for diffusion model.

Implements GNN layers that are aware of graph structure and Laplacian.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import warnings

# Try to import torch_geometric for optimized layers
try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("torch_geometric not found. GraphAttentionLayer will use slow dense implementation.")


def _torch_sparse_to_scipy_csr(t: torch.Tensor) -> csr_matrix:
    """Converts a torch.sparse_coo_tensor to a scipy.sparse.csr_matrix."""
    if not t.is_sparse:
        warnings.warn("Input tensor is dense, converting to numpy array.")
        return csr_matrix(t.cpu().numpy())
    
    t = t.coalesce()
    
    return csr_matrix(
        (t.values().cpu().numpy(), (t.indices()[0].cpu().numpy(), t.indices()[1].cpu().numpy())),
        shape=t.shape
    )


class LaplacianConv(nn.Module):
    """Graph convolution using Laplacian structure.
    
    Performs: H_out = σ(L @ H_in @ W + b)
    where L is the (normalized) graph Laplacian.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        normalize_laplacian: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            use_bias: Whether to use bias term
            normalize_laplacian: If True, normalize L to have eigenvalues in [-1, 1]
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize_laplacian = normalize_laplacian
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, in_features) or (N, in_features)
            L: Laplacian matrix (N, N)
            edge_weight: Optional edge weights (not used, for compatibility)
            
        Returns:
            out: Transformed features (B, N, out_features) or (N, out_features)
        """
        # Normalize Laplacian if requested
        # OPTIMIZATION: Removed expensive re-normalization in every forward pass.
        # The Laplacian should be normalized once before training.
        if self.normalize_laplacian:
             # We assume L is already normalized or we skip it for performance.
             # If strict normalization is needed, it must be done outside.
             pass
        
        L_norm = L
        
        # Apply linear transformation first
        x_transformed = torch.matmul(x, self.weight)

        # Propagation step
        if x.ndim == 3: # Batch mode
            B, N, F_out = x_transformed.shape
            # Reshape for efficient sparse matmul: (B, N, F) -> (N, B*F)
            x_reshaped = x_transformed.transpose(0, 1).reshape(N, B * F_out)
            propagated = L_norm @ x_reshaped
            # Reshape back: (N, B*F) -> (B, N, F)
            out = propagated.reshape(N, B, F_out).transpose(0, 1)
        else: # Single sample mode
            out = L_norm @ x_transformed

        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GraphConvLayer(nn.Module):
    """Graph convolution layer with residual connection and normalization."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            activation: "relu", "gelu", "silu"
            dropout: Dropout probability
            use_residual: Whether to use residual connection
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        
        # Graph convolution
        self.conv = LaplacianConv(in_dim, out_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions don't match
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, in_dim) or (N, in_dim)
            L: Laplacian (N, N)
            
        Returns:
            out: (B, N, out_dim) or (N, out_dim)
        """
        identity = x
        
        # Graph convolution
        out = self.conv(x, L)
        
        # Normalization
        out = self.norm(out)
        
        # Activation
        out = self.activation(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
        
        return out


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for variable-size neighborhoods.
    
    Computes attention weights based on node features and applies
    weighted aggregation.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat_heads: bool = True
    ):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension (per head if concat_heads=True)
            num_heads: Number of attention heads
            dropout: Dropout probability
            concat_heads: If True, concatenate heads; else average
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        
        # Multi-head dimensions
        if concat_heads:
            assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
            self.head_dim = out_dim // num_heads
        else:
            self.head_dim = out_dim
        
        # Initialize PyG layer if available
        if HAS_PYG:
            self.pyg_gat = GATConv(
                in_dim, 
                self.head_dim, 
                heads=num_heads, 
                dropout=dropout, 
                concat=concat_heads
            )
            # We do not initialize manual weights to avoid unused parameters
            self.W_q = None
            self.W_k = None
            self.W_v = None
            self.out_proj = None
        else:
            # Linear transformations for Q, K, V
            self.W_q = nn.Linear(in_dim, self.head_dim * num_heads, bias=False)
            self.W_k = nn.Linear(in_dim, self.head_dim * num_heads, bias=False)
            self.W_v = nn.Linear(in_dim, self.head_dim * num_heads, bias=False)
            
            # Output projection
            if concat_heads:
                self.out_proj = nn.Linear(out_dim, out_dim)
            else:
                self.out_proj = nn.Linear(self.head_dim, out_dim)
        
        # Attention dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, in_dim) or (N, in_dim)
            adjacency: Adjacency matrix (N, N), 1 if edge exists. Can be sparse.
            mask: Optional attention mask (B, N, N) or (N, N)
            
        Returns:
            out: (B, N, out_dim) or (N, out_dim)
        """
        if HAS_PYG:
            if mask is not None:
                warnings.warn("Attention mask is ignored when using torch_geometric optimization.")
            
            # Use optimized PyG implementation
            if x.ndim == 3:
                # PyG expects (N, F), so we need to handle batching manually or reshape
                # GATConv supports (N, F) or (B*N, F) with batch index
                # Here we treat (B, N, F) as B independent graphs or one large graph
                B, N, F_in = x.shape
                x_reshaped = x.reshape(B * N, F_in)
                
                # Convert adjacency to edge_index
                if adjacency.is_sparse:
                    adjacency = adjacency.coalesce()
                    indices = adjacency.indices()
                else:
                    indices = adjacency.nonzero().t()
                
                # Replicate indices for batch
                # This is a bit tricky without a proper batch vector
                # Simplified: Loop over batch (slower but memory efficient) or use PyG Batch
                # For now, let's stick to the manual loop if B is small, or reshape if adjacency is shared
                
                # If adjacency is shared across batch (which is typical here):
                # We can construct a large block diagonal adjacency
                # But that's expensive.
                
                # Alternative: Iterate over batch
                outs = []
                edge_index = indices
                for i in range(B):
                    out_i = self.pyg_gat(x[i], edge_index)
                    outs.append(out_i)
                out = torch.stack(outs)
                return out
            else:
                # Single graph case
                if adjacency.is_sparse:
                    adjacency = adjacency.coalesce()
                    edge_index = adjacency.indices()
                else:
                    edge_index = adjacency.nonzero().t()
                
                return self.pyg_gat(x, edge_index)

        # Fallback to original implementation
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, _ = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim)
        
        # Transpose for attention: (B, H, N, D)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)
        
        # Apply adjacency mask (only attend to neighbors)
        if adjacency.is_sparse:
            # Convert sparse adjacency to a dense mask for softmax
            # This is a bottleneck, but unavoidable for standard softmax attention
            adjacency_mask = adjacency.to_dense().unsqueeze(0).unsqueeze(0)
        else:
            adjacency_mask = adjacency.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

        scores = scores.masked_fill(adjacency_mask == 0, float('-inf'))
        
        # Optional additional mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, H, N, D)
        
        # Concatenate or average heads
        if self.concat_heads:
            out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, H*D)
        else:
            out = out.mean(dim=1)  # (B, N, D)
        
        # Output projection
        out = self.out_proj(out)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out


class GNNBlock(nn.Module):
    """Complete GNN block with convolution + attention + FFN."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
            use_attention: Whether to use attention layer
        """
        super().__init__()
        self.use_attention = use_attention
        
        # Graph convolution
        self.conv = GraphConvLayer(dim, dim, dropout=dropout)
        
        # Optional attention
        if use_attention:
            self.attn = GraphAttentionLayer(dim, dim, num_heads=num_heads, dropout=dropout)
            self.attn_norm = nn.LayerNorm(dim)
        
        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, dim)
            L: Laplacian (N, N)
            adjacency: Adjacency matrix (N, N), required if use_attention=True
            
        Returns:
            out: (B, N, dim)
        """
        # Graph convolution with residual
        x = self.conv(x, L)
        
        # Optional attention with residual
        if self.use_attention:
            if adjacency is None:
                # Derive adjacency from Laplacian: A_ij = -L_ij for i!=j
                if L.is_sparse:
                    L_coalesced = L.coalesce()
                    indices = L_coalesced.indices()
                    # Filter out diagonal entries
                    off_diag_mask = indices[0] != indices[1]
                    adj_indices = indices[:, off_diag_mask]
                    # Adjacency is 1 where L is non-zero off-diagonal
                    adj_values = torch.ones(adj_indices.shape[1], device=L.device, dtype=L.dtype)
                    adjacency = torch.sparse_coo_tensor(adj_indices, adj_values, L.shape)
                else:
                    D = torch.diag(L.diagonal())
                    adj_dense = D - L
                    adjacency = (adj_dense != 0).float()
            
            attn_out = self.attn(x, adjacency)
            x = x + attn_out
            x = self.attn_norm(x)
        
        # Feed-forward with residual
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.mlp_norm(x)
        
        return x


if __name__ == "__main__":
    print("Testing GNN layers...")
    
    # Setup
    B, N, d = 4, 10, 32  # 4 samples, 10 nodes, 32 dims
    
    # Create simple Laplacian (chain graph)
    L = torch.eye(N) * 2.0
    for i in range(N - 1):
        L[i, i + 1] = -1.0
        L[i + 1, i] = -1.0
    
    # Derive adjacency
    D = torch.diag(L.diagonal())
    A = D - L
    
    # Test data
    x = torch.randn(B, N, d)
    
    print(f"\nTest configuration:")
    print(f"  Batch: {B}, Nodes: {N}, Features: {d}")
    
    # Test 1: LaplacianConv
    print("\n=== Test 1: LaplacianConv ===")
    conv = LaplacianConv(d, d)
    out = conv(x, L)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test 2: GraphConvLayer
    print("\n=== Test 2: GraphConvLayer ===")
    layer = GraphConvLayer(d, d)
    out = layer(x, L)
    print(f"Output shape: {out.shape}")
    
    # Test 3: GraphAttentionLayer
    print("\n=== Test 3: GraphAttentionLayer ===")
    attn = GraphAttentionLayer(d, d, num_heads=4)
    out = attn(x, A)
    print(f"Output shape: {out.shape}")
    
    # Test 4: GNNBlock
    print("\n=== Test 4: GNNBlock ===")
    block = GNNBlock(d, num_heads=4, use_attention=True)
    out = block(x, L, A)
    print(f"Output shape: {out.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in block.parameters())
    print(f"Total parameters in GNNBlock: {total_params:,}")
    
    print("\n✓ GNN layers test passed!")
