"""Physics-informed regularizers for diffusion model training.

Implements various regularization terms to enforce physical constraints:
- Energy regularization
- Graph smoothness
- Alignment preservation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class EnergyRegularizer(nn.Module):
    """Energy-based regularization: E(Ŝ₀) = ½||M^(1/2)(Ŝ₀ - S*)||²
    
    Penalizes deviation from equilibrium state S*.
    """
    
    def __init__(
        self,
        equilibrium_state: Optional[torch.Tensor] = None,
        mass_matrix: Optional[torch.Tensor] = None
    ):
        """
        Args:
            equilibrium_state: Target equilibrium S* (N, d), if None uses zero
            mass_matrix: Mass matrix M (N, N) or (N,) for diagonal
        """
        super().__init__()
        
        if equilibrium_state is not None:
            self.register_buffer('equilibrium', equilibrium_state)
        else:
            self.equilibrium = None
        
        if mass_matrix is not None:
            self.register_buffer('mass_matrix', mass_matrix)
        else:
            self.mass_matrix = None
    
    def forward(self, S_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S_pred: Predicted states (B, N, d)
            
        Returns:
            energy: Energy penalty (scalar)
        """
        # Deviation from equilibrium
        if self.equilibrium is not None:
            deviation = S_pred - self.equilibrium.unsqueeze(0)  # (B, N, d)
        else:
            deviation = S_pred
        
        # Apply mass matrix if provided
        if self.mass_matrix is not None:
            if self.mass_matrix.ndim == 1:
                # Diagonal mass matrix
                weighted_dev = deviation * torch.sqrt(self.mass_matrix.unsqueeze(0).unsqueeze(-1))
            else:
                # Full mass matrix: need sqrt(M) @ deviation
                sqrt_M = self._matrix_sqrt(self.mass_matrix)
                weighted_dev = torch.zeros_like(deviation)
                for d in range(deviation.shape[-1]):
                    weighted_dev[:, :, d] = torch.matmul(deviation[:, :, d], sqrt_M.T)
        else:
            weighted_dev = deviation
        
        # Energy: ½||weighted_dev||²
        energy = 0.5 * (weighted_dev ** 2).mean()
        
        return energy
    
    @staticmethod
    def _matrix_sqrt(M: torch.Tensor) -> torch.Tensor:
        """Compute matrix square root via eigendecomposition."""
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        sqrt_M = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
        return sqrt_M


class GraphSmoothnessRegularizer(nn.Module):
    """Graph smoothness regularization: ||L Ŝ₀||²_F
    
    Penalizes rapid variations between neighboring nodes.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, normalize by number of nodes
        """
        super().__init__()
        self.normalize = normalize
    
    def forward(self, S_pred: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S_pred: Predicted states (B, N, d)
            L: Graph Laplacian (N, N)
            
        Returns:
            smoothness: Smoothness penalty (scalar)
        """
        B, N, d = S_pred.shape
        
        # Apply Laplacian: L @ S for each dimension
        smoothness = 0.0
        for dim_idx in range(d):
            LS = torch.matmul(L, S_pred[:, :, dim_idx].T).T  # (B, N)
            smoothness += (LS ** 2).sum()
        
        # Normalize
        if self.normalize:
            smoothness = smoothness / (B * N * d)
        else:
            smoothness = smoothness / B
        
        return smoothness


class AlignmentRegularizer(nn.Module):
    """Alignment regularization: preserves directional field structure.
    
    Enforces that predicted directions align with reference directions.
    """
    
    def __init__(self, use_cosine_similarity: bool = True):
        """
        Args:
            use_cosine_similarity: If True, use cosine similarity; else MSE
        """
        super().__init__()
        self.use_cosine_similarity = use_cosine_similarity
    
    def forward(
        self,
        S_pred: torch.Tensor,
        S_reference: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            S_pred: Predicted states (B, N, d)
            S_reference: Reference states (B, N, d)
            mask: Optional mask for nodes (B, N)
            
        Returns:
            alignment: Alignment penalty (scalar)
        """
        if self.use_cosine_similarity:
            # Cosine similarity between predicted and reference
            # sim = (S_pred · S_ref) / (||S_pred|| ||S_ref||)
            dot_product = (S_pred * S_reference).sum(dim=-1)  # (B, N)
            norm_pred = torch.norm(S_pred, dim=-1) + 1e-8
            norm_ref = torch.norm(S_reference, dim=-1) + 1e-8
            cos_sim = dot_product / (norm_pred * norm_ref)
            
            # Penalty: 1 - cos_sim (0 when aligned, 2 when opposite)
            penalty = 1.0 - cos_sim
        else:
            # MSE between directions
            penalty = ((S_pred - S_reference) ** 2).sum(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            penalty = penalty * mask
            alignment = penalty.sum() / (mask.sum() + 1e-8)
        else:
            alignment = penalty.mean()
        
        return alignment


class DivergenceRegularizer(nn.Module):
    """Divergence regularization: penalizes non-zero divergence.
    
    For physical systems where ∇·S should be small.
    """
    
    def __init__(self, epsilon: float = 1e-3):
        """
        Args:
            epsilon: Step size for finite difference
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, S_pred: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S_pred: Predicted states (B, N, d)
            adjacency: Adjacency matrix (N, N)
            
        Returns:
            divergence: Divergence penalty (scalar)
        """
        B, N, d = S_pred.shape
        
        # Compute approximate divergence using finite differences
        # div(S) ≈ Σ_neighbors (S_neighbor - S_node) / distance
        divergence = 0.0
        
        for i in range(N):
            neighbors = adjacency[i].nonzero(as_tuple=True)[0]
            if len(neighbors) > 0:
                # Average difference with neighbors
                S_i = S_pred[:, i:i+1, :]  # (B, 1, d)
                S_neighbors = S_pred[:, neighbors, :]  # (B, num_neighbors, d)
                
                # Divergence at node i
                div_i = (S_neighbors - S_i).mean(dim=1)  # (B, d)
                divergence += (div_i ** 2).sum()
        
        # Normalize
        divergence = divergence / (B * N * d)
        
        return divergence


class CombinedPhysicsRegularizer(nn.Module):
    """Combined regularizer with multiple physics constraints."""
    
    def __init__(
        self,
        lambda_energy: float = 0.1,
        lambda_smoothness: float = 0.1,
        lambda_alignment: float = 0.0,
        lambda_divergence: float = 0.0,
        equilibrium_state: Optional[torch.Tensor] = None,
        mass_matrix: Optional[torch.Tensor] = None
    ):
        """
        Args:
            lambda_energy: Weight for energy regularization
            lambda_smoothness: Weight for smoothness regularization
            lambda_alignment: Weight for alignment regularization
            lambda_divergence: Weight for divergence regularization
            equilibrium_state: Equilibrium state for energy reg
            mass_matrix: Mass matrix for energy reg
        """
        super().__init__()
        
        self.lambda_energy = lambda_energy
        self.lambda_smoothness = lambda_smoothness
        self.lambda_alignment = lambda_alignment
        self.lambda_divergence = lambda_divergence
        
        # Initialize regularizers
        if lambda_energy > 0:
            self.energy_reg = EnergyRegularizer(equilibrium_state, mass_matrix)
        
        if lambda_smoothness > 0:
            self.smoothness_reg = GraphSmoothnessRegularizer()
        
        if lambda_alignment > 0:
            self.alignment_reg = AlignmentRegularizer()
        
        if lambda_divergence > 0:
            self.divergence_reg = DivergenceRegularizer()
    
    def forward(
        self,
        S_pred: torch.Tensor,
        L: Optional[torch.Tensor] = None,
        S_reference: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            S_pred: Predicted states (B, N, d)
            L: Laplacian (N, N), required if lambda_smoothness > 0
            S_reference: Reference states (B, N, d), required if lambda_alignment > 0
            adjacency: Adjacency (N, N), required if lambda_divergence > 0
            
        Returns:
            reg_dict: Dictionary of regularization terms and total
        """
        total_reg = 0.0
        reg_dict = {}
        
        # Energy regularization
        if self.lambda_energy > 0:
            energy_loss = self.energy_reg(S_pred)
            reg_dict['energy'] = energy_loss.item()
            total_reg += self.lambda_energy * energy_loss
        
        # Smoothness regularization
        if self.lambda_smoothness > 0:
            if L is None:
                raise ValueError("Laplacian L required for smoothness regularization")
            smoothness_loss = self.smoothness_reg(S_pred, L)
            reg_dict['smoothness'] = smoothness_loss.item()
            total_reg += self.lambda_smoothness * smoothness_loss
        
        # Alignment regularization
        if self.lambda_alignment > 0:
            if S_reference is None:
                raise ValueError("Reference state required for alignment regularization")
            alignment_loss = self.alignment_reg(S_pred, S_reference)
            reg_dict['alignment'] = alignment_loss.item()
            total_reg += self.lambda_alignment * alignment_loss
        
        # Divergence regularization
        if self.lambda_divergence > 0:
            if adjacency is None:
                raise ValueError("Adjacency required for divergence regularization")
            divergence_loss = self.divergence_reg(S_pred, adjacency)
            reg_dict['divergence'] = divergence_loss.item()
            total_reg += self.lambda_divergence * divergence_loss
        
        reg_dict['total_reg'] = total_reg.item()
        
        return total_reg, reg_dict


if __name__ == "__main__":
    print("Testing regularizers...")
    
    # Setup
    B, N, d = 4, 10, 3
    
    # Create Laplacian
    L = torch.eye(N) * 2.0
    for i in range(N - 1):
        L[i, i + 1] = -1.0
        L[i + 1, i] = -1.0
    
    # Adjacency
    D = torch.diag(L.diagonal())
    A = D - L
    
    # Test data
    S_pred = torch.randn(B, N, d)
    S_ref = torch.randn(B, N, d)
    
    print(f"\nTest configuration:")
    print(f"  Batch: {B}, Nodes: {N}, Dimensions: {d}")
    
    # Test 1: Energy Regularizer
    print("\n=== Test 1: Energy Regularizer ===")
    energy_reg = EnergyRegularizer()
    energy_loss = energy_reg(S_pred)
    print(f"Energy loss: {energy_loss.item():.6f}")
    
    # With equilibrium
    S_equilibrium = torch.zeros(N, d)
    energy_reg_eq = EnergyRegularizer(equilibrium_state=S_equilibrium)
    energy_loss_eq = energy_reg_eq(S_pred)
    print(f"Energy loss (with equilibrium): {energy_loss_eq.item():.6f}")
    
    # Test 2: Smoothness Regularizer
    print("\n=== Test 2: Smoothness Regularizer ===")
    smoothness_reg = GraphSmoothnessRegularizer()
    smoothness_loss = smoothness_reg(S_pred, L)
    print(f"Smoothness loss: {smoothness_loss.item():.6f}")
    
    # Test with smooth data (should be lower)
    S_smooth = torch.linspace(0, 1, N).unsqueeze(0).unsqueeze(-1).expand(B, N, d)
    smoothness_loss_smooth = smoothness_reg(S_smooth, L)
    print(f"Smoothness loss (smooth data): {smoothness_loss_smooth.item():.6f}")
    
    # Test 3: Alignment Regularizer
    print("\n=== Test 3: Alignment Regularizer ===")
    alignment_reg = AlignmentRegularizer(use_cosine_similarity=True)
    alignment_loss = alignment_reg(S_pred, S_ref)
    print(f"Alignment loss: {alignment_loss.item():.6f}")
    
    # Perfect alignment (should be ~0)
    alignment_loss_perfect = alignment_reg(S_pred, S_pred)
    print(f"Alignment loss (perfect): {alignment_loss_perfect.item():.6f}")
    
    # Test 4: Combined Regularizer
    print("\n=== Test 4: Combined Regularizer ===")
    combined_reg = CombinedPhysicsRegularizer(
        lambda_energy=0.1,
        lambda_smoothness=0.2,
        lambda_alignment=0.05,
        lambda_divergence=0.0
    )
    
    total_reg, reg_dict = combined_reg(
        S_pred=S_pred,
        L=L,
        S_reference=S_ref,
        adjacency=A
    )
    
    print(f"Total regularization: {total_reg.item():.6f}")
    print("Components:")
    for k, v in reg_dict.items():
        print(f"  {k}: {v:.6f}")
    
    # Test 5: Gradient flow
    print("\n=== Test 5: Gradient Flow ===")
    S_pred_grad = torch.randn(B, N, d, requires_grad=True)
    total_reg, _ = combined_reg(S_pred_grad, L, S_ref, A)
    total_reg.backward()
    
    print(f"Gradient norm: {S_pred_grad.grad.norm().item():.6f}")
    print(f"Gradient mean: {S_pred_grad.grad.mean().item():.6f}")
    
    print("\n✓ Regularizers test passed!")
