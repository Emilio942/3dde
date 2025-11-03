"""Loss functions for training the diffusion model."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EpsilonMSELoss(nn.Module):
    """MSE loss for epsilon prediction: ||ε - ε_θ(S_t, t)||²
    
    This is the standard loss for DDPM-style diffusion models.
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            eps_pred: Predicted noise (B, N, d)
            eps_true: True noise (B, N, d)
            mask: Optional mask for nodes (B, N, 1) or (B, N)
            
        Returns:
            loss: Scalar loss
        """
        # Compute squared error
        loss = (eps_pred - eps_true) ** 2
        
        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)  # (B, N, 1)
            loss = loss * mask
            
            # Adjust normalization for masked elements
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() * eps_pred.shape[-1])
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedEpsilonLoss(nn.Module):
    """Time-weighted epsilon loss: λ(t) ||ε - ε_θ(S_t, t)||²
    
    Weights different timesteps differently to balance training.
    """
    
    def __init__(
        self,
        num_steps: int,
        weight_schedule: str = "uniform",
        reduction: str = "mean"
    ):
        """
        Args:
            num_steps: Total number of diffusion steps
            weight_schedule: "uniform", "snr" (signal-to-noise ratio), or "t_weighted"
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.num_steps = num_steps
        self.reduction = reduction
        
        # Compute time weights
        if weight_schedule == "uniform":
            weights = torch.ones(num_steps + 1)
        elif weight_schedule == "snr":
            # Weight by 1 / SNR(t) to balance signal-to-noise
            # Higher noise => more weight
            t = torch.arange(num_steps + 1, dtype=torch.float32)
            weights = 1.0 + t / num_steps  # Increasing with time
        elif weight_schedule == "t_weighted":
            # Weight middle timesteps more
            t = torch.arange(num_steps + 1, dtype=torch.float32)
            weights = 1.0 - torch.abs(t / num_steps - 0.5) * 0.5
        else:
            raise ValueError(f"Unknown weight schedule: {weight_schedule}")
        
        self.register_buffer('weights', weights)
    
    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            eps_pred: Predicted noise (B, N, d)
            eps_true: True noise (B, N, d)
            t: Timestep indices (B,)
            
        Returns:
            loss: Scalar loss
        """
        # Compute MSE per sample
        mse = ((eps_pred - eps_true) ** 2).mean(dim=[1, 2])  # (B,)
        
        # Apply time weights
        time_weights = self.weights[t]  # (B,)
        weighted_loss = mse * time_weights
        
        # Reduction
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class DenoisingScoreMatchingLoss(nn.Module):
    """Denoising score matching loss for score-based models.
    
    Loss: ||s_θ(S_t, t) - ∇ log p(S_t | S_0)||²
    where s_θ is the score network.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        score_pred: torch.Tensor,
        S_t: torch.Tensor,
        S_0: torch.Tensor,
        Sigma_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            score_pred: Predicted score (B, N, d)
            S_t: Noised state (B, N, d)
            S_0: Clean state (B, N, d)
            Sigma_t: Covariance at time t (N, N) or (N,) or (B, N)
            
        Returns:
            loss: Scalar loss
        """
        # True score: ∇ log p(S_t | S_0) = -Σ^(-1) (S_t - Φ S_0)
        # Simplified for this implementation: -(S_t - S_0) / Sigma
        
        if Sigma_t.ndim == 1:
            # Diagonal, shared across batch
            inv_sigma = 1.0 / (Sigma_t + 1e-8)
            score_true = -(S_t - S_0) * inv_sigma.unsqueeze(0).unsqueeze(-1)
        elif Sigma_t.ndim == 2 and Sigma_t.shape[0] != S_t.shape[1]:
            # Diagonal, per batch
            inv_sigma = 1.0 / (Sigma_t + 1e-8)
            score_true = -(S_t - S_0) * inv_sigma.unsqueeze(-1)
        else:
            # Full covariance - use solve
            residual = S_t - S_0
            score_true = torch.zeros_like(residual)
            for d in range(residual.shape[-1]):
                score_true[:, :, d] = -torch.linalg.solve(
                    Sigma_t + 1e-8 * torch.eye(Sigma_t.shape[0], device=Sigma_t.device),
                    residual[:, :, d].T
                ).T
        
        # MSE between predicted and true score
        loss = (score_pred - score_true) ** 2
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """Combined loss with multiple components."""
    
    def __init__(
        self,
        eps_weight: float = 1.0,
        regularization_weight: float = 0.0
    ):
        """
        Args:
            eps_weight: Weight for epsilon MSE loss
            regularization_weight: Weight for physics regularization (if any)
        """
        super().__init__()
        self.eps_weight = eps_weight
        self.reg_weight = regularization_weight
        
        self.eps_loss = EpsilonMSELoss()
    
    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        S_0_pred: Optional[torch.Tensor] = None,
        S_0_true: Optional[torch.Tensor] = None,
        L: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            eps_pred: Predicted noise (B, N, d)
            eps_true: True noise (B, N, d)
            S_0_pred: Optional predicted clean state (B, N, d)
            S_0_true: Optional true clean state (B, N, d)
            L: Optional Laplacian for regularization (N, N)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Epsilon loss
        eps_loss_val = self.eps_loss(eps_pred, eps_true)
        total_loss = self.eps_weight * eps_loss_val
        
        loss_dict = {
            'eps_loss': eps_loss_val.item()
        }
        
        # Optional: Reconstruction loss on S_0
        if S_0_pred is not None and S_0_true is not None:
            s0_loss = ((S_0_pred - S_0_true) ** 2).mean()
            loss_dict['s0_loss'] = s0_loss.item()
            # Could add this to total_loss if desired
        
        # Optional: Physics regularization
        if self.reg_weight > 0 and S_0_pred is not None and L is not None:
            # Graph smoothness: ||L S_0||²
            reg_loss = 0.0
            for d in range(S_0_pred.shape[-1]):
                smoothed = torch.matmul(L, S_0_pred[:, :, d].T).T
                reg_loss += (smoothed ** 2).mean()
            reg_loss = reg_loss / S_0_pred.shape[-1]
            
            loss_dict['reg_loss'] = reg_loss.item()
            total_loss = total_loss + self.reg_weight * reg_loss
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Setup test data
    B, N, d = 4, 10, 3
    
    eps_pred = torch.randn(B, N, d)
    eps_true = torch.randn(B, N, d)
    t = torch.randint(0, 100, (B,))
    
    print(f"\nTest configuration:")
    print(f"  Batch: {B}, Nodes: {N}, Dimensions: {d}")
    
    # Test 1: EpsilonMSELoss
    print("\n=== Test 1: EpsilonMSELoss ===")
    loss_fn = EpsilonMSELoss()
    loss = loss_fn(eps_pred, eps_true)
    print(f"MSE Loss: {loss.item():.6f}")
    
    # Test with mask
    mask = torch.ones(B, N)
    mask[:, -2:] = 0  # Mask last 2 nodes
    loss_masked = loss_fn(eps_pred, eps_true, mask)
    print(f"Masked MSE Loss: {loss_masked.item():.6f}")
    
    # Test 2: WeightedEpsilonLoss
    print("\n=== Test 2: WeightedEpsilonLoss ===")
    weighted_loss_fn = WeightedEpsilonLoss(num_steps=100, weight_schedule="snr")
    loss_weighted = weighted_loss_fn(eps_pred, eps_true, t)
    print(f"Weighted Loss: {loss_weighted.item():.6f}")
    print(f"Time weights for t={t.tolist()}: {weighted_loss_fn.weights[t].tolist()}")
    
    # Test 3: CombinedLoss
    print("\n=== Test 3: CombinedLoss ===")
    combined_loss_fn = CombinedLoss(eps_weight=1.0, regularization_weight=0.1)
    
    # Create Laplacian
    L = torch.eye(N) * 2.0
    for i in range(N - 1):
        L[i, i + 1] = -1.0
        L[i + 1, i] = -1.0
    
    S_0_pred = torch.randn(B, N, d)
    S_0_true = torch.randn(B, N, d)
    
    total_loss, loss_dict = combined_loss_fn(
        eps_pred, eps_true,
        S_0_pred=S_0_pred,
        S_0_true=S_0_true,
        L=L
    )
    
    print(f"Total Loss: {total_loss.item():.6f}")
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    # Test 4: Gradient flow
    print("\n=== Test 4: Gradient Flow ===")
    eps_pred_grad = torch.randn(B, N, d, requires_grad=True)
    loss = loss_fn(eps_pred_grad, eps_true)
    loss.backward()
    
    print(f"Gradient norm: {eps_pred_grad.grad.norm().item():.6f}")
    print(f"Gradient mean: {eps_pred_grad.grad.mean().item():.6f}")
    
    print("\n✓ Loss functions test passed!")
