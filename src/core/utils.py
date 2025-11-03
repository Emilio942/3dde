"""Utility functions for diffusion model operations."""

import torch
import numpy as np
from typing import Optional, Union, Tuple


def normalize_states(
    S: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize node states to zero mean and unit variance.
    
    Args:
        S: States (B, N, d) or (N, d)
        mean: Optional pre-computed mean (N, d) or (d,)
        std: Optional pre-computed std (N, d) or (d,)
        eps: Small constant for numerical stability
        
    Returns:
        S_normalized: Normalized states
        mean: Computed or provided mean
        std: Computed or provided std
    """
    if mean is None:
        if S.ndim == 3:
            mean = S.mean(dim=0, keepdim=True)  # (1, N, d)
        else:
            mean = S.mean(dim=0, keepdim=True)  # (1, d) or (1, N, d)
    
    if std is None:
        if S.ndim == 3:
            std = S.std(dim=0, keepdim=True) + eps
        else:
            std = S.std(dim=0, keepdim=True) + eps
    
    S_normalized = (S - mean) / std
    
    return S_normalized, mean.squeeze(0), std.squeeze(0)


def denormalize_states(
    S_normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    """Reverse normalization.
    
    Args:
        S_normalized: Normalized states
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        S: Original scale states
    """
    if S_normalized.ndim == 3 and mean.ndim == 2:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    elif S_normalized.ndim == 2 and mean.ndim == 1:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return S_normalized * std + mean


def batch_matrix_sqrt(A: torch.Tensor, reg: float = 1e-8) -> torch.Tensor:
    """Compute matrix square root A^(1/2) for batch of matrices.
    
    Uses eigendecomposition: A = Q Λ Q^T => A^(1/2) = Q Λ^(1/2) Q^T
    
    Args:
        A: Symmetric positive definite matrices (B, N, N) or (N, N)
        reg: Regularization for numerical stability
        
    Returns:
        sqrt_A: Matrix square root, same shape as A
    """
    if A.ndim == 2:
        A = A.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    
    # Ensure positive eigenvalues
    eigenvalues = torch.clamp(eigenvalues, min=reg)
    
    # Compute sqrt of eigenvalues
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    
    # Reconstruct: Q Λ^(1/2) Q^T
    sqrt_A = eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    if squeeze_output:
        sqrt_A = sqrt_A.squeeze(0)
    
    return sqrt_A


def check_symmetric(A: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if matrix is symmetric within tolerance.
    
    Args:
        A: Matrix (N, N) or batch (B, N, N)
        tol: Tolerance for symmetry check
        
    Returns:
        is_symmetric: Boolean
    """
    diff = (A - A.transpose(-2, -1)).abs().max()
    return diff.item() < tol


def check_positive_definite(A: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if matrix is positive definite.
    
    Args:
        A: Symmetric matrix (N, N) or batch (B, N, N)
        tol: Minimum eigenvalue threshold
        
    Returns:
        is_pd: Boolean
    """
    eigenvalues = torch.linalg.eigvalsh(A)
    return eigenvalues.min().item() > tol


def exponential_moving_average(
    model_params: torch.nn.Parameter,
    ema_params: torch.Tensor,
    decay: float = 0.999
) -> torch.Tensor:
    """Update EMA parameters for stable model averaging.
    
    EMA: θ_ema = decay * θ_ema + (1 - decay) * θ_model
    
    Args:
        model_params: Current model parameters
        ema_params: EMA parameters (updated in-place)
        decay: EMA decay rate (typically 0.999 or 0.9999)
        
    Returns:
        ema_params: Updated EMA parameters
    """
    with torch.no_grad():
        ema_params.mul_(decay).add_(model_params.data, alpha=1 - decay)
    return ema_params


def cosine_schedule_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr_max: float,
    lr_min: float = 0.0
) -> float:
    """Cosine learning rate schedule with linear warmup.
    
    Args:
        step: Current training step
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        lr_max: Maximum learning rate
        lr_min: Minimum learning rate
        
    Returns:
        lr: Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return lr_max * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        num_params: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradient_norm(model: torch.nn.Module) -> float:
    """Compute total gradient norm across all parameters.
    
    Args:
        model: PyTorch model with gradients
        
    Returns:
        total_norm: L2 norm of all gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
):
    """Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Save path
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], dict]:
    """Load training checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        path: Checkpoint path
        device: Device to load to
        
    Returns:
        model: Model with loaded weights
        optimizer: Optimizer with loaded state (or None)
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint


if __name__ == "__main__":
    print("Testing utils module...")
    
    # Test normalization
    S = torch.randn(10, 5, 3)  # 10 samples, 5 nodes, 3 dims
    S_norm, mean, std = normalize_states(S)
    S_denorm = denormalize_states(S_norm, mean, std)
    
    print(f"Normalization test:")
    print(f"  Original mean: {S.mean():.4f}, std: {S.std():.4f}")
    print(f"  Normalized mean: {S_norm.mean():.4f}, std: {S_norm.std():.4f}")
    print(f"  Reconstruction error: {(S - S_denorm).abs().mean():.6f}")
    
    # Test matrix sqrt
    A = torch.randn(5, 5)
    A = A @ A.T  # Make positive definite
    sqrt_A = batch_matrix_sqrt(A)
    reconstructed = sqrt_A @ sqrt_A
    
    print(f"\nMatrix sqrt test:")
    print(f"  Is A symmetric? {check_symmetric(A)}")
    print(f"  Is A positive definite? {check_positive_definite(A)}")
    print(f"  Reconstruction error: {(A - reconstructed).abs().mean():.6f}")
    
    # Test learning rate schedule
    steps = [0, 10, 50, 100, 200]
    print(f"\nCosine schedule test (total=200, warmup=10, lr_max=1e-3):")
    for step in steps:
        lr = cosine_schedule_with_warmup(step, 200, 10, 1e-3)
        print(f"  Step {step:3d}: lr = {lr:.6f}")
    
    print("\n✓ Utils module test passed!")
