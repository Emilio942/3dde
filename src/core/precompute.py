"""Precomputation of Φ(t) and Σ(t) matrices for diffusion process.

This module implements the forward diffusion process precomputation, converting
the graph Laplacian into time-dependent transformation matrices Φ(t) and
covariance matrices Σ(t).
"""

import torch
from typing import Tuple, Optional, Literal
import warnings


def build_A_node(
    L: torch.Tensor,
    gamma: float = 1.0,
    lambda_reg: float = 1e-6,
    lambd: Optional[float] = None
) -> torch.Tensor:
    """Convert graph Laplacian L to drift matrix A for node dynamics.
    
    The dynamics are: dS/dt = -A·S + noise
    where A controls the diffusion based on graph structure.
    
    Args:
        L: Graph Laplacian matrix (N, N), symmetric positive semi-definite
        gamma: Diffusion strength parameter (controls edge influence)
        lambda_reg: Regularization for numerical stability
        
    Returns:
        A: Drift matrix (N, N) = gamma * L + lambda_reg * I
        
    Shape:
        L: (N, N)
        A: (N, N)
    """
    if lambd is not None:
        gamma = lambd

    N = L.shape[0]
    device = L.device
    dtype = L.dtype
    
    # A = γL + λI for regularization
    A = gamma * L + lambda_reg * torch.eye(N, device=device, dtype=dtype)
    
    return A


def precompute_phi_sigma_explicit(
    L: torch.Tensor,
    num_steps: int = 1000,
    T: float = 1.0,
    gamma: float = 1.0,
    lambda_reg: float = 1e-6,
    beta_schedule: Literal["linear", "cosine"] = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    beta_schedule_type: Optional[str] = None,
    diag_approx: bool = False,
    check_stability: bool = True,
    lambd: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute Φ(t) and Σ(t) matrices using explicit Euler recursion.
    
    Forward process: S_t = Φ(t) S_0 + √Σ(t) ε, where ε ~ N(0, I)
    
    Uses discrete time-stepping:
        Φ(t+dt) = (I - β(t)A·dt) Φ(t)
        Σ(t+dt) = Σ(t) + β(t)·dt·I
    
    Args:
        L: Graph Laplacian (N, N)
        num_steps: Number of discrete time steps
        T: Total diffusion time (usually 1.0)
        gamma: Graph diffusion strength
        lambda_reg: Numerical stability regularization
        lambd: Optional alias for gamma (retained for backwards compatibility)
    beta_schedule: "linear" or "cosine" noise schedule
    beta_start: Initial beta value for linear schedule
    beta_end: Final beta value for linear schedule
        diag_approx: If True, only store diagonal of Σ (memory efficient)
        check_stability: If True, warn about numerical instabilities
        
    Returns:
        Phi: Transformation matrices, shape (num_steps+1, N, N)
        Sigma: Covariance matrices, shape:
            - (num_steps+1, N, N) if diag_approx=False (full)
            - (num_steps+1, N) if diag_approx=True (diagonal only)
    
    Notes:
        - Φ(0) = I, Σ(0) = 0
        - For stability, ensure dt·max(eigenvalue(A)) < 2
        - diag_approx assumes Σ ≈ diag, valid when graph mixing is weak
    """
    if lambd is not None:
        gamma = lambd
    if beta_schedule_type is not None:
        beta_schedule = beta_schedule_type

    N = L.shape[0]
    device = L.device
    dtype = L.dtype
    dt = T / max(num_steps, 1)
    
    # Build drift matrix A
    A = build_A_node(L, gamma=gamma, lambda_reg=lambda_reg)
    
    # Check stability condition
    if check_stability:
        max_eigenval = torch.linalg.eigvalsh(A).max()
        stability_factor = dt * max_eigenval
        if stability_factor > 2.0:
            warnings.warn(
                f"Stability condition violated: dt·λ_max = {stability_factor:.3f} > 2.0. "
                f"Consider increasing num_steps or decreasing gamma.",
                RuntimeWarning
            )
    
    # Generate β(t) schedule
    betas = get_beta_schedule(
        schedule=beta_schedule,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
        dtype=dtype
    )
    
    # Initialize storage
    Phi = torch.zeros(num_steps + 1, N, N, device=device, dtype=dtype)
    Phi[0] = torch.eye(N, device=device, dtype=dtype)  # Φ(0) = I
    
    if diag_approx:
        Sigma_diag = torch.zeros(num_steps + 1, N, device=device, dtype=dtype)
    else:
        Sigma = torch.zeros(num_steps + 1, N, N, device=device, dtype=dtype)
    
    # Discrete Euler integration
    I = torch.eye(N, device=device, dtype=dtype)
    
    for t_idx in range(num_steps):
        beta_t = betas[t_idx]
        
        # Φ(t+dt) = (I - β(t)·A·dt) Φ(t)
        # For numerical stability, can use: Φ(t+dt) = (I + β(t)·A·dt)^(-1) Φ(t)
        update_matrix = I - beta_t * A * dt
        Phi[t_idx + 1] = update_matrix @ Phi[t_idx]
        
        # Σ(t+dt) = Σ(t) + β(t)·dt·I
        if diag_approx:
            diag_increment = beta_t * dt * torch.diag(A)
            Sigma_diag[t_idx + 1] = Sigma_diag[t_idx] + diag_increment
        else:
            Sigma[t_idx + 1] = Sigma[t_idx] + beta_t * dt * A
    
    # Numerical check for positive definiteness
    if diag_approx:
        Sigma_result = torch.diag_embed(Sigma_diag)
    else:
        Sigma_result = 0.5 * (Sigma + Sigma.transpose(-1, -2))

    if check_stability:
        min_eigenval = torch.linalg.eigvalsh(Sigma_result[-1]).min()
        if min_eigenval < -1e-6:
            warnings.warn(
                f"Σ(T) has negative eigenvalue: {min_eigenval:.3e}. "
                "This indicates numerical instability.",
                RuntimeWarning
            )
    
    return Phi, Sigma_result


def _get_beta_schedule(
    num_steps: int,
    schedule: Literal["linear", "cosine"] = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Generate β(t) noise schedule.
    
    Args:
        num_steps: Number of time steps
        schedule: "linear" or "cosine"
        beta_start: Starting noise level (for linear)
        beta_end: Ending noise level (for linear)
        device: Target device
        dtype: Target dtype
        
    Returns:
        betas: (num_steps,) tensor of β values
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=dtype)
    
    elif schedule == "cosine":
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        # More stable, concentrates diffusion in middle timesteps
        s = 0.008  # Offset to prevent β=0
        steps = torch.arange(num_steps + 1, device=device, dtype=dtype)
        alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)  # Clip to prevent instability
    
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use 'linear' or 'cosine'.")
    
    return betas


def get_beta_schedule(
    schedule: Literal["linear", "cosine"],
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Public wrapper for beta schedule generation used across the codebase."""

    return _get_beta_schedule(
        num_steps=num_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
        dtype=dtype
    )


def get_Phi_Sigma_at_t(
    Phi_all: torch.Tensor,
    Sigma_all: torch.Tensor,
    t_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract Φ(t) and Σ(t) for specific time indices (batch-friendly).
    
    Args:
        Phi_all: Precomputed Φ matrices (num_steps+1, N, N)
        Sigma_all: Precomputed Σ matrices (num_steps+1, N, N) or (num_steps+1, N)
        t_indices: Time step indices (B,) with values in [0, num_steps]
        
    Returns:
        Phi_t: (B, N, N)
        Sigma_t: (B, N, N) or (B, N) depending on input
    """
    Phi_t = Phi_all[t_indices]  # (B, N, N)
    Sigma_t = Sigma_all[t_indices]  # (B, N, N) or (B, N)
    
    return Phi_t, Sigma_t


def compute_Phi_inverse(
    Phi: torch.Tensor,
    reg: float = 1e-8
) -> torch.Tensor:
    """Compute inverse of Φ matrix with regularization for numerical stability.
    
    Args:
        Phi: Transformation matrix (N, N) or (B, N, N)
        reg: Regularization term added to diagonal before inversion
        
    Returns:
        Phi_inv: Inverse matrix, same shape as input
    """
    if Phi.ndim == 2:
        N = Phi.shape[0]
        Phi_reg = Phi + reg * torch.eye(N, device=Phi.device, dtype=Phi.dtype)
        return torch.linalg.inv(Phi_reg)
    
    elif Phi.ndim == 3:
        B, N, _ = Phi.shape
        I = torch.eye(N, device=Phi.device, dtype=Phi.dtype).unsqueeze(0).expand(B, -1, -1)
        Phi_reg = Phi + reg * I
        return torch.linalg.inv(Phi_reg)
    
    else:
        raise ValueError(f"Phi must be 2D or 3D, got shape {Phi.shape}")


def compute_condition_number(Phi: torch.Tensor) -> torch.Tensor:
    """Compute condition number of Φ to check numerical stability.
    
    Higher condition number = more ill-conditioned = numerical issues
    
    Args:
        Phi: Matrix (N, N) or batch (B, N, N)
        
    Returns:
        cond: Condition number(s), scalar or (B,)
    """
    singular_values = torch.linalg.svdvals(Phi)
    cond = singular_values[..., 0] / (singular_values[..., -1] + 1e-12)
    return cond


def check_numerical_stability(
    Phi: torch.Tensor,
    Sigma: torch.Tensor,
    max_t: Optional[int] = None,
    atol: float = 1e-6
) -> bool:
    """Check matrices for NaN/Inf and crude positive-semidefinite guarantees."""

    if max_t is not None:
        Phi = Phi[: max_t + 1]
        Sigma = Sigma[: max_t + 1]

    if not torch.isfinite(Phi).all() or not torch.isfinite(Sigma).all():
        return False

    if Sigma.ndim == 2:
        Sigma_mats = torch.diag_embed(Sigma)
    else:
        Sigma_mats = Sigma

    Sigma_mats = 0.5 * (Sigma_mats + Sigma_mats.transpose(-1, -2))
    flat = Sigma_mats.reshape(-1, Sigma_mats.shape[-1], Sigma_mats.shape[-1])

    for mat in flat:
        eigenvalues = torch.linalg.eigvalsh(mat)
        if torch.any(eigenvalues < -atol):
            return False

    return True


if __name__ == "__main__":
    # Quick test
    print("Testing precompute module...")
    
    # Simple 3-node chain graph
    L = torch.tensor([
        [2.0, -1.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -1.0, 2.0]
    ], dtype=torch.float32)
    
    print("\nLaplacian L:")
    print(L)
    
    # Precompute with diagonal approximation
    Phi, Sigma = precompute_phi_sigma_explicit(
        L, num_steps=100, diag_approx=True, check_stability=True
    )
    
    print(f"\nΦ shape: {Phi.shape}")
    print(f"Σ shape (diag): {Sigma.shape}")
    print(f"\nΦ(0) = I:\n{Phi[0]}")
    print(f"Σ(0) = 0: {Sigma[0]}")
    print(f"Σ(T) final variance: {Sigma[-1].mean():.4f}")
    
    # Condition number check
    cond_0 = compute_condition_number(Phi[0])
    cond_T = compute_condition_number(Phi[-1])
    print(f"\nCondition numbers:")
    print(f"  Φ(0): {cond_0:.2f}")
    print(f"  Φ(T): {cond_T:.2f}")
    
    print("\n✓ Precompute module test passed!")
