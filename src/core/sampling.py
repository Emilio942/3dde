"""Reverse sampling process for generating samples from noise.

Implements the reverse diffusion process:
    S_{t-1} = F(t) S_t + Q(t)^(1/2) z
where z ~ N(0, I)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .forward import (
    apply_Phi_node_batch,
    apply_Sigma_sqrt_node_batch,
    compute_hat_S0_from_eps_hat,
)
from .precompute import compute_Phi_inverse


def compute_F_Q_from_PhiSigma(
    Phi: torch.Tensor,
    Sigma: torch.Tensor,
    reg: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute reverse process transition matrices F(t) and Q(t)."""

    num_steps = Phi.shape[0] - 1
    N = Phi.shape[1]
    device = Phi.device
    dtype = Phi.dtype

    F = torch.zeros_like(Phi)
    F[0] = torch.eye(N, device=device, dtype=dtype)

    if Sigma.ndim == 2:
        Sigma_work = torch.diag_embed(Sigma)
    else:
        Sigma_work = Sigma.clone()

    Q = torch.zeros_like(Sigma_work)
    Q[0] = Sigma_work[0]

    for t in range(1, num_steps + 1):
        Phi_t = Phi[t]
        Phi_prev = Phi[t - 1]
        Phi_t_inv = compute_Phi_inverse(Phi_t, reg=reg)
        F[t] = Phi_prev @ Phi_t_inv

        Sigma_t = Sigma_work[t]
        Sigma_prev = Sigma_work[t - 1]
        proposal = Sigma_prev - F[t] @ Sigma_t @ F[t].transpose(-1, -2)
        Q[t] = _ensure_psd(proposal, eps=reg)

    if Sigma.ndim == 2:
        return F, torch.diagonal(Q, dim1=-2, dim2=-1)

    return F, Q


def _ensure_psd(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project matrix onto the positive semi-definite cone via eigen clipping."""
    
    if matrix.is_sparse:
        matrix = matrix.to_dense()

    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(sym)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)


def one_step_reverse(
    S_t: torch.Tensor,
    eps_hat: torch.Tensor,
    t: int,
    F: torch.Tensor,
    Q: torch.Tensor,
    deterministic: bool = False
) -> torch.Tensor:
    """Perform a single reverse diffusion step from t to t-1."""

    if t <= 0:
        return S_t.clone()

    F_t = F[t]
    mean_state = apply_Phi_node_batch(S_t - eps_hat, F_t)

    if deterministic:
        return mean_state

    Q_t = Q[t]
    noise = torch.randn_like(S_t)
    noise_scaled = apply_Sigma_sqrt_node_batch(noise, Q_t)
    return mean_state + noise_scaled


def sample_reverse_from_S_T(
    S_T: torch.Tensor,
    F: torch.Tensor,
    Q: torch.Tensor,
    eps_model: nn.Module,
    L: torch.Tensor,
    num_steps: int,
    use_predictor_corrector: bool = False,
    corrector_steps: int = 1,
    adjacency: Optional[torch.Tensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample from S_T (pure noise) back to S_0 (clean data)."""

    if "num_corrector_steps" in kwargs:
        corrector_steps = kwargs.pop("num_corrector_steps")
    if kwargs:
        unexpected = ", ".join(kwargs.keys())
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    B, N, D = S_T.shape
    device = S_T.device
    dtype = S_T.dtype

    Q_full = Q if Q.ndim == 3 else torch.diag_embed(Q)

    trajectory = torch.zeros(num_steps + 1, B, N, D, device=device, dtype=dtype)
    trajectory[num_steps] = S_T

    current = S_T.clone()

    eps_model.eval()
    with torch.no_grad():
        for step in range(num_steps, 0, -1):
            t_tensor = torch.full((B,), step, device=device, dtype=torch.long)
            eps_pred = eps_model(current, t_tensor, L, adjacency)

            current = one_step_reverse(
                S_t=current,
                eps_hat=eps_pred,
                t=step,
                F=F,
                Q=Q,
                deterministic=False
            )

            if use_predictor_corrector and step > 1 and corrector_steps > 0:
                current = langevin_corrector(
                    S_t=current,
                    t=step - 1,
                    eps_model=eps_model,
                    L=L,
                    Sigma=Q_full,
                    num_steps=corrector_steps,
                    step_size=0.001,
                    adjacency=adjacency
                )

            trajectory[step - 1] = current

    return trajectory[0], trajectory


def langevin_corrector(
    S_t: torch.Tensor,
    t: int,
    eps_model: nn.Module,
    L: torch.Tensor,
    Sigma: Optional[torch.Tensor] = None,
    num_steps: int = 5,
    step_size: float = 0.001,
    adjacency: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Apply Langevin dynamics correction for improved sampling quality.
    
    Performs: S_{k+1} = S_k + α ∇ log p(S_k | t) + √(2α) z
    
    Args:
        S_t: Current state (B, N, d)
        t: Current timestep
        eps_model: Epsilon prediction network
        L: Laplacian (N, N)
        Sigma: Optional covariance schedule for noise scaling
        num_steps: Number of Langevin steps
        step_size: Step size α
        adjacency: Optional adjacency (N, N)
        
    Returns:
    S_corrected: Corrected state (B, N, d)
    """
    if num_steps <= 0:
        return S_t

    B = S_t.shape[0]
    device = S_t.device
    dtype = S_t.dtype

    Sigma_source = Sigma if Sigma is not None else None
    noise_scale = torch.sqrt(torch.tensor(2.0 * step_size, device=device, dtype=dtype))

    result = S_t.clone()
    for _ in range(num_steps):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        score = -eps_model(result, t_tensor, L, adjacency)

        noise = torch.randn_like(result)
        if Sigma_source is not None:
            covariance = Sigma_source[t]
            noise = apply_Sigma_sqrt_node_batch(noise, covariance)

        noise = noise * noise_scale
        result = result + step_size * score + noise

    return result


def ddim_sampling(
    S_T: torch.Tensor,
    Phi: torch.Tensor,
    Sigma: torch.Tensor,
    eps_model: nn.Module,
    L: torch.Tensor,
    num_steps: int,
    eta: float = 0.0,
    adjacency: Optional[torch.Tensor] = None,
    num_sample_steps: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DDIM sampling returning both final sample and trajectory."""

    total_steps = Phi.shape[0] - 1
    sample_steps = num_sample_steps or num_steps

    step_indices = torch.linspace(0, total_steps, steps=sample_steps + 1)
    step_indices = torch.round(step_indices).clamp(0, total_steps).to(dtype=torch.long)

    B, N, D = S_T.shape
    device = S_T.device
    dtype = S_T.dtype

    trajectory = torch.zeros(sample_steps + 1, B, N, D, device=device, dtype=dtype)
    trajectory[sample_steps] = S_T

    current = S_T.clone()

    eps_model.eval()
    with torch.no_grad():
        for idx in range(sample_steps, 0, -1):
            t_curr = int(step_indices[idx].item())
            t_prev = int(step_indices[idx - 1].item())

            t_tensor = torch.full((B,), t_curr, device=device, dtype=torch.long)
            eps_pred = eps_model(current, t_tensor, L, adjacency)

            Phi_curr = Phi[t_curr]
            Sigma_curr = Sigma[t_curr]
            S_0_pred = compute_hat_S0_from_eps_hat(current, eps_pred, Phi_curr, Sigma_curr)

            Phi_prev = Phi[t_prev]
            next_state = apply_Phi_node_batch(S_0_pred, Phi_prev)

            if eta > 0 and idx > 1:
                Sigma_prev = Sigma[t_prev]
                noise = torch.randn_like(current)
                noise = apply_Sigma_sqrt_node_batch(noise, Sigma_prev)
                next_state = next_state + eta * noise

            current = next_state
            trajectory[idx - 1] = current

    return current, trajectory


if __name__ == "__main__":
    print("Testing sampling module...")
    
    # This is a minimal test without trained model
    # Full test requires trained eps_model
    
    from precompute import precompute_phi_sigma_explicit
    from ..data.graph_builder import build_grid_laplacian
    
    # Setup
    N = 25  # 5x5 grid
    d = 3
    num_steps = 100
    
    L = build_grid_laplacian((5, 5))
    Phi, Sigma = precompute_phi_sigma_explicit(L, num_steps=num_steps, diag_approx=True)
    
    print(f"\nTest configuration:")
    print(f"  Nodes: {N}, Dimensions: {d}, Steps: {num_steps}")
    
    # Test F and Q computation
    print("\n=== Test: F and Q computation ===")
    F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
    
    print(f"F shape: {F.shape}")
    print(f"Q shape: {Q.shape}")
    print(f"F[0] determinant: {torch.linalg.det(F[0]):.4f}")
    print(f"Q[0] min/max: {Q[0].min():.6f} / {Q[0].max():.6f}")
    
    # Check Q is positive
    assert (Q >= -1e-6).all(), "Q should be positive"
    
    print("\n✓ Sampling module test passed!")
    print("  Note: Full sampling test requires trained model")
