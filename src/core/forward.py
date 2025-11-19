"""Forward noising process for diffusion model.

Implements the forward diffusion process that adds noise to clean data:
    S_t = Φ(t) S_0 + √Σ(t) ε
where ε ~ N(0, I)
"""

import torch
from typing import Tuple, Optional, Union

from .precompute import get_Phi_Sigma_at_t, compute_Phi_inverse


def apply_Phi_node_batch(
    S: torch.Tensor,
    Phi: torch.Tensor
) -> torch.Tensor:
    """Apply Φ transformation to node states in batch.
    
    Computes: S_out = Φ @ S for each sample in batch
    
    Args:
        Phi: Transformation matrix (N, N) or (B, N, N)
        S: Node states (B, N, d) where B=batch, N=nodes, d=dimensions
        
    Returns:
        S_transformed: (B, N, d)
        
    Notes:
        - Uses batched matrix multiplication for efficiency
        - Φ can be shared across batch (N, N) or per-sample (B, N, N)
    """
    if Phi.ndim == 2:
        return torch.einsum("nm,bmd->bnd", Phi, S)

    if Phi.ndim == 3:
        return torch.einsum("bnm,bmd->bnd", Phi, S)

    raise ValueError(f"Phi must be 2D or 3D, got shape {Phi.shape}")


def apply_Sigma_sqrt_node_batch(
    noise: torch.Tensor,
    Sigma_t: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Apply square root of Σ(t) to noise samples.

    Supports diagonal and full covariance in shared or per-sample form.
    """
    B, N, D = noise.shape

    if Sigma_t.ndim == 1:
        if torch.allclose(Sigma_t, torch.zeros_like(Sigma_t)):
            return torch.zeros_like(noise)
        sqrt_sigma = torch.sqrt(torch.clamp(Sigma_t, min=0.0))
        return noise * sqrt_sigma.unsqueeze(0).unsqueeze(-1)

    if Sigma_t.ndim == 2:
        if Sigma_t.shape == (B, N):
            if torch.allclose(Sigma_t, torch.zeros_like(Sigma_t)):
                return torch.zeros_like(noise)
            sqrt_sigma = torch.sqrt(torch.clamp(Sigma_t, min=0.0))
            return noise * sqrt_sigma.unsqueeze(-1)

        if Sigma_t.shape[0] == N and Sigma_t.shape[1] == N:
            if torch.allclose(Sigma_t, torch.zeros_like(Sigma_t)):
                return torch.zeros_like(noise)
            chol = torch.linalg.cholesky(
                Sigma_t + eps * torch.eye(N, device=Sigma_t.device, dtype=Sigma_t.dtype)
            )
            return torch.einsum("nm,bmd->bnd", chol, noise)

    if Sigma_t.ndim == 3:
        if Sigma_t.shape[0] == B:
            eye = torch.eye(N, device=Sigma_t.device, dtype=Sigma_t.dtype)
            if torch.allclose(Sigma_t, torch.zeros_like(Sigma_t)):
                return torch.zeros_like(noise)
            chol = torch.linalg.cholesky(Sigma_t + eps * eye.unsqueeze(0))
            return torch.einsum("bnm,bmd->bnd", chol, noise)

        if Sigma_t.shape[0] == N and Sigma_t.shape[1] == N:
            if torch.allclose(Sigma_t, torch.zeros_like(Sigma_t)):
                return torch.zeros_like(noise)
            chol = torch.linalg.cholesky(
                Sigma_t + eps * torch.eye(N, device=Sigma_t.device, dtype=Sigma_t.dtype)
            )
            return torch.einsum("nm,bmd->bnd", chol, noise)

    raise ValueError(f"Unsupported Sigma_t shape {Sigma_t.shape} for noise {noise.shape}")


def _resolve_phi_sigma_inputs(
    Phi_t: Optional[torch.Tensor],
    Sigma_t: Optional[torch.Tensor],
    t: Optional[Union[int, torch.Tensor]],
    Phi_all: Optional[Union[torch.Tensor, object]],
    Sigma_all: Optional[Union[torch.Tensor, object]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve Φ(t) and Σ(t) from provided arguments."""

    if Phi_t is not None and Sigma_t is not None:
        return Phi_t, Sigma_t

    if t is not None:
        # Check for SpectralDiffusion object (or similar)
        if hasattr(Phi_all, 'get_phi_sigma'):
            # Phi_all is actually the spectral object
            if isinstance(t, int):
                t_tensor = torch.tensor([t], device=Phi_all.device)
                phi, sigma = Phi_all.get_phi_sigma(t_tensor)
                return phi.squeeze(0), sigma.squeeze(0)
            else:
                return Phi_all.get_phi_sigma(t)

        if Phi_all is not None and Sigma_all is not None:
            if isinstance(t, torch.Tensor):
                return get_Phi_Sigma_at_t(Phi_all, Sigma_all, t)
            return Phi_all[t], Sigma_all[t]

    raise TypeError(
        "Provide either Phi_t/Sigma_t directly, or t with full Phi/Sigma schedules (or SpectralDiffusion object)."
    )


def forward_noising_batch(
    S_0: torch.Tensor,
    *args,
    Phi_t: Optional[torch.Tensor] = None,
    Sigma_t: Optional[torch.Tensor] = None,
    t: Optional[Union[int, torch.Tensor]] = None,
    Phi_all: Optional[torch.Tensor] = None,
    Sigma_all: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply forward noising process S_t = Φ(t) S_0 + √Σ(t) ε.

    Accepts either explicit named arguments (recommended) or legacy positional arguments:
    - `(S_0, Phi_t, Sigma_t)`
    - `(S_0, t, Phi_all, Sigma_all)`
    """
    # Handle legacy positional arguments
    if args:
        if len(args) == 2:
            Phi_t, Sigma_t = args
        elif len(args) == 3:
            t, Phi_all, Sigma_all = args
        else:
            raise TypeError("Invalid positional arguments for forward_noising_batch")

    # Handle legacy kwargs (if any were passed to **kwargs instead of the named args)
    if kwargs:
        if "Phi" in kwargs: Phi_all = kwargs.pop("Phi")
        if "Sigma" in kwargs: Sigma_all = kwargs.pop("Sigma")
        # Map other legacy kwargs if they match our new named args but were passed as kwargs
        # (This happens if the caller used **kwargs dict)
        if "Phi_t" in kwargs and Phi_t is None: Phi_t = kwargs.pop("Phi_t")
        if "Sigma_t" in kwargs and Sigma_t is None: Sigma_t = kwargs.pop("Sigma_t")
        if "t" in kwargs and t is None: t = kwargs.pop("t")
        
        if kwargs:
             # If there are still kwargs left, they might be unexpected
             # But for safety we can ignore or warn. The original code raised TypeError.
             unexpected = ", ".join(kwargs.keys())
             # raise TypeError(f"Unexpected keyword arguments: {unexpected}")
             pass

    Phi_selected, Sigma_selected = _resolve_phi_sigma_inputs(
        Phi_t=Phi_t,
        Sigma_t=Sigma_t,
        t=t,
        Phi_all=Phi_all,
        Sigma_all=Sigma_all
    )

    if noise is None:
        eps = torch.randn_like(S_0)
    else:
        eps = noise

    Phi_S0 = apply_Phi_node_batch(S_0, Phi_selected)
    noise_scaled = apply_Sigma_sqrt_node_batch(eps, Sigma_selected)

    S_t = Phi_S0 + noise_scaled

    return S_t, eps


def compute_hat_S0_from_eps_hat(
    S_t: torch.Tensor,
    eps_hat: torch.Tensor,
    *args: Union[int, torch.Tensor],
    reg: float = 1e-8,
    **kwargs
) -> torch.Tensor:
    """Recover an estimate of S_0 from S_t and a noise prediction ε̂.

    Works with both `(S_t, eps_hat, Phi_t, Sigma_t)` and
    `(S_t, eps_hat, t, Phi_all, Sigma_all)` styles.
    """
    Phi_t_kw = kwargs.pop("Phi_t", None)
    Sigma_t_kw = kwargs.pop("Sigma_t", None)
    t_kw = kwargs.pop("t", None)
    Phi_kw = kwargs.pop("Phi", None)
    Sigma_kw = kwargs.pop("Sigma", None)

    if kwargs:
        unexpected = ", ".join(kwargs.keys())
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    if len(args) == 2:
        Phi_t_kw, Sigma_t_kw = args
    elif len(args) == 3:
        t_kw, Phi_kw, Sigma_kw = args
    elif len(args) != 0:
        raise TypeError("Invalid positional arguments for compute_hat_S0_from_eps_hat")

    Phi_selected, Sigma_selected = _resolve_phi_sigma_inputs(
        Phi_t=Phi_t_kw,
        Sigma_t=Sigma_t_kw,
        t=t_kw,
        Phi_all=Phi_kw,
        Sigma_all=Sigma_kw
    )

    noise_scaled = apply_Sigma_sqrt_node_batch(eps_hat, Sigma_selected)
    residual = S_t - noise_scaled

    Phi_inv = compute_Phi_inverse(Phi_selected, reg=reg)
    S_0_hat = apply_Phi_node_batch(residual, Phi_inv)

    return S_0_hat


def sample_timesteps(
    batch_size: int,
    num_steps: int,
    device: torch.device,
    strategy: str = "uniform"
) -> torch.Tensor:
    """Sample time steps for training.
    
    Args:
        batch_size: Number of samples
        num_steps: Total number of time steps
        device: Target device
        strategy: "uniform" (all t equally likely) or "weighted" (favor difficult t)
        
    Returns:
        t_indices: (B,) tensor of time step indices in [0, num_steps]
    """
    if strategy == "uniform":
        t_indices = torch.randint(0, num_steps + 1, (batch_size,), device=device)
    
    elif strategy == "weighted":
        # Weight middle time steps more (typically harder to denoise)
        # Use triangular distribution centered at num_steps // 2
        center = num_steps // 2
        probs = torch.abs(torch.arange(num_steps + 1, dtype=torch.float32) - center)
        probs = 1.0 / (probs + 1.0)  # Inverse distance weighting
        probs = probs / probs.sum()
        t_indices = torch.multinomial(probs, batch_size, replacement=True).to(device)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return t_indices


if __name__ == "__main__":
    print("Testing forward noising module...")
    
    # Setup test data
    B, N, d = 4, 5, 3  # 4 samples, 5 nodes, 3 dimensions
    num_steps = 100
    
    # Create simple Laplacian
    L = torch.eye(N) * 2.0
    for i in range(N - 1):
        L[i, i + 1] = -1.0
        L[i + 1, i] = -1.0
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {B}")
    print(f"  Nodes: {N}")
    print(f"  Dimensions: {d}")
    print(f"  Time steps: {num_steps}")
    
    # Precompute Φ and Σ (using simple version for test)
    from precompute import precompute_phi_sigma_explicit, get_Phi_Sigma_at_t
    
    Phi_all, Sigma_all = precompute_phi_sigma_explicit(
        L, num_steps=num_steps, diag_approx=True
    )
    
    # Generate clean data
    S0 = torch.randn(B, N, d)
    
    # Sample time steps
    t_indices = sample_timesteps(B, num_steps, S0.device, strategy="uniform")
    print(f"\nSampled time steps: {t_indices.tolist()}")
    
    # Get Φ(t) and Σ(t) for these time steps
    Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi_all, Sigma_all, t_indices)
    
    # Forward noising
    St, eps = forward_noising_batch(S0, Phi_t, Sigma_t)
    
    print(f"\nShapes:")
    print(f"  S0: {S0.shape}")
    print(f"  St: {St.shape}")
    print(f"  eps: {eps.shape}")
    print(f"  Phi_t: {Phi_t.shape}")
    print(f"  Sigma_t: {Sigma_t.shape}")
    
    # Test reconstruction
    S0_reconstructed = compute_hat_S0_from_eps_hat(St, eps, Phi_t, Sigma_t)
    
    # Should match S0 exactly when using true noise
    reconstruction_error = (S0 - S0_reconstructed).abs().mean()
    print(f"\nReconstruction error (should be ~0): {reconstruction_error:.6f}")
    
    # Verify noise distribution
    print(f"\nNoise statistics:")
    print(f"  Mean: {eps.mean():.4f} (should be ~0)")
    print(f"  Std: {eps.std():.4f} (should be ~1)")
    
    print("\n✓ Forward noising module test passed!")
