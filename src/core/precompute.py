"""Precomputation of Φ(t) and Σ(t) matrices for diffusion process.

This module implements the forward diffusion process precomputation, converting
the graph Laplacian into time-dependent transformation matrices Φ(t) and
covariance matrices Σ(t).
"""

import torch
from typing import Tuple, Optional, Literal
import warnings
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix


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
        L: Graph Laplacian matrix (N, N), symmetric positive semi-definite. Can be sparse or dense.
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
    if L.is_sparse:
        I = torch.sparse_coo_tensor(
            torch.arange(N, device=device).unsqueeze(0).repeat(2, 1),
            torch.ones(N, dtype=dtype, device=device),
            (N, N)
        )
        A = gamma * L + lambda_reg * I
    else:
        I = torch.eye(N, device=device, dtype=dtype)
        A = gamma * L + lambda_reg * I
    
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
    lambd: Optional[float] = None,
    noise_type: Literal["isotropic", "anisotropic"] = "anisotropic"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute Φ(t) and Σ(t) matrices using explicit Euler recursion.
    
    Forward process: S_t = Φ(t) S_0 + √Σ(t) ε, where ε ~ N(0, I)
    
    Uses discrete time-stepping:
        Φ(t+dt) = (I - β(t)A·dt) Φ(t)
        Σ(t+dt) = Σ(t) + β(t)·dt·Q
        where Q = I (isotropic) or Q = A (anisotropic)
    
    Args:
        L: Graph Laplacian (N, N). Can be sparse or dense.
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
        noise_type: "isotropic" (standard) or "anisotropic" (graph-dependent)
        
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
        if A.is_sparse:
            import numpy as np
            A_scipy = _torch_sparse_to_scipy_csr(A)
            # eigs returns largest magnitude by default, use which='LR' for largest real part
            # Use a fixed random seed for the starting vector for deterministic results
            rng = np.random.RandomState(0)
            v0 = rng.rand(A_scipy.shape[0])
            max_eigenval = eigs(A_scipy, k=1, which='LR', return_eigenvectors=False, v0=v0)[0].real
        else:
            max_eigenval = torch.linalg.eigvalsh(A).max().item()
            
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
        
        # update_matrix will be sparse if A is sparse
        update_matrix = I - beta_t * A * dt
        
        # Phi becomes dense here, but the multiplication is efficient (sparse @ dense)
        Phi[t_idx + 1] = update_matrix @ Phi[t_idx]
        
        # Σ(t+dt) = Σ(t) + β(t)·dt·Q
        if noise_type == "isotropic":
            noise_cov = I
        else:
            noise_cov = A

        if diag_approx:
            # A.diagonal() does not work for sparse, so convert to dense first for this part
            if noise_type == "isotropic":
                diag_increment = beta_t * dt * torch.ones(N, device=device, dtype=dtype)
            else:
                diag_increment = beta_t * dt * A.to_dense().diagonal()
            Sigma_diag[t_idx + 1] = Sigma_diag[t_idx] + diag_increment
        else:
            # Sigma will become dense
            Sigma[t_idx + 1] = Sigma[t_idx] + beta_t * dt * noise_cov
    
    # Numerical check for positive definiteness
    if diag_approx:
        Sigma_result = torch.diag_embed(Sigma_diag)
    else:
        # Ensure Sigma is symmetric
        Sigma_result = 0.5 * (Sigma + Sigma.transpose(-1, -2))

    if check_stability:
        # This check is expensive and requires a dense matrix.
        # Only perform on the final matrix.
        final_sigma_dense = Sigma_result[-1].to_dense() if Sigma_result[-1].is_sparse else Sigma_result[-1]
        min_eigenval = torch.linalg.eigvalsh(final_sigma_dense).min()
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
    """Generate β(t) noise schedule."""
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=dtype)
    
    elif schedule == "cosine":
        s = 0.008
        steps = torch.arange(num_steps + 1, device=device, dtype=dtype)
        alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
    
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
    """Extract Φ(t) and Σ(t) for specific time indices (batch-friendly)."""
    Phi_t = Phi_all[t_indices]
    Sigma_t = Sigma_all[t_indices]
    
    return Phi_t, Sigma_t


def compute_Phi_inverse(
    Phi: torch.Tensor,
    reg: float = 1e-8
) -> torch.Tensor:
    """Compute inverse of Φ matrix with regularization for numerical stability."""
    if Phi.is_sparse:
        raise NotImplementedError("Inverse of sparse matrix is not supported directly. This would result in a dense matrix.")

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
    """Compute condition number of Φ to check numerical stability."""
    if Phi.is_sparse:
        warnings.warn("Condition number for sparse matrix is not implemented; converting to dense. This may be slow.")
        Phi = Phi.to_dense()
        
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

    if Sigma.is_sparse:
        warnings.warn("Symmetry and positive-semidefinite check on sparse Sigma is not fully implemented.")
        return True # Basic check passed

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


class SpectralDiffusion:
    """Memory-efficient diffusion using spectral decomposition.
    
    Instead of storing T matrices of size NxN (O(T*N^2)), this class stores
    eigenvalues/vectors (O(N^2)) and precomputed time-dependent coefficients (O(T*N)).
    This reduces memory usage by factor of ~N/2 for large graphs.
    """
    
    def __init__(
        self,
        L: torch.Tensor,
        num_steps: int = 1000,
        T: float = 1.0,
        gamma: float = 1.0,
        lambda_reg: float = 1e-6,
        beta_schedule: Literal["linear", "cosine"] = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        if device is None:
            device = L.device
        if dtype is None:
            dtype = L.dtype
            
        self.num_steps = num_steps
        self.device = device
        self.dtype = dtype
        
        # 1. Build drift matrix A
        A = build_A_node(L, gamma=gamma, lambda_reg=lambda_reg)
        if A.is_sparse:
            A = A.to_dense() # Eigen-decomposition requires dense
            
        # 2. Spectral decomposition: A = U @ diag(evals) @ U.T
        # We assume A is symmetric (valid for undirected graphs)
        # If A is not symmetric, we would need eig() instead of eigh()
        self.evals, self.U = torch.linalg.eigh(A)
        
        # 3. Precompute time-dependent coefficients
        dt = T / num_steps
        betas = get_beta_schedule(
            schedule=beta_schedule,
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
            dtype=dtype
        )
        
        # Cumulative integral of beta (B(t))
        # B[t] = sum_{i=0}^{t-1} beta_i * dt
        B_t = torch.cumsum(torch.cat([torch.tensor([0.0], device=device, dtype=dtype), betas]) * dt, dim=0)
        
        # Phi coefficients: exp(-lambda_k * B(t))
        # Shape: (T+1, N)
        self.phi_coeffs = torch.exp(-self.evals.unsqueeze(0) * B_t.unsqueeze(1))
        
        # Sigma coefficients:
        # D_k(t) = integral_0^t exp(-2 * lambda_k * (B(t) - B(tau))) * beta(tau) dtau
        # We compute this iteratively:
        # D_k(t+1) = exp(-2 * lambda_k * beta_t * dt) * D_k(t) + beta_t * dt * (noise_factor)
        # For isotropic noise (noise_cov=I), noise_factor = 1
        # For anisotropic noise (noise_cov=A), noise_factor = lambda_k
        
        # Assuming isotropic noise for now as it's standard for spectral method
        # (Anisotropic is possible but requires checking noise_type)
        
        self.sigma_coeffs = torch.zeros(num_steps + 1, len(self.evals), device=device, dtype=dtype)
        
        for t in range(num_steps):
            beta_t = betas[t]
            decay = torch.exp(-2 * self.evals * beta_t * dt)
            # For isotropic noise:
            increment = beta_t * dt 
            # For anisotropic (if needed): increment = beta_t * dt * self.evals
            
            self.sigma_coeffs[t+1] = self.sigma_coeffs[t] * decay + increment

    def get_phi_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Phi(t) and Sigma(t) on demand for a batch of timesteps.
        
        Args:
            t: Timestep indices (B,)
            
        Returns:
            Phi_t: (B, N, N)
            Sigma_t: (B, N, N)
        """
        B = t.shape[0]
        N = self.U.shape[0]
        
        # Gather coefficients
        # (B, N)
        phi_c = self.phi_coeffs[t]
        sigma_c = self.sigma_coeffs[t]
        
        # Reconstruct matrices
        # Phi = U @ diag(phi_c) @ U.T
        # Sigma = U @ diag(sigma_c) @ U.T
        
        # Efficient computation:
        # Phi = (U * phi_c.unsqueeze(1)) @ U.T
        # But we need (B, N, N).
        
        # U is (N, N), phi_c is (B, N)
        # U_scaled = U.unsqueeze(0) * phi_c.unsqueeze(1)  -> (B, N, N)
        # Phi = U_scaled @ U.T.unsqueeze(0)
        
        U_expanded = self.U.unsqueeze(0) # (1, N, N)
        
        Phi_t = (U_expanded * phi_c.unsqueeze(1)) @ self.U.T.unsqueeze(0)
        Sigma_t = (U_expanded * sigma_c.unsqueeze(1)) @ self.U.T.unsqueeze(0)
        
        return Phi_t, Sigma_t


if __name__ == "__main__":
    # Quick test
    print("Testing precompute module...")
    
    # Simple 3-node chain graph (dense)
    L_dense = torch.tensor([
        [2.0, -1.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -1.0, 2.0]
    ], dtype=torch.float32)
    
    print("\n--- Testing with Dense Laplacian ---")
    print("Laplacian L:")
    print(L_dense)
    
    Phi_d, Sigma_d = precompute_phi_sigma_explicit(
        L_dense, num_steps=100, diag_approx=True, check_stability=True
    )
    
    print(f"\nΦ shape: {Phi_d.shape}")
    print(f"Σ shape (diag): {Sigma_d.shape}")
    
    # Test with sparse matrix
    print("\n--- Testing with Sparse Laplacian ---")
    from src.data.graph_builder import build_laplacian_from_edges
    edges = [(0, 1), (0, 2), (1, 2)]
    L_sparse = build_laplacian_from_edges(edges, 3)
    print("Sparse Laplacian L:")
    print(L_sparse.to_dense())

    Phi_s, Sigma_s = precompute_phi_sigma_explicit(
        L_sparse, num_steps=100, diag_approx=True, check_stability=True
    )
    
    print(f"\nΦ shape: {Phi_s.shape}")
    print(f"Σ shape (diag): {Sigma_s.shape}")

    # Check if results are close
    assert torch.allclose(Phi_d, Phi_s, atol=1e-5), "Dense and sparse Phi results differ!"
    assert torch.allclose(Sigma_d, Sigma_s, atol=1e-5), "Dense and sparse Sigma results differ!"

    print("\n✓ Precompute module test passed!")
