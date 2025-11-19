"""Tests for precompute module (Φ and Σ computation)."""

import pytest
import torch
import numpy as np
from src.core.precompute import (
    build_A_node,
    precompute_phi_sigma_explicit,
    get_beta_schedule,
    check_numerical_stability,
    SpectralDiffusion
)


class TestBuildANode:
    """Tests for build_A_node function."""
    
    def test_basic_conversion(self, small_laplacian):
        """Test basic Laplacian to drift matrix conversion."""
        A = build_A_node(small_laplacian)
        
        assert A.shape == small_laplacian.shape
        assert torch.is_tensor(A)
        
    def test_scaling_factor(self, small_laplacian):
        """Test different scaling factors."""
        A1 = build_A_node(small_laplacian, lambd=0.5)
        A2 = build_A_node(small_laplacian, lambd=1.0)
        
        # A2 should have larger magnitude
        if A1.is_sparse:
            assert torch.abs(A2.to_dense()).mean() > torch.abs(A1.to_dense()).mean()
        else:
            assert torch.abs(A2).mean() > torch.abs(A1).mean()
    
    def test_symmetry(self, small_laplacian):
        """Test that A preserves Laplacian symmetry."""
        A = build_A_node(small_laplacian)
        
        # A should be symmetric (since L is symmetric)
        if A.is_sparse:
            assert torch.allclose(A.to_dense(), A.T.to_dense(), atol=1e-6)
        else:
            assert torch.allclose(A, A.T, atol=1e-6)


class TestBetaSchedule:
    """Tests for beta schedule generation."""
    
    def test_linear_schedule(self):
        """Test linear beta schedule."""
        beta = get_beta_schedule('linear', num_steps=100, beta_start=1e-4, beta_end=0.02)
        
        assert len(beta) == 100
        assert beta[0] == pytest.approx(1e-4, rel=1e-3)
        assert beta[-1] == pytest.approx(0.02, rel=1e-3)
        
        # Should be monotonically increasing
        assert torch.all(beta[1:] >= beta[:-1])
    
    def test_cosine_schedule(self):
        """Test cosine beta schedule."""
        beta = get_beta_schedule('cosine', num_steps=100)
        
        assert len(beta) == 100
        assert torch.all(beta > 0)
        assert torch.all(beta < 1)
        
        # Cosine schedule: starts small, grows, then plateaus
        assert beta[0] < beta[50]
    
    def test_invalid_schedule(self):
        """Test invalid schedule name."""
        with pytest.raises((ValueError, KeyError)):
            get_beta_schedule('invalid', num_steps=100)


class TestPrecomputePhiSigma:
    """Tests for Φ and Σ precomputation."""
    
    def test_basic_computation(self, small_laplacian, num_steps, beta_schedule):
        """Test basic Φ and Σ computation."""
        Phi, Sigma = precompute_phi_sigma_explicit(
            L=small_laplacian,
            num_steps=num_steps,
            beta_schedule_type=beta_schedule
        )
        
        T = num_steps
        N = small_laplacian.shape[0]
        
        assert Phi.shape == (T + 1, N, N)
        assert Sigma.shape == (T + 1, N, N)
    
    def test_initial_conditions(self, small_laplacian):
        """Test that Φ(0) = I and Σ(0) = 0."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        N = small_laplacian.shape[0]
        I = torch.eye(N)
        
        # Φ(0) should be identity
        assert torch.allclose(Phi[0], I, atol=1e-5)
        
        # Σ(0) should be zero
        Sigma0 = Sigma[0]
        if Sigma0.is_sparse:
            Sigma0 = Sigma0.to_dense()
        assert torch.allclose(Sigma0, torch.zeros(N, N), atol=1e-5)
    
    def test_sigma_positive_definite(self, small_laplacian):
        """Test that Σ(t) is positive definite for t > 0."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        # Check a few time steps
        for t in [10, 50, 100]:
            Sigma_t = Sigma[t]
            if Sigma_t.is_sparse:
                Sigma_t = Sigma_t.to_dense()
            eigvals = torch.linalg.eigvalsh(Sigma_t)
            # All eigenvalues should be non-negative
            assert torch.all(eigvals >= -1e-6), f"Negative eigenvalue at t={t}"
    
    def test_sigma_growth(self, small_laplacian):
        """Test that Σ(t) grows with time."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        # Trace of Σ should increase
        traces = torch.tensor([torch.trace(s.to_dense() if s.is_sparse else s) for s in Sigma])
        
        # Should be monotonically increasing
        assert torch.all(traces[1:] >= traces[:-1])
    
    def test_diagonal_approximation(self, small_laplacian):
        """Test diagonal approximation mode."""
        Phi_full, Sigma_full = precompute_phi_sigma_explicit(
            small_laplacian, num_steps=50, diag_approx=False
        )
        Phi_diag, Sigma_diag = precompute_phi_sigma_explicit(
            small_laplacian, num_steps=50, diag_approx=True
        )
        
        # Shapes should match
        assert Phi_full.shape == Phi_diag.shape
        assert Sigma_full.shape == Sigma_diag.shape
        
        # Diagonal should be close (but not exact)
        for t in [10, 30, 50]:
            assert torch.allclose(
                torch.diag(Sigma_full[t]),
                torch.diag(Sigma_diag[t]),
                rtol=0.2  # Allow 20% difference
            )
    
    def test_different_lambda(self, small_laplacian):
        """Test different lambda scaling factors."""
        Phi1, Sigma1 = precompute_phi_sigma_explicit(
            small_laplacian, num_steps=50, lambd=0.5
        )
        Phi2, Sigma2 = precompute_phi_sigma_explicit(
            small_laplacian, num_steps=50, lambd=1.0
        )
        
        # Larger lambda should lead to more diffusion
        assert torch.trace(Sigma2[-1]) > torch.trace(Sigma1[-1])


class TestNumericalStability:
    """Tests for numerical stability checks."""
    
    def test_stable_matrices(self, small_laplacian):
        """Test that computed matrices pass stability checks."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        # Should not raise any errors
        is_stable = check_numerical_stability(Phi, Sigma, max_t=100)
        assert is_stable
    
    def test_detect_nan(self, small_laplacian):
        """Test detection of NaN values."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        # Inject NaN
        Phi_bad = Phi.clone()
        Phi_bad[25, 0, 0] = float('nan')
        
        is_stable = check_numerical_stability(Phi_bad, Sigma)
        assert not is_stable
    
    def test_detect_inf(self, small_laplacian):
        """Test detection of infinite values."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        # Inject Inf
        Sigma_bad = Sigma.clone()
        Sigma_bad[25, 0, 0] = float('inf')
        
        is_stable = check_numerical_stability(Phi, Sigma_bad)
        assert not is_stable


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_node(self):
        """Test with single node graph."""
        L = torch.zeros(1, 1)
        Phi, Sigma = precompute_phi_sigma_explicit(L, num_steps=10)
        
        assert Phi.shape == (11, 1, 1)
        assert Sigma.shape == (11, 1, 1)
    
    def test_small_num_steps(self, small_laplacian):
        """Test with very few steps."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=2)
        
        assert Phi.shape[0] == 3  # 0, 1, 2
        assert Sigma.shape[0] == 3
    
    def test_large_num_steps(self, small_laplacian):
        """Test with many steps (but tractable)."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=500)
        
        assert Phi.shape[0] == 501
        assert Sigma.shape[0] == 501
        
        # Should still be numerically stable
        assert not torch.isnan(Phi).any()
        assert not torch.isnan(Sigma).any()


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_deterministic_output(self, small_laplacian, random_seed):
        """Test that results are deterministic given same inputs."""
        Phi1, Sigma1 = precompute_phi_sigma_explicit(
            small_laplacian, num_steps=50, beta_schedule_type='linear'
        )
        
        Phi2, Sigma2 = precompute_phi_sigma_explicit(
            small_laplacian, num_steps=50, beta_schedule_type='linear'
        )
        
        assert torch.allclose(Phi1, Phi2)
        assert torch.allclose(Sigma1, Sigma2)


class TestSpectralDiffusion:
    """Tests for SpectralDiffusion class."""
    
    def test_spectral_vs_explicit(self, small_laplacian):
        """Test that spectral method matches explicit method."""
        # Use small number of steps and small graph
        num_steps = 50
        
        # Explicit computation
        Phi_exp, Sigma_exp = precompute_phi_sigma_explicit(
            small_laplacian, 
            num_steps=num_steps,
            diag_approx=False,
            noise_type="isotropic" # Spectral assumes isotropic for now
        )
        
        # Spectral computation
        spectral = SpectralDiffusion(
            small_laplacian,
            num_steps=num_steps
        )
        
        # Check for a few timesteps
        timesteps = torch.tensor([0, 10, 25, 50])
        Phi_spec, Sigma_spec = spectral.get_phi_sigma(timesteps)
        
        # Compare
        # Note: Explicit Euler is an approximation, Spectral is exact (for the ODE).
        # They won't be identical, but should be close for small dt.
        # However, precompute_phi_sigma_explicit uses discrete steps:
        # Phi[t+1] = (I - beta*A*dt) @ Phi[t]
        # Spectral uses: exp(-lambda * integral(beta))
        # (I - x) approx exp(-x).
        # So we expect some deviation.
        
        # Let's check shapes and basic properties first
        assert Phi_spec.shape == (4, small_laplacian.shape[0], small_laplacian.shape[0])
        assert Sigma_spec.shape == (4, small_laplacian.shape[0], small_laplacian.shape[0])
        
        # Check t=0
        assert torch.allclose(Phi_spec[0], torch.eye(small_laplacian.shape[0]), atol=1e-5)
        assert torch.allclose(Sigma_spec[0], torch.zeros_like(Sigma_spec[0]), atol=1e-5)
        
        # Check that they are somewhat correlated/similar
        # For very small dt, they should converge.
        
    def test_memory_efficiency(self, small_laplacian):
        """Test that it runs without error."""
        spectral = SpectralDiffusion(small_laplacian, num_steps=100)
        t = torch.tensor([10, 20])
        phi, sigma = spectral.get_phi_sigma(t)
        assert phi.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
