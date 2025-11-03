"""Tests for forward noising process."""

import pytest
import torch
from src.core.precompute import precompute_phi_sigma_explicit
from src.core.forward import (
    apply_Phi_node_batch,
    apply_Sigma_sqrt_node_batch,
    forward_noising_batch,
    compute_hat_S0_from_eps_hat
)


class TestApplyPhi:
    """Tests for Φ application."""
    
    def test_basic_application(self, small_laplacian, sample_states):
        """Test basic Φ application to states."""
        Phi, _ = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        t = 25
        result = apply_Phi_node_batch(sample_states, Phi[t])
        
        assert result.shape == sample_states.shape
        assert torch.is_tensor(result)
    
    def test_identity_at_t0(self, small_laplacian, sample_states):
        """Test that Φ(0) acts as identity."""
        Phi, _ = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        result = apply_Phi_node_batch(sample_states, Phi[0])
        
        assert torch.allclose(result, sample_states, atol=1e-5)
    
    def test_batch_consistency(self, small_laplacian):
        """Test that batch processing is consistent."""
        Phi, _ = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        # Single sample
        single = torch.randn(1, 9, 3)
        result_single = apply_Phi_node_batch(single, Phi[25])
        
        # Batched
        batch = single.repeat(4, 1, 1)
        result_batch = apply_Phi_node_batch(batch, Phi[25])
        
        # All should be identical
        for i in range(4):
            assert torch.allclose(result_batch[i], result_single[0], atol=1e-6)


class TestApplySigmaSqrt:
    """Tests for Σ^(1/2) application."""
    
    def test_basic_application(self, small_laplacian):
        """Test basic Σ^(1/2) application."""
        _, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        noise = torch.randn(4, 9, 3)
        result = apply_Sigma_sqrt_node_batch(noise, Sigma[25])
        
        assert result.shape == noise.shape
    
    def test_zero_at_t0(self, small_laplacian):
        """Test that Σ(0) = 0 leads to zero output."""
        _, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        noise = torch.randn(4, 9, 3)
        result = apply_Sigma_sqrt_node_batch(noise, Sigma[0])
        
        # Should be approximately zero
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)
    
    def test_increases_with_time(self, small_laplacian):
        """Test that noise magnitude increases with time."""
        _, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        noise = torch.randn(4, 9, 3)
        
        result_10 = apply_Sigma_sqrt_node_batch(noise, Sigma[10])
        result_50 = apply_Sigma_sqrt_node_batch(noise, Sigma[50])
        result_100 = apply_Sigma_sqrt_node_batch(noise, Sigma[100])
        
        # Magnitude should increase
        mag_10 = torch.norm(result_10)
        mag_50 = torch.norm(result_50)
        mag_100 = torch.norm(result_100)
        
        assert mag_50 > mag_10
        assert mag_100 > mag_50


class TestForwardNoising:
    """Tests for complete forward noising process."""
    
    def test_basic_noising(self, small_laplacian, sample_states):
        """Test basic forward noising."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        t = 25
        noised, noise = forward_noising_batch(
            S_0=sample_states,
            t=t,
            Phi=Phi,
            Sigma=Sigma
        )
        
        assert noised.shape == sample_states.shape
        assert noise.shape == sample_states.shape
    
    def test_no_noise_at_t0(self, small_laplacian, sample_states):
        """Test that t=0 gives original states."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        noised, noise = forward_noising_batch(
            S_0=sample_states,
            t=0,
            Phi=Phi,
            Sigma=Sigma
        )
        
        assert torch.allclose(noised, sample_states, atol=1e-5)
    
    def test_increasing_noise(self, small_laplacian):
        """Test that noise increases with timestep."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        S_0 = torch.randn(4, 9, 3)
        
        distances = []
        for t in [10, 30, 50, 70, 100]:
            S_t, _ = forward_noising_batch(S_0, t, Phi, Sigma)
            dist = torch.norm(S_t - S_0)
            distances.append(dist.item())
        
        # Distance should generally increase
        for i in range(len(distances) - 1):
            assert distances[i + 1] >= distances[i] * 0.9  # Allow small fluctuations
    
    def test_reproducibility(self, small_laplacian, sample_states, random_seed):
        """Test that noising is reproducible with same seed."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        torch.manual_seed(42)
        noised1, noise1 = forward_noising_batch(sample_states, 25, Phi, Sigma)
        
        torch.manual_seed(42)
        noised2, noise2 = forward_noising_batch(sample_states, 25, Phi, Sigma)
        
        assert torch.allclose(noised1, noised2)
        assert torch.allclose(noise1, noise2)
    
    def test_different_timesteps(self, small_laplacian, sample_states):
        """Test noising at different timesteps."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        for t in [1, 10, 50, 99, 100]:
            noised, noise = forward_noising_batch(sample_states, t, Phi, Sigma)
            
            assert noised.shape == sample_states.shape
            assert not torch.isnan(noised).any()
            assert not torch.isinf(noised).any()


class TestS0Reconstruction:
    """Tests for S_0 reconstruction from epsilon prediction."""
    
    def test_perfect_reconstruction(self, small_laplacian, sample_states):
        """Test reconstruction with perfect epsilon prediction."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        t = 25
        S_t, eps_true = forward_noising_batch(sample_states, t, Phi, Sigma)
        
        # Use true epsilon for reconstruction
        S_0_hat = compute_hat_S0_from_eps_hat(
            S_t=S_t,
            eps_hat=eps_true,
            t=t,
            Phi=Phi,
            Sigma=Sigma
        )
        
        # Should recover original (approximately)
        assert torch.allclose(S_0_hat, sample_states, atol=1e-3)
    
    def test_noisy_reconstruction(self, small_laplacian, sample_states):
        """Test reconstruction with noisy epsilon."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        t = 25
        S_t, eps_true = forward_noising_batch(sample_states, t, Phi, Sigma)
        
        # Add noise to epsilon
        eps_noisy = eps_true + 0.1 * torch.randn_like(eps_true)
        
        S_0_hat = compute_hat_S0_from_eps_hat(S_t, eps_noisy, t, Phi, Sigma)
        
        # Should be close but not perfect
        error = torch.norm(S_0_hat - sample_states)
        assert error < torch.norm(sample_states) * 0.5  # Within 50%
    
    def test_batch_reconstruction(self, small_laplacian):
        """Test reconstruction works for batches."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        batch_size = 8
        S_0 = torch.randn(batch_size, 9, 3)
        
        t = 30
        S_t, eps = forward_noising_batch(S_0, t, Phi, Sigma)
        S_0_hat = compute_hat_S0_from_eps_hat(S_t, eps, t, Phi, Sigma)
        
        assert S_0_hat.shape == (batch_size, 9, 3)
        assert torch.allclose(S_0_hat, S_0, atol=1e-3)


class TestNumericalStability:
    """Tests for numerical stability of forward process."""
    
    def test_no_nan_or_inf(self, small_laplacian):
        """Test that forward process doesn't produce NaN or Inf."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        S_0 = torch.randn(4, 9, 3)
        
        for t in range(0, 101, 10):
            S_t, eps = forward_noising_batch(S_0, t, Phi, Sigma)
            
            assert not torch.isnan(S_t).any(), f"NaN at t={t}"
            assert not torch.isinf(S_t).any(), f"Inf at t={t}"
            assert not torch.isnan(eps).any(), f"NaN in noise at t={t}"
    
    def test_extreme_values(self, small_laplacian):
        """Test with extreme input values."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        # Very large values
        S_0_large = torch.randn(2, 9, 3) * 1000
        S_t, eps = forward_noising_batch(S_0_large, 25, Phi, Sigma)
        
        assert not torch.isnan(S_t).any()
        assert not torch.isinf(S_t).any()
        
        # Very small values
        S_0_small = torch.randn(2, 9, 3) * 1e-6
        S_t, eps = forward_noising_batch(S_0_small, 25, Phi, Sigma)
        
        assert not torch.isnan(S_t).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
