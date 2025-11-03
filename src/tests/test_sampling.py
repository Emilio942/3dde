"""Tests for reverse sampling process."""

import pytest
import torch
from src.core.precompute import precompute_phi_sigma_explicit
from src.core.forward import forward_noising_batch
from src.core.sampling import (
    compute_F_Q_from_PhiSigma,
    one_step_reverse,
    sample_reverse_from_S_T,
    ddim_sampling,
    langevin_corrector
)
from src.models.eps_net import EpsilonNetwork


class TestComputeFQ:
    """Tests for F and Q matrix computation."""
    
    def test_basic_computation(self, small_laplacian):
        """Test basic F and Q computation."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        assert F.shape == Phi.shape
        assert Q.shape == Sigma.shape
    
    def test_terminal_condition(self, small_laplacian):
        """Test that F[T] and Q[T] are properly defined."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        # Should not contain NaN or Inf
        assert not torch.isnan(F[-1]).any()
        assert not torch.isnan(Q[-1]).any()
        assert not torch.isinf(F[-1]).any()
    
    def test_q_positive_definite(self, small_laplacian):
        """Test that Q matrices are positive semi-definite."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        # Check several timesteps
        for t in [10, 25, 40]:
            eigvals = torch.linalg.eigvalsh(Q[t])
            assert torch.all(eigvals >= -1e-6), f"Negative eigenvalue in Q[{t}]"


class TestOneStepReverse:
    """Tests for single reverse step."""
    
    def test_basic_reverse_step(self, small_laplacian):
        """Test basic reverse step."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        S_t = torch.randn(4, 9, 3)
        eps_hat = torch.randn(4, 9, 3)
        
        S_tm1 = one_step_reverse(
            S_t=S_t,
            eps_hat=eps_hat,
            t=25,
            F=F,
            Q=Q
        )
        
        assert S_tm1.shape == S_t.shape
        assert not torch.isnan(S_tm1).any()
    
    def test_deterministic_mode(self, small_laplacian):
        """Test deterministic reverse step (no noise)."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        S_t = torch.randn(4, 9, 3)
        eps_hat = torch.randn(4, 9, 3)
        
        torch.manual_seed(42)
        S_tm1_det = one_step_reverse(S_t, eps_hat, 25, F, Q, deterministic=True)
        
        torch.manual_seed(43)  # Different seed
        S_tm1_det2 = one_step_reverse(S_t, eps_hat, 25, F, Q, deterministic=True)
        
        # Should be identical (deterministic)
        assert torch.allclose(S_tm1_det, S_tm1_det2)
    
    def test_stochastic_mode(self, small_laplacian):
        """Test stochastic reverse step (with noise)."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        S_t = torch.randn(4, 9, 3)
        eps_hat = torch.randn(4, 9, 3)
        
        torch.manual_seed(42)
        S_tm1_stoch1 = one_step_reverse(S_t, eps_hat, 25, F, Q, deterministic=False)
        
        torch.manual_seed(43)
        S_tm1_stoch2 = one_step_reverse(S_t, eps_hat, 25, F, Q, deterministic=False)
        
        # Should be different (stochastic)
        assert not torch.allclose(S_tm1_stoch1, S_tm1_stoch2)


class TestFullSampling:
    """Tests for complete sampling process."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple epsilon network for testing."""
        return EpsilonNetwork(
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            time_dim=16
        )
    
    def test_basic_sampling(self, small_laplacian, simple_model):
        """Test basic full sampling."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=20)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        S_T = torch.randn(2, 9, 3)
        
        S_0, trajectory = sample_reverse_from_S_T(
            S_T=S_T,
            F=F,
            Q=Q,
            eps_model=simple_model,
            L=small_laplacian,
            num_steps=20
        )
        
        assert S_0.shape == (2, 9, 3)
        assert trajectory.shape[0] == 21  # T+1 timesteps
        assert trajectory.shape[1:] == (2, 9, 3)
    
    def test_trajectory_evolution(self, small_laplacian, simple_model):
        """Test that trajectory evolves smoothly."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=20)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        S_T = torch.randn(2, 9, 3)
        
        S_0, trajectory = sample_reverse_from_S_T(
            S_T, F, Q, simple_model, small_laplacian, num_steps=20
        )
        
        # Check trajectory is continuous
        for t in range(len(trajectory) - 1):
            diff = torch.norm(trajectory[t + 1] - trajectory[t])
            assert diff < 10.0  # Reasonable step size
    
    def test_sampling_with_predictor_corrector(self, small_laplacian, simple_model):
        """Test sampling with Langevin corrector."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=20)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        S_T = torch.randn(2, 9, 3)
        
        S_0, _ = sample_reverse_from_S_T(
            S_T, F, Q, simple_model, small_laplacian,
            num_steps=20,
            use_predictor_corrector=True,
            num_corrector_steps=5
        )
        
        assert S_0.shape == (2, 9, 3)
        assert not torch.isnan(S_0).any()


class TestDDIMSampling:
    """Tests for DDIM sampling."""
    
    @pytest.fixture
    def simple_model(self):
        return EpsilonNetwork(input_dim=3, hidden_dim=32, num_layers=2)
    
    def test_ddim_basic(self, small_laplacian, simple_model):
        """Test basic DDIM sampling."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        S_T = torch.randn(2, 9, 3)
        
        S_0, trajectory = ddim_sampling(
            S_T=S_T,
            Phi=Phi,
            Sigma=Sigma,
            eps_model=simple_model,
            L=small_laplacian,
            num_steps=50,
            eta=0.0  # Deterministic
        )
        
        assert S_0.shape == (2, 9, 3)
        assert not torch.isnan(S_0).any()
    
    def test_ddim_deterministic(self, small_laplacian, simple_model):
        """Test DDIM deterministic sampling (eta=0)."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=20)
        
        S_T = torch.randn(2, 9, 3)
        
        torch.manual_seed(42)
        S_0_1, _ = ddim_sampling(S_T, Phi, Sigma, simple_model, small_laplacian, 20, eta=0.0)
        
        torch.manual_seed(43)  # Different seed
        S_0_2, _ = ddim_sampling(S_T, Phi, Sigma, simple_model, small_laplacian, 20, eta=0.0)
        
        # Should be identical (deterministic)
        assert torch.allclose(S_0_1, S_0_2)
    
    def test_ddim_stochastic(self, small_laplacian, simple_model):
        """Test DDIM stochastic sampling (eta>0)."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=20)
        
        S_T = torch.randn(2, 9, 3)
        
        torch.manual_seed(42)
        S_0_1, _ = ddim_sampling(S_T, Phi, Sigma, simple_model, small_laplacian, 20, eta=1.0)
        
        torch.manual_seed(43)
        S_0_2, _ = ddim_sampling(S_T, Phi, Sigma, simple_model, small_laplacian, 20, eta=1.0)
        
        # Should be different (stochastic)
        assert not torch.allclose(S_0_1, S_0_2)
    
    def test_ddim_subsampling(self, small_laplacian, simple_model):
        """Test DDIM with fewer sampling steps than training."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        S_T = torch.randn(2, 9, 3)
        
        # Sample with only 20 steps instead of 100
        S_0, trajectory = ddim_sampling(
            S_T, Phi, Sigma, simple_model, small_laplacian,
            num_steps=100,
            num_sample_steps=20,
            eta=0.0
        )
        
        assert S_0.shape == (2, 9, 3)
        assert trajectory.shape[0] == 21  # 20 steps + initial


class TestLangevinCorrector:
    """Tests for Langevin corrector."""
    
    @pytest.fixture
    def simple_model(self):
        return EpsilonNetwork(input_dim=3, hidden_dim=32, num_layers=2)
    
    def test_basic_correction(self, small_laplacian, simple_model):
        """Test basic Langevin correction."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        S_t = torch.randn(2, 9, 3)
        
        S_corrected = langevin_corrector(
            S_t=S_t,
            t=25,
            eps_model=simple_model,
            L=small_laplacian,
            Sigma=Sigma,
            num_steps=5,
            step_size=1e-3
        )
        
        assert S_corrected.shape == S_t.shape
        assert not torch.isnan(S_corrected).any()
    
    def test_correction_movement(self, small_laplacian, simple_model):
        """Test that correction actually moves the state."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        S_t = torch.randn(2, 9, 3)
        
        S_corrected = langevin_corrector(
            S_t, 25, simple_model, small_laplacian, Sigma,
            num_steps=10,
            step_size=1e-2
        )
        
        # Should have moved
        assert not torch.allclose(S_corrected, S_t)
        
        # But not too far
        dist = torch.norm(S_corrected - S_t)
        assert dist < torch.norm(S_t) * 0.5


class TestRoundTrip:
    """Tests for forward-backward consistency."""
    
    @pytest.fixture
    def simple_model(self):
        return EpsilonNetwork(input_dim=3, hidden_dim=64, num_layers=3)
    
    def test_approximate_roundtrip(self, small_laplacian, simple_model, random_seed):
        """Test approximate round-trip: S_0 -> S_T -> S_0'."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=20)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        S_0_original = torch.randn(2, 9, 3)
        
        # Forward
        S_T, _ = forward_noising_batch(S_0_original, 20, Phi, Sigma)
        
        # Backward (with untrained model - won't be perfect)
        S_0_reconstructed, _ = sample_reverse_from_S_T(
            S_T, F, Q, simple_model, small_laplacian, num_steps=20
        )
        
        # With untrained model, just check shapes and no NaN
        assert S_0_reconstructed.shape == S_0_original.shape
        assert not torch.isnan(S_0_reconstructed).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
