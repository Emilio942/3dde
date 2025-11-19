"""Integration tests for complete diffusion model pipeline."""

import pytest
import torch
from src.core.precompute import precompute_phi_sigma_explicit, get_Phi_Sigma_at_t
from src.core.forward import forward_noising_batch, compute_hat_S0_from_eps_hat
from src.core.sampling import compute_F_Q_from_PhiSigma, sample_reverse_from_S_T
from src.data.graph_builder import build_grid_laplacian, build_molecular_graph
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.eps_net import EpsilonNetwork
from src.training.trainer import DiffusionTrainer


class TestForwardBackwardConsistency:
    """Tests for forward-backward consistency."""
    
    def test_forward_then_perfect_backward(self, small_laplacian, random_seed):
        """Test forward noising followed by perfect reverse."""
        # Setup
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        S_0_original = torch.randn(4, 9, 3)
        t = 25
        
        # Get Phi and Sigma at timestep t
        t_indices = torch.tensor([t] * 4)
        Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t_indices)
        
        # Forward noising
        S_t, eps_true = forward_noising_batch(S_0_original, Phi_t, Sigma_t)
        
        # Perfect reconstruction (using true epsilon)
        S_0_reconstructed = compute_hat_S0_from_eps_hat(
            S_t, eps_true, Phi_t, Sigma_t
        )
        
        # Should match closely
        assert torch.allclose(S_0_original, S_0_reconstructed, atol=1e-3)
    
    def test_multiple_timesteps_consistency(self, small_laplacian):
        """Test consistency across multiple timesteps."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=100)
        
        S_0 = torch.randn(2, 9, 3)
        
        for t in [10, 30, 50, 70, 99]:  # Changed 100 to 99 (0-indexed)
            t_indices = torch.tensor([t] * 2)
            Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t_indices)
            
            S_t, eps = forward_noising_batch(S_0, Phi_t, Sigma_t)
            S_0_hat = compute_hat_S0_from_eps_hat(S_t, eps, Phi_t, Sigma_t)
            
            # Should reconstruct well
            error = torch.norm(S_0 - S_0_hat) / torch.norm(S_0)
            assert error < 0.01  # Less than 1% relative error


class TestGraphStructures:
    """Tests with different graph structures."""
    
    def test_grid_graphs(self):
        """Test with different grid sizes."""
        grid_sizes = [(3, 3), (5, 5), (4, 6), (10, 10)]
        
        for size in grid_sizes:
            L = build_grid_laplacian(size)
            Phi, Sigma = precompute_phi_sigma_explicit(L, num_steps=20)
            
            num_nodes = size[0] * size[1]
            S_0 = torch.randn(2, num_nodes, 3)
            
            t_indices = torch.tensor([10, 10])
            Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t_indices)
            S_t, eps = forward_noising_batch(S_0, Phi_t, Sigma_t)
            
            assert S_t.shape == (2, num_nodes, 3)
            assert not torch.isnan(S_t).any()
    
    def test_molecular_graph(self):
        """Test with molecular graph structure."""
        # Simple molecule: 5 atoms in 3D space
        positions = torch.randn(5, 3)
        
        L, edges = build_molecular_graph(positions, cutoff=2.0)
        Phi, Sigma = precompute_phi_sigma_explicit(L, num_steps=20)
        
        S_0 = torch.randn(2, 5, 3)
        t_indices = torch.tensor([10, 10])
        Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t_indices)
        S_t, eps = forward_noising_batch(S_0, Phi_t, Sigma_t)
        
        assert S_t.shape == (2, 5, 3)
        assert not torch.isnan(S_t).any()


class TestEndToEndPipeline:
    """Tests for complete end-to-end pipeline."""
    
    def test_training_and_sampling(self, small_laplacian):
        """Test complete training and sampling pipeline."""
        # 1. Create dataset
        dataset = SyntheticDataset(
            num_samples=64,
            num_nodes=9,
            node_dim=3,
            mode='smooth'
        )
        
        train_loader, val_loader = create_dataloaders(
            dataset.data,
            batch_size=16,
            val_split=0.2
        )
        
        # 2. Create model
        model = EpsilonNetwork(
            input_dim=3,
            hidden_dim=64,
            num_layers=3
        )
        
        # 3. Create trainer
        trainer = DiffusionTrainer(
            model=model,
            L=small_laplacian,
            num_steps=20,
            learning_rate=1e-3
        )
        
        # 4. Train (just a few steps for testing)
        trainer.train(train_loader, val_loader, num_epochs=2)
        
        # 5. Sample
        if trainer.use_spectral:
            # Reconstruct dense matrices for testing sampling
            t_all = torch.arange(trainer.num_steps + 1, device=trainer.device)
            Phi, Sigma = trainer.spectral_diff.get_phi_sigma(t_all)
        else:
            Phi, Sigma = trainer.Phi, trainer.Sigma
            
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        S_T = torch.randn(4, 9, 3)
        
        model.eval()
        with torch.no_grad():
            S_0, trajectory = sample_reverse_from_S_T(
                S_T, F, Q, model, small_laplacian, num_steps=20
            )
        
        # Check results
        assert S_0.shape == (4, 9, 3)
        assert trajectory.shape == (21, 4, 9, 3)
        assert not torch.isnan(S_0).any()
    
    def test_different_data_modes(self, small_laplacian):
        """Test training with different synthetic data modes."""
        modes = ['random', 'smooth', 'wave', 'cluster']
        
        model = EpsilonNetwork(input_dim=3, hidden_dim=32, num_layers=2)
        
        for mode in modes:
            # Create dataset
            dataset = SyntheticDataset(
                num_samples=32,
                num_nodes=9,
                node_dim=3,
                mode=mode
            )
            
            train_loader, _ = create_dataloaders(
                dataset.data,
                batch_size=8,
                val_split=0.0
            )
            
            # Train for one epoch
            trainer = DiffusionTrainer(
                model=model,
                L=small_laplacian,
                num_steps=20
            )
            
            metrics = trainer.train_epoch(train_loader, epoch=0)
            
            # Should complete without errors
            assert 'train_loss' in metrics
            assert not torch.isnan(torch.tensor(metrics['train_loss']))


class TestScalability:
    """Tests for scalability with larger graphs."""
    
    def test_larger_graph(self):
        """Test with larger graph structure."""
        # 10x10 grid = 100 nodes
        L = build_grid_laplacian((10, 10))
        
        # Use diagonal approximation for speed
        Phi, Sigma = precompute_phi_sigma_explicit(
            L, num_steps=50, diag_approx=True
        )
        
        S_0 = torch.randn(2, 100, 3)
        t_indices = torch.tensor([25, 25])
        Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t_indices)
        S_t, eps = forward_noising_batch(S_0, Phi_t, Sigma_t)
        
        assert S_t.shape == (2, 100, 3)
        assert not torch.isnan(S_t).any()
    
    def test_larger_batch(self, small_laplacian):
        """Test with larger batch size."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        batch_size = 64
        S_0 = torch.randn(batch_size, 9, 3)
        t_indices = torch.tensor([25] * batch_size)
        Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t_indices)
        S_t, eps = forward_noising_batch(S_0, Phi_t, Sigma_t)
        
        assert S_t.shape == (batch_size, 9, 3)
    
    def test_higher_dimension(self, small_laplacian):
        """Test with higher dimensional features."""
        model = EpsilonNetwork(
            input_dim=10,  # Higher dimension
            hidden_dim=64,
            num_layers=3
        )
        
        x = torch.randn(4, 9, 10)
        t = torch.tensor([25, 25, 25, 25])
        
        eps = model(x, t, small_laplacian)
        
        assert eps.shape == (4, 9, 10)


class TestNumericalStability:
    """Tests for numerical stability across the pipeline."""
    
    def test_long_diffusion_process(self, small_laplacian):
        """Test stability with many diffusion steps."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=500)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        # Check all timesteps
        for t in range(0, 500, 50):  # Changed 501 to 500 (max index is 499)
            assert not torch.isnan(Phi[t]).any(), f"NaN in Phi at t={t}"
            assert not torch.isnan(Sigma[t]).any(), f"NaN in Sigma at t={t}"
            assert not torch.isnan(F[t]).any(), f"NaN in F at t={t}"
            assert not torch.isnan(Q[t]).any(), f"NaN in Q at t={t}"
    
    def test_extreme_inputs(self, small_laplacian):
        """Test with extreme input values."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        # Very large values
        S_large = torch.randn(2, 9, 3) * 1000
        t_indices = torch.tensor([25, 25])
        Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t_indices)
        S_t, eps = forward_noising_batch(S_large, Phi_t, Sigma_t)
        assert not torch.isnan(S_t).any()
        
        # Very small values
        S_small = torch.randn(2, 9, 3) * 1e-6
        S_t, eps = forward_noising_batch(S_small, Phi_t, Sigma_t)
        assert not torch.isnan(S_t).any()


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_deterministic_training(self, small_laplacian, random_seed):
        """Test that training is deterministic with fixed seed."""
        def train_model(seed):
            torch.manual_seed(seed)
            
            dataset = SyntheticDataset(
                num_samples=32, num_nodes=9, node_dim=3, seed=seed
            )
            train_loader, _ = create_dataloaders(
                dataset.data, batch_size=8, val_split=0.0
            )
            
            model = EpsilonNetwork(input_dim=3, hidden_dim=32, num_layers=2)
            trainer = DiffusionTrainer(
                model=model, L=small_laplacian, num_steps=20
            )
            
            metrics = trainer.train_epoch(train_loader, epoch=0)
            return metrics['train_loss']
        
        loss1 = train_model(42)
        loss2 = train_model(42)
        
        # Should be very close (may not be exactly identical due to floating point)
        assert abs(loss1 - loss2) < 1e-5
    
    def test_deterministic_sampling(self, small_laplacian, random_seed):
        """Test that sampling is deterministic with fixed seed."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=20)
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        
        model = EpsilonNetwork(input_dim=3, hidden_dim=32, num_layers=2)
        model.eval()
        
        def sample_with_seed(seed):
            torch.manual_seed(seed)
            S_T = torch.randn(2, 9, 3)
            with torch.no_grad():
                S_0, _ = sample_reverse_from_S_T(
                    S_T, F, Q, model, small_laplacian, num_steps=20
                )
            return S_0
        
        S_0_1 = sample_with_seed(42)
        S_0_2 = sample_with_seed(42)
        
        assert torch.allclose(S_0_1, S_0_2, atol=1e-6)


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_mismatched_dimensions(self, small_laplacian):
        """Test error handling for mismatched dimensions."""
        model = EpsilonNetwork(input_dim=3, hidden_dim=32, num_layers=2)
        
        # Wrong node dimension
        x_wrong = torch.randn(4, 5, 3)  # 5 nodes instead of 9
        t = torch.tensor([25, 25, 25, 25])
        
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            model(x_wrong, t, small_laplacian)
    
    def test_invalid_timestep(self, small_laplacian):
        """Test with invalid timestep."""
        Phi, Sigma = precompute_phi_sigma_explicit(small_laplacian, num_steps=50)
        
        S_0 = torch.randn(2, 9, 3)
        
        # Timestep beyond range (max valid index is 49 for num_steps=50)
        t_indices = torch.tensor([100, 100])  # Invalid indices
        with pytest.raises((IndexError, RuntimeError)):
            Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t_indices)
            forward_noising_batch(S_0, Phi_t, Sigma_t)
    
    def test_empty_batch(self, small_laplacian):
        """Test handling of empty batch."""
        model = EpsilonNetwork(input_dim=3, hidden_dim=32, num_layers=2)
        
        x_empty = torch.randn(0, 9, 3)  # Empty batch
        t_empty = torch.tensor([])
        
        # Should handle gracefully or raise clear error
        try:
            out = model(x_empty, t_empty, small_laplacian)
            assert out.shape == (0, 9, 3)
        except (RuntimeError, ValueError):
            pass  # Acceptable to raise error


class TestMemoryEfficiency:
    """Tests for memory efficiency."""
    
    def test_gradient_checkpointing_compatible(self, small_laplacian):
        """Test that model is compatible with gradient checkpointing."""
        model = EpsilonNetwork(input_dim=3, hidden_dim=64, num_layers=4)
        
        x = torch.randn(2, 9, 3, requires_grad=True)
        t = torch.tensor([25, 25])
        
        eps = model(x, t, small_laplacian)
        loss = eps.mean()
        loss.backward()
        
        # Should work without issues
        assert x.grad is not None
    
    def test_inference_mode(self, small_laplacian):
        """Test inference without storing gradients."""
        model = EpsilonNetwork(input_dim=3, hidden_dim=64, num_layers=3)
        model.eval()
        
        x = torch.randn(4, 9, 3)
        t = torch.tensor([25, 25, 25, 25])
        
        with torch.no_grad():
            eps = model(x, t, small_laplacian)
        
        # Should not have gradient information
        assert not eps.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
