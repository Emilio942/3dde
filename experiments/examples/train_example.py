"""Complete training example for 3D Diffusion Model.

This script demonstrates the full pipeline:
1. Create synthetic dataset
2. Build graph structure
3. Initialize model
4. Train with regularization
5. Sample new data
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import our modules
from src.data.graph_builder import build_grid_laplacian
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.eps_net import EpsilonNetwork
from src.training.trainer import DiffusionTrainer
from src.training.losses import CombinedLoss
from src.core.sampling import compute_F_Q_from_PhiSigma, sample_reverse_from_S_T


def main():
    # Configuration
    config = {
        # Data
        'num_samples': 1000,
        'grid_size': (5, 5),  # 5x5 grid = 25 nodes
        'node_dim': 3,  # 3D positions
        'data_mode': 'smooth',
        
        # Model
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 4,
        'time_dim': 64,
        
        # Training
        'num_steps': 1000,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        
        # Regularization
        'lambda_energy': 0.0,
        'lambda_smoothness': 0.1,
        'lambda_alignment': 0.0,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 60)
    print("3D Diffusion Model - Complete Training Example")
    print("=" * 60)
    
    # Step 1: Create graph structure
    print("\n[1/6] Building graph structure...")
    L = build_grid_laplacian(config['grid_size'])
    num_nodes = L.shape[0]
    print(f"  Graph: {config['grid_size']} grid with {num_nodes} nodes")
    print(f"  Laplacian shape: {L.shape}")
    
    # Step 2: Generate synthetic dataset
    print("\n[2/6] Generating synthetic dataset...")
    dataset = SyntheticDataset(
        num_samples=config['num_samples'],
        num_nodes=num_nodes,
        node_dim=config['node_dim'],
        mode=config['data_mode'],
        noise_level=0.1,
        seed=42
    )
    print(f"  Generated {len(dataset)} samples")
    print(f"  Data shape: {dataset[0]['states'].shape}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_data=dataset.data,
        batch_size=config['batch_size'],
        val_split=0.2
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Step 3: Initialize model
    print("\n[3/6] Initializing model...")
    model = EpsilonNetwork(
        input_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        time_dim=config['time_dim'],
        use_attention=True
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: EpsilonNetwork")
    print(f"  Parameters: {num_params:,}")
    print(f"  Device: {config['device']}")
    
    # Step 4: Setup training
    print("\n[4/6] Setting up training...")
    
    # Loss configuration
    loss_config = {
        'eps_weight': 1.0,
        'regularization_weight': 0.0
    }
    
    # Regularizer configuration
    reg_config = {
        'lambda_energy': config['lambda_energy'],
        'lambda_smoothness': config['lambda_smoothness'],
        'lambda_alignment': config['lambda_alignment'],
        'lambda_divergence': 0.0
    }
    
    trainer = DiffusionTrainer(
        model=model,
        L=L,
        num_steps=config['num_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        loss_config=loss_config,
        reg_config=reg_config if any(reg_config.values()) else None,
        device=config['device'],
        checkpoint_dir="./checkpoints",
        log_interval=50
    )
    
    print(f"  Optimizer: AdamW (lr={config['learning_rate']:.6f})")
    print(f"  Diffusion steps: {config['num_steps']}")
    if any(reg_config.values()):
        print(f"  Regularization: ON")
        for k, v in reg_config.items():
            if v > 0:
                print(f"    {k}: {v}")
    
    # Step 5: Train model
    print("\n[5/6] Training...")
    print("-" * 60)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs']
    )
    
    print("-" * 60)
    print("  Training completed!")
    
    # Step 6: Sample new data
    print("\n[6/6] Sampling new data...")
    
    model.eval()
    with torch.no_grad():
        # Compute reverse process matrices
        F, Q = compute_F_Q_from_PhiSigma(trainer.Phi, trainer.Sigma)
        
        # Start from pure noise
        num_samples_to_generate = 4
        S_T = torch.randn(
            num_samples_to_generate,
            num_nodes,
            config['node_dim'],
            device=config['device']
        )
        
        print(f"  Generating {num_samples_to_generate} samples...")
        print(f"  Starting from noise: shape {S_T.shape}")
        
        # Sample (simplified - using basic reverse process)
        S_0, trajectory = sample_reverse_from_S_T(
            S_T=S_T,
            F=F,
            Q=Q,
            eps_model=model,
            L=L.to(config['device']),
            num_steps=config['num_steps'],
            use_predictor_corrector=False
        )
        
        print(f"  Generated samples: shape {S_0.shape}")
        print(f"  Trajectory: {trajectory.shape[0]} steps")
        
        # Statistics
        print(f"\n  Generated data statistics:")
        print(f"    Mean: {S_0.mean():.4f}")
        print(f"    Std:  {S_0.std():.4f}")
        print(f"    Min:  {S_0.min():.4f}")
        print(f"    Max:  {S_0.max():.4f}")
        
        # Save samples for visualization
        torch.save({
            'samples': S_0,
            'trajectory': trajectory
        }, 'generated_samples.pt')
        print(f"  ✓ Saved samples to generated_samples.pt")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Best validation loss: {trainer.best_loss:.6f}")
    print(f"  Total training steps: {trainer.global_step}")
    print(f"  Checkpoints saved to: {trainer.checkpoint_dir}")
    print("\nNext steps:")
    print("  - Run unit tests: pytest src/tests/")
    print("  - Visualize results in experiments/notebooks/")
    print("  - Tune hyperparameters for your specific data")
    print("=" * 60)


if __name__ == "__main__":
    main()
