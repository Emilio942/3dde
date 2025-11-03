"""Visualize training results and generated samples."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup
checkpoint_dir = Path("checkpoints")
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("3D Diffusion Model - Results Visualization")
print("=" * 60)

# Load checkpoint
checkpoint_path = checkpoint_dir / "best.pt"
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n[1/3] Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['best_loss']:.6f}")
    
    # Plot training history
    if 'history' in checkpoint:
        history = checkpoint['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax = axes[0]
        epochs = range(len(history['train_loss']))
        ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss components
        ax = axes[1]
        if 'eps_loss' in history:
            ax.plot(epochs, history['eps_loss'], label='Epsilon Loss', marker='o')
        if 'reg_loss' in history:
            ax.plot(epochs, history['reg_loss'], label='Regularization Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Component')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved training curves to {plot_path}")
        plt.close()

# Visualize generated samples
print("\n[2/3] Visualizing generated samples...")

# Load from train_example output if available
samples_file = Path("generated_samples.pt")
if samples_file.exists():
    data = torch.load(samples_file)
    samples = data['samples']  # (B, N, d)
    trajectory = data.get('trajectory', None)  # (num_steps, B, N, d)
    
    B, N, d = samples.shape
    print(f"  Samples shape: {samples.shape}")
    
    # Plot sample grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < B:
            sample = samples[i].numpy()  # (N, 3)
            
            # Create grid positions (5x5)
            grid_size = int(np.sqrt(N))
            x_grid = np.arange(grid_size)
            y_grid = np.arange(grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.stack([X.flatten(), Y.flatten()], axis=1)
            
            # Plot vectors
            ax.quiver(
                positions[:, 0], positions[:, 1], np.zeros(N),
                sample[:, 0], sample[:, 1], sample[:, 2],
                length=0.3, normalize=True, arrow_length_ratio=0.3
            )
            
            # Scatter points
            ax.scatter(positions[:, 0], positions[:, 1], np.zeros(N),
                      c='red', s=50, alpha=0.6)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Sample {i+1}')
            ax.set_xlim(-1, grid_size)
            ax.set_ylim(-1, grid_size)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / "generated_samples.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved sample visualization to {plot_path}")
    plt.close()
    
    # Plot diffusion trajectory if available
    if trajectory is not None:
        print("\n[3/3] Visualizing diffusion trajectory...")
        
        num_steps = trajectory.shape[0]
        sample_idx = 0  # Show first sample
        
        # Select timesteps to visualize
        timesteps = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
        
        fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 3), subplot_kw={'projection': '3d'})
        
        for idx, t in enumerate(timesteps):
            ax = axes[idx]
            state = trajectory[t, sample_idx].numpy()  # (N, 3)
            
            grid_size = int(np.sqrt(N))
            x_grid = np.arange(grid_size)
            y_grid = np.arange(grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.stack([X.flatten(), Y.flatten()], axis=1)
            
            ax.quiver(
                positions[:, 0], positions[:, 1], np.zeros(N),
                state[:, 0], state[:, 1], state[:, 2],
                length=0.3, normalize=True, arrow_length_ratio=0.3,
                color='blue', alpha=0.7
            )
            
            ax.scatter(positions[:, 0], positions[:, 1], np.zeros(N),
                      c='red', s=30, alpha=0.5)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f't = {t}/{num_steps-1}')
            ax.set_xlim(-1, grid_size)
            ax.set_ylim(-1, grid_size)
        
        plt.tight_layout()
        plot_path = output_dir / "diffusion_trajectory.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved trajectory visualization to {plot_path}")
        plt.close()
else:
    print("  No samples file found. Run train_example.py first with sample saving.")

print("\n" + "=" * 60)
print("Visualization complete!")
print(f"Check the '{output_dir}/' directory for results.")
print("=" * 60)
