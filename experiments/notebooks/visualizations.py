"""
Visualization utilities for 3D Diffusion Model.

This notebook demonstrates how to visualize:
1. Training progress
2. Diffusion process (forward and reverse)
3. Generated samples
4. Graph structures
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path.cwd().parent))

from src.data.graph_builder import build_grid_laplacian
from src.core.precompute import precompute_phi_sigma_explicit
from src.core.forward import forward_noising_batch
from src.core.sampling import compute_F_Q_from_PhiSigma, sample_reverse_from_S_T
from src.models.eps_net import EpsilonNetwork


# ============================================================================
# 1. SETUP
# ============================================================================

print("Setting up...")

# Create graph
grid_size = (5, 5)
L = build_grid_laplacian(grid_size)
num_nodes = L.shape[0]

# Precompute matrices
num_steps = 50
Phi, Sigma = precompute_phi_sigma_explicit(L, num_steps=num_steps)
F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)

print(f"✓ Graph: {grid_size} grid with {num_nodes} nodes")
print(f"✓ Diffusion: {num_steps} steps")


# ============================================================================
# 2. VISUALIZE FORWARD DIFFUSION
# ============================================================================

def visualize_forward_process(S_0, Phi, Sigma, num_steps, timesteps_to_show=None):
    """Visualize forward diffusion process."""
    if timesteps_to_show is None:
        timesteps_to_show = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps]
    
    n_plots = len(timesteps_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    for idx, t in enumerate(timesteps_to_show):
        ax = axes[idx] if n_plots > 1 else axes
        
        # Forward noise to timestep t
        S_t, _ = forward_noising_batch(S_0, t, Phi, Sigma)
        
        # Plot first sample, first dimension
        data = S_t[0, :, 0].detach().numpy()
        
        # Reshape for grid
        grid_data = data.reshape(grid_size)
        
        im = ax.imshow(grid_data, cmap='viridis', aspect='auto')
        ax.set_title(f't = {t}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


print("\n2. Creating sample data and visualizing forward process...")

# Create smooth sample
S_0 = torch.randn(1, num_nodes, 3)
S_0 = S_0 * 5  # Scale up

fig = visualize_forward_process(S_0, Phi, Sigma, num_steps)
plt.savefig('forward_diffusion.png', dpi=150, bbox_inches='tight')
print("✓ Saved: forward_diffusion.png")
plt.close()


# ============================================================================
# 3. VISUALIZE REVERSE SAMPLING
# ============================================================================

def visualize_reverse_process(S_T, F, Q, model, L, num_steps, timesteps_to_show=None):
    """Visualize reverse sampling process."""
    if timesteps_to_show is None:
        # Show equally spaced timesteps
        timesteps_to_show = np.linspace(num_steps, 0, 5, dtype=int)
    
    # Sample with trajectory
    model.eval()
    with torch.no_grad():
        S_0, trajectory = sample_reverse_from_S_T(
            S_T, F, Q, model, L, num_steps
        )
    
    n_plots = len(timesteps_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    for idx, t in enumerate(timesteps_to_show):
        ax = axes[idx] if n_plots > 1 else axes
        
        # Get state at timestep t
        S_t = trajectory[num_steps - t]  # Reverse indexing
        
        # Plot first sample, first dimension
        data = S_t[0, :, 0].detach().numpy()
        grid_data = data.reshape(grid_size)
        
        im = ax.imshow(grid_data, cmap='plasma', aspect='auto')
        ax.set_title(f't = {t}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig, S_0


print("\n3. Visualizing reverse sampling process...")

# Create simple model (untrained for demo)
model = EpsilonNetwork(input_dim=3, hidden_dim=64, num_layers=3)

# Start from noise
S_T = torch.randn(1, num_nodes, 3)

fig, S_0_generated = visualize_reverse_process(
    S_T, F, Q, model, L, num_steps
)
plt.savefig('reverse_sampling.png', dpi=150, bbox_inches='tight')
print("✓ Saved: reverse_sampling.png")
plt.close()


# ============================================================================
# 4. VISUALIZE DIFFUSION MATRICES
# ============================================================================

def visualize_phi_sigma_evolution(Phi, Sigma, num_steps):
    """Visualize evolution of Φ and Σ."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    timesteps = [0, num_steps//2, num_steps]
    
    for idx, t in enumerate(timesteps):
        # Φ matrix
        ax = axes[0, idx]
        im = ax.imshow(Phi[t].detach().numpy(), cmap='RdBu', center=0)
        ax.set_title(f'Φ(t={t})')
        ax.set_xlabel('Node')
        ax.set_ylabel('Node')
        plt.colorbar(im, ax=ax)
        
        # Σ matrix
        ax = axes[1, idx]
        im = ax.imshow(Sigma[t].detach().numpy(), cmap='YlOrRd')
        ax.set_title(f'Σ(t={t})')
        ax.set_xlabel('Node')
        ax.set_ylabel('Node')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


print("\n4. Visualizing Φ and Σ matrices...")

fig = visualize_phi_sigma_evolution(Phi, Sigma, num_steps)
plt.savefig('phi_sigma_evolution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: phi_sigma_evolution.png")
plt.close()


# ============================================================================
# 5. VISUALIZE GRAPH STRUCTURE
# ============================================================================

def visualize_graph_laplacian(L, grid_size):
    """Visualize graph Laplacian."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Laplacian matrix
    ax = axes[0]
    im = ax.imshow(L.numpy(), cmap='RdBu', center=0)
    ax.set_title('Graph Laplacian L')
    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    plt.colorbar(im, ax=ax)
    
    # Eigenvalues
    ax = axes[1]
    eigvals = torch.linalg.eigvalsh(L).numpy()
    ax.plot(eigvals, 'o-')
    ax.set_title('Eigenvalues of L')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


print("\n5. Visualizing graph structure...")

fig = visualize_graph_laplacian(L, grid_size)
plt.savefig('graph_structure.png', dpi=150, bbox_inches='tight')
print("✓ Saved: graph_structure.png")
plt.close()


# ============================================================================
# 6. STATISTICS PLOTS
# ============================================================================

def plot_diffusion_statistics(Phi, Sigma, num_steps):
    """Plot statistics of diffusion process."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trace of Σ (total variance)
    ax = axes[0, 0]
    traces = [torch.trace(Sigma[t]).item() for t in range(num_steps + 1)]
    ax.plot(traces)
    ax.set_title('Total Variance (Trace of Σ)')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Tr(Σ)')
    ax.grid(True, alpha=0.3)
    
    # Norm of Φ
    ax = axes[0, 1]
    phi_norms = [torch.norm(Phi[t]).item() for t in range(num_steps + 1)]
    ax.plot(phi_norms)
    ax.set_title('Norm of Φ')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('||Φ||')
    ax.grid(True, alpha=0.3)
    
    # Diagonal of Σ
    ax = axes[1, 0]
    for node in [0, num_nodes//4, num_nodes//2, num_nodes-1]:
        diag_vals = [Sigma[t, node, node].item() for t in range(num_steps + 1)]
        ax.plot(diag_vals, label=f'Node {node}')
    ax.set_title('Diagonal Elements of Σ')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Σ[i,i]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Condition number
    ax = axes[1, 1]
    cond_nums = []
    for t in range(0, num_steps + 1, max(1, num_steps//20)):
        eigvals = torch.linalg.eigvalsh(Sigma[t] + 1e-6 * torch.eye(num_nodes))
        cond = (eigvals.max() / eigvals.min()).item()
        cond_nums.append((t, cond))
    
    ts, conds = zip(*cond_nums)
    ax.semilogy(ts, conds, 'o-')
    ax.set_title('Condition Number of Σ')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('cond(Σ)')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


print("\n6. Plotting diffusion statistics...")

fig = plot_diffusion_statistics(Phi, Sigma, num_steps)
plt.savefig('diffusion_statistics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: diffusion_statistics.png")
plt.close()


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print("\nGenerated plots:")
print("  1. forward_diffusion.png     - Forward noising process")
print("  2. reverse_sampling.png      - Reverse sampling process")
print("  3. phi_sigma_evolution.png   - Φ and Σ matrices over time")
print("  4. graph_structure.png       - Graph Laplacian and eigenvalues")
print("  5. diffusion_statistics.png  - Diffusion process statistics")
print("\n" + "="*70)
