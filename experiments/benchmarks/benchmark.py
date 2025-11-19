#!/usr/bin/env python
"""
Benchmark Script for 3D Diffusion Model

Compares the diffusion model against baseline methods:
1. Simple Gaussian noise
2. Linear interpolation
3. Random sampling

Metrics:
- Sample quality (MSE, distribution match)
- Physical plausibility (energy, smoothness)
- Computational efficiency (time, memory)
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.data.graph_builder import build_grid_laplacian
from src.core.precompute import precompute_phi_sigma_explicit
from src.core.sampling import sample_reverse_from_S_T
from src.models.eps_net import EpsilonNetwork


def generate_reference_data(num_samples, num_nodes, d_node):
    """Generate reference 'clean' samples for comparison."""
    # Simple synthetic data: smooth vector fields with some structure
    S_ref = torch.randn(num_samples, num_nodes, d_node) * 0.5
    
    # Add smoothness by averaging with neighbors (simple blur)
    grid_size = int(np.sqrt(num_nodes))
    S_ref_grid = S_ref.view(num_samples, grid_size, grid_size, d_node)
    
    # Apply Gaussian blur-like operation
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            neighbors = (
                S_ref_grid[:, i-1, j] + S_ref_grid[:, i+1, j] +
                S_ref_grid[:, i, j-1] + S_ref_grid[:, i, j+1]
            ) / 4
            S_ref_grid[:, i, j] = 0.7 * S_ref_grid[:, i, j] + 0.3 * neighbors
    
    return S_ref_grid.view(num_samples, num_nodes, d_node)


class BaselineGaussian:
    """Baseline: Simple Gaussian noise."""
    def __init__(self, num_nodes, d_node):
        self.num_nodes = num_nodes
        self.d_node = d_node
    
    def sample(self, num_samples):
        """Generate samples from standard Gaussian."""
        return torch.randn(num_samples, self.num_nodes, self.d_node)


class BaselineLinearInterp:
    """Baseline: Linear interpolation from noise to mean."""
    def __init__(self, num_nodes, d_node, reference_mean=None):
        self.num_nodes = num_nodes
        self.d_node = d_node
        self.reference_mean = reference_mean if reference_mean is not None else torch.zeros(num_nodes, d_node)
    
    def sample(self, num_samples):
        """Generate samples by linear interpolation."""
        noise = torch.randn(num_samples, self.num_nodes, self.d_node)
        # Interpolate: 0.3 * noise + 0.7 * target
        alpha = 0.3
        return alpha * noise + (1 - alpha) * self.reference_mean.unsqueeze(0)


class BaselineRandom:
    """Baseline: Random samples from uniform distribution."""
    def __init__(self, num_nodes, d_node, scale=1.0):
        self.num_nodes = num_nodes
        self.d_node = d_node
        self.scale = scale
    
    def sample(self, num_samples):
        """Generate random uniform samples."""
        return (torch.rand(num_samples, self.num_nodes, self.d_node) - 0.5) * 2 * self.scale


def compute_metrics(samples, reference, L=None):
    """
    Compute evaluation metrics.
    
    Args:
        samples: Generated samples, shape (B, N, d)
        reference: Reference samples, shape (B, N, d)
        L: Graph Laplacian for smoothness metric
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # 1. Distribution metrics
    metrics['mean_mse'] = torch.mean((samples.mean(dim=0) - reference.mean(dim=0)) ** 2).item()
    metrics['std_mse'] = torch.mean((samples.std(dim=0) - reference.std(dim=0)) ** 2).item()
    
    # 2. Sample-wise MSE
    if samples.shape[0] == reference.shape[0]:
        metrics['sample_mse'] = torch.mean((samples - reference) ** 2).item()
    
    # 3. Vector magnitude statistics
    mag_samples = torch.norm(samples, dim=2)
    mag_reference = torch.norm(reference, dim=2)
    metrics['magnitude_mean'] = mag_samples.mean().item()
    metrics['magnitude_std'] = mag_samples.std().item()
    metrics['magnitude_mse'] = torch.mean((mag_samples.mean(dim=0) - mag_reference.mean(dim=0)) ** 2).item()
    
    # 4. Graph smoothness (if Laplacian provided)
    if L is not None:
        # Compute ||L @ S||^2 for each sample
        L_torch = torch.from_numpy(L) if isinstance(L, np.ndarray) else L
        smoothness = []
        for i in range(samples.shape[0]):
            S_flat = samples[i].view(-1)
            LS = torch.sparse.mm(L_torch.to_sparse(), samples[i]) if L_torch.is_sparse else L_torch @ samples[i]
            smoothness.append(torch.norm(LS) ** 2)
        metrics['graph_smoothness'] = torch.tensor(smoothness).mean().item()
    
    # 5. Energy (deviation from zero or reference)
    energy = torch.mean(torch.sum(samples ** 2, dim=(1, 2)))
    metrics['energy'] = energy.item()
    
    return metrics


def benchmark_method(method, num_samples, method_name, reference, L, device='cpu'):
    """Benchmark a single method."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {method_name}")
    print(f"{'='*60}")
    
    results = {
        'method': method_name,
        'num_samples': num_samples
    }
    
    # Time the sampling
    start_time = time.time()
    
    if method_name == "Diffusion Model":
        # Diffusion model sampling
        model, Phi, Sigma, beta_schedule, T_steps = method
        with torch.no_grad():
            S_T = torch.randn(num_samples, Phi.shape[1], 3, device=device)
            samples = sample_reverse_from_S_T(
                model=model,
                S_T=S_T,
                Phi=Phi,
                Sigma=Sigma,
                beta_schedule=beta_schedule,
                T_steps=T_steps,
                diag_approx=False,
                return_trajectory=False
            )
    else:
        # Baseline methods
        samples = method.sample(num_samples)
    
    elapsed_time = time.time() - start_time
    results['time'] = elapsed_time
    results['time_per_sample'] = elapsed_time / num_samples
    
    print(f"✓ Generated {num_samples} samples in {elapsed_time:.2f}s")
    print(f"  ({elapsed_time/num_samples*1000:.2f}ms per sample)")
    
    # Compute metrics
    metrics = compute_metrics(samples.cpu(), reference, L)
    results['metrics'] = metrics
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key:20s}: {value:.6f}")
    
    return results, samples


def plot_comparison(results_list, save_dir="visualizations"):
    """Plot comparison of all methods."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    methods = [r['method'] for r in results_list]
    
    # 1. Time comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Time per sample
    times = [r['time_per_sample'] * 1000 for r in results_list]  # Convert to ms
    axes[0, 0].bar(methods, times)
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].set_title('Time per Sample')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MSE metrics
    mean_mse = [r['metrics']['mean_mse'] for r in results_list]
    axes[0, 1].bar(methods, mean_mse)
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Mean MSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    std_mse = [r['metrics']['std_mse'] for r in results_list]
    axes[0, 2].bar(methods, std_mse)
    axes[0, 2].set_ylabel('MSE')
    axes[0, 2].set_title('Std MSE')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Magnitude statistics
    mag_mean = [r['metrics']['magnitude_mean'] for r in results_list]
    axes[1, 0].bar(methods, mag_mean)
    axes[1, 0].set_ylabel('Mean Magnitude')
    axes[1, 0].set_title('Vector Magnitude')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Graph smoothness
    if 'graph_smoothness' in results_list[0]['metrics']:
        smoothness = [r['metrics']['graph_smoothness'] for r in results_list]
        axes[1, 1].bar(methods, smoothness)
        axes[1, 1].set_ylabel('Smoothness')
        axes[1, 1].set_title('Graph Smoothness (lower = smoother)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Energy
    energy = [r['metrics']['energy'] for r in results_list]
    axes[1, 2].bar(methods, energy)
    axes[1, 2].set_ylabel('Energy')
    axes[1, 2].set_title('Total Energy')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {save_dir / 'benchmark_comparison.png'}")
    plt.close()


def main():
    """Run full benchmark suite."""
    print("=" * 70)
    print("3D DIFFUSION MODEL BENCHMARK")
    print("=" * 70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    grid_size = 5
    num_nodes = grid_size * grid_size
    d_node = 3
    num_samples = 100
    
    print(f"Configuration:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Node dimension: {d_node}")
    print(f"  Samples: {num_samples}")
    
    # Generate reference data
    print("\nGenerating reference data...")
    reference = generate_reference_data(num_samples, num_nodes, d_node)
    print(f"✓ Reference data shape: {reference.shape}")
    
    # Create graph
    print("\nCreating graph...")
    L = build_grid_laplacian((grid_size, grid_size))
    print(f"✓ Laplacian shape: {L.shape}")
    
    # Initialize methods
    baselines = {
        'Gaussian': BaselineGaussian(num_nodes, d_node),
        'Linear Interp': BaselineLinearInterp(num_nodes, d_node, reference.mean(dim=0)),
        'Random Uniform': BaselineRandom(num_nodes, d_node, scale=1.0)
    }
    
    # Load diffusion model
    print("\nLoading diffusion model...")
    print("⚠ Diffusion model sampling temporarily disabled (API refactoring in progress)")
    print("  Comparing baseline methods only...")
    diffusion_method = None
    
    # Run benchmarks
    results_list = []
    samples_dict = {}
    
    for name, method in baselines.items():
        results, samples = benchmark_method(method, num_samples, name, reference, L, device)
        results_list.append(results)
        samples_dict[name] = samples
    
    if diffusion_method is not None:
        results, samples = benchmark_method(diffusion_method, num_samples, "Diffusion Model", reference, L, device)
        results_list.append(results)
        samples_dict["Diffusion Model"] = samples
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating comparison plots...")
    plot_comparison(results_list)
    
    # Save results to JSON
    output_file = Path("benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"✓ Saved results to {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for results in results_list:
        print(f"\n{results['method']}:")
        print(f"  Time per sample: {results['time_per_sample']*1000:.2f}ms")
        print(f"  Mean MSE: {results['metrics']['mean_mse']:.6f}")
        print(f"  Graph smoothness: {results['metrics'].get('graph_smoothness', 'N/A')}")
    
    # Find best method
    if diffusion_method is not None:
        best_idx = min(range(len(results_list)), key=lambda i: results_list[i]['metrics']['mean_mse'])
        print(f"\n{'='*60}")
        print(f"WINNER: {results_list[best_idx]['method']} (lowest Mean MSE)")
        print(f"{'='*60}")
    
    print("\n✓ Benchmark complete!")


if __name__ == '__main__':
    main()
