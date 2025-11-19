import torch
import argparse
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.graph_builder import build_grid_laplacian, get_grid_edges
from evaluation.metrics import PhysicalEvaluator
from models.eps_net import EpsilonNetwork
from core.sampling import compute_F_Q_from_PhiSigma, sample_reverse_from_S_T
from core.precompute import SpectralDiffusion

def evaluate_model(config_path: str, checkpoint_path: str, num_samples: int = 16):
    """
    Generates samples and evaluates their physical plausibility.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Setup Graph (Grid for now, as per current status)
    N = 10 # 10x10 grid
    num_nodes = N * N
    print(f"Setting up {N}x{N} grid graph...")
    L = build_grid_laplacian((N, N)).to(device)
    edge_index = get_grid_edges((N, N)).to(device)
    
    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model = EpsilonNetwork(
        input_dim=3,
        hidden_dim=128, # Matching default config
        num_layers=4
    ).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    
    # 3. Prepare Sampling
    print("Precomputing diffusion matrices (Spectral)...")
    num_steps = 1000
    spectral_diff = SpectralDiffusion(L, num_steps=num_steps, device=device)
    
    # Get full matrices for sampling (needed for F, Q computation)
    # Note: For very large graphs, we would need a more memory efficient sampler
    # but for evaluation N=100 is fine.
    t_all = torch.arange(num_steps + 1, device=device)
    Phi, Sigma = spectral_diff.get_phi_sigma(t_all)
    
    print("Computing reverse transition matrices...")
    F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
    
    # 4. Generate Samples
    print(f"Generating {num_samples} samples...")
    S_T = torch.randn(num_samples, num_nodes, 3, device=device)
    
    with torch.no_grad():
        S_0_pred, _ = sample_reverse_from_S_T(
            S_T, F, Q, model, L, num_steps=num_steps
        )
    
    # 5. Evaluate
    print("\nEvaluating Physical Plausibility...")
    evaluator = PhysicalEvaluator(L, edge_index)
    metrics = evaluator.evaluate_batch(S_0_pred)
    
    print("-" * 40)
    print("QUALITY METRICS REPORT")
    print("-" * 40)
    print(f"Connectivity Score: {metrics['connectivity']:.4f} (Target: 1.0)")
    print(f"Smoothness Energy:  {metrics['smoothness']:.4f} (Lower is better)")
    print(f"Edge Length Std:    {metrics['edge_len_std']:.4f} (Lower is better)")
    print("-" * 40)
    
    # Interpretation
    if metrics['connectivity'] > 0.95:
        print("✅ Structure is coherent (fully connected).")
    else:
        print("⚠️ Structure is fragmented (flying artifacts detected).")
        
    if metrics['smoothness'] < 1.0: # Threshold depends on scale, heuristic
        print("✅ Surface is smooth.")
    else:
        print("⚠️ Surface is rough/noisy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint}. Please train the model first.")
    else:
        evaluate_model(None, args.checkpoint)
