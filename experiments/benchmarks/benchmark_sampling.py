import torch
import time
import argparse
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.graph_builder import build_grid_laplacian
from models.eps_net import EpsilonNetwork
from core.sampling import compute_F_Q_from_PhiSigma, sample_reverse_from_S_T
from core.precompute import SpectralDiffusion
from core.advanced_sampling import ddim_sample

def benchmark_sampling(checkpoint_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Setup
    N = 10 # 10x10 grid
    num_nodes = N * N
    L = build_grid_laplacian((N, N)).to(device)
    
    # Load Model
    model = EpsilonNetwork(input_dim=3, hidden_dim=128, num_layers=4).to(device)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded.")
        except:
            print("Could not load checkpoint, using random weights.")
    else:
        print("Checkpoint not found, using random weights.")
    model.eval()
    
    # Precompute
    num_steps = 1000
    spectral_diff = SpectralDiffusion(L, num_steps=num_steps, device=device)
    t_all = torch.arange(num_steps + 1, device=device)
    Phi, Sigma = spectral_diff.get_phi_sigma(t_all)
    
    # Standard Sampling Setup
    print("Precomputing F, Q for standard sampling...")
    F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
    
    batch_size = 16
    shape = (batch_size, num_nodes, 3)
    S_T = torch.randn(shape, device=device)
    
    print(f"\nBenchmarking Sampling (Batch Size: {batch_size})")
    print("-" * 60)
    print(f"{'Method':<20} | {'Steps':<10} | {'Time (s)':<10} | {'Speedup':<10}")
    print("-" * 60)
    
    # 1. Standard Sampling (DDPM-like)
    start_time = time.time()
    with torch.no_grad():
        _ = sample_reverse_from_S_T(S_T, F, Q, model, L, num_steps=num_steps)
    end_time = time.time()
    std_time = end_time - start_time
    print(f"{'Standard (DDPM)':<20} | {num_steps:<10} | {std_time:<10.4f} | {'1.0x':<10}")
    
    # 2. DDIM Sampling (Various steps)
    ddim_steps_list = [100, 50, 20, 10]
    
    for steps in ddim_steps_list:
        start_time = time.time()
        with torch.no_grad():
            _ = ddim_sample(
                model, L, shape, Phi, Sigma, 
                num_inference_steps=steps, 
                eta=0.0, 
                device=device, 
                verbose=False
            )
        end_time = time.time()
        ddim_time = end_time - start_time
        speedup = std_time / ddim_time
        print(f"{'DDIM':<20} | {steps:<10} | {ddim_time:<10.4f} | {speedup:<10.1f}x")
        
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    args = parser.parse_args()
    benchmark_sampling(args.checkpoint)
