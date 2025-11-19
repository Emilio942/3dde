import argparse
import time
import sys
import os
from memory_profiler import profile

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.graph_builder import build_grid_laplacian
from core.precompute import precompute_phi_sigma_explicit
import torch # Import torch for tensor operations

@profile
def run_dense_computation(N: int):
    """
    Runs the dense matrix computation for a given grid size N.
    """
    print(f"Running dense computation for N={N}x{N} grid...")
    
    # Build the Laplacian matrix
    start_time = time.time()
    # Corrected call: build_grid_laplacian expects a tuple for grid_size
    L_torch = build_grid_laplacian((N, N))
    L = L_torch.float() # Ensure it's a float tensor
    laplacian_build_time = time.time() - start_time
    print(f"  Laplacian build time: {laplacian_build_time:.4f} seconds")

    # Precompute Phi and Sigma
    start_time = time.time()
    
    num_steps = 100 # Using a smaller num_steps for benchmarking
    T = 1.0
    gamma = 1.0
    lambda_reg = 1e-6
    
    try:
        Phi, Sigma = precompute_phi_sigma_explicit(
            L, 
            num_steps=num_steps, 
            T=T, 
            gamma=gamma, 
            lambda_reg=lambda_reg
        ) 
        precompute_time = time.time() - start_time
        print(f"  Phi/Sigma precompute time: {precompute_time:.4f} seconds")
        print(f"  Phi shape: {Phi.shape}, Sigma shape: {Sigma.shape}")
        # To avoid unused variable warnings and ensure computation is not optimized away
        _ = Phi, Sigma 
    except Exception as e:
        print(f"  Error during precomputation: {e}")
        precompute_time = -1


    total_time = laplacian_build_time + precompute_time if precompute_time != -1 else laplacian_build_time
    print(f"Total dense computation time: {total_time:.4f} seconds")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the scalability of the 3DDE model's core computation.")
    parser.add_argument("--N", type=int, default=10,
                        help="Size of the N x N grid for the Laplacian matrix. (e.g., N=10 means 10x10 grid)")
    
    args = parser.parse_args()

    print("Note: For memory profiling, run this script with 'python -m memory_profiler'.")
    print(f"Example: .venv/bin/python3.12 -m memory_profiler scalability_benchmark.py --N {args.N}")
    
    run_dense_computation(args.N)
