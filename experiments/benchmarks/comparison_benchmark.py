import time
import sys
import os
import torch
import psutil
import gc
from memory_profiler import memory_usage

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.graph_builder import build_grid_laplacian
from core.precompute import precompute_phi_sigma_explicit, SpectralDiffusion

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_explicit(L, num_steps, device):
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_mem = get_memory_usage()
    start_time = time.time()
    
    try:
        Phi, Sigma = precompute_phi_sigma_explicit(
            L, 
            num_steps=num_steps,
            beta_schedule='linear'
        )
        # Force realization
        if device != 'cpu':
            torch.cuda.synchronize()
            
        end_time = time.time()
        end_mem = get_memory_usage()
        
        # Calculate size of tensors
        tensor_size = (Phi.element_size() * Phi.nelement() + Sigma.element_size() * Sigma.nelement()) / 1024 / 1024
        
        return {
            'time': end_time - start_time,
            'memory_increase': end_mem - start_mem,
            'tensor_size': tensor_size,
            'status': 'Success'
        }
    except Exception as e:
        return {
            'time': 0,
            'memory_increase': 0,
            'tensor_size': 0,
            'status': f'Failed: {str(e)}'
        }

def run_spectral(L, num_steps, device):
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_mem = get_memory_usage()
    start_time = time.time()
    
    try:
        spectral_diff = SpectralDiffusion(
            L, 
            num_steps=num_steps, 
            device=device,
            beta_schedule='linear'
        )
        
        # Simulate usage (getting one step)
        t_idx = torch.tensor([num_steps // 2], device=device)
        phi_t, sigma_t = spectral_diff.get_phi_sigma(t_idx)
        
        if device != 'cpu':
            torch.cuda.synchronize()
            
        end_time = time.time()
        end_mem = get_memory_usage()
        
        # Estimate stored size (eigenvalues + eigenvectors)
        stored_size = (spectral_diff.evals.element_size() * spectral_diff.evals.nelement() + 
                      spectral_diff.U.element_size() * spectral_diff.U.nelement()) / 1024 / 1024
        
        return {
            'time': end_time - start_time,
            'memory_increase': end_mem - start_mem,
            'tensor_size': stored_size,
            'status': 'Success'
        }
    except Exception as e:
        return {
            'time': 0,
            'memory_increase': 0,
            'tensor_size': 0,
            'status': f'Failed: {str(e)}'
        }

def run_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmark on {device.upper()}...")
    print("-" * 80)
    print(f"{'Grid Size':<15} | {'Method':<10} | {'Time (s)':<10} | {'Memory (MB)':<12} | {'Status':<15}")
    print("-" * 80)
    
    # Test cases: (Grid N, Num Steps)
    # Grid N=20 -> 400 nodes
    # Grid N=30 -> 900 nodes
    # Grid N=40 -> 1600 nodes
    configs = [
        (20, 1000),
        (30, 1000)
    ]
    
    results = []
    
    for N, steps in configs:
        num_nodes = N * N
        print(f"Preparing N={N} ({num_nodes} nodes)...")
        L = build_grid_laplacian((N, N)).to(device)
        
        # Explicit
        res_explicit = run_explicit(L, steps, device)
        print(f"{f'{N}x{N} ({steps})':<15} | {'Explicit':<10} | {res_explicit['time']:<10.4f} | {res_explicit['tensor_size']:<12.2f} | {res_explicit['status']:<15}")
        
        # Spectral
        res_spectral = run_spectral(L, steps, device)
        print(f"{f'{N}x{N} ({steps})':<15} | {'Spectral':<10} | {res_spectral['time']:<10.4f} | {res_spectral['tensor_size']:<12.2f} | {res_spectral['status']:<15}")
        
        results.append({
            'N': N,
            'explicit': res_explicit,
            'spectral': res_spectral
        })
        print("-" * 80)

    # Summary
    print("\nSummary of Improvements:")
    for res in results:
        if res['explicit']['status'] == 'Success' and res['spectral']['status'] == 'Success':
            mem_reduction = res['explicit']['tensor_size'] / res['spectral']['tensor_size']
            print(f"Grid {res['N']}x{res['N']}: Spectral is {mem_reduction:.1f}x more memory efficient.")

if __name__ == "__main__":
    run_benchmark()
