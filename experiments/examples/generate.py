import argparse
import torch
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.eps_net import EpsilonNetwork
from data.graph_builder import build_grid_laplacian, build_molecular_graph
from core.precompute import SpectralDiffusion
from core.advanced_sampling import ddim_sample
from core.sampling import sample_reverse_from_S_T, compute_F_Q_from_PhiSigma

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate 3D samples using trained Diffusion Model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--config', type=str, default='experiments/configs/default.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='generated_samples.pt', help='Output file path')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps (for DDIM)')
    parser.add_argument('--method', type=str, default='ddim', choices=['ddim', 'ddpm'], help='Sampling method')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    print(f"Loading config from {args.config}...")
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found. Using defaults.")
        config = {
            'graph': {'type': 'grid', 'grid_size': [10, 10]},
            'model': {'input_dim': 3, 'hidden_dim': 128, 'num_layers': 4, 'dropout': 0.1},
            'diffusion': {'num_steps': 1000}
        }
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Setup Graph
    print("Setting up graph structure...")
    graph_type = config['graph'].get('type', 'grid')
    
    if graph_type == 'grid':
        grid_size = tuple(config['graph'].get('grid_size', [10, 10]))
        print(f"Building {grid_size} grid...")
        L = build_grid_laplacian(grid_size).to(device)
        num_nodes = int(torch.prod(torch.tensor(grid_size)))
    elif graph_type == 'molecular':
        # For inference without a specific molecule, we fallback to a default grid or ask user.
        # In a real app, we would load a template PDB/XYZ here.
        print("Warning: 'molecular' graph type requires a structure definition.")
        print("Falling back to 10x10 grid for demonstration.")
        L = build_grid_laplacian((10, 10)).to(device)
        num_nodes = 100
    else:
        # Fallback
        print(f"Unknown graph type: {graph_type}, defaulting to 10x10 grid")
        L = build_grid_laplacian((10, 10)).to(device)
        num_nodes = 100
        
    # 2. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model_config = config['model']
    model = EpsilonNetwork(
        input_dim=model_config.get('input_dim', 3),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 4),
        dropout=model_config.get('dropout', 0.1)
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print(f"Checkpoint {args.checkpoint} not found!")
        return

    model.eval()
    
    # 3. Precompute Diffusion Matrices
    print("Precomputing diffusion matrices...")
    # We need to know the training num_steps to match the schedule
    train_num_steps = config['diffusion'].get('num_steps', 1000)
    spectral_diff = SpectralDiffusion(L, num_steps=train_num_steps, device=device)
    t_all = torch.arange(train_num_steps + 1, device=device)
    Phi, Sigma = spectral_diff.get_phi_sigma(t_all)
    
    # 4. Generate
    print(f"Generating {args.num_samples} samples using {args.method}...")
    shape = (args.num_samples, num_nodes, model_config.get('input_dim', 3))
    
    if args.method == 'ddim':
        samples, _ = ddim_sample(
            model=model,
            L=L,
            shape=shape,
            Phi_all=Phi,
            Sigma_all=Sigma,
            num_inference_steps=args.steps,
            device=device
        )
    else: # ddpm
        print("Computing F, Q for DDPM...")
        F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
        S_T = torch.randn(shape, device=device)
        samples, _ = sample_reverse_from_S_T(
            S_T, F, Q, model, L, num_steps=train_num_steps
        )
        
    # 5. Save
    print(f"Saving samples to {args.output}...")
    torch.save(samples, args.output)
    
    # Optional: Save as XYZ for visualization if 3D
    if args.output.endswith('.pt'):
        xyz_path = args.output.replace('.pt', '.xyz')
        print(f"Exporting to {xyz_path}...")
        with open(xyz_path, 'w') as f:
            for i in range(min(args.num_samples, 5)): # Save first 5
                f.write(f"{num_nodes}\n")
                f.write(f"Sample {i}\n")
                sample_np = samples[i].cpu().numpy()
                for j in range(num_nodes):
                    f.write(f"C {sample_np[j,0]:.6f} {sample_np[j,1]:.6f} {sample_np[j,2]:.6f}\n")
                    
    print("Done.")

if __name__ == "__main__":
    main()
