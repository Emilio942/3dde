"""Training script with configuration support."""

import argparse
import torch
from pathlib import Path

from experiments.config import load_config
from src.data.graph_builder import build_grid_laplacian, build_molecular_graph
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.eps_net import EpsilonNetwork
from src.training.trainer import DiffusionTrainer


def create_graph_from_config(config):
    """Create graph Laplacian from config."""
    graph_type = config.graph.type
    
    if graph_type == "grid":
        grid_size = tuple(config.graph.grid_size)
        return build_grid_laplacian(grid_size)
    
    elif graph_type == "molecular":
        # For molecular graphs, we need positions
        if hasattr(config.graph, 'num_atoms') and config.graph.num_atoms > 0:
            # Generate random positions for testing/synthetic data
            # In a real scenario, this would load from a file
            positions = torch.randn(config.graph.num_atoms, 3)
            cutoff = getattr(config.graph, 'cutoff', 2.0)
            L, _ = build_molecular_graph(positions, cutoff=cutoff)
            return L
        elif hasattr(config.graph, 'file_path') and config.graph.file_path:
            # TODO: Implement file loading (PDB/XYZ)
            raise NotImplementedError("Loading molecular graph from file not yet implemented")
        else:
            raise ValueError("Molecular graph requires 'num_atoms' in config for synthetic generation")
    
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def create_model_from_config(config):
    """Create model from config."""
    return EpsilonNetwork(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        time_dim=config.model.time_dim,
        use_attention=config.model.use_attention,
        dropout=config.model.dropout
    )


def create_dataset_from_config(config, num_nodes):
    """Create dataset from config."""
    return SyntheticDataset(
        num_samples=config.data.num_samples,
        num_nodes=num_nodes,
        node_dim=config.data.node_dim,
        mode=config.data.mode,
        noise_level=config.data.noise_level,
        seed=config.seed
    )


def create_trainer_from_config(config, model, L):
    """Create trainer from config."""
    # Loss config
    loss_config = {
        'eps_weight': config.training.loss.eps_weight,
        'regularization_weight': config.training.loss.regularization_weight
    }
    
    # Regularization config
    reg_config = {
        'lambda_energy': config.training.regularization.lambda_energy,
        'lambda_smoothness': config.training.regularization.lambda_smoothness,
        'lambda_alignment': config.training.regularization.lambda_alignment,
        'lambda_divergence': config.training.regularization.lambda_divergence
    }
    
    # Only use regularization if any lambda > 0
    if not any(reg_config.values()):
        reg_config = None
    
    # Check for spectral optimization flag in config, default to True if not present
    use_spectral = getattr(config.diffusion, 'use_spectral', True)

    return DiffusionTrainer(
        model=model,
        L=L,
        num_steps=config.diffusion.num_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        loss_config=loss_config,
        reg_config=reg_config,
        device=config.device,
        checkpoint_dir=config.logging.checkpoint_dir,
        log_interval=config.logging.log_interval,
        save_interval=getattr(config.logging, 'save_interval', 0),
        use_spectral=use_spectral,
        beta_schedule_type=config.diffusion.beta_schedule,
        lambd=config.diffusion.lambd,
        diag_approx=config.diffusion.diag_approx
    )


def main():
    parser = argparse.ArgumentParser(description="Train 3D Diffusion Model")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: use default config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    if args.config:
        config = load_config(args.config)
        print(f"  Loaded config from: {args.config}")
    else:
        config = load_config()
        print("  Using default configuration")
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Create graph
    print("\nCreating graph structure...")
    L = create_graph_from_config(config)
    num_nodes = L.shape[0]
    print(f"  Graph: {config.graph.type}")
    print(f"  Nodes: {num_nodes}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = create_dataset_from_config(config, num_nodes)
    train_loader, val_loader = create_dataloaders(
        dataset.data,
        batch_size=config.training.batch_size,
        val_split=config.data.val_split
    )
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model_from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: EpsilonNetwork")
    print(f"  Parameters: {num_params:,}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Layers: {config.model.num_layers}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = create_trainer_from_config(config, model, L)
    print(f"  Diffusion steps: {config.diffusion.num_steps}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Device: {config.device}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        if checkpoint:
            print(f"  Resumed from epoch {checkpoint['epoch']}")
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {trainer.best_loss:.6f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()
