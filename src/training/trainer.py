"""Training pipeline for diffusion model.

Handles training loop, optimization, logging, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import time

from ..core.precompute import precompute_phi_sigma_explicit, get_Phi_Sigma_at_t
from ..core.forward import forward_noising_batch, sample_timesteps
from ..training.losses import CombinedLoss
from ..training.regularizers import CombinedPhysicsRegularizer


class DiffusionTrainer:
    """Trainer for diffusion model."""
    
    def __init__(
        self,
        model: nn.Module,
        L: torch.Tensor,
        num_steps: int = 1000,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_config: Optional[Dict] = None,
        reg_config: Optional[Dict] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100
    ):
        """
        Args:
            model: Epsilon prediction network
            L: Graph Laplacian (N, N)
            num_steps: Number of diffusion steps
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            loss_config: Configuration for loss function
            reg_config: Configuration for regularizers
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_interval: Steps between logging
        """
        self.model = model.to(device)
        self.L = L.to(device)
        self.num_steps = num_steps
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        # Precompute Φ and Σ
        print("Precomputing diffusion matrices...")
        self.Phi, self.Sigma = precompute_phi_sigma_explicit(
            L, num_steps=num_steps, diag_approx=True
        )
        self.Phi = self.Phi.to(device)
        self.Sigma = self.Sigma.to(device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function
        loss_config = loss_config or {}
        self.loss_fn = CombinedLoss(**loss_config)
        
        # Setup regularizers
        reg_config = reg_config or {}
        if reg_config:
            self.regularizer = CombinedPhysicsRegularizer(**reg_config)
        else:
            self.regularizer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_eps_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get clean data (assumes batch is dict with 'states')
            S_0 = batch['states'].to(self.device)  # (B, N, d)
            B = S_0.shape[0]
            
            # Sample timesteps
            t_indices = sample_timesteps(B, self.num_steps, self.device, strategy="uniform")
            
            # Get Φ(t) and Σ(t)
            Phi_t, Sigma_t = get_Phi_Sigma_at_t(self.Phi, self.Sigma, t_indices)
            
            # Forward noising
            S_t, eps_true = forward_noising_batch(S_0, Phi_t, Sigma_t)
            
            # Predict noise
            eps_pred = self.model(S_t, t_indices, self.L)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(eps_pred, eps_true)
            
            # Add regularization if enabled
            if self.regularizer is not None:
                # Reconstruct S_0 from prediction (optional)
                from ..core.forward import compute_hat_S0_from_eps_hat
                S_0_pred = compute_hat_S0_from_eps_hat(S_t, eps_pred, Phi_t, Sigma_t)
                
                reg_loss, reg_dict = self.regularizer(
                    S_pred=S_0_pred,
                    L=self.L,
                    S_reference=S_0
                )
                loss = loss + reg_loss
                total_reg_loss += reg_loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_eps_loss += loss_dict['eps_loss']
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'eps_loss': loss_dict['eps_loss']
            })
            
            # Periodic logging
            if self.global_step % self.log_interval == 0:
                self._log_step(loss.item(), loss_dict)
        
        # Epoch metrics
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_eps_loss': total_eps_loss / num_batches,
            'train_reg_loss': total_reg_loss / num_batches if self.regularizer else 0.0
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_eps_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            S_0 = batch['states'].to(self.device)
            B = S_0.shape[0]
            
            # Sample timesteps
            t_indices = sample_timesteps(B, self.num_steps, self.device)
            
            # Get Φ(t) and Σ(t)
            Phi_t, Sigma_t = get_Phi_Sigma_at_t(self.Phi, self.Sigma, t_indices)
            
            # Forward noising
            S_t, eps_true = forward_noising_batch(S_0, Phi_t, Sigma_t)
            
            # Predict
            eps_pred = self.model(S_t, t_indices, self.L)
            
            # Loss
            loss, loss_dict = self.loss_fn(eps_pred, eps_true)
            
            total_loss += loss.item()
            total_eps_loss += loss_dict['eps_loss']
            num_batches += 1
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_eps_loss': total_eps_loss / num_batches
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {}
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Log epoch
            epoch_time = time.time() - start_time
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)
            
            # Save checkpoint
            is_best = val_metrics.get('val_loss', train_metrics['train_loss']) < self.best_loss
            if is_best:
                self.best_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        print("Training completed!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (loss: {self.best_loss:.6f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            checkpoint: The loaded checkpoint dict
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint
    
    def _log_step(self, loss: float, loss_dict: Dict[str, float]):
        """Log training step."""
        pass  # Can add tensorboard/wandb logging here
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log epoch results."""
        print(f"\nEpoch {epoch}:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train loss: {train_metrics['train_loss']:.6f}")
        if val_metrics:
            print(f"  Val loss: {val_metrics['val_loss']:.6f}")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6e}")


if __name__ == "__main__":
    print("Testing trainer (minimal test without real data)...")
    
    from ..models.eps_net import EpsilonNetwork
    from ..data.graph_builder import build_grid_laplacian
    from torch.utils.data import TensorDataset
    
    # Setup
    N = 25  # 5x5 grid
    d = 3
    
    # Create Laplacian
    L = build_grid_laplacian((5, 5))
    
    # Create model
    model = EpsilonNetwork(input_dim=d, hidden_dim=32, num_layers=2)
    
    print(f"\nTest configuration:")
    print(f"  Nodes: {N}, Dimensions: {d}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy dataset
    dummy_data = torch.randn(100, N, d)
    dataset = TensorDataset(dummy_data)
    # Wrap in dict format expected by trainer
    class DictDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return {'states': self.data[idx]}
    
    train_dataset = DictDataset(dummy_data)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        L=L,
        num_steps=100,
        learning_rate=1e-3,
        device="cpu",
        checkpoint_dir="./test_checkpoints"
    )
    
    print("\n=== Test: Single Epoch ===")
    # Train for one epoch
    metrics = trainer.train_epoch(train_loader, epoch=0)
    print(f"Train loss: {metrics['train_loss']:.6f}")
    
    # Test checkpoint saving/loading
    print("\n=== Test: Checkpoint ===")
    trainer.save_checkpoint(epoch=0, is_best=True)
    print("✓ Checkpoint saved")
    
    trainer.load_checkpoint("./test_checkpoints/best.pt")
    print("✓ Checkpoint loaded")
    
    print("\n✓ Trainer test passed!")
