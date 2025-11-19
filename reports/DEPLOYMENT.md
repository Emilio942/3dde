# Deployment Guide

Complete guide for deploying the 3D Diffusion Model in production environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Docker Deployment](#docker-deployment)
4. [Production Configuration](#production-configuration)
5. [Monitoring & Logging](#monitoring--logging)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (Intel i5 or equivalent)
- **RAM**: 8 GB
- **Storage**: 2 GB free space
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: 8+ cores or **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA 11.8+)
- **RAM**: 16 GB+
- **Storage**: 10 GB+ SSD
- **OS**: Linux (Ubuntu 22.04)
- **Python**: 3.10+

### Software Dependencies
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NumPy 1.23+
- SciPy 1.10+
- Matplotlib 3.7+ (for visualization)

---

## Installation

### 1. Quick Install (pip)

```bash
# Clone repository
git clone https://github.com/your-org/3dde.git
cd 3dde

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
python -c "import src; print('✓ Installation successful')"
pytest src/tests/ -v
```

### 2. Install with GPU Support

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Install from Source (Development)

```bash
# Clone repository
git clone https://github.com/your-org/3dde.git
cd 3dde

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest src/tests/ -v

# Build documentation
cd docs
make html
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package
RUN pip install -e .

# Create directories for data and checkpoints
RUN mkdir -p /app/data /app/checkpoints /app/outputs

# Expose port for API (if applicable)
EXPOSE 8000

# Default command
CMD ["python", "train_example.py"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  diffusion-model:
    build: .
    container_name: 3dde-model
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./outputs:/app/outputs
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
      - OMP_NUM_THREADS=4
    command: python train_example.py
    
  # Optional: Add API service
  api:
    build: .
    container_name: 3dde-api
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
    environment:
      - MODEL_PATH=/app/checkpoints/best.pt
    command: python api_server.py
```

### Build and Run

```bash
# Build Docker image
docker build -t 3dde:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  3dde:latest

# Or use Docker Compose
docker-compose up --build
```

### GPU Support in Docker

For NVIDIA GPU support:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ... rest of Dockerfile
```

```bash
# Run with GPU
docker run --gpus all -it --rm 3dde:latest
```

---

## Production Configuration

### 1. Configuration File

Create `config/production.yaml`:

```yaml
# Model Configuration
model:
  num_nodes: 25
  d_node: 3
  hidden_dim: 256
  num_layers: 4
  dropout: 0.1
  use_layer_norm: true

# Graph Configuration
graph:
  type: "grid"  # or "molecular", "custom"
  grid_height: 5
  grid_width: 5
  gamma: 1.0
  lambda: 0.1

# Diffusion Configuration
diffusion:
  T_steps: 1000
  beta_min: 0.0001
  beta_max: 0.02
  diag_approx: false

# Training Configuration
training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  
  # Learning rate schedule
  lr_schedule:
    type: "cosine"
    warmup_epochs: 5
    min_lr: 0.00001
  
  # Checkpointing
  checkpoint_dir: "checkpoints/"
  save_interval: 10
  keep_last_n: 5

# Loss Configuration
loss:
  type: "combined"
  lambda_t: "linear"  # or "constant", "exponential"
  
  # Regularization
  regularizer:
    lambda_E: 0.1
    lambda_G: 0.01
    lambda_A: 0.05

# Sampling Configuration
sampling:
  num_samples: 100
  return_trajectory: false
  
# Logging Configuration
logging:
  level: "INFO"
  log_dir: "logs/"
  tensorboard: true
  wandb:
    enabled: false
    project: "3d-diffusion"
    entity: "your-org"

# Hardware Configuration
hardware:
  device: "cuda"  # or "cpu"
  num_workers: 4
  pin_memory: true
  mixed_precision: true
```

### 2. Load Configuration

```python
import yaml
from pathlib import Path

def load_config(config_path="config/production.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Usage
config = load_config()
model_config = config['model']
training_config = config['training']
```

### 3. Environment Variables

Create `.env` file:

```bash
# Environment
ENVIRONMENT=production

# Paths
DATA_DIR=/app/data
CHECKPOINT_DIR=/app/checkpoints
LOG_DIR=/app/logs

# Model
MODEL_PATH=/app/checkpoints/best.pt

# Hardware
DEVICE=cuda
NUM_WORKERS=4

# Logging
LOG_LEVEL=INFO
TENSORBOARD_DIR=/app/runs
```

Load in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()

device = os.getenv('DEVICE', 'cpu')
checkpoint_dir = os.getenv('CHECKPOINT_DIR', 'checkpoints/')
```

---

## Monitoring & Logging

### 1. Logging Setup

```python
import logging
from pathlib import Path

def setup_logging(log_dir="logs/", level=logging.INFO):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / 'train.log')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Usage
logger = setup_logging()
logger.info("Training started")
```

### 2. TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/experiment_1')

# Log metrics
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

# Log model graph
writer.add_graph(model, sample_input)

# Close writer
writer.close()
```

### 3. Weights & Biases Integration

```python
import wandb

# Initialize W&B
wandb.init(
    project="3d-diffusion",
    config={
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 64
    }
)

# Log metrics
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "epoch": epoch
})

# Log model
wandb.save('checkpoints/best.pt')
```

### 4. Health Check Endpoint

```python
from flask import Flask, jsonify
import torch

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'cuda_available': torch.cuda.is_available(),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    })

@app.route('/metrics')
def metrics():
    """Model metrics endpoint."""
    checkpoint = torch.load('checkpoints/best.pt', map_location='cpu')
    return jsonify({
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['best_val_loss']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

---

## Performance Optimization

### 1. GPU Optimization

```python
import torch

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = model(batch)
            loss = loss_fn(output, target)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 2. Multi-GPU Training

```python
import torch.nn as nn
from torch.nn.parallel import DataParallel

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

model = model.to(device)
```

### 3. Memory Optimization

```python
# Gradient checkpointing (for large models)
from torch.utils.checkpoint import checkpoint

class EpsilonNetworkWithCheckpointing(nn.Module):
    def forward(self, x):
        # Use checkpointing for memory-intensive layers
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x

# Clear cache periodically
if epoch % 10 == 0:
    torch.cuda.empty_cache()
```

### 4. Data Loading Optimization

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    persistent_workers=True # Keep workers alive
)
```

### 5. Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        output = model(batch)
        loss = loss_fn(output, target)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
batch_size = 32  # or 16, 8

# Enable gradient checkpointing
use_checkpoint = True

# Use mixed precision
use_amp = True

# Clear cache
torch.cuda.empty_cache()
```

#### 2. Slow Training

**Solutions**:
- Increase `num_workers` in DataLoader
- Enable `torch.backends.cudnn.benchmark = True`
- Use mixed precision training
- Profile to identify bottlenecks

#### 3. Model Not Converging

**Check**:
- Learning rate (try 1e-4 to 1e-3)
- Gradient clipping (set to 1.0)
- Batch normalization vs Layer normalization
- Weight initialization

#### 4. NaN Loss

**Solutions**:
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for numerical stability
assert not torch.isnan(loss).any(), "NaN loss detected"

# Lower learning rate
lr = 1e-4
```

### Debugging Tools

```python
# Check model gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Validate data
def check_batch(batch):
    S_0, _ = batch
    assert not torch.isnan(S_0).any(), "NaN in input"
    assert not torch.isinf(S_0).any(), "Inf in input"
    print(f"Batch stats: mean={S_0.mean():.4f}, std={S_0.std():.4f}")
```

---

## Production Checklist

Before deploying to production:

- [ ] All tests passing (`pytest src/tests/ -v`)
- [ ] Model trained and validated
- [ ] Checkpoints saved and tested
- [ ] Configuration file created
- [ ] Docker image built and tested
- [ ] Logging configured
- [ ] Monitoring setup (TensorBoard/W&B)
- [ ] Health check endpoint working
- [ ] Documentation complete
- [ ] Performance profiling done
- [ ] Error handling implemented
- [ ] Security review completed

---

## Example Production Script

```python
#!/usr/bin/env python
"""Production training script with full monitoring."""

import torch
import logging
import yaml
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.data.graph_builder import build_grid_laplacian
from src.core.precompute import precompute_phi_sigma_explicit
from src.models.eps_net import EpsilonNetwork
from src.training.trainer import DiffusionTrainer

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open('config/production.yaml') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    logger.info(f"Using device: {device}")
    
    # Create graph
    L = build_grid_laplacian(
        config['graph']['grid_height'],
        config['graph']['grid_width']
    )
    
    # Precompute matrices
    Phi, Sigma, beta_schedule = precompute_phi_sigma_explicit(
        L=L,
        gamma=config['graph']['gamma'],
        lam=config['graph']['lambda'],
        T_steps=config['diffusion']['T_steps'],
        beta_min=config['diffusion']['beta_min'],
        beta_max=config['diffusion']['beta_max']
    )
    
    # Initialize model
    model = EpsilonNetwork(**config['model']).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config['training']['checkpoint_dir']
    )
    
    # Train
    try:
        history = trainer.train(
            num_epochs=config['training']['num_epochs'],
            Phi=Phi,
            Sigma=Sigma,
            beta_schedule=beta_schedule,
            T_steps=config['diffusion']['T_steps']
        )
        logger.info("✓ Training completed successfully")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise
    
    return history

if __name__ == '__main__':
    main()
```

---

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/your-org/3dde/issues
- **Documentation**: https://3dde.readthedocs.io
- **Email**: support@your-org.com

---

**Last Updated**: November 2025  
**Version**: 1.0.0
