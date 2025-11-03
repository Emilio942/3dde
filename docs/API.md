# API Documentation

## Core Modules

### `src.core.precompute`

Functions for precomputing diffusion matrices Φ(t) and Σ(t).

#### `build_A_node(L, lambd=1.0)`

Convert Laplacian matrix to drift matrix A.

**Parameters:**
- `L` (Tensor): Laplacian matrix (N×N)
- `lambd` (float): Scaling factor for diffusion strength

**Returns:**
- `A` (Tensor): Drift matrix A = -λL

#### `precompute_phi_sigma_explicit(L, num_steps, beta_schedule_type='linear', ...)`

Precompute Φ(t) and Σ(t) matrices for all timesteps.

**Parameters:**
- `L` (Tensor): Laplacian matrix (N×N)
- `num_steps` (int): Number of diffusion steps
- `beta_schedule_type` (str): 'linear' or 'cosine'
- `beta_start` (float): Starting beta value
- `beta_end` (float): Ending beta value
- `lambd` (float): Laplacian scaling
- `diag_approx` (bool): Use diagonal approximation for Σ

**Returns:**
- `Phi` (Tensor): Shape (T+1, N, N), transformation matrices
- `Sigma` (Tensor): Shape (T+1, N, N), covariance matrices

**Example:**
```python
L = build_grid_laplacian((10, 10))
Phi, Sigma = precompute_phi_sigma_explicit(L, num_steps=1000)
```

---

### `src.core.forward`

Forward noising process implementation.

#### `forward_noising_batch(S_0, t, Phi, Sigma)`

Apply forward noising: S_t = Φ(t)S_0 + √Σ(t)ε

**Parameters:**
- `S_0` (Tensor): Clean states (B, N, d)
- `t` (int): Timestep
- `Phi` (Tensor): Precomputed Φ matrices
- `Sigma` (Tensor): Precomputed Σ matrices

**Returns:**
- `S_t` (Tensor): Noised states (B, N, d)
- `eps` (Tensor): Sampled noise (B, N, d)

**Example:**
```python
S_0 = torch.randn(32, 100, 3)  # 32 samples, 100 nodes, 3D
S_t, eps = forward_noising_batch(S_0, t=500, Phi=Phi, Sigma=Sigma)
```

#### `compute_hat_S0_from_eps_hat(S_t, eps_hat, t, Phi, Sigma)`

Reconstruct S_0 from noised state and predicted epsilon.

**Parameters:**
- `S_t` (Tensor): Noised states (B, N, d)
- `eps_hat` (Tensor): Predicted noise (B, N, d)
- `t` (int): Timestep
- `Phi, Sigma` (Tensor): Precomputed matrices

**Returns:**
- `S_0_hat` (Tensor): Reconstructed clean states (B, N, d)

---

### `src.core.sampling`

Reverse sampling implementations.

#### `sample_reverse_from_S_T(S_T, F, Q, eps_model, L, num_steps, ...)`

Sample from noise to clean data.

**Parameters:**
- `S_T` (Tensor): Initial noise (B, N, d)
- `F, Q` (Tensor): Reverse process matrices
- `eps_model` (nn.Module): Epsilon prediction network
- `L` (Tensor): Laplacian matrix
- `num_steps` (int): Number of sampling steps
- `use_predictor_corrector` (bool): Use Langevin corrector
- `num_corrector_steps` (int): Corrector steps per timestep

**Returns:**
- `S_0` (Tensor): Generated samples (B, N, d)
- `trajectory` (Tensor): Full trajectory (T+1, B, N, d)

**Example:**
```python
F, Q = compute_F_Q_from_PhiSigma(Phi, Sigma)
S_T = torch.randn(4, 100, 3)  # Start from noise

model.eval()
with torch.no_grad():
    S_0, trajectory = sample_reverse_from_S_T(
        S_T, F, Q, model, L, num_steps=1000
    )
```

#### `ddim_sampling(S_T, Phi, Sigma, eps_model, L, num_steps, eta=0.0, ...)`

DDIM sampling (deterministic or stochastic).

**Parameters:**
- `eta` (float): Stochasticity (0=deterministic, 1=DDPM)
- `num_sample_steps` (int): Can be < num_steps for faster sampling

**Returns:**
- `S_0, trajectory` (Tensor)

---

## Models

### `src.models.eps_net.EpsilonNetwork`

Main epsilon prediction network.

#### `__init__(input_dim, hidden_dim, num_layers, ...)`

**Parameters:**
- `input_dim` (int): Node feature dimension
- `hidden_dim` (int): Hidden layer size
- `num_layers` (int): Number of GNN layers
- `num_heads` (int): Attention heads
- `time_dim` (int): Time embedding dimension
- `use_attention` (bool): Use graph attention
- `dropout` (float): Dropout rate

#### `forward(x, t, L)`

**Parameters:**
- `x` (Tensor): Node features (B, N, input_dim)
- `t` (Tensor): Timesteps (B,)
- `L` (Tensor): Laplacian matrix (N, N)

**Returns:**
- `eps` (Tensor): Predicted noise (B, N, input_dim)

**Example:**
```python
model = EpsilonNetwork(
    input_dim=3,
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    time_dim=64
)

x = torch.randn(32, 100, 3)
t = torch.randint(0, 1000, (32,))
L = build_grid_laplacian((10, 10))

eps = model(x, t, L)  # Shape: (32, 100, 3)
```

---

## Training

### `src.training.trainer.DiffusionTrainer`

Complete training pipeline.

#### `__init__(model, L, num_steps, learning_rate, ...)`

**Parameters:**
- `model` (nn.Module): Epsilon prediction network
- `L` (Tensor): Graph Laplacian
- `num_steps` (int): Diffusion timesteps
- `learning_rate` (float): Learning rate
- `weight_decay` (float): Weight decay
- `loss_config` (dict): Loss configuration
- `reg_config` (dict): Regularization configuration
- `device` (str): 'cuda' or 'cpu'
- `checkpoint_dir` (str): Checkpoint directory
- `log_interval` (int): Logging frequency

#### `train(train_loader, val_loader, num_epochs)`

Train the model.

**Parameters:**
- `train_loader` (DataLoader): Training data
- `val_loader` (DataLoader): Validation data
- `num_epochs` (int): Number of epochs

**Example:**
```python
from src.data.dataset import SyntheticDataset, create_dataloaders

# Create dataset
dataset = SyntheticDataset(
    num_samples=10000,
    num_nodes=100,
    node_dim=3,
    mode='smooth'
)

train_loader, val_loader = create_dataloaders(
    dataset.data,
    batch_size=32,
    val_split=0.2
)

# Create trainer
trainer = DiffusionTrainer(
    model=model,
    L=L,
    num_steps=1000,
    learning_rate=1e-4
)

# Train
trainer.train(train_loader, val_loader, num_epochs=100)
```

---

## Data

### `src.data.graph_builder`

Graph Laplacian construction.

#### `build_grid_laplacian(grid_size, periodic=False)`

Create grid graph Laplacian.

**Parameters:**
- `grid_size` (tuple): e.g., (10, 10) or (5, 5, 5)
- `periodic` (bool): Periodic boundary conditions

**Returns:**
- `L` (Tensor): Laplacian matrix (N×N)

**Example:**
```python
# 2D grid
L = build_grid_laplacian((10, 10))  # 100 nodes

# 3D grid
L = build_grid_laplacian((5, 5, 5))  # 125 nodes
```

#### `build_molecular_graph(positions, cutoff_radius=None, k_neighbors=None)`

Create molecular graph from 3D positions.

**Parameters:**
- `positions` (Tensor): 3D coordinates (N, 3)
- `cutoff_radius` (float): Distance cutoff
- `k_neighbors` (int): K-nearest neighbors

**Returns:**
- `L` (Tensor): Laplacian matrix (N×N)

### `src.data.dataset.SyntheticDataset`

Generate synthetic data for testing.

#### `__init__(num_samples, num_nodes, node_dim, mode='smooth', ...)`

**Parameters:**
- `num_samples` (int): Number of samples
- `num_nodes` (int): Nodes per sample
- `node_dim` (int): Feature dimension
- `mode` (str): 'random', 'smooth', 'wave', 'cluster'
- `noise_level` (float): Noise to add

**Example:**
```python
dataset = SyntheticDataset(
    num_samples=5000,
    num_nodes=100,
    node_dim=3,
    mode='smooth',
    seed=42
)

# Access data
sample = dataset[0]  # Dict with 'states', 'original'
```

---

## Configuration

### `experiments.config.Config`

Configuration management.

#### `Config.from_yaml(yaml_path)`

Load config from YAML file.

**Example:**
```python
from experiments.config import Config

config = Config.from_yaml('experiments/configs/default.yaml')

print(config.model.hidden_dim)  # 128
print(config.training.batch_size)  # 32
print(config.diffusion.num_steps)  # 1000
```

#### Config structure

```yaml
model:
  input_dim: 3
  hidden_dim: 128
  num_layers: 4

diffusion:
  num_steps: 1000
  beta_schedule: "cosine"

training:
  batch_size: 32
  learning_rate: 0.0001
  
  loss:
    eps_weight: 1.0
    regularization_weight: 0.1
  
  regularization:
    lambda_smoothness: 0.1

data:
  num_samples: 10000
  mode: "smooth"

graph:
  type: "grid"
  grid_size: [10, 10]

device: "cuda"
```

---

## Testing

Run tests with pytest:

```bash
# All tests
pytest src/tests/ -v

# Specific test file
pytest src/tests/test_precompute.py -v

# With coverage
pytest src/tests/ --cov=src --cov-report=html
```

Test categories:
- `test_precompute.py`: Φ and Σ computation
- `test_forward.py`: Forward noising
- `test_sampling.py`: Reverse sampling
- `test_models.py`: Model architecture
- `test_training.py`: Training pipeline
- `test_integration.py`: End-to-end tests

---

## Complete Example

```python
import torch
from src.data.graph_builder import build_grid_laplacian
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.eps_net import EpsilonNetwork
from src.training.trainer import DiffusionTrainer
from src.core.sampling import compute_F_Q_from_PhiSigma, sample_reverse_from_S_T

# 1. Create graph
L = build_grid_laplacian((10, 10))  # 100 nodes

# 2. Create dataset
dataset = SyntheticDataset(
    num_samples=5000,
    num_nodes=100,
    node_dim=3,
    mode='smooth'
)
train_loader, val_loader = create_dataloaders(
    dataset.data, batch_size=32, val_split=0.2
)

# 3. Create model
model = EpsilonNetwork(
    input_dim=3,
    hidden_dim=128,
    num_layers=4
)

# 4. Train
trainer = DiffusionTrainer(
    model=model,
    L=L,
    num_steps=1000,
    learning_rate=1e-4
)
trainer.train(train_loader, val_loader, num_epochs=50)

# 5. Sample
F, Q = compute_F_Q_from_PhiSigma(trainer.Phi, trainer.Sigma)
S_T = torch.randn(10, 100, 3)

model.eval()
with torch.no_grad():
    S_0, _ = sample_reverse_from_S_T(
        S_T, F, Q, model, L, num_steps=1000
    )

print(f"Generated: {S_0.shape}")  # (10, 100, 3)
```
