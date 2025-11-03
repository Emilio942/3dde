# API Documentation

Complete API reference for the 3D Diffusion Model library.

## Table of Contents

1. [Core Modules](#core-modules)
   - [Precomputation](#precomputation)
   - [Forward Process](#forward-process)
   - [Sampling](#sampling)

2. [Data Modules](#data-modules)
   - [Graph Builder](#graph-builder)
   - [Datasets](#datasets)

3. [Model Modules](#model-modules)
   - [Epsilon Network](#epsilon-network)
   - [GNN Layers](#gnn-layers)

4. [Training Modules](#training-modules)
   - [Trainer](#trainer)
   - [Losses](#losses)
   - [Regularizers](#regularizers)

---

## Core Modules

### Precomputation

**Module**: `src.core.precompute`

Functions for precomputing the diffusion matrices Φ(t) and Σ(t).

#### `build_A_node(L, gamma, lam)`

Constructs the drift matrix A from the graph Laplacian.

**Parameters:**
- `L` (torch.Tensor): Graph Laplacian, shape `(N, N)`
- `gamma` (float): Diffusion strength parameter
- `lam` (float): Regularization parameter
- **Returns**: `torch.Tensor` - Drift matrix A, shape `(N, N)`

**Example:**
```python
from src.data.graph_builder import build_grid_laplacian
from src.core.precompute import build_A_node

L = build_grid_laplacian(5, 5)  # 5x5 grid
A = build_A_node(L, gamma=1.0, lam=0.1)
```

---

#### `precompute_phi_sigma_explicit(L, gamma, lam, T_steps, beta_min, beta_max, diag_approx=False)`

Precomputes Φ(t) and Σ(t) for all timesteps using explicit Euler recursion.

**Parameters:**
- `L` (torch.Tensor): Graph Laplacian, shape `(N, N)`
- `gamma` (float): Diffusion strength parameter (γ)
- `lam` (float): Regularization parameter (λ)
- `T_steps` (int): Number of diffusion timesteps
- `beta_min` (float): Minimum noise schedule value
- `beta_max` (float): Maximum noise schedule value
- `diag_approx` (bool, optional): Use diagonal approximation for Σ (default: False)

**Returns:**
- `Phi` (torch.Tensor): Shape `(T_steps+1, N, N)` - Forward transition matrices
- `Sigma` (torch.Tensor): Shape `(T_steps+1, N, N)` or `(T_steps+1, N)` - Covariance matrices
- `beta_schedule` (torch.Tensor): Shape `(T_steps+1,)` - Noise schedule

**Example:**
```python
Phi, Sigma, beta = precompute_phi_sigma_explicit(
    L=L,
    gamma=1.0,
    lam=0.1,
    T_steps=1000,
    beta_min=0.0001,
    beta_max=0.02,
    diag_approx=False
)
print(f"Phi shape: {Phi.shape}")  # (1001, 25, 25)
print(f"Sigma shape: {Sigma.shape}")  # (1001, 25, 25)
```

---

#### `get_Phi_Sigma_at_t(Phi, Sigma, t)`

Helper function to extract Φ(t) and Σ(t) at a specific timestep.

**Parameters:**
- `Phi` (torch.Tensor): Precomputed Φ matrices
- `Sigma` (torch.Tensor): Precomputed Σ matrices
- `t` (int or torch.Tensor): Timestep(s) to extract

**Returns:**
- `Phi_t` (torch.Tensor): Φ(t) matrix
- `Sigma_t` (torch.Tensor): Σ(t) matrix

---

### Forward Process

**Module**: `src.core.forward`

Functions for the forward noising process.

#### `apply_Phi_node_batch(Phi_t, S, d_node)`

Applies Φ(t) to a batch of node states using Kronecker structure.

**Parameters:**
- `Phi_t` (torch.Tensor): Forward matrix, shape `(N, N)`
- `S` (torch.Tensor): Node states, shape `(B, N, d_node)`
- `d_node` (int): Dimension per node

**Returns:**
- `torch.Tensor`: Transformed states, shape `(B, N, d_node)`

**Mathematical Operation:**
```
For each batch: S' = Φ(t) @ S
Vectorized: (I_d ⊗ Φ(t)) @ vec(S)
```

---

#### `forward_noising_batch(S_0, Phi_t, Sigma_t)`

Applies forward noising: S_t = Φ(t)S_0 + √Σ(t)ε

**Parameters:**
- `S_0` (torch.Tensor): Clean states, shape `(B, N, d_node)`
- `Phi_t` (torch.Tensor): Forward matrix Φ(t), shape `(N, N)`
- `Sigma_t` (torch.Tensor): Covariance Σ(t), shape `(N, N)` or `(N,)`

**Returns:**
- `torch.Tensor`: Noised states S_t, shape `(B, N, d_node)`

**Example:**
```python
from src.core.forward import forward_noising_batch

# Generate clean sample
S_0 = torch.randn(4, 25, 3)  # 4 samples, 25 nodes, 3D vectors

# Get matrices at timestep 500
Phi_t, Sigma_t = get_Phi_Sigma_at_t(Phi, Sigma, t=500)

# Apply forward noising
S_t = forward_noising_batch(S_0, Phi_t, Sigma_t)
```

---

#### `compute_hat_S0_from_eps_hat(S_t, eps_hat, Phi_t, Sigma_t, sqrt_Sigma_inv)`

Reconstructs Ŝ₀ from noised state and predicted noise.

**Parameters:**
- `S_t` (torch.Tensor): Noised states, shape `(B, N, d_node)`
- `eps_hat` (torch.Tensor): Predicted noise, shape `(B, N, d_node)`
- `Phi_t` (torch.Tensor): Forward matrix Φ(t)
- `Sigma_t` (torch.Tensor): Covariance Σ(t)
- `sqrt_Sigma_inv` (torch.Tensor): Inverse of √Σ(t)

**Returns:**
- `torch.Tensor`: Reconstructed Ŝ₀, shape `(B, N, d_node)`

**Formula:**
```
Ŝ₀ = Φ(t)⁻¹ [S_t - √Σ(t) ε̂]
```

---

### Sampling

**Module**: `src.core.sampling`

Functions for reverse diffusion sampling.

#### `sample_reverse_from_S_T(model, S_T, Phi, Sigma, beta_schedule, T_steps, diag_approx=False, return_trajectory=False)`

Performs reverse diffusion sampling from noise to clean samples.

**Parameters:**
- `model` (nn.Module): Trained epsilon network
- `S_T` (torch.Tensor): Starting noise, shape `(B, N, d_node)`
- `Phi` (torch.Tensor): Precomputed Φ matrices, shape `(T_steps+1, N, N)`
- `Sigma` (torch.Tensor): Precomputed Σ matrices
- `beta_schedule` (torch.Tensor): Noise schedule, shape `(T_steps+1,)`
- `T_steps` (int): Number of diffusion steps
- `diag_approx` (bool, optional): Use diagonal approximation
- `return_trajectory` (bool, optional): Return full trajectory

**Returns:**
- `S_0` (torch.Tensor): Generated samples, shape `(B, N, d_node)`
- `trajectory` (torch.Tensor, optional): Full reverse process, shape `(B, T_steps+1, N, d_node)`

**Example:**
```python
from src.core.sampling import sample_reverse_from_S_T

# Start from pure noise
S_T = torch.randn(4, 25, 3)

# Sample reverse process
samples, trajectory = sample_reverse_from_S_T(
    model=model,
    S_T=S_T,
    Phi=Phi,
    Sigma=Sigma,
    beta_schedule=beta_schedule,
    T_steps=1000,
    return_trajectory=True
)

print(f"Generated {samples.shape[0]} samples")
print(f"Trajectory has {trajectory.shape[1]} timesteps")
```

---

## Data Modules

### Graph Builder

**Module**: `src.data.graph_builder`

#### `build_grid_laplacian(height, width)`

Creates a Laplacian for a 2D grid graph with 4-connectivity.

**Parameters:**
- `height` (int): Grid height
- `width` (int): Grid width

**Returns:**
- `torch.Tensor`: Laplacian matrix, shape `(height*width, height*width)`

**Example:**
```python
from src.data.graph_builder import build_grid_laplacian

L = build_grid_laplacian(5, 5)  # 5x5 grid = 25 nodes
print(L.shape)  # (25, 25)
```

---

#### `build_molecular_graph(positions, atom_types, cutoff=5.0)`

Creates a molecular graph from 3D atomic positions.

**Parameters:**
- `positions` (torch.Tensor): Atomic coordinates, shape `(N, 3)`
- `atom_types` (torch.Tensor): Atom type indices, shape `(N,)`
- `cutoff` (float, optional): Distance cutoff for edges

**Returns:**
- `L` (torch.Tensor): Graph Laplacian
- `edge_index` (torch.Tensor): Edge connectivity, shape `(2, E)`
- `edge_attr` (torch.Tensor): Edge features (distances)

---

### Datasets

**Module**: `src.data.datasets`

#### `SyntheticVectorFieldDataset`

Dataset of synthetic vector fields on graphs.

**Constructor:**
```python
dataset = SyntheticVectorFieldDataset(
    L=L,                    # Graph Laplacian
    num_samples=1000,       # Number of samples
    d_node=3,               # Vector dimension
    noise_level=0.1         # Noise in training data
)
```

**Methods:**
- `__len__()`: Returns dataset size
- `__getitem__(idx)`: Returns `(S_0, graph_info)` tuple

---

## Model Modules

### Epsilon Network

**Module**: `src.models.eps_net`

#### `EpsilonNetwork`

Neural network for predicting ε in the diffusion process.

**Constructor:**
```python
model = EpsilonNetwork(
    num_nodes=25,           # Number of graph nodes
    d_node=3,               # Dimension per node
    hidden_dim=128,         # Hidden layer size
    num_layers=3,           # Number of layers
    dropout=0.1,            # Dropout rate
    use_layer_norm=True     # Use layer normalization
)
```

**Forward:**
```python
eps_pred = model(
    S_t=S_t,                # Noised states, shape (B, N, d_node)
    t=t,                    # Timesteps, shape (B,)
    edge_index=edge_index   # Optional graph edges
)
# Returns: Predicted noise, shape (B, N, d_node)
```

**Example:**
```python
from src.models.eps_net import EpsilonNetwork

model = EpsilonNetwork(
    num_nodes=25,
    d_node=3,
    hidden_dim=128,
    num_layers=3
)

# Forward pass
S_t = torch.randn(4, 25, 3)
t = torch.randint(0, 1000, (4,))
eps_pred = model(S_t, t)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

### GNN Layers

**Module**: `src.models.gnn_layers`

#### `GraphConvLayer`

Graph convolution layer with edge features.

**Constructor:**
```python
layer = GraphConvLayer(
    in_dim=128,
    out_dim=128,
    edge_dim=16,            # Edge feature dimension
    use_attention=True      # Use attention mechanism
)
```

---

## Training Modules

### Trainer

**Module**: `src.training.trainer`

#### `DiffusionTrainer`

Training manager for the diffusion model.

**Constructor:**
```python
trainer = DiffusionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device='cpu',
    checkpoint_dir='checkpoints/'
)
```

**Methods:**

##### `train(num_epochs, ...)`

Runs training loop.

**Parameters:**
- `num_epochs` (int): Number of epochs
- `log_interval` (int, optional): Logging frequency
- `save_interval` (int, optional): Checkpoint save frequency

**Returns:**
- `history` (dict): Training metrics history

**Example:**
```python
history = trainer.train(
    num_epochs=10,
    log_interval=10,
    save_interval=5
)

print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
```

##### `train_epoch(epoch, Phi, Sigma, beta_schedule, T_steps)`

Trains for one epoch.

##### `validate(Phi, Sigma, beta_schedule, T_steps)`

Runs validation.

##### `save_checkpoint(epoch, metrics, filename)`

Saves model checkpoint.

##### `load_checkpoint(checkpoint_path)`

Loads model from checkpoint.

---

### Losses

**Module**: `src.training.losses`

#### `SimpleMSELoss`

Basic MSE loss for epsilon prediction.

**Constructor:**
```python
loss_fn = SimpleMSELoss()
```

**Forward:**
```python
loss, loss_dict = loss_fn(eps_pred, eps_true)
# Returns: (scalar loss, dict with 'mse' key)
```

---

#### `CombinedLoss`

MSE loss with time-dependent weighting.

**Constructor:**
```python
loss_fn = CombinedLoss(
    lambda_t_fn=lambda t: 1.0,  # Time weighting function
    reduction='mean'
)
```

**Forward:**
```python
loss, loss_dict = loss_fn(
    eps_pred=eps_pred,
    eps_true=eps_true,
    t=t,                        # Timesteps
    S_hat_0=S_hat_0            # Optional reconstructed S_0
)
```

---

### Regularizers

**Module**: `src.training.regularizers`

#### `CombinedPhysicsRegularizer`

Combines energy, graph smoothness, and alignment regularization.

**Constructor:**
```python
regularizer = CombinedPhysicsRegularizer(
    lambda_E=0.1,           # Energy weight
    lambda_G=0.01,          # Graph smoothness weight
    lambda_A=0.05,          # Alignment weight
    L=L,                    # Graph Laplacian
    S_star=None             # Target state (optional)
)
```

**Forward:**
```python
reg_total, reg_dict = regularizer(S_hat_0)
# Returns: (total regularization, dict with individual terms)
```

**Regularization Terms:**
- `energy`: E(Ŝ₀) = ½‖M^(1/2)(Ŝ₀ - S*)‖²
- `graph`: ‖L Ŝ₀‖²_F (Laplacian smoothness)
- `alignment`: Orientation preservation

---

## Complete Training Example

```python
import torch
from torch.utils.data import DataLoader

from src.data.graph_builder import build_grid_laplacian
from src.data.datasets import SyntheticVectorFieldDataset
from src.core.precompute import precompute_phi_sigma_explicit
from src.models.eps_net import EpsilonNetwork
from src.training.trainer import DiffusionTrainer
from src.training.losses import CombinedLoss
from src.training.regularizers import CombinedPhysicsRegularizer

# 1. Create graph
L = build_grid_laplacian(5, 5)

# 2. Precompute matrices
Phi, Sigma, beta_schedule = precompute_phi_sigma_explicit(
    L=L, gamma=1.0, lam=0.1, T_steps=1000,
    beta_min=0.0001, beta_max=0.02
)

# 3. Create datasets
train_dataset = SyntheticVectorFieldDataset(L, num_samples=1000, d_node=3)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. Initialize model
model = EpsilonNetwork(num_nodes=25, d_node=3, hidden_dim=128, num_layers=3)

# 5. Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = CombinedLoss()

trainer = DiffusionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=train_loader,  # Using same for simplicity
    optimizer=optimizer,
    loss_fn=loss_fn,
    device='cpu'
)

# 6. Train
history = trainer.train(
    num_epochs=10,
    Phi=Phi,
    Sigma=Sigma,
    beta_schedule=beta_schedule,
    T_steps=1000
)

# 7. Sample
from src.core.sampling import sample_reverse_from_S_T

S_T = torch.randn(4, 25, 3)
samples = sample_reverse_from_S_T(
    model=model,
    S_T=S_T,
    Phi=Phi,
    Sigma=Sigma,
    beta_schedule=beta_schedule,
    T_steps=1000
)

print(f"Generated {samples.shape[0]} samples!")
```

---

## See Also

- [Quickstart Notebook](../../notebooks/quickstart.ipynb) - Interactive tutorial
- [Formulas](../formulas.md) - Mathematical details
- [Success Summary](../../SUCCESS_SUMMARY.md) - Test results and achievements
