# Experiments

This directory contains configuration files, training scripts, and notebooks for experiments with the 3D Diffusion Model.

## Structure

```
experiments/
├── configs/           # Configuration files
│   ├── default.yaml   # Default configuration
│   ├── fast.yaml      # Fast training (for testing)
│   └── high_quality.yaml  # High-quality training
├── notebooks/         # Jupyter notebooks for visualization
├── config.py          # Configuration loader
└── train_with_config.py  # Training script with config support
```

## Quick Start

### 1. Training with Config

```bash
# Train with default config
python -m experiments.train_with_config

# Train with custom config
python -m experiments.train_with_config --config experiments/configs/fast.yaml

# Resume from checkpoint
python -m experiments.train_with_config --config experiments/configs/default.yaml --resume checkpoints/best_model.pt
```

### 2. Available Configurations

**default.yaml**: Balanced configuration for general use
- 128 hidden dims, 4 layers
- 1000 diffusion steps
- 100 epochs training

**fast.yaml**: Quick experiments and testing
- 64 hidden dims, 3 layers
- 100 diffusion steps (with DDIM sampling)
- 10 epochs training
- Diagonal Σ approximation

**high_quality.yaml**: Best quality results
- 256 hidden dims, 8 layers
- 2000 diffusion steps
- 500 epochs training
- Predictor-corrector sampling

### 3. Creating Custom Configs

Copy and modify any config file:

```bash
cp experiments/configs/default.yaml experiments/configs/my_config.yaml
# Edit my_config.yaml
python -m experiments.train_with_config --config experiments/configs/my_config.yaml
```

### 4. Config Structure

```yaml
model:
  input_dim: 3          # Node feature dimension
  hidden_dim: 128       # Hidden layer size
  num_layers: 4         # Number of GNN layers
  num_heads: 4          # Attention heads
  time_dim: 64          # Time embedding dimension
  use_attention: true   # Use graph attention
  dropout: 0.1

diffusion:
  num_steps: 1000              # Diffusion timesteps
  beta_schedule: "cosine"      # "linear" or "cosine"
  lambd: 1.0                   # Laplacian scaling
  diag_approx: false           # Diagonal Σ approximation

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  
  loss:
    eps_weight: 1.0
    regularization_weight: 0.1
    use_time_weighting: true
  
  regularization:
    lambda_energy: 0.0         # Energy conservation
    lambda_smoothness: 0.1     # Graph smoothness
    lambda_alignment: 0.0      # Field alignment
    lambda_divergence: 0.0     # Divergence constraint

data:
  num_samples: 10000
  node_dim: 3
  mode: "smooth"              # "random", "smooth", "wave", "cluster"
  val_split: 0.2

graph:
  type: "grid"                # "grid", "molecular", "custom"
  grid_size: [10, 10]

sampling:
  method: "ddpm"              # "ddpm", "ddim", "predictor_corrector"
  num_sample_steps: 1000
  eta: 0.0                    # DDIM stochasticity

device: "cuda"
seed: 42
```

## Hyperparameter Tuning

### Model Size
- **Small** (fast, less capacity): `hidden_dim: 64, num_layers: 2-3`
- **Medium** (balanced): `hidden_dim: 128, num_layers: 4-6`
- **Large** (slow, more capacity): `hidden_dim: 256, num_layers: 8-12`

### Diffusion Steps
- **Fast** (quick sampling): `num_steps: 100` with DDIM
- **Medium** (balanced): `num_steps: 500-1000`
- **Quality** (best results): `num_steps: 2000+`

### Regularization
Start with `lambda_smoothness: 0.1`, adjust based on results:
- Too smooth → reduce lambda
- Too noisy → increase lambda

### Learning Rate
- Large models: `1e-5` to `1e-4`
- Small models: `1e-4` to `1e-3`
- Use warmup for large models

## Monitoring Training

Training progress is logged to console. Key metrics:
- **Train Loss**: Should decrease steadily
- **Val Loss**: Should follow train loss
- **Step Time**: Time per training step

Checkpoints are saved to `checkpoint_dir` (default: `./checkpoints/`)

## Tips

1. **Start with fast.yaml** to verify everything works
2. **Use diagonal approximation** (`diag_approx: true`) for large graphs
3. **Reduce num_steps** during development, increase for final training
4. **Use DDIM sampling** (`method: "ddim"`) for faster generation
5. **Monitor GPU memory** - reduce `batch_size` or `hidden_dim` if OOM

## Examples

```bash
# Quick test run
python -m experiments.train_with_config --config experiments/configs/fast.yaml

# Production training
python -m experiments.train_with_config --config experiments/configs/high_quality.yaml

# Custom grid size
# Edit config to set grid_size: [20, 20], then:
python -m experiments.train_with_config --config experiments/configs/my_config.yaml
```
