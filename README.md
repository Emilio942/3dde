# 3D Diffusion Model (3DDE)

A physics-informed 3D diffusion model for physical systems with graph structure.

## Overview

This project implements a diffusion model specifically designed for 3D physical systems that can be represented as graphs. The model uses:
- **Graph Neural Networks** for spatial relationships
- **Physics-informed regularization** to maintain physical constraints
- **Efficient Kronecker-structured operations** for computational efficiency
- **Flexible noise schedules** for controlled diffusion processes

## Installation

### From Source

```bash
git clone https://github.com/emilio/3dde.git
cd 3dde
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from core.precompute import precompute_phi_sigma_explicit
from core.forward import forward_noising_batch
from models.eps_net import EpsilonNetwork

# Setup graph and parameters
N = 100  # Number of nodes
d = 3    # Dimensions per node
L = build_grid_laplacian(N)  # Graph Laplacian

# Precompute diffusion matrices
Phi, Sigma = precompute_phi_sigma_explicit(L, num_steps=1000)

# Initialize model
model = EpsilonNetwork(input_dim=d, hidden_dim=128, num_layers=4)

# Training loop
# ... (see examples in experiments/notebooks/)
```

## 🎯 Features

- ✅ **Graph-strukturierte Diffusion** mit Laplace-Operator
- ✅ **Vollständige DDPM/DDIM Implementation**
- ✅ **Flexible GNN-Architektur** mit Attention
- ✅ **Physics-informed Regularization**
- ✅ **Multiple Sampling-Methoden** (DDPM, DDIM, Predictor-Corrector)
- ✅ **Config-basiertes Training**
- ✅ **Umfassende Test-Suite** (pytest)
- ✅ **Visualisierungs-Tools**

## 📁 Projekt-Struktur

```
3dde/
├── src/                      # Haupt-Code (27 Module, ~6.500 Zeilen)
│   ├── core/                # Diffusion Core
│   │   ├── precompute.py   # Φ(t), Σ(t) Berechnung
│   │   ├── forward.py      # Forward Noising
│   │   ├── sampling.py     # Reverse Sampling (DDPM, DDIM, PC)
│   │   └── utils.py        # Hilfsfunktionen
│   ├── models/             # Neuronale Netze
│   │   ├── gnn_layers.py   # GNN Layer (Laplacian Conv, Attention)
│   │   └── eps_net.py      # Epsilon-Netz
│   ├── training/           # Training
│   │   ├── losses.py       # Loss-Funktionen
│   │   ├── regularizers.py # Physics Regularisierer
│   │   ├── trainer.py      # Trainer
│   │   └── dataset.py      # Daten-Loading
│   ├── data/               # Daten-Utilities
│   │   └── graph_builder.py # Graph-Konstruktion
│   └── tests/              # Unit Tests (6 Test-Dateien, pytest)
│       ├── test_precompute.py
│       ├── test_forward.py
│       ├── test_sampling.py
│       ├── test_models.py
│       ├── test_training.py
│       └── test_integration.py
├── experiments/            # Experimente
│   ├── configs/           # YAML Configs (default, fast, high_quality)
│   ├── config.py          # Config-Loader
│   ├── train_with_config.py # Training-Script
│   └── notebooks/         # Visualisierungen
├── docs/                  # Dokumentation
│   ├── formulas.md       # Mathematik (vollständige Herleitung)
│   ├── STATUS.md         # Projekt-Status (Phasen 1-7 ✅)
│   ├── API.md            # API-Dokumentation
│   └── aufgabenliste.md  # Development Roadmap
├── train_example.py      # Einfaches End-to-End Beispiel
├── QUICKSTART.md         # Schnellstart-Guide
├── run_tests.sh          # Test-Runner
└── requirements.txt      # Dependencies
```

## 🧪 Testing

```bash
# Alle Tests ausführen
pytest src/tests/ -v

# Mit Coverage
pytest src/tests/ --cov=src --cov-report=html

# Schnell-Tests mit Scripts
./run_tests.sh              # Alle Tests mit Coverage
./run_tests_detailed.sh     # Tests nach Kategorie
```

**Test-Abdeckung:**
- ✅ Core-Module (precompute, forward, sampling)
- ✅ Model-Architektur
- ✅ Training-Pipeline  
- ✅ Integration Tests
- ✅ Numerical Stability
- ✅ Reproducibility

## 🎛️ Training mit Configs

```bash
# Schnelles Training (zum Testen)
python -m experiments.train_with_config --config experiments/configs/fast.yaml

# Standard-Training
python -m experiments.train_with_config --config experiments/configs/default.yaml

# High-Quality Training
python -m experiments.train_with_config --config experiments/configs/high_quality.yaml

# Von Checkpoint fortsetzen
python -m experiments.train_with_config --config config.yaml --resume checkpoints/best.pt
```

## 📊 Visualisierung

```bash
# Visualisierungs-Script ausführen
cd experiments/notebooks
python visualizations.py
```

Generiert:
- Forward Diffusion Process
- Reverse Sampling Process
- Φ und Σ Evolution
- Graph Structure
- Diffusion Statistics

## 📚 Weitere Dokumentation

- **Mathematik:** `docs/formulas.md` - Vollständige mathematische Herleitung
- **Status:** `docs/STATUS.md` - Aktueller Implementierungs-Stand (Phasen 1-7 ✅)
- **API:** `docs/API.md` - Vollständige API-Dokumentation
- **Schnellstart:** `QUICKSTART.md` - Detaillierte Schnellstart-Anleitung
- **Experimente:** `experiments/README.md` - Experiment-Dokumentation

## Development Status

✅ **Phasen 1-7 KOMPLETT:**
- ✅ Phase 1: Projekt-Struktur
- ✅ Phase 2: Core-Module (precompute, forward, sampling)
- ✅ Phase 3: Model-Architektur (GNN, Epsilon-Net)
- ✅ Phase 4: Training-Pipeline (losses, regularizers, trainer)
- ✅ Phase 5: Sampling-Algorithmen (DDPM, DDIM, Predictor-Corrector)
- ✅ Phase 6: Unit Tests (vollständige Test-Suite)
- ✅ Phase 7: Experimente (Configs, Training-Scripts, Visualisierung)

**Nächste Schritte (Optional):**
- [ ] Tutorial Jupyter Notebooks
- [ ] Pre-trained Models
- [ ] Multi-GPU Support
- [ ] Weights & Biases Integration

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- See `requirements.txt` for full list

## License

MIT License (see LICENSE file)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{3dde2025,
  author = {Emilio},
  title = {3D Diffusion Model for Physical Systems},
  year = {2025},
  url = {https://github.com/emilio/3dde}
}
```

## Contributing

Contributions welcome! Please see development guidelines in `docs/`.

## Contact

For questions or issues, please open an issue on GitHub.
