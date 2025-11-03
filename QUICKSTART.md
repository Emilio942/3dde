# 3D Diffusion Model - Schnellstart

## 🚀 Installation

```bash
# 1. Repository klonen (falls nicht lokal)
cd /home/emilio/Documents/ai/3dde

# 2. Virtual Environment erstellen (optional aber empfohlen)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# oder: .venv\Scripts\activate  # Windows

# 3. Dependencies installieren
pip install -r requirements.txt

# 4. Package im Development-Modus installieren
pip install -e .
```

## 🎯 Schnelltest

```bash
# Test der Core-Module
python -m src.core.precompute
python -m src.core.forward
python -m src.data.graph_builder

# Test der Model-Module
python -m src.models.gnn_layers
python -m src.models.eps_net

# Test der Training-Module
python -m src.training.losses
python -m src.training.regularizers
```

## 🏃 Training-Beispiel

```bash
# Vollständiges Training-Beispiel ausführen
python train_example.py
```

Oder in Python:

```python
import torch
from src.data.graph_builder import build_grid_laplacian
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.eps_net import EpsilonNetwork
from src.training.trainer import DiffusionTrainer

# 1. Graph erstellen (5x5 Grid)
L = build_grid_laplacian((5, 5))

# 2. Daten generieren
dataset = SyntheticDataset(
    num_samples=500,
    num_nodes=25,
    node_dim=3,
    mode='smooth'
)
train_loader, val_loader = create_dataloaders(
    dataset.data,
    batch_size=16,
    val_split=0.2
)

# 3. Modell erstellen
model = EpsilonNetwork(
    input_dim=3,
    hidden_dim=64,
    num_layers=3
)

# 4. Trainer initialisieren
trainer = DiffusionTrainer(
    model=model,
    L=L,
    num_steps=500,
    learning_rate=1e-4
)

# 5. Trainieren!
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10
)
```

## 📊 Sampling (Generierung)

```python
from src.core.sampling import compute_F_Q_from_PhiSigma, sample_reverse_from_S_T

# Nach dem Training:
F, Q = compute_F_Q_from_PhiSigma(trainer.Phi, trainer.Sigma)

# Starte von Rauschen
S_T = torch.randn(4, 25, 3)  # 4 Samples, 25 Nodes, 3D

# Generiere neue Daten
S_0, trajectory = sample_reverse_from_S_T(
    S_T=S_T,
    F=F,
    Q=Q,
    eps_model=model,
    L=L,
    num_steps=500
)

print(f"Generated: {S_0.shape}")  # (4, 25, 3)
```

## 🧪 Mit echten Daten

```python
# Eigene Daten laden
your_data = torch.load('your_data.pt')  # (N_samples, N_nodes, d)

# Eigene Graph-Struktur
from src.data.graph_builder import build_laplacian_from_edges

edges = [(0, 1), (1, 2), (2, 3), ...]  # Deine Edges
L = build_laplacian_from_edges(edges, num_nodes=N_nodes)

# Rest wie oben...
```

## 📂 Projekt-Struktur

```
3dde/
├── src/                    # Haupt-Code
│   ├── core/              # Diffusion-Core (Φ, Σ, Forward, Sampling)
│   ├── models/            # Neuronale Netze (GNN, Epsilon-Net)
│   ├── training/          # Training (Losses, Regularizer, Trainer)
│   └── data/              # Daten (Graphs, Datasets)
├── docs/                  # Dokumentation
├── experiments/           # Notebooks & Configs
├── train_example.py       # Beispiel-Training
└── requirements.txt       # Dependencies
```

## 🎛️ Hyperparameter

**Modell:**
- `hidden_dim`: 64-256 (je nach Graph-Größe)
- `num_layers`: 3-6
- `num_heads`: 4-8 (für Attention)

**Training:**
- `num_steps`: 500-2000 (mehr Steps = bessere Qualität, aber langsamer)
- `learning_rate`: 1e-4 bis 1e-3
- `batch_size`: 16-64 (abhängig von GPU-Speicher)

**Regularisierung:**
- `lambda_smoothness`: 0.01-0.5 (Graph-Glättung)
- `lambda_energy`: 0.0-0.1 (Energie-Erhaltung)

## 🐛 Troubleshooting

**Import-Fehler:**
```bash
pip install -e .  # Package installieren
```

**CUDA Out of Memory:**
- Reduziere `batch_size`
- Reduziere `hidden_dim` oder `num_layers`
- Nutze `diag_approx=True` für Σ

**Training instabil:**
- Reduziere `learning_rate`
- Erhöhe `num_steps`
- Prüfe Input-Normalisierung

## 📚 Weitere Infos

- **Mathematik:** `docs/formulas.md`
- **Aufgaben:** `docs/aufabenliste.md`
- **Status:** `docs/STATUS.md`
- **Main README:** `README.md`

## 🎉 Fertig!

Du hast jetzt ein vollständiges 3D Diffusion Model für Graph-strukturierte Daten!

Viel Erfolg! 🚀
