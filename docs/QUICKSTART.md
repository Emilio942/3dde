# Quick Start Guide

## Phase 1 & 2 Complete! ✅

Wir haben erfolgreich die **Grundlagen und Core-Implementierung** abgeschlossen:

### ✅ Phase 1: Projektstruktur
- Vollständige Verzeichnisstruktur erstellt
- `requirements.txt` mit allen Dependencies
- `setup.py` für Package-Installation
- README mit Projektübersicht

### ✅ Phase 2: Core-Module
Drei essenzielle Module implementiert:

#### 1. **`src/core/precompute.py`** - Φ(t) und Σ(t) Berechnung
- `build_A_node()` - Konvertiert Graph-Laplacian zu Drift-Matrix
- `precompute_phi_sigma_explicit()` - Berechnet Diffusions-Matrizen
- Unterstützt linear & cosine β-schedules
- Diagonal-Approximation für Speicher-Effizienz
- Automatische Stabilitätsprüfung

#### 2. **`src/core/forward.py`** - Forward-Noising
- `apply_Phi_node_batch()` - Batch-fähige Transformation
- `forward_noising_batch()` - St = Φ(t)S0 + √Σ(t)ε
- `compute_hat_S0_from_eps_hat()` - Rekonstruktion von Clean-State
- Flexible Zeitschritt-Sampling-Strategien

#### 3. **`src/data/graph_builder.py`** - Graph-Konstruktion
- `build_grid_laplacian()` - 1D/2D/3D Grids mit periodischen BC
- `build_laplacian_from_edges()` - Generische Edge-List zu Laplacian
- `build_molecular_graph()` - 3D Molekular-Strukturen mit Cutoff
- Graph-Statistiken und Visualisierung

#### 4. **`src/core/utils.py`** - Utility-Funktionen
- Normalisierung und Denormalisierung
- Matrix-Operationen (sqrt, Symmetrie-Check)
- EMA und Learning-Rate Scheduling
- Checkpoint-Management

## Nächste Schritte

### Sofort verfügbar:
1. Dependencies installieren: `pip install -r requirements.txt`
2. Package installieren: `pip install -e .`
3. Module testen: `python -m src.core.precompute`

### Phase 3: Modell-Architektur (nächste Priorität)
- Epsilon-Netzwerk (ε-prediction)
- Graph Neural Network Layers
- Zeit-Embedding

### Phase 4: Training
- Loss-Funktionen
- Physikalische Regularisierer
- Training-Pipeline

## Code-Beispiel

```python
import torch
from src.core.precompute import precompute_phi_sigma_explicit
from src.core.forward import forward_noising_batch, sample_timesteps
from src.data.graph_builder import build_grid_laplacian

# Setup: 5x5 grid, 100 timesteps
L = build_grid_laplacian((5, 5))
Phi, Sigma = precompute_phi_sigma_explicit(L, num_steps=100, diag_approx=True)

# Generate data: 4 samples, 25 nodes, 3 dimensions
S0 = torch.randn(4, 25, 3)

# Sample time steps and apply noising
t_indices = sample_timesteps(4, 100, S0.device)
Phi_t = Phi[t_indices]
Sigma_t = Sigma[t_indices]
St, eps = forward_noising_batch(S0, Phi_t, Sigma_t)

print(f"Clean: {S0.shape}, Noised: {St.shape}, Noise: {eps.shape}")
```

## Projektstruktur

```
3dde/
├── docs/
│   ├── formulas.md          ✅ Mathematische Formeln
│   ├── aufabenliste.md      ✅ Aufgabenliste (wird aktualisiert)
│   └── QUICKSTART.md        ✅ Diese Datei
├── src/
│   ├── core/
│   │   ├── precompute.py    ✅ Φ/Σ Berechnung
│   │   ├── forward.py       ✅ Forward-Noising
│   │   ├── utils.py         ✅ Utilities
│   │   └── sampling.py      ⏳ Nächster Schritt
│   ├── models/
│   │   ├── eps_net.py       ⏳ Phase 3
│   │   └── gnn_layers.py    ⏳ Phase 3
│   ├── training/            ⏳ Phase 4
│   ├── data/
│   │   ├── graph_builder.py ✅ Graph-Konstruktion
│   │   └── dataset.py       ⏳ Später
│   └── tests/               ⏳ Phase 6
├── requirements.txt         ✅
├── setup.py                 ✅
└── README.md                ✅
```

## Status

- **Phase 1**: ✅ Komplett (Projektstruktur)
- **Phase 2**: ✅ Komplett (Core-Module)
- **Phase 3**: ⏳ Bereit zu starten (Modell-Architektur)
- **Phase 4-8**: 🔜 Geplant

## Technische Highlights

### Numerische Stabilität
- Automatische Konditionsprüfung für Φ-Matrizen
- Regularisierung bei Matrix-Inversionen
- Cholesky-Dekomposition für Σ^(1/2)
- Clipping bei β-Schedules

### Effizienz
- Diagonal-Approximation für Σ (reduziert O(N²) → O(N))
- Batch-fähige Matrix-Operationen
- Optional: Sparse-Matrix Support (vorbereitet)

### Flexibilität
- Multiple β-Schedules (linear, cosine)
- Unterstützt 1D/2D/3D Graphen
- Generische Laplacian-Konstruktion
- Periodische & offene Randbedingungen

## Nächster Befehl

Willst du weitermachen mit **Phase 3** (Modell-Architektur)?
Oder sollen wir zuerst die Dependencies installieren und Tests laufen lassen?
