# 3D Diffusions-Modell - Aufgabenliste

## Projektübersicht

Entwicklung eines 3D Diffusions-Modells für physikalische Systeme mit Graph-Struktur.

## Verzeichnisstruktur

```
3dde/
├── docs/
│   ├── formulas.md              # Mathematische Formeln (✓ erstellt)
│   ├── aufgabenliste.md         # Diese Datei
│   └── architecture.md          # Systemarchitektur
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── precompute.py        # Φ(t) und Σ(t) Berechnungen
│   │   ├── forward.py           # Forward-Noising Prozess
│   │   ├── sampling.py          # Reverse-Sampling
│   │   └── utils.py             # Hilfsfunktionen
│   ├── models/
│   │   ├── __init__.py
│   │   ├── eps_net.py           # ε-Vorhersage Netzwerk
│   │   ├── score_net.py         # Score-basiertes Netzwerk
│   │   └── gnn_layers.py        # Graph Neural Network Schichten
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py            # Loss-Funktionen
│   │   ├── regularizers.py      # Physikalische Regularisierer
│   │   └── trainer.py           # Training-Pipeline
│   ├── data/
│   │   ├── __init__.py
│   │   ├── graph_builder.py     # Graph-Konstruktion
│   │   └── dataset.py           # Datenloader
│   └── tests/
│       ├── __init__.py
│       ├── test_precompute.py
│       ├── test_forward.py
│       ├── test_sampling.py
│       └── test_integration.py
├── experiments/
│   ├── notebooks/
│   │   ├── exploration.ipynb    # Daten-Exploration
│   │   ├── validation.ipynb     # Model-Validierung
│   │   └── results.ipynb        # Ergebnis-Analyse
│   └── configs/
│       ├── base_config.yaml
│       ├── small_test.yaml
│       └── production.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Aufgaben

### Phase 1: Grundlagen und Mathematik ✅

#### 1.1 Mathematische Grundlagen ✅
- [x] **1.1.1** Formeln aus Paper extrahiert und in `docs/formulas.md` dokumentiert
- [x] **1.1.2** Tensor-Shapes und PyTorch-Dimensionen definiert
- [x] **1.1.3** Kronecker-Struktur für Effizienz spezifiziert

#### 1.2 Projektstruktur ✅
- [x] **1.2.1** Verzeichnisstruktur erstellen
- [x] **1.2.2** `requirements.txt` mit Dependencies erstellen
- [x] **1.2.3** `setup.py` für Package-Installation erstellen

### Phase 2: Core-Implementierung ✅

#### 2.1 Precomputation (Φ und Σ Matrizen) ✅
- [x] **2.1.1** `src/core/precompute.py` implementieren
  - [x] `build_A_node()` - Graph-Laplacian zu A-Matrix
  - [x] `precompute_phi_sigma_explicit()` - Diskrete Euler-Rekursion
  - [x] Unterstützung für diag_approx und full matrix
  - [x] Numerische Stabilität (Regularisierung, Konditionierung)

#### 2.2 Forward-Noising Prozess ✅
- [x] **2.2.1** `src/core/forward.py` implementieren
  - [x] `apply_Phi_node_batch()` - Batch-fähige Φ-Anwendung
  - [x] `forward_noising_batch()` - St = Φ(t)S0 + √Σ(t)ε
  - [x] `compute_hat_S0_from_eps_hat()` - Rekonstruktion von Ŝ0
  - [x] Support für diag und full Σ Varianten

#### 2.3 Graph-Utilities ✅
- [x] **2.3.1** `src/data/graph_builder.py` implementieren
  - [x] `build_grid_laplacian()` - Einfacher Grid-Graph für Tests
  - [x] `build_from_edges()` - Graph aus Edge-Liste
  - [x] `build_molecular_graph()` - Molekül-spezifische Topologie
  - [x] Sparse-Matrix Unterstützung

### Phase 3: Modell-Architektur ✅

#### 3.1 Epsilon-Netzwerk (ε-MSE Ansatz) ✅
- [x] **3.1.1** `src/models/eps_net.py` implementieren
  - [x] Basis MLP pro Knoten
  - [x] Graph-Convolution Layer für Nachbarschafts-Information
  - [x] Zeit-Embedding für t-Abhängigkeit
  - [x] Batch-Normalisierung und Residual-Connections

#### 3.2 Score-Netzwerk (Alternative)
- [ ] **3.2.1** `src/models/score_net.py` implementieren
  - [ ] Score-Funktion s_θ(St, t) ≈ ∇ log p(St)
  - [ ] Kompatibel mit ε-Net über s = -Σ^(-1/2) ε Beziehung

#### 3.3 GNN-Komponenten ✅
- [x] **3.3.1** `src/models/gnn_layers.py` implementieren
  - [x] Graph-Convolution mit Laplacian-Awareness
  - [x] Attention-Mechanismus für variable Nachbarschaftsgrößen
  - [x] Feature-Normalisierung

### Phase 4: Training und Losses ✅

#### 4.1 Loss-Funktionen ✅
- [x] **4.1.1** `src/training/losses.py` implementieren
  - [x] MSE Loss für ε-Vorhersage: ||ε - ε_θ(St,t)||²
  - [x] Score-Matching Loss (Denoising Score Matching)
  - [x] Zeitgewichtung λ(t) Support

#### 4.2 Physikalische Regularisierer ✅
- [x] **4.2.1** `src/training/regularizers.py` implementieren
  - [x] Energie-Regularisierer: E(Ŝ0) = ½||M^(1/2)(Ŝ0 - S*)||²
  - [x] Graph-Regularisierer: ||L Ŝ0||²_F für Smoothness
  - [x] Alignment-Regularisierer: Richtungsfeld-Erhaltung
  - [x] Kombinierte Regularisierung mit Gewichtsfaktoren λE, λG, λA

#### 4.3 Training-Pipeline ✅
- [x] **4.3.1** `src/training/trainer.py` implementieren
  - [x] Training-Loop mit Batch-Processing
  - [x] Zeitschritt-Sampling (uniform/gewichtet)
  - [x] Optimizer-Setup (Adam mit lr-Schedule)
  - [x] Gradient-Clipping für Stabilität
  - [x] Logging und Metrics-Tracking

#### 4.3 Training-Pipeline
- [ ] **4.3.1** `src/training/trainer.py` implementieren
  - [ ] Training-Loop mit Batch-Processing
  - [ ] Zeitschritt-Sampling (uniform/gewichtet)
  - [ ] Optimizer-Setup (Adam mit lr-Schedule)
  - [ ] Gradient-Clipping für Stabilität
  - [ ] Logging und Metrics-Tracking

### Phase 5: Sampling und Inferenz ✅

#### 5.1 Reverse-Sampling ✅
- [x] **5.1.1** `src/core/sampling.py` implementieren
  - [x] `compute_F_Q_from_PhiSigma()` - Diskrete Übergangsmatrizen
  - [x] `sample_reverse_from_S_T()` - Vollständiger Sampling-Loop
  - [x] Posterior-Konditionierung für Gaussian-Steps
  - [x] Support für sowohl diag_approx als auch full Σ

#### 5.2 Sampling-Strategien ✅
- [x] **5.2.1** Erweiterte Sampling-Methoden
  - [x] DDIM deterministic sampling
  - [x] Predictor-Corrector (Langevin) für höhere Qualität
  - [x] Flexible Schritt-Größen
  - [x] Conditional sampling (über ConditionalEpsilonNetwork)

### Phase 6: Testing und Validierung

#### 6.1 Unit-Tests
- [ ] **6.1.1** `src/tests/test_precompute.py`
  - [ ] Φ-Matrix Eigenschaften (Invertierbarkeit, Konditionierung)
  - [ ] Σ-Matrix Positive Definiteness
  - [ ] Konsistenz zwischen diag_approx und full
  
- [ ] **6.1.2** `src/tests/test_forward.py`
  - [ ] Roundtrip-Test: S0 → St → Ŝ0 ≈ S0 mit korrektem ε
  - [ ] Batch-Konsistenz
  - [ ] Shape-Validierung

- [ ] **6.1.3** `src/tests/test_sampling.py`
  - [ ] Posterior-Mittelwert Korrektheit
  - [ ] Sampling-Verteilung vs. Ground-Truth
  - [ ] Numerische Stabilität bei extremen Parametern

#### 6.2 Integrationstests
- [ ] **6.2.1** `src/tests/test_integration.py`
  - [ ] End-to-end Training auf synthetischen Daten
  - [ ] Sampling-Qualität metrics
  - [ ] Physikalische Constraint-Erfüllung

### Phase 7: Experimente und Optimierung

#### 7.1 Baseline-Experimente
- [ ] **7.1.1** `experiments/notebooks/exploration.ipynb`
  - [ ] Datenverteilung-Analyse
  - [ ] Hyperparameter-Sensitivität
  - [ ] Architektur-Ablations (MLP vs GNN)

#### 7.2 Hyperparameter-Tuning
- [ ] **7.2.1** Parameter-Optimierung
  - [ ] γ, λ Werte für Graph-Laplacian Gewichtung
  - [ ] β(t) Schedule für Noise-Stärke
  - [ ] λE, λG, λA für Regularisierer-Balance
  - [ ] Netzwerk-Größe und Tiefe

#### 7.3 Performance-Optimierung
- [ ] **7.3.1** Effizienz-Verbesserungen
  - [ ] Sparse-Matrix Operationen optimieren
  - [ ] Memory-efficient Σ Approximationen (Low-rank)
  - [ ] GPU-Memory Management für große Graphs
  - [ ] Iterative Linear-Solver statt direkter Inversion

### Phase 8: Dokumentation und Deployment

#### 8.1 Dokumentation
- [ ] **8.1.1** Code-Dokumentation vervollständigen
- [ ] **8.1.2** `README.md` mit Usage-Examples
- [ ] **8.1.3** `docs/architecture.md` mit System-Design
- [ ] **8.1.4** Tutorial-Notebooks für neue Nutzer

#### 8.2 Packaging
- [ ] **8.2.1** PyPI-Package vorbereiten
- [ ] **8.2.2** Docker-Container für reproduzierbare Environments
- [ ] **8.2.3** CI/CD Pipeline für automatische Tests

## Identifizierte Logikfehler aus ursprünglichem Dokument

### ❌ Strukturelle Probleme:
1. **Chat-Format**: Original war Copy-Paste von ChatGPT Konversation
2. **Keine klaren Aufgaben**: Nur Beschreibungen, keine actionable Items
3. **Fehlende Priorisierung**: Alles gleich wichtig, keine Abhängigkeiten
4. **Keine Verzeichnisstruktur**: Code-Organisation unklar

### ❌ Technische Inkonsistenzen:
1. **Mixing von Approximations**: diag_approx vs full Σ ohne klare Strategie
2. **Memory-Probleme**: (B*d, N, N) Solve ohne Speicher-Bedenken
3. **Numerische Stabilität**: Regularisierung nur als Nachgedanke
4. **Testing fehlt**: Keine Validierungs-Strategie

### ❌ Implementierungs-Lücken:
1. **Modulare Struktur fehlt**: Alles in einer Datei geplant
2. **Error-Handling**: Keine Robustheit gegen fehlerhafte Inputs
3. **Konfiguration**: Hardcoded Parameter statt konfigurierbarer Settings
4. **Logging/Monitoring**: Keine Observability für Training-Prozess

## Prioritäten-Matrix

| Phase | Kritisch | Hoch | Mittel | Niedrig |
|-------|----------|------|--------|---------|
| 1 | 1.2.* | - | - | - |
| 2 | 2.1.1, 2.2.1 | 2.3.1 | - | - |
| 3 | 3.1.1 | 3.3.1 | 3.2.1 | - |
| 4 | 4.1.1, 4.2.1 | 4.3.1 | - | - |
| 5 | 5.1.1 | 5.2.1 | - | - |
| 6 | 6.1.*, 6.2.1 | - | - | - |
| 7 | - | 7.1.1 | 7.2.1 | 7.3.1 |
| 8 | - | - | 8.1.* | 8.2.* |

## Nächste Schritte

1. **Sofort**: Verzeichnisstruktur erstellen (Aufgabe 1.2.1)
2. **Diese Woche**: Precompute-Modul implementieren (Aufgabe 2.1.1)
3. **Nächste Woche**: Forward-Noising und erste Tests (Aufgaben 2.2.1, 6.1.2)
4. **Monat 1**: Vollständiges Training-System (Phasen 1-4)
5. **Monat 2**: Sampling und Validierung (Phasen 5-6)
6. **Monat 3**: Optimierung und Dokumentation (Phasen 7-8)