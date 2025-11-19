# Fehlerbericht und Verbesserungsvorschläge (Version 2)

Diese zweite Analyse wurde nach der Umstellung auf Sparse-Matrizen durchgeführt. Sie hebt neue Engpässe und verbleibende Probleme hervor.

### 1. Kritisches Performance-Problem in den GNN-Layern (`src/models/gnn_layers.py`)

Nach der Optimierung der Datenstrukturen sind die GNN-Schichten nun der größte Leistungsengpass. Mehrere Probleme tragen dazu bei:

*   **Ineffiziente Normalisierung:** Die `LaplacianConv`-Schicht berechnet bei **jedem einzelnen Forward-Pass** die Eigenwerte des Laplacians neu, um ihn zu normalisieren. Dies ist extrem ineffizient, da der Laplacian während des Trainings konstant ist.
*   **Nicht-skalierbare Attention-Maske:** Die `GraphAttentionLayer` konvertiert die dünn besetzte Adjazenzmatrix in eine dichte Matrix, nur um sie als Maske zu verwenden. Bei großen Graphen führt dies unweigerlich zu "Out of Memory"-Fehlern.
*   **Redundante Adjazenz-Berechnung:** Der `GNNBlock` leitet die Adjazenzmatrix bei jedem Aufruf erneut aus dem Laplacian ab.

**Lösungsvorschlag:**
*   **Vor-Berechnung:** Die Normalisierung des Laplacians und die Erstellung der Adjazenzmatrix sollten **einmalig** vor dem Training stattfinden. Die Ergebnisse können dann an das Modell übergeben und in jedem Forward-Pass wiederverwendet werden.
*   **Optimierte Attention:** Anstatt eine dichte Maske zu erstellen, sollte eine sparse-fähige Attention-Implementierung verwendet werden. Bibliotheken wie `torch_geometric` bieten hierfür optimierte "Sparse-to-Dense"-Attention-Mechanismen.
*   **Alternative Bibliotheken:** Eine Umstellung der GNN-Layer auf eine spezialisierte Bibliothek wie `torch_geometric` würde diese Probleme wahrscheinlich vollständig beheben, da diese für solche Anwendungsfälle optimiert sind.

### 2. Potenzielle Ressourcen-Lecks im `HDF5Dataset` (`src/data/dataset.py`)

**Problem:** Die `HDF5Dataset`-Klasse öffnet Dateihandles, verlässt sich aber auf die `__del__`-Methode, um sie zu schließen. In Python ist `__del__` kein verlässlicher Destruktor, insbesondere wenn mehrere Worker in einem `DataLoader` verwendet werden. Dies kann dazu führen, dass Dateihandles offen bleiben, was zu Ressourcen-Lecks und potenziellen Fehlern führt.

**Lösungsvorschlag:**
*   **Context Manager:** Die Dateihandles sollten explizit in den `__getitem__`-Methoden geöffnet und geschlossen werden, idealerweise unter Verwendung eines `with`-Statements.
*   **Worker-spezifische Handles:** Alternativ kann für jeden Worker ein eigenes Dateihandle im `worker_init_fn` des `DataLoader` initialisiert und dort auch wieder geschlossen werden.

### 3. Langsame Eigenwert-Berechnung in `precompute.py`

**Problem:** Die Stabilitätsprüfung in `precompute_phi_sigma_explicit` verwendet `scipy.sparse.linalg.eigs`, was einen Wechsel zwischen PyTorch- und SciPy-Datenstrukturen erfordert und relativ langsam ist.

**Lösungsvorschlag:**
*   **PyTorch-native Alternativen:** Es sollte geprüft werden, ob neuere PyTorch-Versionen eine native, schnellere Methode zur Berechnung von Eigenwerten für Sparse-Matrizen bieten, um den Overhead zu reduzieren. Da dies jedoch eine einmalige Berechnung bei der Initialisierung ist, hat dieses Problem eine niedrigere Priorität als die GNN-Performance.
