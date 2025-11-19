# Fehlerbericht und Verbesserungsvorschläge

Dieses Dokument listet potenzielle Fehler, konzeptionelle Schwächen und Bereiche für Verbesserungen auf, die während der Analyse des Projekts identifiziert wurden.

### 1. Kritisches Skalierbarkeitsproblem durch dichte Matrizen

**Problem:** Die Kernlogik des Diffusionsmodells in `src/core/precompute.py` und `src/data/graph_builder.py` verwendet ausschließlich dichte `(N, N)`-Matrizen für die Graphen-Laplacians und alle nachfolgenden Berechnungen. Dies führt zu einer inakzeptablen Komplexität:

*   **Rechenaufwand:** `O(N³)` für Matrixoperationen (Multiplikation, Inversion).
*   **Speicherbedarf:** `O(N² * num_steps)`.

Diese Komplexität macht das Modell für alle bis auf die kleinsten Graphen praktisch unbrauchbar und verhindert jegliche Skalierung.

**Lösungsvorschlag:**
*   **Sparse Matrizen:** Umstellung der Graphenrepräsentation und aller Berechnungen auf dünn besetzte (sparse) Matrizen. Bibliotheken wie `scipy.sparse` sind hierfür ideal.
*   **Implementierung von Kronecker-Operationen:** Die `README.md` erwähnt "Efficient Kronecker-structured operations". Diese sollten für Gittergraphen implementiert werden, um die Skalierbarkeit massiv zu verbessern.

### 2. Potenzieller konzeptioneller Fehler im Noising-Prozess

**Problem:** In `src/core/precompute.py` wird die Kovarianzmatrix des Rauschens (`Sigma`) so berechnet, dass sie von der Graphen-Laplacian-Matrix `A` abhängt (`... + dt * A`). Üblicherweise wird hier isotropes Rauschen verwendet, das von der Identitätsmatrix `I` abhängt (`... + dt * I`). Diese unkonventionelle Wahl bedeutet, dass das Rauschen anisotrop ist und von der Struktur des Graphen abhängt. Es ist unklar, ob dies beabsichtigt ist und ob es theoretisch fundiert ist. Es könnte zu unerwarteten und potenziell schlechten Ergebnissen führen.

**Lösungsvorschlag:**
*   **Validierung des Ansatzes:** Überprüfen, ob dieses anisotrope Rauschen beabsichtigt ist und welche theoretische Grundlage es hat.
*   **Standardimplementierung:** Testweise auf eine Standardimplementierung mit isotropem Rauschen umstellen und die Ergebnisse vergleichen.

### 3. Unklare und fehleranfällige Funktionssignaturen

**Problem:** Funktionen wie `forward_noising_batch` in `src/core/forward.py` verwenden eine Mischung aus `*args` und `**kwargs`, um verschiedene Aufrufmuster zu ermöglichen. Dies macht den Code schwer lesbar, schwer zu debuggen und fehleranfällig.

**Lösungsvorschlag:**
*   **Refactoring:** Die Funktionen sollten refaktorisiert werden, um klare und explizite Argumente zu haben. Überladen oder separate Funktionen für unterschiedliche Anwendungsfälle sind eine bessere Alternative.

### 4. Ineffizientes Sampling

**Problem:** Die Sampling-Methoden (z.B. `ddim_sampling` in `src/core/sampling.py`) erben die Ineffizienz der dichten Matrizen. Operationen wie Matrixinversionen und -multiplikationen sind extrem langsam.

**Lösungsvorschlag:**
*   Die Behebung des Skalierbarkeitsproblems (Punkt 1) durch die Verwendung von Sparse-Matrizen und iterativen Lösern wird dieses Problem automatisch lösen.

### 5. Irreführender `NotImplementedError` für molekulare Graphen

**Problem:** Das Trainingsskript `experiments/train_with_config.py` wirft einen `NotImplementedError` für den Graphentyp 'molecular'. Allerdings existiert die Logik zum Erstellen solcher Graphen bereits in `src/data/graph_builder.py`. Lediglich die Datenpipeline im Trainingsskript ist nicht dafür konfiguriert.

**Lösungsvorschlag:**
*   **Anpassung der Datenpipeline:** Das Skript so anpassen, dass es die notwendigen Atompositionen für den Bau von molekularen Graphen akzeptiert und verarbeitet.
*   **Fehlermeldung präzisieren:** Die Fehlermeldung sollte klarstellen, dass nicht die Graphenerstellung, sondern die Datenzufuhr das Problem ist.
