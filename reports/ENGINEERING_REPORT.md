# Engineering Post-Mortem & Future Architecture

**Perspektive:** Software & ML Engineering
**Vorfall:** Ungeplanter Trainingsabbruch (Process Termination)
**Systemzustand:** Stabil, aber optimierungsbedürftig.

---

## 1. Was ist passiert? (Root Cause Analysis)

Aus Ingenieurssicht lag hier ein **"Process Isolation Failure"** vor.

*   **Das Problem:** Der Trainingsprozess (`train_with_config.py`) lief als Kindprozess der interaktiven Shell (VS Code Terminal).
*   **Die Schwachstelle:** Es gab keine Trennung zwischen "Control Plane" (Terminal-Eingabe) und "Data Plane" (Training). Ein Fehler in der Eingabe (`Ctrl+C` / Signal Injection) hat direkt die Ausführungsebene getroffen.
*   **Der "Fix" (Best Practice):** Langläufer-Prozesse sollten **entkoppelt** laufen.
    *   *Local:* `nohup`, `tmux`, `screen` oder als `systemd` Service.
    *   *Cloud:* Docker Container, Kubernetes Job.

---

## 2. Ressourcen-Effizienz (The "Stagnation" Issue)

Das System hat Rechenleistung verschwendet (Compute Waste).

*   **Beobachtung:** Konvergenz bei Epoche ~280, Abbruch bei ~390.
*   **Ineffizienz:** ~110 Epochen (ca. 20-30% der Laufzeit) waren redundant.
*   **Engineering Solution:** Implementierung eines **Early Stopping Callbacks**.
    *   *Logik:* Überwache `val_loss`. Wenn `min_delta` über `patience=50` Epochen nicht erreicht wird -> `sys.exit(0)`.

---

## 3. Daten-Pipeline (Mocking vs. Production)

Das aktuelle Setup ist ein klassischer **"Integration Test"**.

*   **Status Quo:** Wir nutzen `SyntheticDataset` (Sinus-Wellen).
*   **Ingenieurs-Sicht:** Das ist ein **Mock**. Wir testen die Pipeline (Datenfluss -> Modell -> Loss -> Backprop), ohne externe Abhängigkeiten (echte Dateien).
*   **Nächster Schritt:** Austausch des Mocks durch einen **Production Data Loader**.
    *   Format: HDF5 oder Parquet für große 3D-Datensätze.
    *   Validierung: Pydantic-Modelle, um sicherzustellen, dass echte Daten das richtige Format (Shape, Typ) haben.

---

## 4. Empfohlene Architektur-Upgrades (Roadmap)

### A. Graceful Shutdown (Signal Handling)
Das Skript sollte nicht einfach "sterben", wenn es unterbrochen wird.

```python
import signal
import sys

def signal_handler(sig, frame):
    print('Graceful Shutdown eingeleitet... Speichere Checkpoint.')
    trainer.save_checkpoint("emergency_save.pt")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
```

### B. Monitoring & Observability
Statt roher Text-Logs sollten wir strukturierte Metriken nutzen.
*   **TensorBoard:** Standard für ML-Visualisierung.
*   **JSON-Logs:** Maschinenlesbare Logs für automatisierte Auswertung.

### C. Config Management
Die `overnight.yaml` ist gut, aber wir sollten **Hydra** oder **Pydantic** nutzen, um Konfigurationen typsicher zu machen und Validierungsfehler vor dem Start abzufangen.

---

## Fazit

Das System funktioniert ("It works"), ist aber noch nicht "Production Ready". Es ist ein solider **Prototyp**. Der nächste Schritt wäre die **Härtung** (Hardening) des Codes gegen Fehler und die Anbindung an echte Datenquellen.
