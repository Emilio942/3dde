# Fehlerbericht & Trainingsanalyse: Overnight Run

**Datum:** 19. November 2025  
**Status:** Training manuell beendet (Epoche 389/5000)  
**Ergebnis:** Erfolgreich konvergiert, Daten gesichert.

---

## 1. Vorfallbericht: Manueller Abbruch
Das Training wurde in Epoche 389 durch einen manuellen Eingriff (versehentliches Senden eines Befehls in das laufende Terminal) abrupt beendet.
*   **Auswirkung:** Der Prozess wurde gestoppt.
*   **Datenverlust:** **Keiner.**
    *   `best.pt` (Bestes Modell) wurde sicher in Epoche 277 gespeichert.
    *   `latest.pt` (Letzter Stand) wurde Sekunden vor dem Abbruch gespeichert.

---

## 2. Analyse der Trainingswerte ("Warum diese Werte?")

Der Nutzer stellte fest, dass der Training Loss (~7.6) sehr hoch war, während der Validation Loss (~1.1) niedrig blieb und stagnierte.

### A. Die Diskrepanz (7.6 vs 1.1)
Der massive Unterschied zwischen Training- und Validation-Loss ist **kein Fehler**, sondern eine Folge der Konfiguration in `overnight.yaml`:

```yaml
loss:
  eps_weight: 1.0
  regularization_weight: 0.1  # <--- Der Grund
regularization:
  lambda_smoothness: 0.1      # <--- Zusätzliche Strafe
```

*   **Validation Loss (~1.11):** Misst nur, wie gut das Modell das Rauschen vorhersagt (Reconstruction Error). Das ist der "echte" Leistungswert.
*   **Training Loss (~7.62):** Setzt sich zusammen aus:
    1.  Reconstruction Error (~1.1)
    2.  **PLUS** Regularisierungs-Strafe (Physik-Constraints, Glattheit).
    
Das Modell kämpft also während des Trainings gegen "harte" physikalische Regeln an, was den Wert künstlich hochtreibt.

### B. Die Stagnation (Plateau ab Epoche 277)
Das Modell hat bei Epoche 277 seinen optimalen Punkt erreicht (`Val Loss: 1.09`). Danach schwankten die Werte nur noch (Oszillation zwischen 1.10 und 1.12). Das bedeutet, das Modell ist **konvergiert**. Es konnte aus den Daten nichts mehr lernen.

---

## 3. Warum wurde es nicht früher erkannt? ("Weswegen nicht früher gestoppt?")

Das Training lief über 100 Epochen "umsonst" weiter. Gründe dafür:

1.  **Fehlendes "Early Stopping":**
    Der Trainings-Code (`trainer.py`) enthält keinen Mechanismus, der das Training automatisch stoppt, wenn sich der Validation Loss für X Epochen nicht verbessert.
    
2.  **Harte Vorgabe der Epochenzahl:**
    In der Config war `num_epochs: 5000` eingestellt. Ohne Early Stopping versucht das Skript stur, diese 5000 Epochen zu erreichen, egal ob das Modell schon fertig ist oder nicht.

3.  **Überwachung:**
    Da es ein "Overnight"-Training war, wurde es nicht aktiv überwacht. Ein automatischer Wächter (Monitor) fehlte.

---

## 4. Empfehlungen für die Zukunft

Um Energie und Zeit zu sparen, sollten folgende Änderungen implementiert werden:

1.  **Early Stopping implementieren:**
    Das Skript sollte abbrechen, wenn sich `best.pt` seit 50 Epochen nicht aktualisiert hat.
    
2.  **Getrennte Logs:**
    Der Training-Loss sollte im Log aufgeteilt werden in `Recon Loss` und `Reg Loss`, um Verwirrung über die hohen Werte zu vermeiden.

3.  **Dynamische Lernrate:**
    Wenn das Plateau erreicht ist, sollte die Lernrate automatisch reduziert werden (ReduceLROnPlateau), um vielleicht doch noch kleine Verbesserungen zu erzielen.
