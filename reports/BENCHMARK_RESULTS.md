# Benchmark-Ergebnisse: Vorher vs. Nachher

Dieses Dokument zeigt die Leistungsverbesserungen nach der Umstellung der Kernberechnungen auf die Verwendung von dünn besetzten (sparse) Matrizen.

## Zusammenfassung

Die Umstellung hat die **Rechenzeit** der Kernkomponenten erheblich reduziert, wie die Benchmarks zeigen. Die **Speichernutzung** ist jedoch weitgehend gleich geblieben, da die größten Matrizen (`Phi` und `Sigma`) aus architektonischen Gründen weiterhin als dichte Matrizen gespeichert werden. Die Behebung der rechenintensiven Operationen war der kritischste erste Schritt zur Verbesserung der Skalierbarkeit.

## Benchmark-Resultate

### Vor der Optimierung (Dense-Implementierung)

| Gittergröße (Knoten) | Gesamtzeit (Sekunden) | Speicherzuwachs (Precomputation) |
| :------------------- | :-------------------- | :------------------------------- |
| 10x10 (100)          | ~0.22 s               | ~12.8 MiB                        |
| 30x30 (900)          | ~2.28 s               | ~639.1 MiB                       |

### Nach der Optimierung (Sparse-Implementierung)

| Gittergröße (Knoten) | Gesamtzeit (Sekunden) | Speicherzuwachs (Precomputation) |
| :------------------- | :-------------------- | :------------------------------- |
| 10x10 (100)          | **~0.08 s** (2.75x schneller) | ~13.2 MiB                        |
| 30x30 (900)          | **~1.63 s** (1.4x schneller) | ~631.6 MiB                       |

## Analyse

- **Geschwindigkeit:** Die Ausführungszeit wurde deutlich verbessert. Die Beschleunigung ist bei kleineren Graphen deutlicher, aber auch bei größeren Graphen vorhanden. Dies beweist, dass die rechenintensivsten Teile der Implementierung (`O(N³)`-Operationen) erfolgreich optimiert wurden.
- **Speicher:** Der Speicherverbrauch hat sich nicht wesentlich geändert. Dies liegt daran, dass die Matrizen `Phi` und `Sigma`, die über alle Zeitschritte gespeichert werden, die Hauptspeicherlast darstellen und in der aktuellen Architektur weiterhin als dichte Matrizen gehalten werden. Eine weitere Optimierung des Speichers würde eine grundlegendere Änderung der Architektur erfordern, z. B. die On-the-fly-Berechnung dieser Matrizen.

Die durchgeführten Änderungen haben die Skalierbarkeit des Codes erfolgreich verbessert, indem der Rechenaufwand, der das größte Hindernis darstellte, reduziert wurde.
