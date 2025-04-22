# Technische Dokumentation: Neuronales Netz

## Übersicht
Das Modul `train_nn.py` implementiert das Training und die Evaluation eines neuronalen Netzes für die Textklassifikation. Es bietet Funktionen zum Trainieren, Evaluieren und Speichern des Modells.

## Hauptfunktionen

### `train_and_save_model(target_column, max_features=1000, test_size=0.2, epochs=30, patience=5)`
Trainiert ein neuronales Netz mit den angegebenen Parametern und speichert es.

- **Parameter**:
  - `target_column`: Name der Zielspalte
  - `max_features`: Maximale Anzahl der Features für TF-IDF (Standard: 1000)
  - `test_size`: Anteil der Testdaten (Standard: 0.2)
  - `epochs`: Anzahl der Trainingsepochen (Standard: 30)
  - `patience`: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird (Standard: 5)
- **Rückgabewert**: Tuple mit Metriken (accuracy, precision, recall, f1, report, conf_matrix)

## Technische Details

### Datenverarbeitung
Das Modul verwendet das `data_loader`-Modul, um die Daten zu laden und vorzuverarbeiten:
- Laden der Textdaten und der Zielvariable
- TF-IDF-Vektorisierung der Textdaten
- Aufteilung in Trainings- und Testdaten
- Konvertierung der Zielvariable in One-Hot-Encoding

### Modellarchitektur
Das neuronale Netz wird mit PyTorch implementiert:
- Eingabeschicht: Größe entspricht der Anzahl der Features
- Versteckte Schicht: 128 Neuronen mit ReLU-Aktivierung
- Ausgabeschicht: Größe entspricht der Anzahl der Klassen mit Softmax-Aktivierung
- Dropout (0.2) zur Vermeidung von Overfitting

### Modelltraining
Das Training erfolgt mit folgenden Komponenten:
- Verlustfunktion: Cross-Entropy-Loss
- Optimierer: Adam mit Lernrate 0.001
- Early Stopping: Training wird beendet, wenn sich die Validierungsgenauigkeit für `patience` Epochen nicht verbessert
- Batch-Größe: 64

### Modellspeicherung
Das trainierte Modell wird mit PyTorch serialisiert und in einer Datei gespeichert:
- Speicherort: `models/nn_{target_column}_model.pt`
- Enthält die trainierten Gewichte des Modells

### Evaluation
Das Modul berechnet verschiedene Metriken zur Bewertung der Modellleistung:
- Accuracy: Gesamtgenauigkeit des Modells
- Precision: Präzision (Verhältnis der korrekt positiven Vorhersagen zu allen positiven Vorhersagen)
- Recall: Trefferquote (Verhältnis der korrekt positiven Vorhersagen zu allen tatsächlich positiven Fällen)
- F1-Score: Harmonisches Mittel aus Precision und Recall
- Klassifikationsbericht: Detaillierte Metriken für jede Klasse
- Konfusionsmatrix: Visualisierung der Vorhersageergebnisse

## Abhängigkeiten
- `PyTorch`: Für die Implementierung und das Training des neuronalen Netzes
- `scikit-learn`: Für Evaluation und Datenverarbeitung
- `numpy`: Für numerische Operationen
- Projektmodule:
  - `data_loader`: Zum Laden und Vorverarbeiten der Daten
- Standard-Python-Bibliotheken: `os`, `sys`, `time`