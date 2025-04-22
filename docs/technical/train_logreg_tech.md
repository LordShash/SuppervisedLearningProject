# Technische Dokumentation: Logistische Regression

## Übersicht
Das Modul `train_logreg.py` implementiert das Training und die Evaluation eines logistischen Regressionsmodells für die Textklassifikation. Es bietet Funktionen zum Trainieren, Evaluieren und Speichern des Modells.

## Hauptfunktionen

### `train_and_save_model(target_column, max_features=1000, test_size=0.2, max_iter=1000, C=1.0, solver='lbfgs')`
Trainiert ein logistisches Regressionsmodell mit den angegebenen Parametern und speichert es.

- **Parameter**:
  - `target_column`: Name der Zielspalte
  - `max_features`: Maximale Anzahl der Features für TF-IDF (Standard: 1000)
  - `test_size`: Anteil der Testdaten (Standard: 0.2)
  - `max_iter`: Maximale Anzahl von Iterationen (Standard: 1000)
  - `C`: Regularisierungsparameter (Standard: 1.0)
  - `solver`: Algorithmus für die Optimierung (Standard: 'lbfgs')
- **Rückgabewert**: Tuple mit Metriken (accuracy, precision, recall, f1, report, conf_matrix)

## Technische Details

### Datenverarbeitung
Das Modul verwendet das `data_loader`-Modul, um die Daten zu laden und vorzuverarbeiten:
- Laden der Textdaten und der Zielvariable
- TF-IDF-Vektorisierung der Textdaten
- Aufteilung in Trainings- und Testdaten

### Modelltraining
Für das Training wird die `LogisticRegression`-Klasse aus scikit-learn verwendet:
- Konfigurierbare Hyperparameter: max_iter, C, solver
- Multi-Class-Klassifikation mit 'multinomial' als Strategie (wenn unterstützt)
- Fallback auf 'ovr' (One-vs-Rest) für Solver, die 'multinomial' nicht unterstützen

### Modellspeicherung
Das trainierte Modell wird mit pickle serialisiert und in einer Datei gespeichert:
- Speicherort: `models/logreg_{target_column}_model.pkl`
- Enthält das trainierte Modell mit allen Parametern

### Evaluation
Das Modul berechnet verschiedene Metriken zur Bewertung der Modellleistung:
- Accuracy: Gesamtgenauigkeit des Modells
- Precision: Präzision (Verhältnis der korrekt positiven Vorhersagen zu allen positiven Vorhersagen)
- Recall: Trefferquote (Verhältnis der korrekt positiven Vorhersagen zu allen tatsächlich positiven Fällen)
- F1-Score: Harmonisches Mittel aus Precision und Recall
- Klassifikationsbericht: Detaillierte Metriken für jede Klasse
- Konfusionsmatrix: Visualisierung der Vorhersageergebnisse

## Abhängigkeiten
- `scikit-learn`: Für Modelltraining, Evaluation und Datenverarbeitung
- `numpy`: Für numerische Operationen
- `pickle`: Für Modellserialisierung
- Projektmodule:
  - `data_loader`: Zum Laden und Vorverarbeiten der Daten
- Standard-Python-Bibliotheken: `os`, `sys`, `time`