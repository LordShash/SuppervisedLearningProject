# Logistische Regression - Technische Referenz

## Übersicht

Das Modul `train_logreg.py` implementiert das Training und die Speicherung eines logistischen Regressionsmodells für die Textklassifikation. Es verwendet scikit-learn für die Modellimplementierung und Evaluierung.

## Funktionen

### train_logistic_regression()

**Beschreibung**: Trainiert ein logistisches Regressionsmodell mit Standardskalierung.

**Inputs**:
- `X` (np.ndarray): Feature-Matrix.
- `y` (np.ndarray): Zielvariable.
- `target_name` (str, optional): Name der Zielvariable für Reporting. Standard: "Fits_Topic_Code".
- `test_size` (float, optional): Anteil der Testdaten. Standard: 0.2.
- `random_state` (int, optional): Seed für die Reproduzierbarkeit. Standard: 42.

**Outputs**:
- `pipeline` (Pipeline): Trainiertes Modell als scikit-learn Pipeline.
- `accuracy` (float): Genauigkeit auf dem Testset.
- `precision` (float): Precision-Score auf dem Testset.
- `recall` (float): Recall-Score auf dem Testset.
- `f1` (float): F1-Score auf dem Testset.
- `report` (str): Detaillierter Klassifikationsbericht.
- `conf_matrix` (np.ndarray): Konfusionsmatrix.

**Verhalten**:
1. Teilt die Daten in Trainings- und Testsets auf.
2. Erstellt eine Pipeline mit Standardskalierung und logistischer Regression.
3. Trainiert das Modell mit den Trainingsdaten.
4. Evaluiert das Modell mit den Testdaten.
5. Berechnet Genauigkeit und Klassifikationsbericht.

### save_model()

**Beschreibung**: Speichert das trainierte Modell als PKL-Datei.

**Inputs**:
- `model` (Pipeline): Trainiertes Modell.
- `target_column` (str, optional): Name der Zielspalte für den Dateinamen. Standard: "Fits_Topic_Code".

**Outputs**: Keine direkten Rückgabewerte.

**Verhalten**:
1. Ermittelt den Pfad zum models-Verzeichnis.
2. Erstellt das Verzeichnis, falls es nicht existiert.
3. Generiert einen Dateinamen basierend auf der Zielspalte.
4. Speichert das Modell mit joblib.

### train_and_save_model()

**Beschreibung**: Lädt Daten, trainiert ein Modell und speichert es.

**Inputs**:
- `target_column` (str, optional): Name der Zielspalte. Standard: "Fits_Topic_Code".

**Outputs**:
- `accuracy` (float): Genauigkeit des Modells.
- `precision` (float): Precision-Score des Modells.
- `recall` (float): Recall-Score des Modells.
- `f1` (float): F1-Score des Modells.
- `report` (str): Detaillierter Klassifikationsbericht.
- `conf_matrix` (np.ndarray): Konfusionsmatrix.

**Verhalten**:
1. Lädt die Daten mit der angegebenen Zielspalte.
2. Trainiert ein logistisches Regressionsmodell.
3. Speichert das trainierte Modell.
4. Gibt Genauigkeit, Precision, Recall, F1-Score, Klassifikationsbericht und Konfusionsmatrix zurück.

## Abhängigkeiten

- **Externe Bibliotheken**:
  - `os`: Für Pfadoperationen.
  - `sys`: Für Systemfunktionen.
  - `joblib`: Zum Speichern des Modells.
  - `numpy`: Für numerische Operationen.
  - `sklearn`: Für Modelltraining, Evaluation und Datenverarbeitung.

- **Interne Module**:
  - `data_loader`: Für das Laden der Daten.

## Modelldetails

- **Modelltyp**: Logistische Regression
- **Vorverarbeitung**: StandardScaler (with_mean=False für Sparse-Matrizen)
- **Hyperparameter**:
  - `random_state`: 42 (für Reproduzierbarkeit)
  - `max_iter`: 1000 (erhöhte Anzahl von Iterationen für Konvergenz)

## Fehlerbehandlung

Das Modul verwendet Try-Except-Blöcke mit `sys.exit()` für verschiedene Fehlersituationen:
- Fehler beim Training des Modells
- Fehler beim Speichern des Modells
- Fehler beim Laden der Daten

## Beispielverwendung

```python
from train_logreg import train_and_save_model

# Modell mit Standardzielspalte trainieren
accuracy, precision, recall, f1, report, conf_matrix = train_and_save_model()
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(report)
print("Konfusionsmatrix:")
print(conf_matrix)

# Modell mit anderer Zielspalte trainieren
accuracy, precision, recall, f1, report, conf_matrix = train_and_save_model(target_column="Urgency_Code")
```

## Ausgabedateien

- Modellspeicherung: `models/logreg_<target_column>_model.pkl`
