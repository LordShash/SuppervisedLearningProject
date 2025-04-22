# Technische Dokumentation: Hauptmodul

## Übersicht
Das Modul `main.py` dient als Einstiegspunkt für das Projekt und orchestriert das Training verschiedener Modelle für die Textklassifikation. Es ermöglicht die Konfiguration und Ausführung von Trainingsläufen über die Kommandozeile.

## Hauptfunktionen

### `parse_arguments()`
Parst die Kommandozeilenargumente und gibt ein Namespace-Objekt mit den konfigurierten Parametern zurück.

- **Rückgabewert**: `argparse.Namespace` - Geparste Argumente
- **Unterstützte Argumente**:
  - `--model`: Zu trainierendes Modell (logreg, nn oder all)
  - `--target`: Zielvariable für das Training
  - `--epochs`: Anzahl der Trainingsepochen für neuronale Netze
  - `--max-features`: Maximale Anzahl der Features für TF-IDF
  - `--test-size`: Anteil der Testdaten
  - `--max-iter`: Maximale Anzahl von Iterationen für die logistische Regression
  - `--C`: Regularisierungsparameter für die logistische Regression
  - `--solver`: Algorithmus für die Optimierung der logistischen Regression
  - `--patience`: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird

### `train_models(model_type, target_column, ...)`
Trainiert die angegebenen Modelle mit den angegebenen Parametern.

- **Parameter**:
  - `model_type`: Zu trainierendes Modell ('logreg', 'nn' oder 'all')
  - `target_column`: Zielvariable für das Training
  - `epochs`: Anzahl der Trainingsepochen für neuronale Netze
  - `max_features`: Maximale Anzahl der Features für TF-IDF
  - `test_size`: Anteil der Testdaten
  - `max_iter`: Maximale Anzahl von Iterationen für die logistische Regression
  - `C`: Regularisierungsparameter für die logistische Regression
  - `solver`: Algorithmus für die Optimierung der logistischen Regression
  - `patience`: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird
- **Rückgabewert**: `Dict[str, Dict[str, Any]]` - Dictionary mit den Trainingsergebnissen

### `main()`
Hauptfunktion des Programms, die die Argumente parst und das Training startet.

## Technische Details

### Modelltypen
Das Modul unterstützt zwei Arten von Modellen:
1. Logistische Regression (`logreg`): Implementiert durch das Modul `train_logreg.py`
2. Neuronales Netz (`nn`): Implementiert durch das Modul `train_nn.py`

### Zielvariablen
Das Modul kann mit verschiedenen Zielvariablen arbeiten:
- Standardmäßig: 'Fits_Topic_Code'
- Weitere Optionen: 'Urgency_Code', 'Categorie_Code'
- Option 'all': Trainiert Modelle für alle verfügbaren Zielvariablen

### Fehlerbehandlung
Das Modul implementiert eine robuste Fehlerbehandlung:
- Abfangen von Ausnahmen während des Trainings einzelner Modelle
- Globale Ausnahmebehandlung in der `main()`-Funktion
- Spezielle Behandlung von Tastaturunterbrechungen (`KeyboardInterrupt`)

## Abhängigkeiten
- `argparse`: Für die Verarbeitung von Kommandozeilenargumenten
- `typing`: Für Typ-Annotationen
- Projektmodule:
  - `data_loader`: Zum Laden der Daten
  - `train_logreg`: Für das Training logistischer Regressionsmodelle
  - `train_nn`: Für das Training neuronaler Netze
- Standard-Python-Bibliotheken: `os`, `sys`