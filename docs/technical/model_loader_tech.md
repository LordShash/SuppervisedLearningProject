# Technische Dokumentation: Modelllademodul

## Übersicht
Das Modul `model_loader.py` ist verantwortlich für das Laden von trainierten Modellen für das Textklassifikationsprojekt. Es stellt Funktionen bereit, um verschiedene Arten von Modellen (Logistische Regression und neuronale Netze) zu laden und Informationen über den Ladeprozess zu erhalten.

## Hauptfunktionen

### `load_model(model_type=None, target_column=None, model_path=None)`
Lädt ein trainiertes Modell von der Festplatte mit flexiblen Pfadoptionen.

- **Parameter**:
  - `model_type`: Typ des zu ladenden Modells ('logreg' oder 'nn'). Erforderlich, wenn model_path nicht angegeben ist.
  - `target_column`: Zielspalte, die für das Training des Modells verwendet wurde. Erforderlich, wenn model_path nicht angegeben ist.
  - `model_path`: Direkter Pfad zur Modelldatei. Wenn angegeben, werden model_type und target_column ignoriert.
- **Rückgabewert**: `Optional[Any]` - Das geladene Modell oder None, wenn das Modell nicht geladen werden konnte.
- **Exceptions**:
  - Fängt alle Exceptions ab und gibt None zurück, wenn ein Fehler auftritt.
- **Logging**: Verwendet das Python-Logging-System, um Informationen und Fehler zu protokollieren.

### `load_model_with_info(model_type=None, target_column=None, model_path=None)`
Lädt ein Modell und gibt sowohl das Modell als auch Informationen über den Ladeprozess zurück.

- **Parameter**: Gleich wie bei `load_model`
- **Rückgabewert**: `Dict[str, Any]` - Ein Dictionary mit folgenden Schlüsseln:
  - 'model': Das geladene Modell oder None, wenn das Laden fehlgeschlagen ist
  - 'success': Boolean, der angibt, ob das Laden erfolgreich war
  - 'message': Informationsnachricht über den Ladeprozess
- **Exceptions**:
  - Fängt alle Exceptions ab und gibt ein Dictionary mit Fehlerinformationen zurück.

## Technische Details

### Unterstützte Modelltypen
Das Modul unterstützt zwei Arten von Modellen:
1. Logistische Regression: Gespeichert als .pkl-Dateien mit joblib
2. Neuronale Netze: Gespeichert als .pt-Dateien mit PyTorch

### Modellpfadkonvention
Wenn kein direkter Pfad angegeben wird, verwendet das Modul folgende Konvention für Modellpfade:
- Logistische Regression: `models/logreg_{target_column}_model.pkl`
- Neuronale Netze: `models/nn_{target_column}_model.pt`

### Fehlerbehandlung
Das Modul verwendet eine robuste Fehlerbehandlung:
1. Überprüfung der Eingabeparameter
2. Überprüfung der Existenz von Modelldateien
3. Abfangen von Ausnahmen während des Ladevorgangs
4. Detaillierte Fehlerberichte über das Logging-System

## Abhängigkeiten
- `joblib`: Für das Laden von Logistic Regression-Modellen
- `torch`: Für das Laden von neuronalen Netzmodellen
- Standard-Python-Bibliotheken: `os`, `logging`, `typing`