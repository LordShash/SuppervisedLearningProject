# Tutorial: Modelle laden und verwenden

## Einführung
In diesem Tutorial lernen Sie, wie Sie trainierte Modelle für die Textklassifikation laden und verwenden können. Das Modul `model_loader.py` bietet einfache Funktionen, um verschiedene Arten von Modellen zu laden und Informationen über den Ladeprozess zu erhalten.

## Voraussetzungen
- Ein trainiertes Modell (logistische Regression oder neuronales Netz)
- Grundlegende Python-Kenntnisse

## Schritt 1: Importieren des Modelllademodul
Zuerst müssen Sie das Modelllademodul importieren:

```python
from src.model_loader import load_model, load_model_with_info
```

## Schritt 2: Ein Modell laden
Es gibt zwei Hauptmethoden, um ein Modell zu laden:

### Methode 1: Laden mit Modelltyp und Zielspalte
Wenn Sie ein Modell mit dem Standardnamensmuster laden möchten, können Sie den Modelltyp und die Zielspalte angeben:

```python
# Logistische Regression laden
model = load_model(model_type="logreg", target_column="Fits_Topic_Code")

# Neuronales Netz laden
model = load_model(model_type="nn", target_column="Urgency_Code")
```

### Methode 2: Laden mit direktem Pfad
Wenn sich Ihr Modell an einem benutzerdefinierten Speicherort befindet, können Sie den direkten Pfad angeben:

```python
# Modell mit direktem Pfad laden
model = load_model(model_path="models/mein_eigenes_modell.pkl")
```

## Schritt 3: Fehlerbehandlung und Statusinformationen
Um mehr Informationen über den Ladeprozess zu erhalten, können Sie die Funktion `load_model_with_info` verwenden:

```python
# Modell mit Statusinformationen laden
result = load_model_with_info(model_type="logreg", target_column="Fits_Topic_Code")

# Prüfen, ob das Laden erfolgreich war
if result['success']:
    print(f"Erfolg: {result['message']}")
    model = result['model']
    # Modell für Vorhersagen verwenden
else:
    print(f"Fehler: {result['message']}")
```

## Schritt 4: Modell für Vorhersagen verwenden
Nachdem Sie das Modell geladen haben, können Sie es für Vorhersagen verwenden:

```python
# Für logistische Regression
if model is not None:
    # Angenommen, X_test enthält Ihre Testdaten
    predictions = model.predict(X_test)
    print(f"Vorhersagen: {predictions}")
```

## Tipps und Tricks

### Tipp 1: Überprüfen Sie, ob das Modell existiert
Bevor Sie versuchen, ein Modell zu laden, können Sie prüfen, ob die Modelldatei existiert:

```python
import os

model_path = os.path.join("models", "logreg_Fits_Topic_Code_model.pkl")
if os.path.exists(model_path):
    model = load_model(model_path=model_path)
else:
    print(f"Modell nicht gefunden: {model_path}")
```

### Tipp 2: Verschiedene Modelle für verschiedene Aufgaben
Sie können mehrere Modelle für verschiedene Klassifikationsaufgaben laden:

```python
# Modell für Themenklassifikation
topic_model = load_model(model_type="logreg", target_column="Fits_Topic_Code")

# Modell für Dringlichkeitsklassifikation
urgency_model = load_model(model_type="nn", target_column="Urgency_Code")
```

## Fehlerbehebung

### Problem: Modell kann nicht geladen werden
Wenn ein Modell nicht geladen werden kann, überprüfen Sie Folgendes:

1. Existiert die Modelldatei am angegebenen Speicherort?
2. Haben Sie den richtigen Modelltyp und die richtige Zielspalte angegeben?
3. Wurde das Modell mit dem richtigen Format gespeichert (.pkl für logistische Regression, .pt für neuronale Netze)?

### Problem: Fehler bei der Verwendung des Modells
Wenn Sie Fehler bei der Verwendung des Modells erhalten, stellen Sie sicher, dass:

1. Das Modell erfolgreich geladen wurde (nicht None).
2. Die Eingabedaten das gleiche Format haben wie die Trainingsdaten.
3. Sie die richtige Methode für Vorhersagen verwenden (z.B. `predict` für logistische Regression).

## Zusammenfassung
In diesem Tutorial haben Sie gelernt, wie Sie:
- Trainierte Modelle mit dem Modelllademodul laden
- Statusinformationen über den Ladeprozess erhalten
- Geladene Modelle für Vorhersagen verwenden
- Häufige Probleme beim Laden und Verwenden von Modellen beheben

Mit diesen Kenntnissen können Sie nun trainierte Modelle in Ihren eigenen Anwendungen einsetzen.