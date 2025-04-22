# Tutorial: Daten laden und verarbeiten

In diesem Tutorial lernen Sie, wie Sie das Modul `data_loader.py` verwenden, um Daten für Ihr Textklassifikationsprojekt zu laden und vorzubereiten.

## Einführung

Das Datenlademodul ist ein wichtiger Bestandteil des Projekts, da es die Daten aus CSV-Dateien lädt und für das Training der Modelle vorbereitet. Es bietet Funktionen zum Laden der Daten, zur Extraktion von Features und zur Identifizierung verfügbarer Zielvariablen.

## Daten laden

Die Hauptfunktion zum Laden der Daten ist `load_data()`. Hier ist ein einfaches Beispiel:

```python
from data_loader import load_data

# Daten mit Standardparametern laden
X, y = load_data()

# Ausgabe der Dimensionen
print(f"Feature-Matrix: {X.shape}")
print(f"Zielvariable: {y.shape}")
```

### Parameter anpassen

Sie können verschiedene Parameter anpassen:

```python
# Daten mit angepassten Parametern laden
X, y = load_data(
    target_column="Urgency_Code",  # Andere Zielvariable wählen
    max_features=2000              # Mehr Features verwenden
)
```

## Verfügbare Zielvariablen ermitteln

Wenn Sie wissen möchten, welche Zielvariablen in den Daten verfügbar sind, können Sie die Funktion `get_available_targets()` verwenden:

```python
from data_loader import get_available_targets

# Verfügbare Zielvariablen abrufen
targets = get_available_targets()

# Ausgabe der verfügbaren Zielvariablen
print("Verfügbare Zielvariablen:")
for target, info in targets.items():
    print(f"- {target}: {info['description']}")
    print(f"  Anzahl einzigartiger Werte: {info['unique_values']}")
```

## Fehlerbehandlung

Das Modul enthält robuste Fehlerbehandlung. Hier ist ein Beispiel, wie Sie mit möglichen Fehlern umgehen können:

```python
from data_loader import load_data, get_available_targets

try:
    # Versuche, Daten zu laden
    X, y = load_data(target_column="Nicht_Existierende_Spalte")
except KeyError as e:
    print(f"Fehler: Die angegebene Zielspalte existiert nicht. {e}")
except FileNotFoundError as e:
    print(f"Fehler: Die Datendatei wurde nicht gefunden. {e}")
except ValueError as e:
    print(f"Fehler: Problem beim Laden der Daten. {e}")
```

## Tipps und Tricks

1. **Caching**: Das Modul verwendet intern Caching-Mechanismen, um wiederholte Aufrufe zu optimieren. Sie müssen sich darum nicht kümmern.

2. **Cache leeren**: Wenn Sie Speicher freigeben möchten oder sich die Daten geändert haben, können Sie den TF-IDF-Cache leeren:

```python
from data_loader import clear_tfidf_cache

# TF-IDF-Cache leeren
clear_tfidf_cache()
```

3. **Große Datensätze**: Bei großen Datensätzen kann die Erhöhung von `max_features` die Genauigkeit verbessern, aber auch die Verarbeitungszeit erhöhen.

4. **Dateistruktur**: Das Modul unterstützt sowohl eine einzelne große CSV-Datei als auch mehrere geteilte Dateien. Sie können das Skript `split_csv.py` verwenden, um große Dateien aufzuteilen.

## Nächste Schritte

Nachdem Sie die Daten geladen haben, können Sie:

- Ein logistisches Regressionsmodell trainieren (siehe Tutorial `train_logreg_einfach.md`)
- Ein neuronales Netz trainieren (siehe Tutorial `train_nn_einfach.md`)
- Das Hauptmodul verwenden, um verschiedene Modelle zu trainieren (siehe Tutorial `main_einfach.md`)
