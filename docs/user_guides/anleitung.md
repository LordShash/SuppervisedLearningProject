# Benutzeranleitung: Textklassifikationsprojekt

Diese Anleitung bietet einen Überblick über das Textklassifikationsprojekt und erklärt, wie Sie die verschiedenen Komponenten verwenden können.

## Einführung

Das Textklassifikationsprojekt ist eine Anwendung zur automatischen Klassifizierung von Texten in verschiedene Kategorien. Es verwendet Techniken des maschinellen Lernens, um Muster in Textdaten zu erkennen und Vorhersagen zu treffen.

## Systemanforderungen

- Python 3.7 oder höher
- Ausreichend RAM für die Verarbeitung der Datensätze (mindestens 4 GB empfohlen)
- Für beschleunigtes Training: NVIDIA GPU mit CUDA-Unterstützung (optional)

## Installation

1. Klonen Sie das Repository oder laden Sie es herunter
2. Installieren Sie die erforderlichen Abhängigkeiten:

```bash
pip install -r requirements.txt
```

## Projektstruktur

Das Projekt ist wie folgt strukturiert:

```
SuppervisedLearningProject/
├── data/                  # Datendateien
├── docs/                  # Dokumentation
│   ├── technical/         # Technische Dokumentation
│   ├── tutorials/         # Tutorials
│   └── user_guides/       # Benutzeranleitungen
├── models/                # Gespeicherte Modelle
├── scripts/               # Hilfsskripte
├── src/                   # Quellcode
│   ├── data_loader.py     # Datenlademodul
│   ├── gui.py             # Grafische Benutzeroberfläche
│   ├── main.py            # Hauptmodul
│   ├── model_loader.py    # Modelllademodul
│   ├── train_logreg.py    # Logistische Regression
│   └── train_nn.py        # Neuronales Netz
├── tests/                 # Tests
├── run_gui.py             # Startskript für die GUI
└── README.md              # Projektübersicht
```

## Schnellstart

### Grafische Benutzeroberfläche starten

Die einfachste Möglichkeit, mit dem Projekt zu arbeiten, ist die Verwendung der grafischen Benutzeroberfläche:

```bash
python run_gui.py
```

### Modelle über die Kommandozeile trainieren

Sie können Modelle auch direkt über die Kommandozeile trainieren:

```bash
# Beide Modelltypen mit Standardparametern trainieren
python src/main.py

# Nur logistische Regression trainieren
python src/main.py --model logreg

# Nur neuronales Netz trainieren
python src/main.py --model nn
```

## Daten

Das Projekt erwartet Textdaten im CSV-Format mit mindestens zwei Spalten:
- `BODY`: Der zu klassifizierende Text
- Eine oder mehrere Zielspalten (z.B. `Fits_Topic_Code`, `Urgency_Code`)

Die Daten sollten im Verzeichnis `data/` gespeichert sein. Standardmäßig wird die Datei `data/Daten_UTF8_Clean_encoded.csv` verwendet.

Bei großen Datensätzen können Sie das Skript `scripts/split_csv.py` verwenden, um die Datei in kleinere Teile aufzuteilen:

```bash
python scripts/split_csv.py
```

## Modelle trainieren

### Logistische Regression

Die logistische Regression ist ein einfaches, aber effektives Modell für Textklassifikation:

```bash
python src/main.py --model logreg --target Fits_Topic_Code
```

### Neuronales Netz

Neuronale Netze können komplexere Muster erkennen, benötigen aber möglicherweise mehr Rechenleistung:

```bash
python src/main.py --model nn --target Fits_Topic_Code
```

## Parameter anpassen

Sie können verschiedene Parameter anpassen, um die Leistung der Modelle zu optimieren:

```bash
python src/main.py --model logreg --target Urgency_Code --max-features 2000 --test-size 0.3 --max-iter 2000 --C 0.5 --solver newton-cg
```

```bash
python src/main.py --model nn --target Urgency_Code --max-features 2000 --test-size 0.3 --epochs 50 --patience 10
```

## Ergebnisse interpretieren

Nach dem Training werden verschiedene Metriken ausgegeben:

- **Accuracy**: Der Anteil der korrekt klassifizierten Beispiele
- **Precision**: Der Anteil der korrekt positiven Vorhersagen an allen positiven Vorhersagen
- **Recall**: Der Anteil der korrekt positiven Vorhersagen an allen tatsächlich positiven Beispielen
- **F1-Score**: Das harmonische Mittel aus Precision und Recall

Zusätzlich werden ein detaillierter Klassifikationsbericht und eine Konfusionsmatrix ausgegeben.

## Gespeicherte Modelle verwenden

Die trainierten Modelle werden im Verzeichnis `models/` gespeichert:

- Logistische Regression: `models/logreg_{target_column}_model.pkl`
- Neuronales Netz: `models/nn_{target_column}_model.pt`

Das Projekt bietet ein spezielles Modul `model_loader.py` zum einfachen Laden dieser Modelle:

```python
from src.model_loader import load_model

# Logistische Regression laden
logreg_model = load_model(model_type="logreg", target_column="Fits_Topic_Code")

# Neuronales Netz laden
nn_model = load_model(model_type="nn", target_column="Urgency_Code")

# Alternativ: Direkter Pfad
custom_model = load_model(model_path="models/mein_eigenes_modell.pkl")
```

Das Modul bietet auch eine erweiterte Funktion mit Statusinformationen:

```python
from src.model_loader import load_model_with_info

result = load_model_with_info(model_type="logreg", target_column="Fits_Topic_Code")
if result['success']:
    model = result['model']
    print(f"Modell erfolgreich geladen: {result['message']}")
else:
    print(f"Fehler beim Laden: {result['message']}")
```

Ausführliche Beispiele und weitere Informationen finden Sie im Tutorial `docs/tutorials/model_loader_einfach.md`.

## Fehlerbehebung

### Speicherprobleme

Bei großen Datensätzen können Speicherprobleme auftreten. Versuchen Sie in diesem Fall:

1. Die Datei mit `scripts/split_csv.py` aufzuteilen
2. Den Parameter `max_features` zu reduzieren
3. Einen kleineren Wert für `test_size` zu verwenden

### Trainingszeit

Das Training kann je nach Datensatzgröße und Modellkomplexität einige Zeit in Anspruch nehmen. Für schnellere Ergebnisse:

1. Verwenden Sie eine GPU, wenn verfügbar
2. Reduzieren Sie den Parameter `max_features`
3. Verwenden Sie für erste Tests die logistische Regression statt des neuronalen Netzes

## Weiterführende Dokumentation

- Technische Dokumentation: Siehe Verzeichnis `docs/technical/`
- Tutorials: Siehe Verzeichnis `docs/tutorials/`
- GUI-Anleitung: Siehe `docs/user_guides/gui_anleitung.md`
