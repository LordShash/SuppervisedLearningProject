# Anleitung zur Verwendung des Textklassifikationsprojekts

Diese Anleitung erklärt, wie das Textklassifikationsprojekt verwendet werden kann, um verschiedene Modelle für die Klassifikation von Texten zu trainieren und zu evaluieren.

## Projektstruktur

Das Projekt ist wie folgt strukturiert:

```
mein_projekt/
├── data/
│   ├── Daten_UTF8_Clean_encoded.csv          # Hauptdatensatz mit Texten und Klassifikationen
│   ├── Daten_UTF8_Clean_encoded_part1.csv    # Geteilter Datensatz (Teil 1)
│   └── Daten_UTF8_Clean_encoded_part2.csv    # Geteilter Datensatz (Teil 2)
├── docs/
│   ├── user_guides/                          # Benutzeranleitungen
│   │   └── anleitung.md                      # Diese Anleitung
│   ├── technical/                            # Technische Dokumentation
│   │   ├── data_loader_tech.md               # Technische Details zu data_loader.py
│   │   ├── main_tech.md                      # Technische Details zu main.py
│   │   ├── split_csv_tech.md                 # Technische Details zu split_csv.py
│   │   ├── train_logreg_tech.md              # Technische Details zu train_logreg.py
│   │   └── train_nn_tech.md                  # Technische Details zu train_nn.py
│   └── tutorials/                            # Einfache Anleitungen
│       ├── data_loader_einfach.md            # Einfache Erklärung zu data_loader.py
│       ├── main_einfach.md                   # Einfache Erklärung zu main.py
│       ├── split_csv_einfach.md              # Einfache Erklärung zu split_csv.py
│       ├── train_logreg_einfach.md           # Einfache Erklärung zu train_logreg.py
│       └── train_nn_einfach.md               # Einfache Erklärung zu train_nn.py
├── models/                                   # Verzeichnis für trainierte Modelle (wird automatisch erstellt)
├── scripts/
│   └── split_csv.py                          # Skript zum Teilen der CSV-Datei
├── src/
│   ├── data_loader.py                        # Modul zum Laden und Verarbeiten der Daten
│   ├── main.py                               # Hauptmodul zur Orchestrierung des Trainings
│   ├── train_logreg.py                       # Modul zum Training logistischer Regressionsmodelle
│   └── train_nn.py                           # Modul zum Training neuronaler Netze
├── tests/
│   └── test_main.py                          # Tests für das Hauptmodul
└── requirements.txt                          # Abhängigkeiten des Projekts
```

## Installation

Um das Projekt zu verwenden, müssen zunächst die Abhängigkeiten installiert werden:

```bash
pip install -r requirements.txt
```

## Verwendung

Das Projekt kann über das Hauptmodul `main.py` gesteuert werden. Dieses Modul bietet verschiedene Optionen zum Training der Modelle.

### Grundlegende Verwendung

Um alle Modelle mit der Standardzielvariable zu trainieren:

```bash
python src/main.py
```

### Optionen

Das Hauptmodul bietet folgende Optionen:

- `--model`: Zu trainierendes Modell (logreg, nn oder all)
  - `logreg`: Logistische Regression
  - `nn`: Neuronales Netz
  - `all`: Beide Modelle (Standard)

- `--target`: Zielvariable für das Training
  - `Fits_Topic_Code`: Ob das Thema passt (0 oder 1) (Standard)
  - `Urgency_Code`: Dringlichkeitsstufe (1, 2 oder 3)
  - `Categorie_Code`: Kategorie (1, 2, 3 oder 4)
  - `all`: Alle Zielvariablen

- `--epochs`: Anzahl der Trainingsepochen für neuronale Netze (Standard: 30)

### Beispiele

Nur logistische Regression mit der Zielvariable "Urgency_Code" trainieren:

```bash
python src/main.py --model logreg --target Urgency_Code
```

Nur neuronales Netz mit der Zielvariable "Categorie_Code" und 50 Epochen trainieren:

```bash
python src/main.py --model nn --target Categorie_Code --epochs 50
```

Alle Modelle mit allen Zielvariablen trainieren:

```bash
python src/main.py --model all --target all
```

## Funktionsweise

### Datenverarbeitung

Der Datensatz enthält Texte in der Spalte "BODY" und verschiedene Klassifikationen wie "Fits_Topic_Code", "Urgency_Code" und "Categorie_Code". Die Texte werden mittels TF-IDF-Vektorisierung in numerische Features umgewandelt, die dann für das Training der Modelle verwendet werden.

### Modelle

#### Logistische Regression

Die logistische Regression ist ein einfaches, aber effektives Modell für Textklassifikation. Es wird mit scikit-learn implementiert und verwendet eine Pipeline mit Standardskalierung und logistischer Regression.

#### Neuronales Netz

Das neuronale Netz ist ein Feed-Forward-Netzwerk mit zwei Hidden-Layern und Dropout zur Vermeidung von Overfitting. Es wird mit PyTorch implementiert und kann sowohl für binäre als auch für Mehrklassenklassifikation verwendet werden.

### Evaluation

Die Modelle werden anhand verschiedener Metriken evaluiert:
- Genauigkeit (Accuracy): Anteil der korrekt klassifizierten Beispiele
- Precision: Anteil der korrekten positiven Vorhersagen an allen positiven Vorhersagen
- Recall: Anteil der korrekten positiven Vorhersagen an allen tatsächlich positiven Beispielen
- F1-Score: Harmonisches Mittel aus Precision und Recall
- Klassifikationsbericht: Detaillierte Aufschlüsselung von Precision, Recall und F1-Score für jede Klasse
- Konfusionsmatrix: Tabelle, die zeigt, wie viele Beispiele jeder tatsächlichen Klasse als welche Klasse vorhergesagt wurden

## Dokumentation

Die Dokumentation des Projekts ist in drei Bereiche unterteilt:

1. **Benutzeranleitungen** (`docs/user_guides/`): Enthält diese Anleitung, die einen Überblick über das Projekt und seine Verwendung gibt.

2. **Technische Dokumentation** (`docs/technical/`): Enthält detaillierte technische Beschreibungen der einzelnen Komponenten:
   - `data_loader_tech.md`: Technische Details zum Datenlademodul
   - `main_tech.md`: Technische Details zum Hauptmodul
   - `split_csv_tech.md`: Technische Details zum CSV-Teilungsskript
   - `train_logreg_tech.md`: Technische Details zum logistischen Regressionsmodell
   - `train_nn_tech.md`: Technische Details zum neuronalen Netzwerk

3. **Tutorials** (`docs/tutorials/`): Enthält einfache, alltagstaugliche Erklärungen der einzelnen Komponenten:
   - `data_loader_einfach.md`: Einfache Erklärung des Datenladeprozesses
   - `main_einfach.md`: Einfache Erklärung des Hauptmoduls
   - `split_csv_einfach.md`: Einfache Erklärung des CSV-Teilungsskripts
   - `train_logreg_einfach.md`: Einfache Erklärung des logistischen Regressionsmodells
   - `train_nn_einfach.md`: Einfache Erklärung des neuronalen Netzwerks

## Erweiterungen

Das Projekt kann auf verschiedene Weise erweitert werden:

- Hinzufügen weiterer Modelle (z.B. Random Forest, SVM)
- Verbessern der Textverarbeitung (z.B. Lemmatisierung, Entfernung von Stoppwörtern)
- Implementierung von Kreuzvalidierung zur robusteren Evaluation
- Hyperparameter-Optimierung zur Verbesserung der Modellleistung
