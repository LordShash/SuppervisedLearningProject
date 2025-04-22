# Supervised Learning Projekt

Dieses Projekt implementiert verschiedene Modelle für die Textklassifikation mit Supervised Learning.

## Projektstruktur

Die Projektstruktur wurde nach Best-Practice-Prinzipien organisiert:

- `src/`: Quellcode des Projekts
  - `data_loader.py`: Funktionen zum Laden der Daten
  - `main.py`: Hauptmodul zur Orchestrierung des Trainings
  - `train_logreg.py`: Training von logistischen Regressionsmodellen
  - `train_nn.py`: Training von neuronalen Netzwerken
  - `gui.py`: Grafische Benutzeroberfläche für die Anwendung
- `run_gui.py`: Startskript für die grafische Benutzeroberfläche
- `data/`: Datendateien
  - `Daten_UTF8_Clean_encoded.csv`: Hauptdatendatei
  - `Daten_UTF8_Clean_encoded_part1.csv`, `Daten_UTF8_Clean_encoded_part2.csv`: Geteilte Datendateien
- `models/`: Gespeicherte Modelle
  - `logreg_Fits_Topic_Code_model.pkl`: Trainiertes logistisches Regressionsmodell
- `docs/`: Dokumentation
  - `user_guides/`: Benutzeranleitungen
    - `anleitung.md`: Allgemeine Anleitung
  - `technical/`: Technische Dokumentation
    - Für jede Python-Datei: `<datei>_tech.md`
  - `tutorials/`: Einfache Anleitungen
    - Für jede Python-Datei: `<datei>_einfach.md`
- `scripts/`: Hilfsskripte
  - `split_csv.py`: Skript zum Teilen der CSV-Datei
- `tests/`: Testskripte
  - `test_main.py`: Test für das Hauptmodul

## Änderungen

Folgende Änderungen wurden an der ursprünglichen Codebasis vorgenommen:

1. **Redundanzmanagement**:
   - Gemeinsame Funktionalität zum Laden von Daten in `data_loader.py` extrahiert
   - Duplizierter Code entfernt

2. **Ordnerstruktur optimiert**:
   - Testdateien in einen eigenen `tests/`-Ordner verschoben
   - README.md zur Dokumentation der Projektstruktur erstellt

3. **Kommentierung verbessert**:
   - Standardisierte deutsche Kommentare hinzugefügt
   - Typannotationen und Docstrings ergänzt

4. **Dokumentation erstellt**:
   - Für jede Python-Datei wurden zwei Markdown-Dateien erstellt:
     - `<datei>_einfach.md`: Alltagstaugliche Erklärung für Nicht-Coder
     - `<datei>_tech.md`: Technische Referenz
   - Dokumentation in semantische Unterordner strukturiert:
     - `user_guides/`: Benutzeranleitungen für Endnutzer
     - `technical/`: Technische Dokumentation für Entwickler
     - `tutorials/`: Einfache Anleitungen für Nicht-Coder

5. **Grafische Benutzeroberfläche hinzugefügt**:
   - Benutzerfreundliche GUI mit tkinter implementiert
   - Formular zur Konfiguration von Modellparametern
   - Visualisierung von Trainingsergebnissen (Metriken, Konfusionsmatrix)
   - Dokumentation zur Verwendung der GUI in `docs/user_guides/gui_anleitung.md`
   - Einfaches Startskript `run_gui.py` für den schnellen Einstieg

## Verwendung der GUI

Um die grafische Benutzeroberfläche zu starten, führen Sie das folgende Kommando im Hauptverzeichnis des Projekts aus:

```bash
python run_gui.py
```

Die GUI bietet folgende Funktionen:

- Auswahl des Modelltyps (logistische Regression, neuronales Netz oder beide)
- Konfiguration der Hyperparameter
- Auswahl der Zielvariable
- Visualisierung der Trainingsergebnisse
- Anzeige von Metriken, Konfusionsmatrix und Klassifikationsbericht

Eine detaillierte Anleitung zur Verwendung der GUI finden Sie in der Datei `docs/user_guides/gui_anleitung.md`.