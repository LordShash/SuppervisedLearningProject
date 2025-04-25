# Supervised Learning Projekt

Dieses Projekt implementiert verschiedene Modelle für die Textklassifikation mit Supervised Learning.

## Projektstruktur

Die Projektstruktur wurde nach Best-Practice-Prinzipien organisiert:

- `suppervisedlearningproject/`: Hauptpaket des Projekts
  - `core/`: Kernfunktionalität
    - `data_loader.py`: Funktionen zum Laden der Daten
  - `models/`: Modelltraining
    - `train_logreg.py`: Training von logistischen Regressionsmodellen
    - `train_nn.py`: Training von neuronalen Netzwerken
  - `ui/`: Benutzeroberflächen
    - `gui.py`: Basis der modernen GUI (PyQt5)
    - `gui_tabs.py`: Tab-Implementierungen für die moderne GUI
    - `gui_complete.py`: Vollständige moderne GUI (PyQt5)
  - `utils/`: Hilfsfunktionen
    - `config.py`: Konfigurationseinstellungen
  - `icons/`: Icons für die GUI
  - `main.py`: Hauptmodul zur Orchestrierung des Trainings
- `run_gui.py`: Startskript für die moderne GUI (PyQt5)
- `data/`: Datendateien
  - `Daten_UTF8_Clean_encoded.csv`: Hauptdatendatei
  - `Daten_UTF8_Clean_encoded_part1.csv`, `Daten_UTF8_Clean_encoded_part2.csv`: Geteilte Datendateien
- `models/`: Gespeicherte Modelle (werden während des Trainings generiert)
  - Beispiel: `logreg_[target_column]_model.pkl`: Trainiertes logistisches Regressionsmodell
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

5. **Moderne grafische Benutzeroberfläche hinzugefügt**:
   - Benutzerfreundliche GUI mit PyQt5 implementiert
   - Modernes Design mit intuitivem Look-and-Feel
   - Formular zur Konfiguration von Modellparametern
   - Verbesserte Visualisierung von Trainingsergebnissen (Metriken, Konfusionsmatrix)
   - Dokumentation zur Verwendung der GUI in `docs/user_guides/gui_anleitung.md`
   - Einfaches Startskript `run_gui.py` für den schnellen Einstieg

## Verwendung der GUI

Um die moderne GUI zu starten, führen Sie das folgende Kommando im Hauptverzeichnis des Projekts aus:

```bash
python run_gui.py
```

Die moderne GUI bietet folgende Funktionen:

- Auswahl des Modelltyps (logistische Regression, neuronales Netz oder beide)
- Konfiguration der Hyperparameter
- Auswahl der Zielvariable
- Visualisierung der Trainingsergebnisse
- Anzeige von Metriken, Konfusionsmatrix und Klassifikationsbericht

### Besonderheiten der modernen GUI

Die moderne GUI zeichnet sich durch folgende Merkmale aus:

- Flaches, modernes Design mit ansprechenden Farben und Schattierungen
- Verbesserte Benutzerfreundlichkeit durch intuitivere Bedienelemente
- Responsives Layout, das sich an verschiedene Bildschirmgrößen anpasst
- Verbesserte Visualisierungen mit interaktiven Elementen

### PyQt5-Abhängigkeit

Für die Verwendung der GUI wird PyQt5 benötigt. Das Startskript prüft automatisch, ob PyQt5 installiert ist und bietet folgende Optionen:

1. **Automatische Installation**: Sie können PyQt5 direkt aus dem Skript heraus installieren, wenn Sie dazu aufgefordert werden.
2. **Manuelle Installation**: Sie können PyQt5 auch manuell mit folgendem Befehl installieren:
   ```bash
   pip install PyQt5>=5.15.0
   ```

Eine detaillierte Anleitung zur Verwendung der GUI finden Sie in der Datei `docs/user_guides/gui_anleitung.md`.
