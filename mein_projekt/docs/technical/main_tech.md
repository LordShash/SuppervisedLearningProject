# Hauptmodul - Technische Referenz

## Übersicht

Das Modul `main.py` dient als Einstiegspunkt für das Projekt und orchestriert das Training verschiedener Modelle für die Textklassifikation. Es bietet eine Kommandozeilenschnittstelle zur Konfiguration des Trainings und ruft die entsprechenden Funktionen aus anderen Modulen auf.

## Funktionen

### parse_arguments()

**Beschreibung**: Parst die Kommandozeilenargumente für die Konfiguration des Trainings.

**Inputs**: Keine direkten Eingabeparameter.

**Outputs**:
- `argparse.Namespace`: Objekt mit den geparsten Argumenten.

**Verhalten**:
1. Definiert einen ArgumentParser mit Beschreibung.
2. Fügt Argumente für Modelltyp, Zielvariable und Epochenanzahl hinzu.
3. Parst die Kommandozeilenargumente und gibt sie zurück.

### train_models()

**Beschreibung**: Trainiert die angegebenen Modelle mit der angegebenen Zielvariable.

**Inputs**:
- `model_type` (str): Typ des zu trainierenden Modells ("logreg", "nn" oder "all").
- `target_column` (str): Name der Zielspalte.
- `epochs` (int, optional): Anzahl der Trainingsepochen für neuronale Netze. Standard: 30.

**Outputs**:
- `Dict[str, Any]`: Dictionary mit Trainingsergebnissen.

**Verhalten**:
1. Initialisiert ein leeres Dictionary für die Ergebnisse.
2. Wenn "logreg" oder "all" ausgewählt ist, trainiert das logistische Regressionsmodell.
3. Wenn "nn" oder "all" ausgewählt ist, trainiert das neuronale Netzwerk.
4. Gibt die Ergebnisse zurück.

### main()

**Beschreibung**: Hauptfunktion des Programms.

**Inputs**: Keine direkten Eingabeparameter.

**Outputs**: Keine direkten Rückgabewerte.

**Verhalten**:
1. Parst die Kommandozeilenargumente.
2. Zeigt Informationen über verfügbare Zielvariablen an.
3. Trainiert die Modelle basierend auf den Argumenten.
4. Gibt eine Abschlussmeldung aus.

## Abhängigkeiten

- **Externe Bibliotheken**:
  - `os`: Für Pfadoperationen.
  - `sys`: Für Systemfunktionen.
  - `argparse`: Für das Parsen von Kommandozeilenargumenten.
  - `typing`: Für Typannotationen.

- **Interne Module**:
  - `data_loader`: Für das Laden der Daten und Informationen über Zielvariablen.
  - `train_logreg`: Für das Training des logistischen Regressionsmodells.
  - `train_nn`: Für das Training des neuronalen Netzwerks.

## Kommandozeilenargumente

- `--model`: Zu trainierendes Modell (logreg, nn oder all). Standard: "all".
- `--target`: Zielvariable für das Training (Fits_Topic_Code, Urgency_Code, Categorie_Code oder all). Standard: "Fits_Topic_Code".
- `--epochs`: Anzahl der Trainingsepochen für neuronale Netze. Standard: 30.

## Beispielverwendung

```bash
# Alle Modelle mit Standardzielvariable trainieren
python src/main.py

# Nur logistische Regression mit Urgency_Code trainieren
python src/main.py --model logreg --target Urgency_Code

# Nur neuronales Netz mit 50 Epochen trainieren
python src/main.py --model nn --epochs 50

# Alle Modelle mit allen Zielvariablen trainieren
python src/main.py --model all --target all
```

## Fehlerbehandlung

Das Modul verwendet einen globalen Try-Except-Block in der `if __name__ == "__main__"` Sektion, um Fehler abzufangen und eine Fehlermeldung auszugeben. Spezifische Fehlerbehandlung für das Daten-Loading und Modelltraining wird in den entsprechenden Modulen durchgeführt.