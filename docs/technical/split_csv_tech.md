# Technische Dokumentation: CSV-Teilungsskript

## Übersicht
Das Skript `split_csv.py` dient dazu, eine große CSV-Datei in kleinere Teile aufzuteilen, um die Verarbeitung zu erleichtern. Dies ist besonders nützlich bei großen Datensätzen, die möglicherweise zu Speicher- oder Verarbeitungsproblemen führen könnten.

## Hauptfunktionen

### `split_csv_file()`
Teilt die große CSV-Datei in kleinere Teile.

- **Rückgabewert**: `bool` - True bei Erfolg, False bei Fehler
- **Funktionsweise**:
  1. Lädt die ursprüngliche CSV-Datei mit pandas
  2. Teilt das DataFrame in zwei gleich große Teile
  3. Speichert die Teile in separate CSV-Dateien

## Technische Details

### Dateipfade
Das Skript verwendet relative Pfade, um die Dateien zu finden und zu speichern:
- Eingabedatei: `data/Daten_UTF8_Clean_encoded.csv`
- Ausgabedateien:
  - `data/Daten_UTF8_Clean_encoded_part1.csv`
  - `data/Daten_UTF8_Clean_encoded_part2.csv`

### Fehlerbehandlung
Das Skript implementiert eine robuste Fehlerbehandlung:
- Überprüfung der Existenz der Eingabedatei
- Try-Except-Block für die gesamte Verarbeitungslogik
- Detaillierte Fehlermeldungen bei Problemen

### Leistungsoptimierung
- Verwendet pandas für effiziente Datenverarbeitung
- Teilt die Daten in-memory, bevor sie in Dateien geschrieben werden
- Behält die Spaltenstruktur und Datentypen bei

## Verwendung
Das Skript kann direkt von der Kommandozeile ausgeführt werden:
```bash
python scripts/split_csv.py
```

Bei erfolgreicher Ausführung wird eine Bestätigungsmeldung mit Details zu den erstellten Dateien ausgegeben.

## Abhängigkeiten
- `pandas`: Für Datenmanipulation und -analyse
- Standard-Python-Bibliotheken: `os`