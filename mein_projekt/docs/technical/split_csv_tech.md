# CSV-Teilungsskript - Technische Referenz

## Übersicht

Das Skript `split_csv.py` ist ein Hilfswerkzeug zur Aufteilung einer großen CSV-Datei in zwei kleinere Teildateien. Es verwendet pandas für die Datenmanipulation und teilt die Datei in der Mitte, basierend auf der Anzahl der Zeilen.

## Funktionen

### split_csv_file()

**Beschreibung**: Teilt eine CSV-Datei in zwei gleich große Teile.

**Inputs**: Keine direkten Eingabeparameter.

**Outputs**:
- `bool`: True bei erfolgreicher Ausführung, False bei Fehlern.

**Verhalten**:
1. Ermittelt den Pfad zur Originaldatei (`Daten_UTF8_Clean_encoded.csv`).
2. Prüft, ob die Datei existiert.
3. Lädt die Datei mit pandas.
4. Berechnet den Mittelpunkt basierend auf der Zeilenanzahl.
5. Teilt das DataFrame in zwei Teile.
6. Speichert die Teile als separate CSV-Dateien.
7. Gibt eine Erfolgsmeldung mit Informationen zu den erstellten Dateien aus.

## Abhängigkeiten

- **Externe Bibliotheken**:
  - `pandas`: Für das Laden, Teilen und Speichern der CSV-Daten.
  - `os`: Für Pfadoperationen und Dateisystemzugriffe.

- **Interne Module**: Keine

## Datentypen

- **Eingabedaten**: Eine CSV-Datei (`Daten_UTF8_Clean_encoded.csv`).
- **Ausgabedaten**: Zwei CSV-Dateien:
  - `Daten_UTF8_Clean_encoded_part1.csv`: Erste Hälfte der Zeilen.
  - `Daten_UTF8_Clean_encoded_part2.csv`: Zweite Hälfte der Zeilen.

## Fehlerbehandlung

Das Skript implementiert folgende Fehlerbehandlung:
- Prüfung, ob die Originaldatei existiert.
- Try-Except-Block für das Laden und Verarbeiten der Datei.
- Rückgabe von Boolean-Werten (True/False) zur Anzeige des Erfolgs oder Misserfolgs.
- Ausgabe von Fehlermeldungen bei Problemen.

## Verwendung

Das Skript kann direkt ausgeführt werden:
```
python scripts/split_csv.py
```

Es kann auch als Modul importiert und die Funktion `split_csv_file()` aufgerufen werden:
```python
from scripts.split_csv import split_csv_file
success = split_csv_file()
```

## Hinweise

- Die Aufteilung erfolgt genau in der Mitte der Datei, unabhängig vom Inhalt.
- Die Originaldatei bleibt unverändert.
- Die Teildateien werden im selben Verzeichnis wie die Originaldatei gespeichert.
- Das Datenlademodul (`data_loader.py`) kann automatisch erkennen, ob die geteilten Dateien vorhanden sind, und sie bei Bedarf verwenden.