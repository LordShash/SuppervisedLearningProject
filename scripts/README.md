# Hilfsskripte

Dieses Verzeichnis enthält Hilfsskripte für das Projekt.

## split_csv.py

Dieses Skript teilt eine große CSV-Datei in kleinere Teile auf, um die Verarbeitung zu erleichtern.

### Verwendung

```bash
python split_csv.py
```

Das Skript liest die Datei `Daten_UTF8_Clean_encoded.csv` aus dem `data`-Verzeichnis und teilt sie in zwei Teile auf:
- `Daten_UTF8_Clean_encoded_part1.csv`
- `Daten_UTF8_Clean_encoded_part2.csv`

Diese Dateien werden ebenfalls im `data`-Verzeichnis gespeichert.