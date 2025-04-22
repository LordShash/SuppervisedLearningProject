# CSV-Teilungsskript - Einfache Erklärung

## Zweck

Das CSV-Teilungsskript (`split_csv.py`) ist ein Hilfswerkzeug, das eine große Datendatei in kleinere Teile aufteilt. Dies ist nützlich, wenn die ursprüngliche Datei zu groß ist, um sie effizient zu verarbeiten oder wenn Sie die Daten auf mehrere Systeme verteilen möchten.

## Ablauf

1. **Datei finden**: 
   - Das Skript sucht nach der Hauptdatendatei (`Daten_UTF8_Clean_encoded.csv`) im `data`-Verzeichnis.

2. **Datei teilen**:
   - Die Datei wird in der Mitte geteilt, sodass zwei etwa gleich große Teile entstehen.
   - Der erste Teil enthält die erste Hälfte der Zeilen.
   - Der zweite Teil enthält die zweite Hälfte der Zeilen.

3. **Neue Dateien speichern**:
   - Die beiden Teile werden als separate Dateien gespeichert:
     - `Daten_UTF8_Clean_encoded_part1.csv`
     - `Daten_UTF8_Clean_encoded_part2.csv`
   - Diese Dateien werden ebenfalls im `data`-Verzeichnis abgelegt.

## Beispiel-Workflow

Stellen Sie sich vor, Sie haben eine große CSV-Datei mit 10.000 Zeilen:

1. Sie führen das Skript aus: `python scripts/split_csv.py`
2. Das Skript findet die Datei und teilt sie in zwei Teile:
   - Teil 1 enthält die Zeilen 1-5.000
   - Teil 2 enthält die Zeilen 5.001-10.000
3. Das Skript speichert die beiden Teile als separate Dateien.
4. Sie erhalten eine Bestätigung, dass die Teilung erfolgreich war, mit Informationen darüber, wie viele Zeilen jeder Teil enthält.

Jetzt können Sie mit den kleineren Dateien arbeiten, was in manchen Fällen schneller und effizienter sein kann. Das Datenlademodul (`data_loader.py`) kann automatisch erkennen, ob die geteilten Dateien vorhanden sind, und sie bei Bedarf wieder zusammenführen.