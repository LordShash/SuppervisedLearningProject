# Tutorial: Große CSV-Dateien aufteilen

In diesem Tutorial lernen Sie, wie Sie das Skript `split_csv.py` verwenden, um große CSV-Dateien in kleinere Teile aufzuteilen.

## Einführung

Bei der Arbeit mit großen Datensätzen kann es manchmal zu Speicher- oder Verarbeitungsproblemen kommen. Das Skript `split_csv.py` hilft Ihnen, eine große CSV-Datei in kleinere, besser handhabbare Teile aufzuteilen.

## Grundlegende Verwendung

Die Verwendung des Skripts ist sehr einfach. Sie müssen es lediglich von der Kommandozeile aus aufrufen:

```bash
python scripts/split_csv.py
```

Das Skript sucht automatisch nach der Datei `data/Daten_UTF8_Clean_encoded.csv` und teilt sie in zwei gleich große Teile auf:
- `data/Daten_UTF8_Clean_encoded_part1.csv`
- `data/Daten_UTF8_Clean_encoded_part2.csv`

## Ausgabe verstehen

Bei erfolgreicher Ausführung gibt das Skript eine Bestätigungsmeldung aus, die etwa so aussieht:

```
Die Datei wurde erfolgreich in zwei Teile aufgeteilt:
Teil 1: C:\...\data\Daten_UTF8_Clean_encoded_part1.csv (500 Zeilen)
Teil 2: C:\...\data\Daten_UTF8_Clean_encoded_part2.csv (500 Zeilen)
Dateiaufteilung erfolgreich abgeschlossen.
```

## Fehlerbehandlung

Das Skript enthält robuste Fehlerbehandlung. Wenn Probleme auftreten, werden aussagekräftige Fehlermeldungen angezeigt:

- Wenn die Eingabedatei nicht gefunden wird:
  ```
  Fehler: Die Datei '...\data\Daten_UTF8_Clean_encoded.csv' wurde nicht gefunden.
  Dateiaufteilung fehlgeschlagen.
  ```

- Bei anderen Problemen:
  ```
  Fehler beim Aufteilen der Datei: [spezifische Fehlermeldung]
  Dateiaufteilung fehlgeschlagen.
  ```

## Anpassung des Skripts

Wenn Sie das Skript anpassen möchten, um beispielsweise andere Dateien zu teilen oder mehr als zwei Teile zu erstellen, können Sie den Quellcode in `scripts/split_csv.py` bearbeiten:

1. Ändern Sie den Pfad zur Eingabedatei:
   ```python
   original_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Ihre_Datei.csv')
   ```

2. Ändern Sie die Aufteilung, um mehr als zwei Teile zu erstellen:
   ```python
   # Für drei gleich große Teile
   chunk_size = total_rows // 3
   df_part1 = df.iloc[:chunk_size]
   df_part2 = df.iloc[chunk_size:2*chunk_size]
   df_part3 = df.iloc[2*chunk_size:]
   
   # Speichern Sie den dritten Teil
   part3_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Daten_UTF8_Clean_encoded_part3.csv')
   df_part3.to_csv(part3_path, index=False)
   ```

## Integration mit dem Datenlademodul

Das Datenlademodul (`data_loader.py`) ist so konzipiert, dass es automatisch erkennt, ob die Daten in einer einzelnen Datei oder in mehreren Teilen vorliegen. Nach dem Aufteilen der Datei können Sie das Datenlademodul wie gewohnt verwenden:

```python
from data_loader import load_data

# Das Modul erkennt automatisch die geteilten Dateien
X, y = load_data()
```

## Tipps und Tricks

1. **Speichereffizienz**: Das Aufteilen großer Dateien kann helfen, Speicherprobleme zu vermeiden, insbesondere auf Systemen mit begrenztem RAM.

2. **Verarbeitungsgeschwindigkeit**: Kleinere Dateien können in manchen Fällen schneller verarbeitet werden, besonders wenn Ihr System Parallelverarbeitung unterstützt.

3. **Datensicherheit**: Das Skript erstellt neue Dateien und lässt die Originaldatei unverändert. So können Sie immer zur ursprünglichen Datei zurückkehren, falls nötig.

## Nächste Schritte

Nachdem Sie die Datei aufgeteilt haben, können Sie:

- Die Daten mit dem Datenlademodul laden (siehe Tutorial `data_loader_einfach.md`)
- Modelle mit den geteilten Daten trainieren (siehe Tutorials `train_logreg_einfach.md` und `train_nn_einfach.md`)
- Das Hauptmodul verwenden, um verschiedene Modelle zu trainieren (siehe Tutorial `main_einfach.md`)