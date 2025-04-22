# Datenlademodul - Technische Referenz

## Übersicht

Das Modul `data_loader.py` stellt Funktionen zum Laden und Vorverarbeiten von Textdaten für Klassifikationsaufgaben bereit. Es unterstützt das Laden von Daten aus einer einzelnen CSV-Datei oder aus mehreren geteilten Dateien.

## Funktionen

### _load_dataframe()

**Beschreibung**: Private Hilfsfunktion zum Laden der Daten aus CSV-Dateien.

**Inputs**: Keine direkten Eingabeparameter.

**Outputs**:
- `pd.DataFrame`: Ein pandas DataFrame mit den geladenen Daten.

**Verhalten**:
1. Sucht nach geteilten Dateien (`Daten_UTF8_Clean_encoded_part*.csv`).
2. Falls vorhanden, lädt und kombiniert diese.
3. Andernfalls lädt die Originaldatei (`Daten_UTF8_Clean_encoded.csv`).

### load_data()

**Beschreibung**: Lädt die Daten und bereitet sie für das Modelltraining vor.

**Inputs**:
- `target_column` (str, optional): Name der Zielspalte. Standard: 'Fits_Topic_Code'.

**Outputs**:
- `X` (np.ndarray): TF-IDF-Vektoren der Texte.
- `y` (np.ndarray): Zielvariable.

**Verhalten**:
1. Lädt die Daten mit `_load_dataframe()`.
2. Prüft, ob die angegebene Zielspalte existiert.
3. Wandelt die Texte in der Spalte 'BODY' in TF-IDF-Vektoren um.
4. Extrahiert die Zielvariable aus der angegebenen Spalte.

### get_available_targets()

**Beschreibung**: Gibt Informationen über verfügbare Zielspalten zurück.

**Inputs**: Keine direkten Eingabeparameter.

**Outputs**:
- `Dict[str, Any]`: Dictionary mit Informationen zu den Zielspalten.

**Verhalten**:
1. Lädt die Daten mit `_load_dataframe()`.
2. Prüft für jede potenzielle Zielspalte ('Urgency_Code', 'Categorie_Code', 'Fits_Topic_Code'), ob sie existiert.
3. Sammelt für jede vorhandene Spalte die eindeutigen Werte und deren Häufigkeiten.

## Abhängigkeiten

- **Externe Bibliotheken**:
  - `numpy`: Für numerische Operationen
  - `pandas`: Für Datenmanipulation
  - `sklearn.feature_extraction.text.TfidfVectorizer`: Für die Umwandlung von Text in TF-IDF-Vektoren

- **Interne Module**: Keine

## Datentypen

- **Eingabedaten**: CSV-Dateien mit mindestens einer Spalte 'BODY' und einer oder mehreren Zielspalten.
- **Ausgabedaten**: 
  - Features (X): Sparse-Matrix mit TF-IDF-Vektoren
  - Zielvariable (y): NumPy-Array mit Klassenlabels

## Fehlerbehandlung

Das Modul verwendet `sys.exit()` mit aussagekräftigen Fehlermeldungen für verschiedene Fehlersituationen:
- Datei nicht gefunden
- Fehler beim Laden der Daten
- Fehlende Zielspalte
- Allgemeine Verarbeitungsfehler