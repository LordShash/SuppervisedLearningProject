# Technische Dokumentation: Datenlademodul

## Übersicht
Das Modul `data_loader.py` ist verantwortlich für das Laden und die Vorverarbeitung der Daten für das Textklassifikationsprojekt. Es stellt Funktionen bereit, um Daten aus CSV-Dateien zu laden, zu verarbeiten und für das Training von Modellen vorzubereiten.

## Hauptfunktionen

### `_load_dataframe()`
Eine interne Hilfsfunktion, die die Daten aus den CSV-Dateien lädt und ein DataFrame zurückgibt.

- **Rückgabewert**: `pd.DataFrame` - Geladenes DataFrame mit allen Daten
- **Exceptions**:
  - `FileNotFoundError`: Wenn keine Datendateien gefunden wurden
  - `ValueError`: Wenn Probleme beim Laden der Daten auftreten
- **Caching**: Verwendet `functools.lru_cache` zur Optimierung wiederholter Aufrufe

### `clear_tfidf_cache()`
Leert den TF-IDF-Cache vollständig. Nützlich, wenn sich die Daten geändert haben oder Speicher freigegeben werden soll.

- **Rückgabewert**: Keiner
- **Seiteneffekte**: Leert die globalen Cache-Variablen `_tfidf_cache` und `_tfidf_cache_access_times`

### `load_data(target_column='Fits_Topic_Code', max_features=1000)`
Lädt die Daten aus den CSV-Dateien und gibt Features und Zielvariable zurück.

- **Parameter**:
  - `target_column`: Name der Zielspalte (Standard: 'Fits_Topic_Code')
  - `max_features`: Maximale Anzahl der Features für TF-IDF (Standard: 1000)
- **Rückgabewert**: `Tuple[np.ndarray, np.ndarray]` - Features (X) und Zielvariable (y)
- **Exceptions**:
  - `FileNotFoundError`: Wenn keine Datendateien gefunden wurden
  - `ValueError`: Wenn Probleme beim Laden oder Verarbeiten der Daten auftreten
  - `KeyError`: Wenn die angegebene Zielspalte nicht im DataFrame vorhanden ist
- **Caching**: Verwendet einen internen Cache für die TF-IDF-Vektorisierung

### `get_available_targets()`
Gibt ein Dictionary mit Informationen über verfügbare Zielspalten zurück.

- **Rückgabewert**: `Dict[str, Dict[str, Any]]` - Dictionary mit Informationen über verfügbare Zielspalten
- **Exceptions**:
  - `FileNotFoundError`: Wenn keine Datendateien gefunden wurden
  - `ValueError`: Wenn Probleme beim Laden der Daten auftreten

## Technische Details

### Datenquellen
Das Modul unterstützt zwei Arten von Datenquellen:
1. Eine einzelne CSV-Datei: `data/Daten_UTF8_Clean_encoded.csv`
2. Geteilte CSV-Dateien: `data/Daten_UTF8_Clean_encoded_part*.csv`

### Featuregenerierung
Für die Featuregenerierung wird die TF-IDF-Vektorisierung (Term Frequency-Inverse Document Frequency) verwendet, implementiert durch `sklearn.feature_extraction.text.TfidfVectorizer`. Diese Methode wandelt Textdaten in numerische Features um, die für maschinelles Lernen geeignet sind.

### Caching-Strategie
Das Modul verwendet zwei Caching-Mechanismen:
1. LRU-Cache für das Laden des DataFrames
2. Einen manuell implementierten Cache für die TF-IDF-Matrizen, begrenzt auf die drei zuletzt verwendeten Konfigurationen

## Abhängigkeiten
- `pandas`: Für Datenmanipulation und -analyse
- `numpy`: Für numerische Operationen
- `scikit-learn`: Für die TF-IDF-Vektorisierung
- Standard-Python-Bibliotheken: `os`, `sys`, `glob`, `functools`, `typing`
