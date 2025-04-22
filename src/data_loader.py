#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul zum Laden und Verarbeiten der Daten für das Projekt.

Dieses Modul stellt Funktionen bereit, um Daten aus CSV-Dateien zu laden,
zu verarbeiten und für das Training von Modellen vorzubereiten.
"""

import os
import sys
import glob
import functools
import logging
from typing import Tuple, Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ausgabe in die Konsole
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'data_loader.log'), 
                           mode='a', encoding='utf-8', delay=True)  # Ausgabe in eine Datei
    ]
)
logger = logging.getLogger(__name__)

# Stelle sicher, dass das Logs-Verzeichnis existiert
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)


@functools.lru_cache(maxsize=1)
def _load_dataframe() -> pd.DataFrame:
    """
    Lädt die Daten aus den CSV-Dateien und gibt ein DataFrame zurück.
    Unterstützt sowohl die ursprüngliche Datei als auch geteilte Dateien.

    Die Funktion verwendet einen Cache, um wiederholte Aufrufe zu optimieren.

    Returns:
        pd.DataFrame: Geladenes DataFrame mit allen Daten

    Raises:
        FileNotFoundError: Wenn keine Datendateien gefunden wurden
        ValueError: Wenn Probleme beim Laden der Daten auftreten
    """
    # Basisverzeichnis für Daten ermitteln
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    # Prüfen, ob geteilte Dateien existieren
    part_files = glob.glob(os.path.join(data_dir, 'Daten_UTF8_Clean_encoded_part*.csv'))

    if part_files:
        # Geteilte Dateien laden und kombinieren
        dfs = []
        for file_path in sorted(part_files):
            if os.path.exists(file_path):
                try:
                    # Datei mit pandas einlesen und zum Listenarray hinzufügen
                    df_part = pd.read_csv(file_path)
                    dfs.append(df_part)
                except Exception as e:
                    # Fehlerbehandlung bei Problemen mit einzelnen Dateien
                    raise ValueError(f"Fehler beim Laden der Datei '{file_path}': {str(e)}")

        # Prüfung, ob Dateien erfolgreich geladen wurden
        if not dfs:
            raise FileNotFoundError("Fehler: Keine Datendateien konnten geladen werden.")

        # Dataframes zu einem gemeinsamen DataFrame kombinieren
        df = pd.concat(dfs, ignore_index=True)
    else:
        # Originaldatei laden, falls geteilte Dateien nicht existieren
        file_path = os.path.join(data_dir, 'Daten_UTF8_Clean_encoded.csv')

        # Prüfen, ob die Originaldatei existiert
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden.")

        try:
            # Daten mit pandas aus der Originaldatei einlesen
            df = pd.read_csv(file_path)
        except Exception as e:
            # Fehlerbehandlung bei Problemen mit der Originaldatei
            raise ValueError(f"Fehler beim Laden der Daten: {str(e)}")

    # Rückgabe des geladenen DataFrames
    return df


# Cache für TF-IDF-Vektorisierung mit Zeitstempel für LRU-Implementierung
_tfidf_cache = {}
_tfidf_cache_access_times = {}
_tfidf_cache_max_size = 3

def clear_tfidf_cache():
    """
    Leert den TF-IDF-Cache vollständig.
    Nützlich, wenn sich die Daten geändert haben oder Speicher freigegeben werden soll.
    """
    global _tfidf_cache, _tfidf_cache_access_times
    _tfidf_cache.clear()
    _tfidf_cache_access_times.clear()
    logger.info("TF-IDF-Cache wurde geleert.")

def load_data(target_column: str = 'Fits_Topic_Code', max_features: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lädt die Daten aus den CSV-Dateien und gibt Features und Zielvariable zurück.
    Unterstützt sowohl die ursprüngliche Datei als auch geteilte Dateien.

    Die Funktion verwendet einen Cache für die TF-IDF-Vektorisierung, um wiederholte
    Aufrufe mit den gleichen Parametern zu optimieren.

    Args:
        target_column: Name der Zielspalte (Standard: 'Fits_Topic_Code')
        max_features: Maximale Anzahl der Features für TF-IDF (Standard: 1000)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) und Zielvariable (y)

    Raises:
        FileNotFoundError: Wenn keine Datendateien gefunden wurden
        ValueError: Wenn Probleme beim Laden oder Verarbeiten der Daten auftreten
        KeyError: Wenn die angegebene Zielspalte nicht im DataFrame vorhanden ist
    """
    try:
        # Laden des DataFrames mit der Hilfsfunktion (verwendet bereits LRU-Cache)
        df = _load_dataframe()

        # Validierung der angegebenen Zielspalte
        if target_column not in df.columns:
            raise KeyError(f"Fehler: Die Spalte '{target_column}' fehlt in der Datei. Bitte stellen Sie sicher, dass die Zielspalte vorhanden ist.")

        # Validierung der BODY-Spalte
        if 'BODY' not in df.columns:
            raise KeyError("Fehler: Die Spalte 'BODY' fehlt in der Datei. Diese wird für die Feature-Extraktion benötigt.")

        # Erstelle einen eindeutigen Cache-Schlüssel basierend auf max_features und einer Prüfsumme der Daten
        # Dies stellt sicher, dass der Cache ungültig wird, wenn sich die Daten ändern
        data_hash = hash(tuple(df['BODY'].iloc[:100].values))  # Verwende die ersten 100 Zeilen für die Prüfsumme
        cache_key = f"max_features_{max_features}_data_{data_hash}"

        # Prüfen, ob die TF-IDF-Matrix bereits im Cache ist
        if cache_key in _tfidf_cache:
            logger.info(f"Verwende gecachte TF-IDF-Matrix mit {max_features} Features")
            # Aktualisiere den Zeitstempel für den LRU-Algorithmus
            _tfidf_cache_access_times[cache_key] = pd.Timestamp.now()
            X = _tfidf_cache[cache_key]
        else:
            logger.info(f"Berechne neue TF-IDF-Matrix mit {max_features} Features")
            # Umwandlung der Textdaten in numerische Features mittels TF-IDF
            vectorizer = TfidfVectorizer(max_features=max_features)
            X = vectorizer.fit_transform(df['BODY'].values)

            # Speichern der Matrix im Cache mit aktuellem Zeitstempel
            _tfidf_cache[cache_key] = X
            _tfidf_cache_access_times[cache_key] = pd.Timestamp.now()

            # Cache-Größe begrenzen (maximal _tfidf_cache_max_size verschiedene Einträge)
            if len(_tfidf_cache) > _tfidf_cache_max_size:
                # Entferne den am längsten nicht verwendeten Eintrag (LRU-Strategie)
                lru_key = min(_tfidf_cache_access_times.items(), key=lambda x: x[1])[0]
                del _tfidf_cache[lru_key]
                del _tfidf_cache_access_times[lru_key]
                logger.info(f"Cache-Eintrag mit Schlüssel {lru_key} entfernt (LRU-Strategie)")

        # Zielvariable extrahieren
        y = df[target_column].values

        # Rückgabe der Features und der Zielvariable
        return X, y

    except (FileNotFoundError, ValueError, KeyError):
        # Spezifische Fehler direkt weiterleiten ohne zusätzlichen try-except Block
        raise
    except Exception as e:
        # Unerwartete Fehler in ValueError umwandeln
        raise ValueError(f"Unerwarteter Fehler beim Laden der Daten: {str(e)}")


def get_available_targets() -> Dict[str, Dict[str, Any]]:
    """
    Gibt ein Dictionary mit Informationen über verfügbare Zielspalten zurück.

    Identifiziert kategoriale Spalten (mit weniger als 100 einzigartigen Werten)
    und gibt detaillierte Informationen über diese zurück.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mit Informationen über verfügbare Zielspalten
            - 'unique_values': Anzahl der einzigartigen Werte
            - 'value_counts': Häufigkeitsverteilung der Werte
            - 'description': Beschreibung der Spalte
            - 'data_type': Datentyp der Spalte

    Raises:
        FileNotFoundError: Wenn keine Datendateien gefunden wurden
        ValueError: Wenn Probleme beim Laden der Daten auftreten
    """
    try:
        # Laden des DataFrames mit der Hilfsfunktion (verwendet bereits LRU-Cache)
        df = _load_dataframe()

        # Potenzielle Zielspalten (kategoriale Spalten mit weniger als 100 einzigartigen Werten)
        target_columns = {}

        # Überspringe die BODY-Spalte, da diese die Textdaten enthält
        columns_to_check = [col for col in df.columns if col != 'BODY']

        # Berechne nunique für alle Spalten auf einmal (effizienter)
        unique_counts = df[columns_to_check].nunique()

        # Filtere Spalten mit weniger als 100 einzigartigen Werten
        categorical_columns = unique_counts[unique_counts < 100].index.tolist()

        # Sammle Informationen für jede kategoriale Spalte
        for col in categorical_columns:
            unique_values = unique_counts[col]
            value_counts = df[col].value_counts()

            # Speichere Informationen über die Zielspalte
            target_columns[col] = {
                'unique_values': unique_values,
                'value_counts': value_counts.to_dict(),
                'description': f"Spalte mit {unique_values} einzigartigen Werten",
                'data_type': str(df[col].dtype)
            }

        # Rückgabe der verfügbaren Zielspalten
        return target_columns

    except (FileNotFoundError, ValueError):
        # Spezifische Fehler direkt weiterleiten
        raise
    except Exception as e:
        # Unerwartete Fehler in ValueError umwandeln
        raise ValueError(f"Unerwarteter Fehler beim Ermitteln der verfügbaren Zielspalten: {str(e)}")


if __name__ == "__main__":
    # Beispielausführung, wenn das Skript direkt ausgeführt wird
    try:
        # Verfügbare Zielspalten anzeigen
        targets = get_available_targets()
        logger.info("Verfügbare Zielspalten:")
        for target, info in targets.items():
            logger.info(f"- {target}: {info['description']}")

        # Beispiel: Daten für die erste verfügbare Zielspalte laden
        if targets:
            first_target = next(iter(targets))
            X, y = load_data(target_column=first_target)
            logger.info(f"\nDaten für Zielspalte '{first_target}' geladen:")
            logger.info(f"X: {type(X)}, Form: {X.shape}")
            logger.info(f"y: {type(y)}, Länge: {len(y)}")
    except Exception as e:
        logger.error(f"Fehler: {str(e)}")
