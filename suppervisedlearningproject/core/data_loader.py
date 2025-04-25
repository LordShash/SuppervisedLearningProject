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
from typing import Tuple, Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Importiere die Konfiguration und richte das Logging ein
# Angepasst für die neue Paketstruktur
from suppervisedlearningproject.utils import setup_logging, DATA_DIR, LOGS_DIR

# Konfiguration des Loggings mit dem zentralen Setup
logger = setup_logging(__name__)


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
    # Prüfen, ob geteilte Dateien existieren
    part_files = glob.glob(os.path.join(DATA_DIR, 'Daten_UTF8_Clean_encoded_part*.csv'))

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
        file_path = os.path.join(DATA_DIR, 'Daten_UTF8_Clean_encoded.csv')

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


# Cache für TF-IDF-Vektorisierung mit functools.lru_cache
_tfidf_cache_enabled = True
_tfidf_vectorizers = {}  # Speichert die Vektorisierer für spätere Verwendung

@functools.lru_cache(maxsize=3)
def _get_tfidf_matrix(max_features: int, data_hash: int):
    """
    Berechnet die TF-IDF-Matrix für die gegebenen Parameter.
    Diese Funktion wird durch lru_cache automatisch gecacht.

    Args:
        max_features: Maximale Anzahl der Features für TF-IDF
        data_hash: Hash der Daten zur Identifikation

    Returns:
        np.ndarray: TF-IDF-Matrix
    """
    logger.info(f"Berechne neue TF-IDF-Matrix mit {max_features} Features")
    df = _load_dataframe()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['BODY'].values)

    # Speichere den Vektorisierer für spätere Verwendung
    _tfidf_vectorizers[(max_features, data_hash)] = vectorizer

    return X

def clear_tfidf_cache():
    """
    Leert den TF-IDF-Cache vollständig.
    Nützlich, wenn sich die Daten geändert haben oder Speicher freigegeben werden soll.
    """
    _get_tfidf_matrix.cache_clear()
    _tfidf_vectorizers.clear()
    logger.info("TF-IDF-Cache wurde geleert.")

def load_data(target_column: str = 'Fits_Topic_Code', max_features: int = 1000, return_feature_names: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    """
    Lädt die Daten aus den CSV-Dateien und gibt Features und Zielvariable zurück.
    Unterstützt sowohl die ursprüngliche Datei als auch geteilte Dateien.

    Die Funktion verwendet einen Cache für die TF-IDF-Vektorisierung, um wiederholte
    Aufrufe mit den gleichen Parametern zu optimieren.

    Args:
        target_column: Name der Zielspalte (Standard: 'Fits_Topic_Code')
        max_features: Maximale Anzahl der Features für TF-IDF (Standard: 1000)
        return_feature_names: Wenn True, werden auch die Feature-Namen zurückgegeben (Standard: False)

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[list]]: Features (X), Zielvariable (y) und optional Feature-Namen

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

        # Erstelle einen eindeutigen Hash basierend auf einer Prüfsumme der Daten
        # Dies stellt sicher, dass der Cache ungültig wird, wenn sich die Daten ändern
        data_hash = hash(tuple(df['BODY'].iloc[:100].values))  # Verwende die ersten 100 Zeilen für die Prüfsumme

        # Verwende die gecachte Funktion, um die TF-IDF-Matrix zu erhalten
        if _tfidf_cache_enabled:
            X = _get_tfidf_matrix(max_features, data_hash)
            logger.info(f"TF-IDF-Matrix mit {max_features} Features geladen (Cache: {'Hit' if _get_tfidf_matrix.cache_info().hits > 0 else 'Miss'})")
        else:
            # Falls Cache deaktiviert ist, berechne die Matrix direkt
            logger.info(f"Berechne TF-IDF-Matrix mit {max_features} Features (Cache deaktiviert)")
            vectorizer = TfidfVectorizer(max_features=max_features)
            X = vectorizer.fit_transform(df['BODY'].values)
            _tfidf_vectorizers[(max_features, data_hash)] = vectorizer

        # Zielvariable extrahieren
        y = df[target_column].values

        # Rückgabe der Features, der Zielvariable und optional der Feature-Namen
        if return_feature_names:
            # Hole den Vectorizer aus dem Cache oder erstelle einen neuen
            data_hash = hash(tuple(df['BODY'].iloc[:100].values))
            vectorizer = _tfidf_vectorizers.get((max_features, data_hash))

            # Wenn kein Vectorizer im Cache ist, wurde er oben erstellt
            if vectorizer is None and not _tfidf_cache_enabled:
                vectorizer = _tfidf_vectorizers.get((max_features, data_hash))

            # Wenn immer noch kein Vectorizer gefunden wurde, erstelle einen neuen
            if vectorizer is None:
                logger.warning("Vectorizer nicht im Cache gefunden, erstelle einen neuen für Feature-Namen")
                vectorizer = TfidfVectorizer(max_features=max_features)
                vectorizer.fit(df['BODY'].values)

            # Extrahiere Feature-Namen
            feature_names = vectorizer.get_feature_names_out()
            return X, y, feature_names
        else:
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
