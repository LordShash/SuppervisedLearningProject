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


# Cache für TF-IDF-Vektorisierung
_tfidf_cache = {}

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

        # Cache-Schlüssel für die aktuelle Konfiguration erstellen
        cache_key = f"max_features_{max_features}"

        # Prüfen, ob die TF-IDF-Matrix bereits im Cache ist
        if cache_key in _tfidf_cache:
            print(f"Verwende gecachte TF-IDF-Matrix mit {max_features} Features")
            X = _tfidf_cache[cache_key]
        else:
            print(f"Berechne neue TF-IDF-Matrix mit {max_features} Features")
            # Umwandlung der Textdaten in numerische Features mittels TF-IDF
            vectorizer = TfidfVectorizer(max_features=max_features)
            X = vectorizer.fit_transform(df['BODY'].values)

            # Speichern der Matrix im Cache
            _tfidf_cache[cache_key] = X

            # Cache-Größe begrenzen (maximal 3 verschiedene max_features-Werte)
            if len(_tfidf_cache) > 3:
                # Ältesten Eintrag entfernen
                oldest_key = next(iter(_tfidf_cache))
                del _tfidf_cache[oldest_key]

        # Extraktion der Zielvariable aus dem DataFrame
        y = df[target_column].values

        # Rückgabe der Features und Zielvariable
        return X, y

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Spezifische Fehler weiterleiten
        raise
    except Exception as e:
        # Unerwartete Fehler in ValueError umwandeln
        raise ValueError(f"Unerwarteter Fehler beim Verarbeiten der Daten: {str(e)}")


def get_available_targets() -> Dict[str, Any]:
    """
    Gibt eine Liste der verfügbaren Zielspalten und deren Werte zurück.
    Unterstützt sowohl die ursprüngliche Datei als auch geteilte Dateien.

    Returns:
        Dict[str, Any]: Dictionary mit Zielspalten und deren Werten

    Raises:
        FileNotFoundError: Wenn keine Datendateien gefunden wurden
        ValueError: Wenn Probleme beim Laden oder Verarbeiten der Daten auftreten
    """
    try:
        # Laden des DataFrames mit der Hilfsfunktion
        df = _load_dataframe()

        # Definition der potenziellen Zielspalten für die Analyse
        target_columns = ['Urgency_Code', 'Categorie_Code', 'Fits_Topic_Code']

        # Erstellung eines Dictionaries zur Speicherung der Zielspaltendaten
        targets = {}
        for col in target_columns:
            # Prüfung, ob die Spalte im DataFrame existiert
            if col in df.columns:
                # Speicherung der eindeutigen Werte und deren Häufigkeiten
                targets[col] = {
                    'unique_values': df[col].unique().tolist(),
                    'value_counts': df[col].value_counts().to_dict()
                }

        # Prüfung, ob mindestens eine Zielspalte gefunden wurde
        if not targets:
            raise ValueError("Keine der erwarteten Zielspalten wurde in den Daten gefunden.")

        # Rückgabe des erstellten Dictionaries
        return targets

    except (FileNotFoundError, ValueError) as e:
        # Spezifische Fehler weiterleiten
        raise
    except Exception as e:
        # Unerwartete Fehler in ValueError umwandeln
        raise ValueError(f"Unerwarteter Fehler beim Verarbeiten der Daten: {str(e)}")


if __name__ == "__main__":
    # Smoke-Test: Ausführung eines einfachen Tests zum Laden der Daten
    try:
        # Abruf und Anzeige der verfügbaren Zielspalten
        targets = get_available_targets()
        print("Verfügbare Zielspalten:")
        for col, info in targets.items():
            print(f"- {col}: {info['unique_values']}")
            print(f"  Verteilung: {info['value_counts']}")

        # Laden der Daten mit der Standard-Zielspalte
        X, y = load_data()
        print(f"\nDaten erfolgreich geladen:")
        print(f"X: Form {X.shape}, Typ {type(X)}")
        print(f"y: Form {y.shape}, Typ {type(y)}")

        # Demonstration der Flexibilität: Laden mit alternativer Zielspalte
        X_urgency, y_urgency = load_data(target_column='Urgency_Code')
        print(f"\nDaten mit Zielspalte 'Urgency_Code' erfolgreich geladen:")
        print(f"X: Form {X_urgency.shape}, Typ {type(X_urgency)}")
        print(f"y: Form {y_urgency.shape}, Typ {type(y_urgency)}")
    except SystemExit as e:
        # Ausgabe von Fehlermeldungen
        print(e)
