#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests für das data_loader Modul.

Dieses Modul enthält Tests für die Funktionalität des data_loader Moduls,
insbesondere für das Laden und Verarbeiten von Daten.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Importiere die zu testenden Funktionen aus dem neuen Paket
from suppervisedlearningproject.core.data_loader import load_data, get_available_targets, clear_tfidf_cache

# Fixture für ein Mock-DataFrame
@pytest.fixture
def mock_dataframe():
    """Erstellt ein Mock-DataFrame für Tests."""
    data = {
        'BODY': [
            "Dies ist ein Testtext für die Kategorie A.",
            "Ein weiterer Text für die Kategorie B.",
            "Noch ein Text für die Kategorie A.",
            "Dieser Text gehört zur Kategorie B."
        ],
        'Fits_Topic_Code': [0, 1, 0, 1],
        'Another_Target': [1, 0, 1, 0]
    }
    return pd.DataFrame(data)

# Test für load_data mit Mock-Daten
@patch('suppervisedlearningproject.core.data_loader._load_dataframe')
@patch('suppervisedlearningproject.core.data_loader._get_tfidf_matrix')
def test_load_data_with_mock(mock_get_tfidf, mock_load_df, mock_dataframe):
    """
    Testet die load_data Funktion mit gemockten Daten.

    Überprüft, ob die Funktion die richtigen Daten zurückgibt und
    die richtigen Aufrufe an die Hilfsfunktionen macht.
    """
    # Konfiguriere die Mocks
    mock_load_df.return_value = mock_dataframe
    mock_tfidf_matrix = np.random.rand(4, 10)  # 4 Dokumente, 10 Features
    mock_get_tfidf.return_value = mock_tfidf_matrix

    # Rufe die zu testende Funktion auf
    X, y = load_data(target_column='Fits_Topic_Code', max_features=10)

    # Überprüfe die Ergebnisse
    assert X is mock_tfidf_matrix
    assert np.array_equal(y, mock_dataframe['Fits_Topic_Code'].values)

    # Überprüfe, ob die Mocks korrekt aufgerufen wurden
    mock_load_df.assert_called_once()
    mock_get_tfidf.assert_called_once()

# Test für load_data mit ungültigem Zielspalten-Namen
@patch('suppervisedlearningproject.core.data_loader._load_dataframe')
def test_load_data_with_invalid_target(mock_load_df, mock_dataframe):
    """
    Testet die load_data Funktion mit einem ungültigen Zielspalten-Namen.

    Überprüft, ob die Funktion eine KeyError-Exception wirft, wenn
    die angegebene Zielspalte nicht im DataFrame vorhanden ist.
    """
    # Konfiguriere den Mock
    mock_load_df.return_value = mock_dataframe

    # Überprüfe, ob die Funktion eine KeyError-Exception wirft
    with pytest.raises(KeyError):
        load_data(target_column='NonExistentColumn')

# Test für get_available_targets
@patch('suppervisedlearningproject.core.data_loader._load_dataframe')
def test_get_available_targets(mock_load_df, mock_dataframe):
    """
    Testet die get_available_targets Funktion.

    Überprüft, ob die Funktion die richtigen Zielspalten zurückgibt.
    """
    # Konfiguriere den Mock
    mock_load_df.return_value = mock_dataframe

    # Rufe die zu testende Funktion auf
    targets = get_available_targets()

    # Überprüfe die Ergebnisse
    assert 'Fits_Topic_Code' in targets
    assert 'Another_Target' in targets
    assert len(targets) == 2

    # Überprüfe die Struktur der zurückgegebenen Daten
    for target_name, target_info in targets.items():
        assert 'unique_values' in target_info
        assert 'value_counts' in target_info
        assert 'description' in target_info
        assert 'data_type' in target_info

# Test für clear_tfidf_cache
@patch('suppervisedlearningproject.core.data_loader._get_tfidf_matrix.cache_clear')
def test_clear_tfidf_cache(mock_cache_clear):
    """
    Testet die clear_tfidf_cache Funktion.

    Überprüft, ob die Funktion die Cache-Clear-Methode aufruft.
    """
    # Rufe die zu testende Funktion auf
    clear_tfidf_cache()

    # Überprüfe, ob die Cache-Clear-Methode aufgerufen wurde
    mock_cache_clear.assert_called_once()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
