#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul zum Training eines neuronalen Netzwerks.

Dieses Modul stellt Funktionen und Klassen bereit, um ein neuronales Netzwerk
zu definieren, zu trainieren, zu evaluieren und zu speichern.
"""

import os
import sys
import datetime
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Importiere die Datenladefunktion aus dem core Paket
from suppervisedlearningproject.core.data_loader import load_data, get_available_targets

# Importiere die Konfiguration und richte das Logging ein
from suppervisedlearningproject.utils import setup_logging, LOGS_DIR, PLOTS_DIR, CHECKPOINTS_DIR, MODELS_DIR

# Konfiguration des Loggings mit dem zentralen Setup
logger = setup_logging(__name__)

@dataclass
class NNModelResults:
    """Datenklasse zur Speicherung der Modellergebnisse."""
    model: nn.Module
    accuracy: float
    precision: float
    recall: float
    f1: float
    report: str
    conf_matrix: np.ndarray
    class_names: list
    training_history: Dict[str, List[float]]
    device: str


class FeedForwardNN(nn.Module):
    """
    Feed-Forward neuronales Netzwerk mit konfigurierbarer Architektur.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int] = [128, 128], 
                 output_dim: int = 1, 
                 multi_class: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 batch_norm: bool = True):
        """
        Initialisiert das neuronale Netzwerk mit konfigurierbarer Architektur.

        Args:
            input_dim: Anzahl der Eingabefeatures
            hidden_dims: Liste mit Anzahl der Neuronen in den Hidden-Layern
            output_dim: Anzahl der Ausgabeneuronen
            multi_class: Flag für Multiclass-Klassifikation
            dropout_rate: Dropout-Rate zur Vermeidung von Overfitting
            activation: Aktivierungsfunktion ('relu', 'leaky_relu', 'elu', 'tanh')
            batch_norm: Flag für Batch-Normalisierung
        """
        super(FeedForwardNN, self).__init__()

        # Speichere Konfiguration
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.multi_class = multi_class
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.batch_norm = batch_norm

        # Aktivierungsfunktion basierend auf dem Parameter wählen
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            logger.warning(f"Unbekannte Aktivierungsfunktion '{activation}', verwende ReLU")
            self.activation = nn.ReLU()
            self.activation_name = 'relu'

        # Dropout-Layer
        self.dropout = nn.Dropout(dropout_rate)

        # Ausgabeaktivierung für binäre Klassifikation
        self.sigmoid = nn.Sigmoid() if not multi_class else None

        # Dynamische Erstellung der Layer basierend auf hidden_dims
        layers = []

        # Eingabe-Layer
        if batch_norm:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(self.activation)
            layers.append(self.dropout)
        else:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(self.activation)
            layers.append(self.dropout)

        # Hidden-Layer
        for i in range(len(hidden_dims) - 1):
            if batch_norm:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                layers.append(self.activation)
                layers.append(self.dropout)
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(self.activation)
                layers.append(self.dropout)

        # Ausgabe-Layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Kombiniere alle Layer in einer Sequential
        self.model = nn.Sequential(*layers)

        # Initialisierung der Gewichte
        self._init_weights()

        # Logge Modellarchitektur
        logger.info(f"Modellarchitektur erstellt: input_dim={input_dim}, hidden_dims={hidden_dims}, "
                   f"output_dim={output_dim}, multi_class={multi_class}, dropout={dropout_rate}, "
                   f"activation={activation}, batch_norm={batch_norm}")

    def _init_weights(self):
        """Initialisiert die Gewichte des Netzwerks mit He-Initialisierung."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Netzwerk.

        Args:
            x: Eingabetensor

        Returns:
            torch.Tensor: Ausgabetensor
        """
        # Forward-Pass durch das Sequential-Modell
        x = self.model(x)

        # Sigmoid für binäre Klassifikation
        if not self.multi_class:
            x = self.sigmoid(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Gibt die Konfiguration des Modells zurück."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'multi_class': self.multi_class,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_name,
            'batch_norm': self.batch_norm
        }


def train_and_save_model(
    target_column: str = "Fits_Topic_Code", 
    epochs: int = 50, 
    max_features: int = 1000, 
    test_size: float = 0.2,
    val_size: float = 0.1,
    patience: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_dims: List[int] = [128, 128],
    dropout_rate: float = 0.2,
    activation: str = 'relu',
    batch_norm: bool = True,
    weight_decay: float = 0.0001,
    lr_scheduler: bool = True,
    lr_factor: float = 0.5,
    lr_patience: int = 3,
    visualize: bool = True,
    save_checkpoints: bool = True,
    version: Optional[str] = None,
    overwrite: bool = True,
    class_names: Optional[List[str]] = None,
    standardize: bool = True,
    save_model_flag: bool = True
) -> NNModelResults:
    """
    Lädt Daten, trainiert ein neuronales Netzwerk und speichert es optional.

    Args:
        target_column: Name der Zielspalte
        epochs: Maximale Anzahl der Trainingsepochen
        max_features: Maximale Anzahl der Features für TF-IDF
        test_size: Anteil der Testdaten
        val_size: Anteil der Validierungsdaten
        patience: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird
        batch_size: Größe der Batches für DataLoader
        learning_rate: Initiale Lernrate für den Optimizer
        hidden_dims: Liste mit Anzahl der Neuronen in den Hidden-Layern
        dropout_rate: Dropout-Rate zur Vermeidung von Overfitting
        activation: Aktivierungsfunktion ('relu', 'leaky_relu', 'elu', 'tanh')
        batch_norm: Flag für Batch-Normalisierung
        weight_decay: L2-Regularisierungsparameter
        lr_scheduler: Flag für Learning Rate Scheduler
        lr_factor: Faktor, um den die Lernrate reduziert wird
        lr_patience: Anzahl der Epochen ohne Verbesserung, bevor die Lernrate reduziert wird
        visualize: Flag für Visualisierung des Trainingsfortschritts
        save_checkpoints: Flag für Speicherung von Checkpoints
        version: Optionale Versionsbezeichnung für das Modell
        overwrite: Wenn True, werden vorhandene Modelldateien überschrieben
        class_names: Liste mit Namen für die Klassen (falls None, werden automatisch generiert)
        standardize: Flag, ob die Daten standardisiert werden sollen
        save_model_flag: Wenn True, wird das Modell gespeichert

    Returns:
        NNModelResults: Objekt mit trainiertem Modell und allen Evaluationsmetriken

    Raises:
        FileNotFoundError: Wenn keine Datendateien gefunden wurden
        ValueError: Wenn Probleme beim Laden, Verarbeiten oder Speichern der Daten auftreten
        KeyError: Wenn die angegebene Zielspalte nicht im DataFrame vorhanden ist
        FileExistsError: Wenn die Modelldatei bereits existiert und overwrite=False
    """
    # Dummy-Implementierung, die ein leeres NNModelResults-Objekt zurückgibt
    # Diese Funktion wird später vollständig implementiert
    logger.warning("Die Funktion train_and_save_model ist noch nicht vollständig implementiert.")

    # Erstelle ein leeres NNModelResults-Objekt
    results = NNModelResults(
        model=FeedForwardNN(input_dim=10, output_dim=1),
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        report="",
        conf_matrix=np.array([]),
        class_names=[],
        training_history={},
        device="cpu"
    )

    return results
