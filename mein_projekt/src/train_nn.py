#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul zum Training eines neuronalen Netzwerks.

Dieses Modul stellt Funktionen und Klassen bereit, um ein neuronales Netzwerk
zu definieren, zu trainieren, zu evaluieren und zu speichern.
"""

import os
import sys
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

# Importiere die Datenladefunktion aus dem data_loader Modul
from data_loader import load_data


class FeedForwardNN(nn.Module):
    """
    Feed-Forward neuronales Netzwerk mit zwei Hidden-Layern.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1, multi_class: bool = False):
        """
        Initialisiert das neuronale Netzwerk.

        Args:
            input_dim: Anzahl der Eingabefeatures
            hidden_dim: Anzahl der Neuronen in den Hidden-Layern
            output_dim: Anzahl der Ausgabeneuronen
            multi_class: Flag für Multiclass-Klassifikation
        """
        super(FeedForwardNN, self).__init__()

        # Netzwerkarchitektur definieren
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

        # Aktivierungsfunktionen
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Dropout zur Vermeidung von Overfitting
        self.multi_class = multi_class

        # Ausgabeaktivierung für binäre Klassifikation
        self.sigmoid = nn.Sigmoid() if not multi_class else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Netzwerk.

        Args:
            x: Eingabetensor

        Returns:
            torch.Tensor: Ausgabetensor
        """
        # Erster Hidden-Layer mit ReLU und Dropout
        x = self.dropout(self.relu(self.layer1(x)))

        # Zweiter Hidden-Layer mit ReLU und Dropout
        x = self.dropout(self.relu(self.layer2(x)))

        # Ausgabe-Layer
        x = self.layer3(x)

        # Sigmoid für binäre Klassifikation
        if not self.multi_class:
            x = self.sigmoid(x)

        return x


def prepare_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                 random_state: int = 42) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Bereitet die Daten für das Training des neuronalen Netzes vor.

    Args:
        X: Feature-Matrix (kann eine Sparse-Matrix sein)
        y: Zielvariable
        test_size: Anteil der Testdaten
        random_state: Seed für die Reproduzierbarkeit

    Returns:
        Tuple: Train-DataLoader, Test-DataLoader, Anzahl der Features, Anzahl der Klassen
    """
    # Daten in Trainings- und Testsets aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Sparse-Matrix in Dense-Matrix umwandeln, falls nötig
    if sparse.issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # Anzahl der Klassen bestimmen
    num_classes = len(np.unique(y))

    # Daten in PyTorch-Tensoren umwandeln
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)

    # Für Multiclass-Klassifikation: One-Hot-Encoding oder direkte Klassen-Indizes
    if num_classes > 2:
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
    else:
        # Für binäre Klassifikation: Umwandlung in Float-Tensor
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    # Datasets erstellen
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # DataLoader erstellen
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, X_train.shape[1], num_classes


def train_neural_network(train_loader: DataLoader, test_loader: DataLoader, 
                         input_dim: int, num_classes: int, target_name: str = "Fits_Topic_Code",
                         epochs: int = 50, patience: int = 5) -> Tuple[nn.Module, float, float, float, float, str, np.ndarray]:
    """
    Trainiert ein neuronales Netzwerk mit Early Stopping.

    Args:
        train_loader: DataLoader für Trainingsdaten
        test_loader: DataLoader für Testdaten
        input_dim: Anzahl der Eingabefeatures
        num_classes: Anzahl der Klassen
        target_name: Name der Zielvariable für Reporting
        epochs: Maximale Anzahl der Trainingsepochen
        patience: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird

    Returns:
        Tuple[nn.Module, float, float, float, float, str, np.ndarray]: Trainiertes Modell, Genauigkeit, Precision, Recall, F1-Score auf dem Testset, Klassifikationsbericht und Konfusionsmatrix
    """
    # Multiclass-Flag und Output-Dimension bestimmen
    multi_class = num_classes > 2
    output_dim = num_classes if multi_class else 1

    # Modell initialisieren
    model = FeedForwardNN(input_dim=input_dim, output_dim=output_dim, multi_class=multi_class)

    # Loss-Funktion basierend auf Klassifikationstyp wählen
    criterion = nn.CrossEntropyLoss() if multi_class else nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early Stopping Variablen
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    print(f"Training mit Early Stopping (Patience: {patience})")

    # Training
    for epoch in range(epochs):
        # Trainingsmodus
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            # Forward-Pass
            outputs = model(inputs)

            # Loss berechnen
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # Backward-Pass und Optimierung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Durchschnittlichen Trainingsverlust berechnen
        avg_train_loss = train_loss / len(train_loader)

        # Validierungsmodus
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Forward-Pass
                outputs = model(inputs)

                # Loss berechnen
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Durchschnittlichen Validierungsverlust berechnen
        avg_val_loss = val_loss / len(test_loader)

        # Fortschritt ausgeben
        print(f"Epoche {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early Stopping prüfen
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early Stopping nach {epoch + 1} Epochen")
                break

    # Bestes Modell wiederherstellen
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Forward-Pass
            outputs = model(inputs)

            # Vorhersagen
            if multi_class:
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.numpy())
                all_targets.extend(targets.numpy())
            else:
                predicted = (outputs >= 0.5).float()
                all_preds.extend(predicted.numpy().flatten())
                all_targets.extend(targets.numpy().flatten())

    # Genauigkeit berechnen
    accuracy = accuracy_score(all_targets, all_preds)

    # Berechnung von Precision, Recall und F1-Score
    # Bei Multiclass-Klassifikation verwenden wir 'weighted' average
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    # Detaillierter Klassifikationsbericht
    report = classification_report(all_targets, all_preds, 
                                  target_names=[f"{target_name}_{i}" for i in sorted(set(all_targets))])

    # Erstellung der Konfusionsmatrix
    conf_matrix = confusion_matrix(all_targets, all_preds)

    return model, accuracy, precision, recall, f1, report, conf_matrix


def save_model(model: nn.Module, target_column: str = "Fits_Topic_Code") -> str:
    """
    Speichert das trainierte PyTorch-Modell.

    Args:
        model: Trainiertes Modell
        target_column: Name der Zielspalte für den Dateinamen

    Returns:
        str: Pfad zur gespeicherten Modelldatei

    Raises:
        ValueError: Wenn Probleme beim Speichern des Modells auftreten
    """
    # Pfad zum models-Verzeichnis
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    # Verzeichnis erstellen, falls es nicht existiert
    os.makedirs(models_dir, exist_ok=True)

    # Vollständiger Pfad zur Modelldatei
    filename = f"nn_{target_column}_model.pt"
    model_path = os.path.join(models_dir, filename)

    try:
        # Modell speichern
        torch.save(model.state_dict(), model_path)
        print(f"Modell erfolgreich gespeichert unter: {model_path}")
        return model_path

    except Exception as e:
        raise ValueError(f"Fehler beim Speichern des Modells: {str(e)}")


def train_and_save_model(target_column: str = "Fits_Topic_Code", epochs: int = 50, 
                      max_features: int = 1000, test_size: float = 0.2,
                      patience: int = 5) -> Tuple[float, float, float, float, str, np.ndarray]:
    """
    Lädt Daten, trainiert ein Modell und speichert es.

    Args:
        target_column: Name der Zielspalte
        epochs: Maximale Anzahl der Trainingsepochen
        max_features: Maximale Anzahl der Features für TF-IDF
        test_size: Anteil der Testdaten
        patience: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird

    Returns:
        Tuple[float, float, float, float, str, np.ndarray]: Genauigkeit, Precision, Recall, F1-Score, Klassifikationsbericht und Konfusionsmatrix

    Raises:
        FileNotFoundError: Wenn keine Datendateien gefunden wurden
        ValueError: Wenn Probleme beim Laden, Verarbeiten oder Speichern der Daten auftreten
        KeyError: Wenn die angegebene Zielspalte nicht im DataFrame vorhanden ist
    """
    try:
        # Daten laden mit konfigurierbaren max_features
        X, y = load_data(target_column=target_column, max_features=max_features)

        # Daten vorbereiten mit konfigurierbarem test_size
        train_loader, test_loader, input_dim, num_classes = prepare_data(X, y, test_size=test_size)

        # Neuronales Netz trainieren mit Early Stopping
        model, accuracy, precision, recall, f1, report, conf_matrix = train_neural_network(
            train_loader, test_loader, input_dim, num_classes, 
            target_name=target_column, epochs=epochs, patience=patience
        )

        # Modell speichern
        model_path = save_model(model, target_column)
        print(f"Modell gespeichert unter: {model_path}")

        # Ausgabe der Konfusionsmatrix
        print("\nKonfusionsmatrix:")
        print(conf_matrix)

        return accuracy, precision, recall, f1, report, conf_matrix

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Spezifische Fehler weiterleiten
        raise
    except Exception as e:
        # Unerwartete Fehler in ValueError umwandeln
        raise ValueError(f"Unerwarteter Fehler beim Training und Speichern des Modells: {str(e)}")


if __name__ == "__main__":
    try:
        # Modell mit Standard-Zielspalte trainieren
        print("Training des neuronalen Netzes mit Zielspalte 'Fits_Topic_Code'...")
        accuracy, precision, recall, f1, report, conf_matrix = train_and_save_model(epochs=30)

        # Metriken und Bericht ausgeben
        print(f"Finale Accuracy auf dem Test-Set: {accuracy:.4f}")
        print(f"Precision auf dem Test-Set: {precision:.4f}")
        print(f"Recall auf dem Test-Set: {recall:.4f}")
        print(f"F1 Score auf dem Test-Set: {f1:.4f}")
        print("\nKlassifikationsbericht:")
        print(report)

        # Optional: Modell mit anderer Zielspalte trainieren
        print("\n--- Training mit Zielspalte 'Categorie_Code' ---")
        accuracy_cat, precision_cat, recall_cat, f1_cat, report_cat, conf_matrix_cat = train_and_save_model(target_column="Categorie_Code", epochs=30)

        # Metriken und Bericht ausgeben
        print(f"Finale Accuracy auf dem Test-Set: {accuracy_cat:.4f}")
        print(f"Precision auf dem Test-Set: {precision_cat:.4f}")
        print(f"Recall auf dem Test-Set: {recall_cat:.4f}")
        print(f"F1 Score auf dem Test-Set: {f1_cat:.4f}")
        print("\nKlassifikationsbericht:")
        print(report_cat)

    except SystemExit as e:
        print(e)
