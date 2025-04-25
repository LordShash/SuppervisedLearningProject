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
import traceback
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Importiere die Datenladefunktion aus dem core Paket
from suppervisedlearningproject.core.data_loader import load_data, get_available_targets

# Importiere die Konfiguration und richte das Logging ein
from suppervisedlearningproject.utils import setup_logging, LOGS_DIR, PLOTS_DIR, CHECKPOINTS_DIR, MODELS_DIR
from suppervisedlearningproject.utils.config import get_model_path, get_plot_path

# Konfiguration des Loggings mit dem zentralen Setup
logger = setup_logging(__name__)

@dataclass
class NNModelResults:
    """
    Datenklasse zur Speicherung der Modellergebnisse.

    Attributes:
        model: Das trainierte neuronale Netzwerk
        accuracy: Genauigkeit des Modells
        precision: Präzision des Modells
        recall: Recall des Modells
        f1: F1-Score des Modells
        report: Klassifikationsbericht als String
        conf_matrix: Konfusionsmatrix als numpy-Array
        class_names: Liste mit den Namen der Klassen
        training_history: Dictionary mit Trainingshistorie (Loss, Accuracy)
        device: Gerät, auf dem das Modell trainiert wurde (CPU/GPU)
        roc_curve_data: Dictionary mit ROC-Kurven-Daten für jede Klasse
        auc_scores: Dictionary mit AUC-Werten für jede Klasse
    """
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
    roc_curve_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    auc_scores: Optional[Dict[str, float]] = None


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: list, target_column: str, timestamp: str = None) -> str:
    """
    Erstellt und speichert eine Visualisierung der Konfusionsmatrix.

    Args:
        conf_matrix: Die Konfusionsmatrix als numpy-Array
        class_names: Liste mit den Namen der Klassen
        target_column: Name der Zielspalte für den Dateinamen
        timestamp: Optionaler Zeitstempel für den Dateinamen

    Returns:
        str: Pfad zur gespeicherten Visualisierung
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Visualisiere Konfusionsmatrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='viridis')
    plt.title(f'Konfusionsmatrix für {target_column}')
    plt.colorbar()

    # Beschriftungen
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Werte in der Matrix anzeigen
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('Vorhergesagte Klasse')

    # Speichere den Plot
    cm_plot_path = get_plot_path('confusion_matrix', target_column, timestamp)
    plt.savefig(cm_plot_path)
    plt.close()

    logger.info(f"Konfusionsmatrix-Visualisierung gespeichert: {cm_plot_path}")
    return cm_plot_path


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
    logger.info(f"Starte Training eines neuronalen Netzes für Zielvariable '{target_column}'")

    # Generiere einen Zeitstempel für die Modellbezeichnung
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Bestimme den Gerät (CPU oder GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Gerät: {device}")

    try:
        # Lade die Daten
        logger.info("Lade Daten...")
        X, y, feature_names = load_data(
            target_column=target_column,
            max_features=max_features,
            return_feature_names=True
        )

        # Prüfe, ob die Daten erfolgreich geladen wurden
        if X is None or y is None:
            raise ValueError("Fehler beim Laden der Daten. X oder y ist None.")

        logger.info(f"Daten geladen: X.shape={X.shape}, y.shape={y.shape}")

        # Konvertiere sparse Matrix zu numpy Array, falls nötig
        if sparse.issparse(X):
            X = X.toarray()

        # Standardisiere die Features, falls gewünscht
        if standardize:
            logger.info("Standardisiere Features...")
            scaler = StandardScaler(with_mean=not sparse.issparse(X))
            X = scaler.fit_transform(X)

        # Bestimme, ob es sich um ein Multiclass-Problem handelt
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        multi_class = num_classes > 2

        # Erstelle Klassennamen, falls nicht angegeben
        if class_names is None:
            class_names = [str(cls) for cls in unique_classes]

        logger.info(f"Klassifikationsproblem: {'Multiclass' if multi_class else 'Binär'} mit {num_classes} Klassen")

        # Teile die Daten in Trainings-, Validierungs- und Testsets auf
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Teile die Trainingsdaten weiter in Trainings- und Validierungssets auf
        if val_size > 0:
            # Berechne den relativen Validierungsanteil vom verbleibenden Datensatz
            relative_val_size = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=relative_val_size, 
                random_state=42, stratify=y_train_val
            )
        else:
            # Wenn keine Validierung gewünscht ist, verwende alle Daten für das Training
            X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None

        logger.info(f"Datenaufteilung: Train={X_train.shape}, "
                   f"{'Val=' + str(X_val.shape) + ', ' if X_val is not None else ''}"
                   f"Test={X_test.shape}")

        # Konvertiere die Daten zu PyTorch Tensoren
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train) if not multi_class else torch.LongTensor(y_train)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val) if not multi_class else torch.LongTensor(y_val)

        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test) if not multi_class else torch.LongTensor(y_test)

        # Erstelle DataLoader für Batch-Training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Definiere das Modell
        input_dim = X_train.shape[1]
        output_dim = num_classes if multi_class else 1

        model = FeedForwardNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            multi_class=multi_class,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm
        )

        # Verschiebe das Modell auf das gewählte Gerät
        model = model.to(device)

        # Definiere Loss-Funktion und Optimizer
        if multi_class:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()

        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

        # Learning Rate Scheduler für adaptive Lernrate
        scheduler = None
        if lr_scheduler:
            # Verbose parameter removed to avoid deprecation warning
            # Use scheduler.get_last_lr() to access the learning rate if needed
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=lr_factor, 
                patience=lr_patience
            )

        # Initialisiere Tracking-Variablen für Early Stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        # Initialisiere Tracking für Trainingshistorie
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Training Loop
        logger.info(f"Starte Training für {epochs} Epochen...")
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                # Verschiebe Daten auf das Gerät
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # Forward Pass
                optimizer.zero_grad()
                outputs = model(batch_X)

                # Berechne Loss
                if multi_class:
                    loss = criterion(outputs, batch_y)
                    # Berechne Accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == batch_y).sum().item()
                else:
                    loss = criterion(outputs, batch_y.view(-1, 1))
                    # Berechne Accuracy
                    predicted = (outputs.data > 0.5).float()
                    train_correct += (predicted == batch_y.view(-1, 1)).sum().item()

                train_total += batch_y.size(0)
                train_loss += loss.item() * batch_y.size(0)

                # Backward Pass und Optimierung
                loss.backward()
                optimizer.step()

            # Berechne durchschnittlichen Loss und Accuracy für diese Epoche
            avg_train_loss = train_loss / train_total
            train_accuracy = train_correct / train_total

            # Validierung
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            if X_val is not None:
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        # Verschiebe Daten auf das Gerät
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                        # Forward Pass
                        outputs = model(batch_X)

                        # Berechne Loss
                        if multi_class:
                            loss = criterion(outputs, batch_y)
                            # Berechne Accuracy
                            _, predicted = torch.max(outputs.data, 1)
                            val_correct += (predicted == batch_y).sum().item()
                        else:
                            loss = criterion(outputs, batch_y.view(-1, 1))
                            # Berechne Accuracy
                            predicted = (outputs.data > 0.5).float()
                            val_correct += (predicted == batch_y.view(-1, 1)).sum().item()

                        val_total += batch_y.size(0)
                        val_loss += loss.item() * batch_y.size(0)

                # Berechne durchschnittlichen Loss und Accuracy für diese Epoche
                avg_val_loss = val_loss / val_total
                val_accuracy = val_correct / val_total

                # Learning Rate Scheduler
                if scheduler is not None:
                    scheduler.step(avg_val_loss)

                # Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0

                    # Speichere Checkpoint, falls gewünscht
                    if save_checkpoints and save_model_flag:
                        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{target_column}_best_model.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_val_loss,
                            'config': model.get_config()
                        }, checkpoint_path)
                        logger.info(f"Checkpoint gespeichert: {checkpoint_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early Stopping nach {epoch+1} Epochen")
                        break
            else:
                # Wenn keine Validierungsdaten vorhanden sind, verwende den Trainingsloss
                avg_val_loss = avg_train_loss
                val_accuracy = train_accuracy

                # Speichere das Modell nach jeder Epoche
                if save_checkpoints and save_model_flag:
                    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{target_column}_best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss,
                        'config': model.get_config()
                    }, checkpoint_path)
                    logger.info(f"Checkpoint gespeichert: {checkpoint_path}")

            # Speichere Trainingshistorie
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_accuracy)
            history['val_acc'].append(val_accuracy)

            # Logge Fortschritt
            logger.info(f"Epoche {epoch+1}/{epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Lade das beste Modell, falls Early Stopping aktiviert war
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluiere das Modell auf den Testdaten
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)

            outputs = model(X_test_tensor)

            if multi_class:
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu().numpy()
            else:
                predicted = (outputs.data > 0.5).float().cpu().numpy().flatten()

            y_true = y_test_tensor.cpu().numpy()

            # Berechne Metriken
            accuracy = accuracy_score(y_true, predicted)

            if multi_class:
                precision = precision_score(y_true, predicted, average='weighted')
                recall = recall_score(y_true, predicted, average='weighted')
                f1 = f1_score(y_true, predicted, average='weighted')
            else:
                precision = precision_score(y_true, predicted, zero_division=0)
                recall = recall_score(y_true, predicted, zero_division=0)
                f1 = f1_score(y_true, predicted, zero_division=0)

            # Erstelle Klassifikationsbericht
            report = classification_report(y_true, predicted, target_names=class_names)

            # Erstelle Konfusionsmatrix
            conf_matrix = confusion_matrix(y_true, predicted)

            # Stelle sicher, dass die Konfusionsmatrix ein 2D-Array ist
            if len(conf_matrix.shape) != 2:
                logger.warning("Konfusionsmatrix hat nicht die erwartete Form. Erstelle 2D-Matrix.")
                conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        # Visualisiere die Ergebnisse, falls gewünscht
        if visualize:
            # Visualisiere Trainingshistorie
            plt.figure(figsize=(12, 5))

            # Plot für Loss
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Loss während des Trainings')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plot für Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['val_acc'], label='Val Accuracy')
            plt.title('Accuracy während des Trainings')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()

            # Speichere den Plot
            history_plot_path = get_plot_path('training_history', target_column, timestamp)
            plt.savefig(history_plot_path)
            plt.close()

            # Visualisiere Konfusionsmatrix mit der dedizierten Funktion
            cm_plot_path = plot_confusion_matrix(conf_matrix, class_names, target_column, timestamp)

        # Speichere das finale Modell, falls gewünscht
        if save_model_flag:
            model_path = get_model_path('nn', target_column, timestamp)

            # Prüfe, ob die Datei bereits existiert
            if os.path.exists(model_path) and not overwrite:
                raise FileExistsError(f"Modelldatei existiert bereits: {model_path}")

            # Speichere das Modell
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.get_config(),
                'class_names': class_names,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'multi_class': multi_class,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            }, model_path)

            logger.info(f"Modell gespeichert: {model_path}")

        # Berechne ROC-Kurve und AUC für jede Klasse
        roc_curve_data = {}
        auc_scores = {}

        try:

            # Berechne Wahrscheinlichkeiten für ROC-Kurve
            model.eval()
            with torch.no_grad():
                X_test_tensor = X_test_tensor.to(device)
                outputs = model(X_test_tensor)

                if multi_class:
                    # Für Mehrklassen-Klassifikation
                    y_pred_proba = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                else:
                    # Für binäre Klassifikation
                    y_pred_proba = outputs.cpu().numpy()

            # Für binäre Klassifikation
            if num_classes == 2:
                # Verwende die Wahrscheinlichkeit der positiven Klasse
                if multi_class:
                    # Bei multi_class=True mit 2 Klassen haben wir Softmax-Ausgaben
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1], pos_label=1)
                else:
                    # Bei multi_class=False haben wir Sigmoid-Ausgaben
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba.flatten(), pos_label=1)

                roc_auc = auc(fpr, tpr)

                roc_curve_data["binary"] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds
                }
                auc_scores["binary"] = roc_auc

                logger.info(f"AUC für binäre Klassifikation: {roc_auc:.4f}")

            # Für Mehrklassen-Klassifikation (One-vs-Rest)
            else:
                # Berechne ROC-Kurve und AUC für jede Klasse
                for i, class_name in enumerate(class_names):
                    # One-vs-Rest: aktuelle Klasse vs. alle anderen
                    y_test_binary = (y_true == i).astype(int)
                    y_score = y_pred_proba[:, i]

                    fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
                    roc_auc = auc(fpr, tpr)

                    roc_curve_data[str(i)] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds
                    }
                    auc_scores[str(i)] = roc_auc

                    logger.info(f"AUC für Klasse '{class_name}': {roc_auc:.4f}")

                # Berechne auch den gewichteten Durchschnitt der AUC-Werte
                try:
                    weighted_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    auc_scores["weighted"] = weighted_auc
                    logger.info(f"Gewichteter AUC: {weighted_auc:.4f}")
                except Exception as auc_error:
                    logger.warning(f"Konnte gewichteten AUC nicht berechnen: {str(auc_error)}")

        except Exception as roc_error:
            logger.warning(f"Konnte ROC-Kurve nicht berechnen: {str(roc_error)}")
            roc_curve_data = None
            auc_scores = None

        # Erstelle und gib das Ergebnisobjekt zurück
        results = NNModelResults(
            model=model,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            report=report,
            conf_matrix=conf_matrix,
            class_names=class_names,
            training_history=history,
            device=str(device),
            roc_curve_data=roc_curve_data,
            auc_scores=auc_scores
        )

        logger.info(f"Training abgeschlossen. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return results

    except Exception as e:
        logger.error(f"Fehler beim Training des neuronalen Netzes: {str(e)}")
        logger.error(traceback.format_exc())

        # Werfe den Fehler weiter, anstatt ein leeres Ergebnisobjekt zurückzugeben
        raise ValueError(f"Fehler beim Training des neuronalen Netzes: {str(e)}")
