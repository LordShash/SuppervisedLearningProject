#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul zum Training eines neuronalen Netzwerks.

Dieses Modul stellt Funktionen und Klassen bereit, um ein neuronales Netzwerk
zu definieren, zu trainieren, zu evaluieren und zu speichern.
"""

import os
import sys
import logging
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

# Importiere die Datenladefunktion aus dem data_loader Modul
from data_loader import load_data

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ausgabe in die Konsole
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'train_nn.log'), 
                           mode='a', encoding='utf-8', delay=True)  # Ausgabe in eine Datei
    ]
)
logger = logging.getLogger(__name__)

# Stelle sicher, dass das Logs-Verzeichnis existiert
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)

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


def prepare_data(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 42,
    batch_size: int = 64,
    val_size: Optional[float] = None,
    standardize: bool = True,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Bereitet die Daten für das Training des neuronalen Netzes vor.

    Args:
        X: Feature-Matrix (kann eine Sparse-Matrix sein)
        y: Zielvariable
        test_size: Anteil der Testdaten
        random_state: Seed für die Reproduzierbarkeit
        batch_size: Größe der Batches für DataLoader
        val_size: Anteil der Validierungsdaten (falls None, wird der Test-Set für Validierung verwendet)
        standardize: Flag, ob die Daten standardisiert werden sollen
        class_names: Liste mit Namen für die Klassen (falls None, werden automatisch generiert)

    Returns:
        Dict[str, Any]: Dictionary mit DataLoadern, Dimensionen und Metadaten:
            - 'train_loader': DataLoader für Trainingsdaten
            - 'val_loader': DataLoader für Validierungsdaten
            - 'test_loader': DataLoader für Testdaten
            - 'input_dim': Anzahl der Features
            - 'num_classes': Anzahl der Klassen
            - 'class_names': Liste mit Klassennamen
            - 'scaler': Standardisierungs-Objekt (falls standardize=True)
            - 'unique_classes': Liste der einzigartigen Klassen
    """
    logger.info(f"Bereite Daten vor: X.shape={X.shape}, y.shape={y.shape}, test_size={test_size}")

    # Bestimme einzigartige Klassen und ihre Anzahl
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    logger.info(f"Gefundene Klassen: {num_classes} ({unique_classes})")

    # Generiere Klassennamen, falls nicht angegeben
    if class_names is None:
        class_names = [f"class_{i}" for i in unique_classes]

    # Prüfe, ob die Anzahl der Klassennamen mit der Anzahl der Klassen übereinstimmt
    if len(class_names) != num_classes:
        logger.warning(f"Anzahl der Klassennamen ({len(class_names)}) stimmt nicht mit der Anzahl der Klassen ({num_classes}) überein")
        class_names = [f"class_{i}" for i in unique_classes]

    # Wenn val_size angegeben ist, teile die Daten in Train, Val und Test auf
    if val_size is not None:
        # Zuerst in Train+Val und Test aufteilen
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Dann Train+Val in Train und Val aufteilen
        # Berechne den relativen Anteil von val_size an (1-test_size)
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
        )

        logger.info(f"Daten aufgeteilt: {X_train.shape[0]} Train, {X_val.shape[0]} Val, {X_test.shape[0]} Test")
    else:
        # Ohne val_size teilen wir nur in Train und Test auf (Test wird auch für Validierung verwendet)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_val, y_val = X_test, y_test

        logger.info(f"Daten aufgeteilt: {X_train.shape[0]} Train, {X_test.shape[0]} Test/Val")

    # Sparse-Matrix in Dense-Matrix umwandeln, falls nötig
    if sparse.issparse(X_train):
        logger.info("Konvertiere sparse Matrix zu dense Matrix")
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    # Standardisierung der Features, falls gewünscht
    scaler = None
    if standardize:
        logger.info("Standardisiere Features")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Multiclass-Klassifikation
    if num_classes > 2:
        logger.info("Multiclass-Klassifikation erkannt")
        # Für Multiclass-Klassifikation verwenden wir direkt die Klassenindizes
        # Wir müssen sicherstellen, dass die Klassen von 0 bis num_classes-1 gehen
        class_mapping = {c: i for i, c in enumerate(unique_classes)}
        y_train = np.array([class_mapping[c] for c in y_train])
        y_val = np.array([class_mapping[c] for c in y_val])
        y_test = np.array([class_mapping[c] for c in y_test])
    # Binäre Klassifikation
    else:
        logger.info("Binäre Klassifikation erkannt")
        # Für binäre Klassifikation verwenden wir 0 und 1
        y_train = (y_train == unique_classes[1]).astype(np.float32)
        y_val = (y_val == unique_classes[1]).astype(np.float32)
        y_test = (y_test == unique_classes[1]).astype(np.float32)

    # Konvertierung zu PyTorch-Tensoren
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    logger.info(f"Tensoren erstellt: X_train: {X_train_tensor.shape}, y_train: {y_train_tensor.shape}")

    # Erstellung der Datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Erstellung der DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"DataLoader erstellt: {len(train_loader)} Train-Batches, {len(val_loader)} Val-Batches, {len(test_loader)} Test-Batches")

    # Rückgabe aller relevanten Daten als Dictionary
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_dim': X_train.shape[1],
        'num_classes': num_classes,
        'class_names': class_names,
        'scaler': scaler,
        'unique_classes': unique_classes.tolist()
    }


def train_neural_network(
    data: Dict[str, Any],
    target_name: str = "Fits_Topic_Code",
    epochs: int = 50,
    patience: int = 5,
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
    checkpoint_dir: Optional[str] = None
) -> NNModelResults:
    """
    Trainiert ein neuronales Netzwerk mit den gegebenen Daten und erweiterten Funktionen.

    Args:
        data: Dictionary mit DataLoadern und Metadaten (von prepare_data)
        target_name: Name der Zielspalte für Ausgabezwecke
        epochs: Maximale Anzahl der Trainingsepochen
        patience: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird
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
        checkpoint_dir: Verzeichnis für Checkpoints (falls None, wird ein Standardverzeichnis verwendet)

    Returns:
        NNModelResults: Objekt mit trainiertem Modell und allen Evaluationsmetriken
    """
    # Extrahiere Daten aus dem Dictionary
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    input_dim = data['input_dim']
    num_classes = data['num_classes']
    class_names = data['class_names']

    # Gerät für das Training bestimmen (CPU oder GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Gerät: {device}")

    # Speicherverbrauch überwachen
    if device.type == 'cuda':
        logger.info(f"GPU-Speicher vor dem Training: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # Modell initialisieren
    multi_class = num_classes > 2
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
    model.to(device)
    logger.info(f"Modell erstellt und auf {device} verschoben")

    # Verlustfunktion definieren
    if multi_class:
        criterion = nn.CrossEntropyLoss()
        logger.info("Verwende CrossEntropyLoss für Multiclass-Klassifikation")
    else:
        criterion = nn.BCELoss()
        logger.info("Verwende BCELoss für binäre Klassifikation")

    # Optimizer mit Weight Decay (L2-Regularisierung)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info(f"Optimizer: Adam mit lr={learning_rate}, weight_decay={weight_decay}")

    # Learning Rate Scheduler
    scheduler = None
    if lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True
        )
        logger.info(f"Learning Rate Scheduler: ReduceLROnPlateau mit factor={lr_factor}, patience={lr_patience}")

    # Checkpoint-Verzeichnis erstellen, falls nötig
    if save_checkpoints:
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoints werden gespeichert in: {checkpoint_dir}")

    # Early Stopping Variablen
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    early_stop = False

    # Trainingshistorie für Visualisierung
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Training des Modells
    logger.info(f"Starte Training für {epochs} Epochen mit Early Stopping (patience={patience})")
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward Pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Verlust berechnen
            if multi_class:
                loss = criterion(outputs, labels.long())
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels.long()).sum().item()
            else:
                loss = criterion(outputs.squeeze(), labels)
                predicted = (outputs.squeeze() > 0.5).float()
                train_correct += (predicted == labels).sum().item()

            train_total += labels.size(0)

            # Backward Pass und Optimierung
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Durchschnittlicher Verlust und Accuracy für diese Epoche
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)

        # Validierung
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Verlust berechnen
                if multi_class:
                    loss = criterion(outputs, labels.long())
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels.long()).sum().item()
                else:
                    loss = criterion(outputs.squeeze(), labels)
                    predicted = (outputs.squeeze() > 0.5).float()
                    val_correct += (predicted == labels).sum().item()

                val_total += labels.size(0)
                val_loss += loss.item() * inputs.size(0)

        # Durchschnittlicher Validierungsverlust und Accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Learning Rate Scheduler aktualisieren
        if scheduler is not None:
            scheduler.step(val_loss)

        # Ausgabe des Fortschritts
        logger.info(f'Epoche {epoch+1}/{epochs}, Train-Loss: {epoch_loss:.4f}, Val-Loss: {val_loss:.4f}, '
                   f'Train-Acc: {train_acc:.4f}, Val-Acc: {val_acc:.4f}')

        # Checkpoint speichern, falls es das beste Modell ist
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0

            if save_checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, f"{target_name}_best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'model_config': model.get_config()
                }, checkpoint_path)
                logger.info(f"Bestes Modell gespeichert: {checkpoint_path}")
        else:
            epochs_no_improve += 1

        # Early Stopping Check
        if epochs_no_improve >= patience:
            logger.info(f'Early Stopping nach {epoch+1} Epochen')
            early_stop = True
            break

    # Lade das beste Modell zurück
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Bestes Modell wiederhergestellt")

    # Visualisierung des Trainingsfortschritts
    if visualize:
        try:
            # Erstelle Verzeichnis für Plots
            plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            # Plot für Verlust
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'Loss für {target_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plot für Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.title(f'Accuracy für {target_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Speichere den Plot
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f'training_history_{target_name}_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Trainingshistorie-Plot gespeichert: {plot_path}")
        except Exception as e:
            logger.warning(f"Fehler bei der Visualisierung: {str(e)}")

    # Evaluation des Modells auf dem Testset
    logger.info("Evaluiere Modell auf dem Testset")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if multi_class:
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
            else:
                predicted = (outputs.squeeze() > 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

    # Berechnung der Metriken
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    logger.info(f"\nFinale Metriken für Zielspalte '{target_name}':")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Detaillierter Klassifikationsbericht
    report = classification_report(all_targets, all_preds, target_names=class_names)
    logger.info(f"\nKlassifikationsbericht:\n{report}")

    # Erstellung der Konfusionsmatrix
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # Visualisierung der Konfusionsmatrix
    if visualize:
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Konfusionsmatrix für {target_name}')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            plt.xlabel('Vorhergesagte Klasse')
            plt.ylabel('Tatsächliche Klasse')
            plt.tight_layout()

            # Speichere die Visualisierung
            conf_matrix_path = os.path.join(plots_dir, f'confusion_matrix_{target_name}_{timestamp}.png')
            plt.savefig(conf_matrix_path)
            plt.close()
            logger.info(f"Konfusionsmatrix-Visualisierung gespeichert: {conf_matrix_path}")
        except Exception as e:
            logger.warning(f"Fehler bei der Visualisierung der Konfusionsmatrix: {str(e)}")

    # Speicherbereinigung, falls GPU verwendet wird
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info(f"GPU-Speicher nach dem Training: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # Erstelle und gib ModelResults-Objekt zurück
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
        device=device.type
    )

    return results


def save_model(
    model: nn.Module, 
    target_column: str = "Fits_Topic_Code",
    version: Optional[str] = None,
    include_timestamp: bool = True,
    include_metadata: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = True
) -> str:
    """
    Speichert das trainierte PyTorch-Modell mit Metadaten.

    Args:
        model: Trainiertes Modell
        target_column: Name der Zielspalte für den Dateinamen
        version: Optionale Versionsbezeichnung (z.B. "v1.0")
        include_timestamp: Wenn True, wird ein Zeitstempel im Dateinamen verwendet
        include_metadata: Wenn True, werden Metadaten mit dem Modell gespeichert
        metadata: Dictionary mit zusätzlichen Metadaten
        overwrite: Wenn True, werden vorhandene Dateien überschrieben

    Returns:
        str: Pfad zur gespeicherten Modelldatei

    Raises:
        ValueError: Wenn Probleme beim Speichern des Modells auftreten
        FileExistsError: Wenn die Datei bereits existiert und overwrite=False
    """
    # Pfad zum models-Verzeichnis
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    # Verzeichnis erstellen, falls es nicht existiert
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Speichere Modell in Verzeichnis: {models_dir}")

    # Generierung des Dateinamens basierend auf der Zielspalte und optionalen Parametern
    timestamp = ""
    if include_timestamp:
        timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    version_str = ""
    if version:
        version_str = f"_{version}"

    filename = f"nn_{target_column}{version_str}{timestamp}_model.pt"
    model_path = os.path.join(models_dir, filename)

    # Prüfen, ob die Datei bereits existiert
    if os.path.exists(model_path) and not overwrite:
        error_msg = f"Die Modelldatei '{model_path}' existiert bereits und overwrite=False."
        logger.error(error_msg)
        raise FileExistsError(error_msg)

    try:
        # Vorbereitung der Metadaten
        model_info = {
            'model_type': 'NeuralNetwork',
            'target_column': target_column,
            'created_at': datetime.datetime.now().isoformat(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
        }

        # Modellkonfiguration hinzufügen, falls verfügbar
        if hasattr(model, 'get_config'):
            model_info['model_config'] = model.get_config()

        # Hinzufügen der zusätzlichen Metadaten, falls vorhanden
        if include_metadata and metadata:
            model_info.update(metadata)

        # Speicherung des Modells mit oder ohne Metadaten
        if include_metadata:
            # Speicherung des Modells mit Metadaten
            save_data = {
                'model_state_dict': model.state_dict(),
                'metadata': model_info
            }
            torch.save(save_data, model_path)
            logger.info(f"Modell mit Metadaten gespeichert unter: {model_path}")
        else:
            # Nur das Modell speichern (für Kompatibilität mit älteren Versionen)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Modell ohne Metadaten gespeichert unter: {model_path}")

        return model_path

    except Exception as e:
        error_msg = f"Fehler beim Speichern des Modells: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


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
    try:
        logger.info(f"Starte Training und Speicherung für Zielspalte '{target_column}'")

        # Daten laden mit konfigurierbaren max_features
        logger.info(f"Lade Daten mit max_features={max_features}")
        X, y = load_data(target_column=target_column, max_features=max_features)
        logger.info(f"Daten geladen: X.shape={X.shape}, y.shape={y.shape}")

        # Daten vorbereiten mit konfigurierbaren Parametern
        logger.info(f"Bereite Daten vor mit test_size={test_size}, val_size={val_size}, batch_size={batch_size}")
        data = prepare_data(
            X, y, 
            test_size=test_size, 
            val_size=val_size,
            batch_size=batch_size,
            standardize=standardize,
            class_names=class_names
        )
        logger.info(f"Daten vorbereitet: {len(data['train_loader'])} Train-Batches, {len(data['val_loader'])} Val-Batches, {len(data['test_loader'])} Test-Batches")

        # Neuronales Netz trainieren mit erweiterten Funktionen
        logger.info("Starte Modelltraining")
        results = train_neural_network(
            data=data,
            target_name=target_column,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            visualize=visualize,
            save_checkpoints=save_checkpoints
        )
        logger.info("Modelltraining abgeschlossen")

        # Modell speichern, falls gewünscht
        if save_model_flag:
            # Erstelle ein Dictionary mit den Metriken für die Speicherung
            metadata = {
                'accuracy': results.accuracy,
                'precision': results.precision,
                'recall': results.recall,
                'f1': results.f1,
                'training_parameters': {
                    'epochs': epochs,
                    'max_features': max_features,
                    'test_size': test_size,
                    'val_size': val_size,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_dims': hidden_dims,
                    'dropout_rate': dropout_rate,
                    'activation': activation,
                    'batch_norm': batch_norm,
                    'weight_decay': weight_decay,
                    'lr_scheduler': lr_scheduler,
                    'patience': patience
                }
            }

            # Speichere das Modell mit den Metriken
            model_path = save_model(
                model=results.model,
                target_column=target_column,
                version=version,
                include_timestamp=True,
                include_metadata=True,
                metadata=metadata,
                overwrite=overwrite
            )
            logger.info(f"Modell gespeichert unter: {model_path}")
        else:
            logger.info("Modell wurde nicht gespeichert (save_model_flag=False)")

        # Rückgabe der ModelResults
        return results

    except (FileNotFoundError, ValueError, KeyError, FileExistsError) as e:
        # Spezifische Fehler weiterleiten
        logger.error(f"Fehler beim Training und Speichern des Modells: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        # Unerwartete Fehler in ValueError umwandeln
        error_msg = f"Unerwarteter Fehler beim Training und Speichern des Modells: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


if __name__ == "__main__":
    import argparse

    # Kommandozeilenargumente parsen
    parser = argparse.ArgumentParser(description="Trainiert und speichert ein neuronales Netzwerk")
    parser.add_argument("--target", default="Fits_Topic_Code", 
                        help="Zielspalte für das Training (Standard: Fits_Topic_Code)")
    parser.add_argument("--max-features", type=int, default=1000, 
                        help="Maximale Anzahl der Features für TF-IDF (Standard: 1000)")
    parser.add_argument("--test-size", type=float, default=0.2, 
                        help="Anteil der Testdaten (Standard: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1, 
                        help="Anteil der Validierungsdaten (Standard: 0.1)")
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Größe der Batches (Standard: 64)")
    parser.add_argument("--epochs", type=int, default=30, 
                        help="Maximale Anzahl der Trainingsepochen (Standard: 30)")
    parser.add_argument("--patience", type=int, default=5, 
                        help="Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird (Standard: 5)")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Initiale Lernrate (Standard: 0.001)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128], 
                        help="Anzahl der Neuronen in den Hidden-Layern (Standard: 128 128)")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout-Rate (Standard: 0.2)")
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "elu", "tanh"], default="relu", 
                        help="Aktivierungsfunktion (Standard: relu)")
    parser.add_argument("--no-batch-norm", action="store_true", 
                        help="Deaktiviert Batch-Normalisierung")
    parser.add_argument("--weight-decay", type=float, default=0.0001, 
                        help="L2-Regularisierungsparameter (Standard: 0.0001)")
    parser.add_argument("--no-lr-scheduler", action="store_true", 
                        help="Deaktiviert Learning Rate Scheduler")
    parser.add_argument("--lr-factor", type=float, default=0.5, 
                        help="Faktor für Learning Rate Scheduler (Standard: 0.5)")
    parser.add_argument("--lr-patience", type=int, default=3, 
                        help="Patience für Learning Rate Scheduler (Standard: 3)")
    parser.add_argument("--no-visualize", action="store_true", 
                        help="Deaktiviert Visualisierung")
    parser.add_argument("--no-checkpoints", action="store_true", 
                        help="Deaktiviert Speicherung von Checkpoints")
    parser.add_argument("--version", 
                        help="Versionsbezeichnung für das Modell")
    parser.add_argument("--no-overwrite", action="store_true", 
                        help="Vorhandene Modelldateien nicht überschreiben")
    parser.add_argument("--no-standardize", action="store_true", 
                        help="Deaktiviert Standardisierung der Features")
    parser.add_argument("--no-save", action="store_true", 
                        help="Modell nicht speichern")
    parser.add_argument("--multi-target", action="store_true", 
                        help="Trainiere Modelle für mehrere Zielspalten")
    parser.add_argument("--verbose", action="store_true", 
                        help="Ausführliche Ausgabe")

    args = parser.parse_args()

    # Logging-Level anpassen, falls verbose aktiviert ist
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug-Modus aktiviert")

    try:
        # Wenn multi-target aktiviert ist, trainiere Modelle für mehrere Zielspalten
        if args.multi_target:
            logger.info("Trainiere Modelle für mehrere Zielspalten")

            # Verfügbare Zielspalten ermitteln
            from data_loader import get_available_targets
            targets = get_available_targets()

            # Ergebnisse für jede Zielspalte speichern
            results_dict = {}

            # Für jede verfügbare Zielspalte ein Modell trainieren
            for target_column in targets.keys():
                logger.info(f"\n\n{'=' * 50}")
                logger.info(f"Training für Zielspalte: {target_column}")
                logger.info(f"{'=' * 50}")

                try:
                    # Training des Modells mit den angegebenen Parametern
                    results = train_and_save_model(
                        target_column=target_column,
                        epochs=args.epochs,
                        max_features=args.max_features,
                        test_size=args.test_size,
                        val_size=args.val_size,
                        batch_size=args.batch_size,
                        patience=args.patience,
                        learning_rate=args.learning_rate,
                        hidden_dims=args.hidden_dims,
                        dropout_rate=args.dropout,
                        activation=args.activation,
                        batch_norm=not args.no_batch_norm,
                        weight_decay=args.weight_decay,
                        lr_scheduler=not args.no_lr_scheduler,
                        lr_factor=args.lr_factor,
                        lr_patience=args.lr_patience,
                        visualize=not args.no_visualize,
                        save_checkpoints=not args.no_checkpoints,
                        version=args.version,
                        overwrite=not args.no_overwrite,
                        standardize=not args.no_standardize,
                        save_model_flag=not args.no_save
                    )

                    # Ergebnisse speichern
                    results_dict[target_column] = {
                        'accuracy': results.accuracy,
                        'precision': results.precision,
                        'recall': results.recall,
                        'f1': results.f1
                    }

                except Exception as target_error:
                    logger.error(f"Fehler beim Training für Zielspalte '{target_column}': {str(target_error)}")
                    results_dict[target_column] = {'error': str(target_error)}

            # Zusammenfassung der Ergebnisse
            logger.info("\n\n" + "=" * 50)
            logger.info("Zusammenfassung der Trainingsergebnisse:")
            logger.info("=" * 50)

            for target, result in results_dict.items():
                if 'error' in result:
                    logger.info(f"Zielspalte '{target}': Fehlgeschlagen - {result['error']}")
                else:
                    logger.info(f"Zielspalte '{target}':")
                    logger.info(f"  Accuracy: {result['accuracy']:.4f}")
                    logger.info(f"  Precision: {result['precision']:.4f}")
                    logger.info(f"  Recall: {result['recall']:.4f}")
                    logger.info(f"  F1 Score: {result['f1']:.4f}")

        else:
            # Training eines einzelnen Modells mit den angegebenen Parametern
            logger.info(f"Trainiere Modell für Zielspalte '{args.target}'")

            results = train_and_save_model(
                target_column=args.target,
                epochs=args.epochs,
                max_features=args.max_features,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                patience=args.patience,
                learning_rate=args.learning_rate,
                hidden_dims=args.hidden_dims,
                dropout_rate=args.dropout,
                activation=args.activation,
                batch_norm=not args.no_batch_norm,
                weight_decay=args.weight_decay,
                lr_scheduler=not args.no_lr_scheduler,
                lr_factor=args.lr_factor,
                lr_patience=args.lr_patience,
                visualize=not args.no_visualize,
                save_checkpoints=not args.no_checkpoints,
                version=args.version,
                overwrite=not args.no_overwrite,
                standardize=not args.no_standardize,
                save_model_flag=not args.no_save
            )

            # Ausgabe der Ergebnisse
            logger.info("\nZusammenfassung der Ergebnisse:")
            logger.info(f"Accuracy: {results.accuracy:.4f}")
            logger.info(f"Precision: {results.precision:.4f}")
            logger.info(f"Recall: {results.recall:.4f}")
            logger.info(f"F1 Score: {results.f1:.4f}")

            logger.info("\nTraining erfolgreich abgeschlossen!")

    except Exception as e:
        # Fehlerbehandlung bei Problemen während der Ausführung
        logger.error(f"Fehler: {str(e)}", exc_info=True)
        sys.exit(1)

    sys.exit(0)
