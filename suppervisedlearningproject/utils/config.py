"""
Konfigurationsmodul für die Textklassifikationsanwendung.

Dieses Modul stellt zentrale Konfigurationseinstellungen und -funktionen
für die gesamte Anwendung bereit, um Konsistenz und einfache Wartung zu gewährleisten.
"""

import os
import logging
from typing import Dict, Any, Optional
import sys

# Basispfade
# Angepasst für die neue Paketstruktur
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Stellen Sie sicher, dass die erforderlichen Verzeichnisse existieren
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Standarddateipfade
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, 'Daten_UTF8_Clean_encoded.csv')

# Modellkonfigurationen
DEFAULT_LOGREG_CONFIG = {
    'max_iter': 1000,
    'C': 1.0,
    'solver': 'lbfgs'
}

DEFAULT_NN_CONFIG = {
    'hidden_size': 100,
    'num_epochs': 10,
    'batch_size': 64,
    'learning_rate': 0.001,
    'dropout_rate': 0.5
}

# Trainingskonfigurationen
DEFAULT_TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 1000
}

# GUI-Konfigurationen
GUI_CONFIG = {
    'theme': 'arc',
    'font_size': 10,
    'window_size': '1024x768'
}

def setup_logging(module_name: str, log_level: int = logging.INFO, 
                 console_output: bool = True) -> logging.Logger:
    """
    Richtet das Logging für ein Modul ein.
    
    Args:
        module_name: Name des Moduls, für das das Logging eingerichtet wird
        log_level: Logging-Level (Standard: INFO)
        console_output: Ob Logs auch in der Konsole ausgegeben werden sollen
        
    Returns:
        Logger-Instanz für das angegebene Modul
    """
    logger = logging.getLogger(module_name)
    
    # Wenn der Logger bereits Handler hat, geben wir ihn einfach zurück
    if logger.handlers:
        return logger
        
    logger.setLevel(log_level)
    
    # Formatierung der Logeinträge
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Datei-Handler
    log_file = os.path.join(LOGS_DIR, f"{module_name.split('.')[-1]}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Konsolen-Handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_model_path(model_type: str, target_column: str, timestamp: str) -> str:
    """
    Generiert den Pfad für ein zu speicherndes Modell.
    
    Args:
        model_type: Typ des Modells ('logreg' oder 'nn')
        target_column: Name der Zielspalte
        timestamp: Zeitstempel für den Dateinamen
        
    Returns:
        Vollständiger Pfad zur Modelldatei
    """
    if model_type.lower() == 'logreg':
        return os.path.join(MODELS_DIR, f"logreg_{target_column}_{timestamp}_model.pkl")
    elif model_type.lower() == 'nn':
        return os.path.join(CHECKPOINTS_DIR, f"{target_column}_best_model.pt")
    else:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}")

def get_plot_path(plot_type: str, target_column: str, timestamp: str) -> str:
    """
    Generiert den Pfad für eine zu speichernde Visualisierung.
    
    Args:
        plot_type: Typ der Visualisierung ('confusion_matrix' oder 'training_history')
        target_column: Name der Zielspalte
        timestamp: Zeitstempel für den Dateinamen
        
    Returns:
        Vollständiger Pfad zur Bilddatei
    """
    return os.path.join(PLOTS_DIR, f"{plot_type}_{target_column}_{timestamp}.png")