"""
Hilfsfunktionen und Konfiguration für die Textklassifikationsanwendung.

Dieses Paket enthält Hilfsfunktionen und Konfigurationseinstellungen für die
Anwendung, wie Logging-Funktionen und Pfadkonfigurationen.
"""

# Importiere wichtige Funktionen und Konstanten aus dem config-Modul
from .config import (
    setup_logging,
    get_model_path,
    get_plot_path,
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    PLOTS_DIR,
    CHECKPOINTS_DIR,
    DEFAULT_DATA_FILE,
    DEFAULT_LOGREG_CONFIG,
    DEFAULT_NN_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    GUI_CONFIG
)

# Definiere, was bei "from suppervisedlearningproject.utils import *" importiert wird
__all__ = [
    'setup_logging',
    'get_model_path',
    'get_plot_path',
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'PLOTS_DIR',
    'CHECKPOINTS_DIR',
    'DEFAULT_DATA_FILE',
    'DEFAULT_LOGREG_CONFIG',
    'DEFAULT_NN_CONFIG',
    'DEFAULT_TRAINING_CONFIG',
    'GUI_CONFIG'
]
