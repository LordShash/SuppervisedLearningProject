"""
Core-Funktionalität für die Textklassifikationsanwendung.

Dieses Paket enthält die Kernfunktionalität der Anwendung, wie das Laden von Daten
und das Laden von Modellen.
"""

# Importiere wichtige Funktionen aus dem data_loader-Modul
from .data_loader import (
    load_data,
    get_available_targets,
    clear_tfidf_cache
)

# Importiere wichtige Funktionen aus dem model_loader-Modul
from .model_loader import (
    load_model,
    load_model_with_info
)

# Definiere, was bei "from suppervisedlearningproject.core import *" importiert wird
__all__ = [
    'load_data',
    'get_available_targets',
    'clear_tfidf_cache',
    'load_model',
    'load_model_with_info'
]
