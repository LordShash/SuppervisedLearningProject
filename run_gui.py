#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Startskript für die grafische Benutzeroberfläche der Textklassifikationsanwendung.

Dieses Skript dient als Einstiegspunkt für die GUI-Anwendung und startet
die grafische Benutzeroberfläche.
"""

import os
import sys

# Importiere die Konfiguration aus dem neuen Paket
from suppervisedlearningproject.utils import setup_logging

# Konfiguriere das Logging für dieses Skript
logger = setup_logging("run_gui")

# Importiere das GUI-Modul aus dem neuen Paket
from suppervisedlearningproject.ui.gui import main

if __name__ == "__main__":
    # Starte die GUI-Anwendung
    main()
