#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Startskript für die grafische Benutzeroberfläche der Textklassifikationsanwendung.

Dieses Skript dient als Einstiegspunkt für die GUI-Anwendung und startet
die grafische Benutzeroberfläche.
"""

import os
import sys

# Füge das src-Verzeichnis zum Pfad hinzu, damit die Module gefunden werden
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

# Importiere das GUI-Modul
from gui import main

if __name__ == "__main__":
    # Starte die GUI-Anwendung
    main()