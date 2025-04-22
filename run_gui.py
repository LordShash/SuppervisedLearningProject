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
# Dies ermöglicht den direkten Import von Modulen aus dem src-Verzeichnis
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

# Importiere das GUI-Modul
# Hinweis: Der Import funktioniert, weil das src-Verzeichnis zum Pfad hinzugefügt wurde
# IDE-Hinweis: Das Modul 'gui' befindet sich in 'src/gui.py'
from gui import main

if __name__ == "__main__":
    # Starte die GUI-Anwendung
    main()
