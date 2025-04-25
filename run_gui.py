#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Startskript für die grafische Benutzeroberfläche der Textklassifikationsanwendung.

Dieses Skript dient als Einstiegspunkt für die moderne PyQt5-basierte GUI-Anwendung
und startet die grafische Benutzeroberfläche mit einem modernen Look-and-Feel.
"""

import os
import sys
import subprocess
import importlib.util
import time

# Importiere die Konfiguration aus dem neuen Paket
from suppervisedlearningproject.utils import setup_logging, GUI_CONFIG

# Konfiguriere das Logging für dieses Skript
logger = setup_logging("run_gui")

# Informiere den Benutzer über die Anforderungen
logger.info("=================================================================")
logger.info("Moderne GUI für die Textklassifikationsanwendung")
logger.info("=================================================================")
logger.info("Prüfe und installiere benötigte Abhängigkeiten automatisch...")
logger.info("=================================================================\n")

def check_and_install_dependencies():
    """
    Prüft, ob alle benötigten Abhängigkeiten installiert sind und installiert fehlende Pakete automatisch.

    Returns:
        bool: True, wenn alle Abhängigkeiten verfügbar sind (nach Installation), sonst False
    """
    # Liste der benötigten Pakete
    required_packages = [
        ("PyQt5", "PyQt5>=5.15.0"),
        ("PyQt5.QtChart", "PyQtChart>=5.15.0")
    ]

    all_installed = True
    packages_to_install = []

    # Prüfe, welche Pakete fehlen
    for package_name, install_spec in required_packages:
        try:
            # Versuche, das Paket zu importieren
            if importlib.util.find_spec(package_name) is None:
                logger.warning(f"{package_name} ist nicht installiert.")
                packages_to_install.append(install_spec)
                all_installed = False
            else:
                logger.info(f"{package_name} ist bereits installiert.")
        except Exception as e:
            logger.error(f"Fehler beim Prüfen von {package_name}: {e}")
            packages_to_install.append(install_spec)
            all_installed = False

    # Installiere fehlende Pakete
    if not all_installed:
        logger.info("Installiere fehlende Abhängigkeiten...")

        # Entferne Duplikate aus der Liste
        packages_to_install = list(set(packages_to_install))

        for package_spec in packages_to_install:
            try:
                logger.info(f"Installiere {package_spec}...")
                # Stille Installation ohne Benutzerinteraktion
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package_spec],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info(f"{package_spec} wurde erfolgreich installiert!")
            except subprocess.CalledProcessError as e:
                logger.error(f"Fehler bei der Installation von {package_spec}: {e}")
                return False
            except Exception as e:
                logger.error(f"Unerwarteter Fehler bei der Installation von {package_spec}: {e}")
                return False

        # Kurze Pause, um sicherzustellen, dass die Installation abgeschlossen ist
        time.sleep(1)

        # Prüfe erneut, ob alle Pakete jetzt installiert sind
        for package_name, _ in required_packages:
            if importlib.util.find_spec(package_name) is None:
                logger.error(f"{package_name} konnte nicht installiert werden.")
                return False

        logger.info("Alle benötigten Abhängigkeiten wurden erfolgreich installiert!")

    return True

# Prüfe und installiere Abhängigkeiten
dependencies_ok = check_and_install_dependencies()

if not dependencies_ok:
    logger.error("Die benötigten Abhängigkeiten konnten nicht installiert werden.")
    logger.error("Die Anwendung kann nicht gestartet werden.")
    sys.exit(0)  # Beende ohne Fehlercode, aber mit Logging

# Versuche, die moderne GUI zu importieren und zu starten
try:
    # Importiere die benötigten Module
    from PyQt5.QtWidgets import QApplication
    from suppervisedlearningproject.ui.gui_complete import ModernTextClassificationGUI
except ImportError as e:
    logger.error(f"Fehler beim Importieren der benötigten Module trotz Installation: {e}")
    logger.error("Die Anwendung kann nicht gestartet werden.")
    sys.exit(0)  # Beende ohne Fehlercode, aber mit Logging

def main():
    """
    Hauptfunktion zum Starten der modernen GUI.

    Diese Funktion erstellt die QApplication, setzt das App-Icon (falls verfügbar)
    und startet die moderne GUI ohne weitere Benutzerinteraktion.
    """
    try:
        # Erstelle die QApplication
        app = QApplication(sys.argv)

        # Setze das App-Icon
        try:
            # Importiere die get_app_icon-Funktion aus dem icons-Paket
            from suppervisedlearningproject.icons import get_app_icon

            # Setze das Icon für die Anwendung
            app_icon = get_app_icon()
            app.setWindowIcon(app_icon)
        except Exception as e:
            logger.warning(f"Hinweis: App-Icon konnte nicht geladen werden: {str(e)}")
            logger.info("Die Anwendung wird ohne Icon gestartet.")

        # Erstelle und zeige das Hauptfenster
        logger.info("Starte die moderne GUI...")
        window = ModernTextClassificationGUI()

        # Setze das Icon auch für das Hauptfenster
        if 'app_icon' in locals():
            window.setWindowIcon(app_icon)

        # Zeige das Hauptfenster
        window.show()

        # Starte die Anwendung
        return app.exec_()
    except Exception as e:
        logger.error(f"Fehler beim Starten der GUI: {e}")
        return 1

if __name__ == "__main__":
    # Starte die GUI direkt ohne weitere Benutzerinteraktion
    sys.exit(main())
