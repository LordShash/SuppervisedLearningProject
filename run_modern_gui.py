#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Startskript für die moderne grafische Benutzeroberfläche der Textklassifikationsanwendung.

Dieses Skript dient als Einstiegspunkt für die moderne PyQt5-basierte GUI-Anwendung
und startet die grafische Benutzeroberfläche mit einem modernen Look-and-Feel.
"""

import os
import sys
import subprocess
import importlib.util

# Importiere die Konfiguration aus dem neuen Paket
from suppervisedlearningproject.utils import setup_logging, GUI_CONFIG

# Konfiguriere das Logging für dieses Skript
logger = setup_logging("run_modern_gui")

# Informiere den Benutzer über die Anforderungen
logger.info("=================================================================")
logger.info("Moderne GUI für die Textklassifikationsanwendung")
logger.info("=================================================================")
logger.info("Diese Anwendung benötigt PyQt5 für die moderne Benutzeroberfläche.")
logger.info("Falls PyQt5 nicht installiert ist, werden Sie aufgefordert, es zu installieren.")
logger.info("Alternativ können Sie auch die Standard-GUI verwenden, die keine")
logger.info("zusätzlichen Abhängigkeiten benötigt.")
logger.info("=================================================================\n")

# Prüfe, ob PyQt5 installiert ist
pyqt5_installed = importlib.util.find_spec("PyQt5") is not None

if not pyqt5_installed:
    logger.warning("PyQt5 ist nicht installiert. Die moderne GUI benötigt PyQt5.")
    logger.info("Sie können PyQt5 mit folgendem Befehl installieren:")
    logger.info("\npip install PyQt5\n")

    # Frage den Benutzer, ob PyQt5 automatisch installiert werden soll
    try:
        user_input = input("Möchten Sie PyQt5 jetzt installieren? (j/n): ")
        if user_input.lower() in ['j', 'ja', 'y', 'yes']:
            logger.info("Installiere PyQt5...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5>=5.15.0"])
                logger.info("PyQt5 wurde erfolgreich installiert!")
                pyqt5_installed = True
            except subprocess.CalledProcessError:
                logger.error("Fehler bei der Installation von PyQt5.")
                logger.error("Bitte installieren Sie PyQt5 manuell mit 'pip install PyQt5'.")
                sys.exit(1)
        else:
            logger.warning("Installation abgebrochen. Die moderne GUI kann nicht gestartet werden.")
            logger.info("Bitte installieren Sie PyQt5 manuell mit 'pip install PyQt5'.")

            # Frage den Benutzer, ob die Standard-GUI gestartet werden soll
            try:
                user_input = input("Möchten Sie stattdessen die Standard-GUI starten? (j/n): ")
                if user_input.lower() in ['j', 'ja', 'y', 'yes']:
                    logger.info("Starte die Standard-GUI...")
                    # Importiere und starte die Standard-GUI
                    try:
                        from suppervisedlearningproject.ui.gui import main as gui_main
                        gui_main()
                        sys.exit(0)
                    except ImportError as e:
                        logger.error(f"Fehler beim Starten der Standard-GUI: {e}")
                        sys.exit(1)
                else:
                    logger.info("Programm wird beendet.")
                    sys.exit(1)
            except KeyboardInterrupt:
                logger.info("\nProgramm wird beendet.")
                sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nInstallation abgebrochen. Die moderne GUI kann nicht gestartet werden.")

        # Frage den Benutzer, ob die Standard-GUI gestartet werden soll
        try:
            user_input = input("Möchten Sie stattdessen die Standard-GUI starten? (j/n): ")
            if user_input.lower() in ['j', 'ja', 'y', 'yes']:
                logger.info("Starte die Standard-GUI...")
                # Importiere und starte die Standard-GUI
                try:
                    from suppervisedlearningproject.ui.gui import main as gui_main
                    gui_main()
                    sys.exit(0)
                except ImportError as e:
                    logger.error(f"Fehler beim Starten der Standard-GUI: {e}")
                    sys.exit(1)
            else:
                logger.info("Programm wird beendet.")
                sys.exit(1)
        except KeyboardInterrupt:
            logger.info("\nProgramm wird beendet.")
            sys.exit(1)

# Importiere das moderne GUI-Modul
try:
    from PyQt5.QtWidgets import QApplication
    from suppervisedlearningproject.ui.modern_gui_complete import ModernTextClassificationGUI
except ImportError as e:
    logger.error(f"Fehler beim Importieren der benötigten Module: {e}")
    logger.error("Bitte stellen Sie sicher, dass alle Abhängigkeiten installiert sind.")

    # Frage den Benutzer, ob die Standard-GUI gestartet werden soll
    try:
        user_input = input("Möchten Sie stattdessen die Standard-GUI starten? (j/n): ")
        if user_input.lower() in ['j', 'ja', 'y', 'yes']:
            logger.info("Starte die Standard-GUI...")
            # Importiere und starte die Standard-GUI
            try:
                from suppervisedlearningproject.ui.gui import main as gui_main
                gui_main()
                sys.exit(0)
            except ImportError as e2:
                logger.error(f"Fehler beim Starten der Standard-GUI: {e2}")
                sys.exit(1)
        else:
            logger.info("Programm wird beendet.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nProgramm wird beendet.")
        sys.exit(1)

def main():
    """
    Hauptfunktion zum Starten der modernen GUI.
    """
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
    window = ModernTextClassificationGUI()

    # Setze das Icon auch für das Hauptfenster
    if 'app_icon' in locals():
        window.setWindowIcon(app_icon)

    window.show()

    # Starte die Anwendung
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
