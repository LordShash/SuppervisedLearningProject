#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testskript für das Hauptmodul.

Dieses Skript testet das Hauptmodul mit einer minimalen Konfiguration,
um sicherzustellen, dass alles wie erwartet funktioniert.
"""

import os
import sys
import subprocess

def run_test_for_model(model_type):
    """
    Führt einen Test des Hauptmoduls für einen bestimmten Modelltyp durch.

    Args:
        model_type (str): Der zu testende Modelltyp ('logreg', 'nn' oder 'all')

    Returns:
        bool: True, wenn der Test erfolgreich war, sonst False
    """
    # Standardisierter deutscher Kommentar: Start des Tests für ein bestimmtes Modell
    print(f"Starte Test des Hauptmoduls mit Modell '{model_type}'...")

    # Pfad zum Hauptmodul (angepasst für die neue Verzeichnisstruktur)
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "main.py")

    # Standardisierter deutscher Kommentar: Prüfung der Existenz der Datei
    if not os.path.exists(main_path):
        print(f"Fehler: Die Datei '{main_path}' wurde nicht gefunden.")
        return False

    # Standardisierter deutscher Kommentar: Vorbereitung des Kommandos
    cmd = [sys.executable, main_path, "--model", model_type, "--target", "Fits_Topic_Code"]

    try:
        # Standardisierter deutscher Kommentar: Ausführung des Hauptmoduls
        print(f"Führe Kommando aus: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Standardisierter deutscher Kommentar: Anzeige der Ausgabe
        print("\nAusgabe:")
        print(process.stdout)

        # Standardisierter deutscher Kommentar: Prüfung des Ergebnisses
        if process.returncode == 0:
            print(f"\nTest für Modell '{model_type}' erfolgreich abgeschlossen!")
            return True
        else:
            print(f"\nFehler: Das Hauptmodul wurde mit Rückgabecode {process.returncode} beendet.")
            print(f"Fehlerausgabe: {process.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        # Standardisierter deutscher Kommentar: Fehlerbehandlung bei Prozessausführung
        print(f"\nFehler beim Ausführen des Hauptmoduls: {str(e)}")
        print(f"Fehlerausgabe: {e.stderr}")
        return False

    except Exception as e:
        # Standardisierter deutscher Kommentar: Allgemeine Fehlerbehandlung
        print(f"\nUnerwarteter Fehler: {str(e)}")
        return False


def run_test():
    """
    Führt Tests des Hauptmoduls für alle verfügbaren Modellkonfigurationen durch.

    Returns:
        bool: True, wenn alle Tests erfolgreich waren, sonst False
    """
    # Standardisierter deutscher Kommentar: Start der Tests
    print("Starte Tests des Hauptmoduls für alle Modellkonfigurationen...")

    # Liste der zu testenden Modelltypen
    model_types = ["logreg", "nn", "all"]

    # Ergebnisse für jeden Modelltyp speichern
    results = {}

    # Jeden Modelltyp testen
    for model_type in model_types:
        print(f"\n{'=' * 50}")
        print(f"Test für Modelltyp: {model_type}")
        print(f"{'=' * 50}")
        results[model_type] = run_test_for_model(model_type)

    # Zusammenfassung der Ergebnisse
    print("\n\n" + "=" * 50)
    print("Zusammenfassung der Testergebnisse:")
    print("=" * 50)

    all_successful = True
    for model_type, success in results.items():
        status = "Erfolgreich" if success else "Fehlgeschlagen"
        print(f"Modell '{model_type}': {status}")
        if not success:
            all_successful = False

    return all_successful


if __name__ == "__main__":
    # Standardisierter deutscher Kommentar: Hauptausführung des Tests
    success = run_test()
    sys.exit(0 if success else 1)
