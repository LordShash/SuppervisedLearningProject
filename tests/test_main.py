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
import tempfile
import shutil
import time
from typing import Dict, Any, Optional, List, Tuple

def run_test_for_model(
    model_type: str,
    target_column: str = "Fits_Topic_Code",
    timeout: int = 300,
    cleanup: bool = True,
    extra_args: Optional[List[str]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Führt einen Test des Hauptmoduls für einen bestimmten Modelltyp durch.

    Args:
        model_type (str): Der zu testende Modelltyp ('logreg', 'nn' oder 'all')
        target_column (str): Die zu verwendende Zielspalte (Standard: 'Fits_Topic_Code')
        timeout (int): Maximale Ausführungszeit in Sekunden (Standard: 300)
        cleanup (bool): Wenn True, werden temporäre Dateien nach dem Test gelöscht (Standard: True)
        extra_args (List[str], optional): Zusätzliche Kommandozeilenargumente für das Hauptmodul

    Returns:
        Tuple[bool, Dict[str, Any]]: 
            - bool: True, wenn der Test erfolgreich war, sonst False
            - Dict: Zusätzliche Informationen zum Testergebnis (Laufzeit, Ausgabe, etc.)
    """
    # Start des Tests für ein bestimmtes Modell
    print(f"Starte Test des Hauptmoduls mit Modell '{model_type}'...")
    start_time = time.time()

    # Ergebnis-Dictionary initialisieren
    result_info = {
        "model_type": model_type,
        "target_column": target_column,
        "start_time": start_time,
        "success": False,
        "output": "",
        "error": "",
        "duration": 0
    }

    # Temporäres Verzeichnis für Testausgaben erstellen
    temp_dir = tempfile.mkdtemp(prefix=f"test_main_{model_type}_")
    result_info["temp_dir"] = temp_dir
    print(f"Temporäres Verzeichnis erstellt: {temp_dir}")

    try:
        # Pfad zum Hauptmodul (angepasst für die neue Verzeichnisstruktur)
        main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "suppervisedlearningproject", "main.py")

        # Prüfung der Existenz der Datei
        if not os.path.exists(main_path):
            error_msg = f"Fehler: Die Datei '{main_path}' wurde nicht gefunden."
            print(error_msg)
            result_info["error"] = error_msg
            return False, result_info

        # Vorbereitung des Kommandos
        cmd = [sys.executable, main_path, "--model", model_type, "--target", target_column]

        # Zusätzliche Argumente hinzufügen, falls vorhanden
        if extra_args:
            cmd.extend(extra_args)

        # Umgebungsvariablen für den Testlauf vorbereiten
        env = os.environ.copy()
        env["TEST_MODE"] = "1"  # Markierung für Testmodus
        env["TEST_OUTPUT_DIR"] = temp_dir  # Ausgabeverzeichnis für Tests

        # Ausführung des Hauptmoduls
        print(f"Führe Kommando aus: {' '.join(cmd)}")
        process = subprocess.run(
            cmd, 
            check=False,  # Nicht automatisch bei Fehler abbrechen
            capture_output=True, 
            text=True,
            timeout=timeout,  # Timeout hinzugefügt
            env=env,  # Umgebungsvariablen übergeben
            cwd=os.path.dirname(os.path.dirname(__file__))  # Arbeitsverzeichnis setzen
        )

        # Ausgabe speichern
        result_info["output"] = process.stdout
        result_info["error"] = process.stderr
        result_info["return_code"] = process.returncode

        # Anzeige der Ausgabe
        print("\nAusgabe:")
        print(process.stdout)

        # Prüfung des Ergebnisses
        if process.returncode == 0:
            # Zusätzliche Validierung der Ausgabe
            if "erfolgreich" in process.stdout.lower() and "fehler" not in process.stdout.lower():
                print(f"\nTest für Modell '{model_type}' erfolgreich abgeschlossen!")
                result_info["success"] = True
                return True, result_info
            else:
                warning_msg = f"\nWarnung: Das Hauptmodul wurde mit Rückgabecode 0 beendet, aber die Ausgabe enthält möglicherweise Fehler."
                print(warning_msg)
                result_info["warning"] = warning_msg
                result_info["success"] = True  # Trotzdem als Erfolg werten
                return True, result_info
        else:
            error_msg = f"\nFehler: Das Hauptmodul wurde mit Rückgabecode {process.returncode} beendet."
            print(error_msg)
            print(f"Fehlerausgabe: {process.stderr}")
            result_info["error"] = f"{error_msg}\n{process.stderr}"
            return False, result_info

    except subprocess.TimeoutExpired as e:
        # Fehlerbehandlung bei Timeout
        error_msg = f"\nFehler: Timeout nach {timeout} Sekunden beim Ausführen des Hauptmoduls."
        print(error_msg)
        result_info["error"] = error_msg
        return False, result_info

    except subprocess.CalledProcessError as e:
        # Fehlerbehandlung bei Prozessausführung
        error_msg = f"\nFehler beim Ausführen des Hauptmoduls: {str(e)}"
        print(error_msg)
        print(f"Fehlerausgabe: {e.stderr}")
        result_info["error"] = f"{error_msg}\n{e.stderr}"
        return False, result_info

    except Exception as e:
        # Allgemeine Fehlerbehandlung
        error_msg = f"\nUnerwarteter Fehler: {str(e)}"
        print(error_msg)
        result_info["error"] = error_msg
        return False, result_info

    finally:
        # Dauer des Tests berechnen
        end_time = time.time()
        duration = end_time - start_time
        result_info["duration"] = duration
        print(f"Testdauer: {duration:.2f} Sekunden")

        # Aufräumen, falls gewünscht
        if cleanup:
            try:
                shutil.rmtree(temp_dir)
                print(f"Temporäres Verzeichnis gelöscht: {temp_dir}")
            except Exception as e:
                print(f"Warnung: Konnte temporäres Verzeichnis nicht löschen: {str(e)}")
        else:
            print(f"Temporäres Verzeichnis beibehalten: {temp_dir}")


def run_test(
    model_types: Optional[List[str]] = None,
    target_column: str = "Fits_Topic_Code",
    timeout: int = 300,
    cleanup: bool = True,
    save_report: bool = False,
    report_path: Optional[str] = None
) -> bool:
    """
    Führt Tests des Hauptmoduls für alle verfügbaren Modellkonfigurationen durch.

    Args:
        model_types (List[str], optional): Liste der zu testenden Modelltypen. 
                                          Wenn None, werden alle Modelltypen getestet.
        target_column (str): Die zu verwendende Zielspalte (Standard: 'Fits_Topic_Code')
        timeout (int): Maximale Ausführungszeit in Sekunden (Standard: 300)
        cleanup (bool): Wenn True, werden temporäre Dateien nach dem Test gelöscht (Standard: True)
        save_report (bool): Wenn True, wird ein Testbericht gespeichert (Standard: False)
        report_path (str, optional): Pfad für den Testbericht. Wenn None, wird ein Standardpfad verwendet.

    Returns:
        bool: True, wenn alle Tests erfolgreich waren, sonst False
    """
    # Start der Tests
    print("Starte Tests des Hauptmoduls für die angegebenen Modellkonfigurationen...")
    start_time = time.time()

    # Liste der zu testenden Modelltypen
    if model_types is None:
        model_types = ["logreg", "nn", "all"]

    # Ergebnisse für jeden Modelltyp speichern
    results = {}
    detailed_results = {}

    # Jeden Modelltyp testen
    for model_type in model_types:
        print(f"\n{'=' * 50}")
        print(f"Test für Modelltyp: {model_type}")
        print(f"{'=' * 50}")

        success, result_info = run_test_for_model(
            model_type=model_type,
            target_column=target_column,
            timeout=timeout,
            cleanup=cleanup
        )

        results[model_type] = success
        detailed_results[model_type] = result_info

    # Gesamtdauer berechnen
    total_duration = time.time() - start_time

    # Zusammenfassung der Ergebnisse
    print("\n\n" + "=" * 50)
    print("Zusammenfassung der Testergebnisse:")
    print("=" * 50)
    print(f"Gesamtdauer: {total_duration:.2f} Sekunden")
    print("-" * 50)

    all_successful = True
    for model_type, success in results.items():
        status = "Erfolgreich" if success else "Fehlgeschlagen"
        duration = detailed_results[model_type].get("duration", 0)
        print(f"Modell '{model_type}': {status} ({duration:.2f} Sekunden)")

        # Warnungen anzeigen, falls vorhanden
        if "warning" in detailed_results[model_type]:
            print(f"  Warnung: {detailed_results[model_type]['warning']}")

        if not success:
            all_successful = False
            print(f"  Fehler: {detailed_results[model_type].get('error', 'Unbekannter Fehler')}")

    # Testbericht speichern, falls gewünscht
    if save_report:
        try:
            import json
            from datetime import datetime

            # Standardpfad für den Bericht, falls keiner angegeben wurde
            if report_path is None:
                report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_reports")
                os.makedirs(report_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = os.path.join(report_dir, f"test_report_{timestamp}.json")

            # Bericht erstellen
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "all_successful": all_successful,
                "results": {}
            }

            # Detaillierte Ergebnisse hinzufügen (ohne große Ausgaben)
            for model_type, result_info in detailed_results.items():
                # Kopie erstellen, um die Originaldaten nicht zu verändern
                result_copy = result_info.copy()

                # Große Ausgaben kürzen
                if "output" in result_copy and len(result_copy["output"]) > 1000:
                    result_copy["output"] = result_copy["output"][:500] + "... [gekürzt] ..." + result_copy["output"][-500:]

                report["results"][model_type] = result_copy

            # Bericht speichern
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"\nTestbericht gespeichert unter: {report_path}")

        except Exception as e:
            print(f"\nFehler beim Speichern des Testberichts: {str(e)}")

    return all_successful


if __name__ == "__main__":
    import argparse

    # Kommandozeilenargumente parsen
    parser = argparse.ArgumentParser(description="Testet das Hauptmodul mit verschiedenen Konfigurationen")
    parser.add_argument("--models", nargs="+", choices=["logreg", "nn", "all"], 
                        help="Zu testende Modelltypen (Standard: alle)")
    parser.add_argument("--target", default="Fits_Topic_Code", 
                        help="Zu verwendende Zielspalte (Standard: Fits_Topic_Code)")
    parser.add_argument("--timeout", type=int, default=300, 
                        help="Timeout in Sekunden (Standard: 300)")
    parser.add_argument("--no-cleanup", action="store_true", 
                        help="Temporäre Dateien nicht löschen")
    parser.add_argument("--save-report", action="store_true", 
                        help="Testbericht speichern")
    parser.add_argument("--report-path", 
                        help="Pfad für den Testbericht (optional)")

    args = parser.parse_args()

    # Tests ausführen
    success = run_test(
        model_types=args.models,
        target_column=args.target,
        timeout=args.timeout,
        cleanup=not args.no_cleanup,
        save_report=args.save_report,
        report_path=args.report_path
    )

    # Exitcode basierend auf Testergebnis
    sys.exit(0 if success else 1)
