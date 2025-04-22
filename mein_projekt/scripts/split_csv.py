import os
import pandas as pd

def split_csv_file() -> bool:
    """
    Teilt die große CSV-Datei in kleinere Teile.

    Returns:
        bool: True bei Erfolg, False bei Fehler
    """
    # Pfad zur ursprünglichen Datendatei
    original_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Daten_UTF8_Clean_encoded.csv')

    # Prüfung, ob die Datei existiert
    if not os.path.exists(original_file_path):
        print(f"Fehler: Die Datei '{original_file_path}' wurde nicht gefunden.")
        return False

    try:
        # Einlesen der CSV-Datei mit pandas
        df = pd.read_csv(original_file_path)

        # Ermittlung der Gesamtzahl der Zeilen
        total_rows = len(df)

        # Berechnung des Mittelpunkts für die Teilung
        middle_point = total_rows // 2

        # Aufteilung des Dataframes in zwei Teile
        df_part1 = df.iloc[:middle_point]
        df_part2 = df.iloc[middle_point:]

        # Erstellung der Pfade für die neuen Dateien
        part1_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Daten_UTF8_Clean_encoded_part1.csv')
        part2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Daten_UTF8_Clean_encoded_part2.csv')

        # Speicherung der Teile in neue CSV-Dateien
        df_part1.to_csv(part1_path, index=False)
        df_part2.to_csv(part2_path, index=False)

        # Ausgabe der Erfolgsmeldung
        print(f"Die Datei wurde erfolgreich in zwei Teile aufgeteilt:")
        print(f"Teil 1: {part1_path} ({len(df_part1)} Zeilen)")
        print(f"Teil 2: {part2_path} ({len(df_part2)} Zeilen)")

        return True

    except Exception as e:
        # Fehlerbehandlung bei Problemen
        print(f"Fehler beim Aufteilen der Datei: {str(e)}")
        return False

if __name__ == "__main__":
    # Hauptausführung des Skripts
    success = split_csv_file()
    if success:
        print("Dateiaufteilung erfolgreich abgeschlossen.")
    else:
        print("Dateiaufteilung fehlgeschlagen.")
