import os
import shutil
import pandas as pd
from typing import Union, List, Optional

def split_csv_file(
    file_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_parts: int = 2,
    max_rows_per_part: Optional[int] = None,
    overwrite: bool = True
) -> bool:
    """
    Teilt eine große CSV-Datei in kleinere Teile.

    Args:
        file_path: Pfad zur CSV-Datei. Wenn None, wird die Standarddatei verwendet.
        output_dir: Ausgabeverzeichnis für die geteilten Dateien. Wenn None, wird das Datenverzeichnis verwendet.
        num_parts: Anzahl der zu erstellenden Teile (Standard: 2).
        max_rows_per_part: Maximale Anzahl von Zeilen pro Teil. Wenn angegeben, überschreibt dies num_parts.
        overwrite: Wenn True, werden vorhandene Dateien überschrieben. Wenn False, wird ein Fehler ausgegeben.

    Returns:
        bool: True bei Erfolg, False bei Fehler

    Raises:
        FileNotFoundError: Wenn die Eingabedatei nicht gefunden wird
        ValueError: Bei ungültigen Parametern
        OSError: Bei Problemen mit dem Dateisystem
    """
    try:
        # Standardpfad zur Datendatei, wenn kein Pfad angegeben wurde
        if file_path is None:
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Daten_UTF8_Clean_encoded.csv')

        # Standardausgabeverzeichnis, wenn keines angegeben wurde
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

        # Prüfung der Parameter
        if num_parts < 1:
            raise ValueError("Die Anzahl der Teile muss mindestens 1 sein.")

        if max_rows_per_part is not None and max_rows_per_part < 1:
            raise ValueError("Die maximale Anzahl von Zeilen pro Teil muss mindestens 1 sein.")

        # Prüfung, ob die Datei existiert
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Die Datei '{file_path}' wurde nicht gefunden.")

        # Prüfung, ob das Ausgabeverzeichnis existiert
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Ausgabeverzeichnis '{output_dir}' wurde erstellt.")

        # Prüfung des verfügbaren Speicherplatzes
        file_size = os.path.getsize(file_path)
        free_space = shutil.disk_usage(output_dir).free

        if free_space < file_size * 1.5:  # Sicherheitsfaktor von 1.5
            print(f"Warnung: Möglicherweise nicht genügend Speicherplatz verfügbar. "
                  f"Benötigt: {file_size * 1.5 / (1024**2):.2f} MB, Verfügbar: {free_space / (1024**2):.2f} MB")

        # Einlesen der CSV-Datei mit pandas
        # Verwende einen Chunk-Iterator für große Dateien
        print(f"Lese Datei '{file_path}'...")
        df = pd.read_csv(file_path)

        # Ermittlung der Gesamtzahl der Zeilen
        total_rows = len(df)
        print(f"Datei enthält {total_rows} Zeilen.")

        # Berechnung der Anzahl der Zeilen pro Teil
        if max_rows_per_part is not None:
            rows_per_part = max_rows_per_part
            num_parts = (total_rows + rows_per_part - 1) // rows_per_part  # Aufrunden
        else:
            rows_per_part = (total_rows + num_parts - 1) // num_parts  # Aufrunden

        print(f"Teile Datei in {num_parts} Teile mit jeweils ca. {rows_per_part} Zeilen...")

        # Liste für die Ausgabepfade
        output_paths = []

        # Basisname der Ausgabedatei
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Aufteilung und Speicherung der Teile
        for i in range(num_parts):
            # Berechnung des Bereichs für diesen Teil
            start_idx = i * rows_per_part
            end_idx = min((i + 1) * rows_per_part, total_rows)

            # Wenn der letzte Teil leer wäre, überspringen
            if start_idx >= total_rows:
                break

            # Aufteilung des Dataframes
            df_part = df.iloc[start_idx:end_idx]

            # Erstellung des Pfads für die neue Datei
            part_path = os.path.join(output_dir, f"{base_name}_part{i+1}.csv")
            output_paths.append(part_path)

            # Prüfung, ob die Datei bereits existiert
            if os.path.exists(part_path) and not overwrite:
                raise FileExistsError(f"Die Datei '{part_path}' existiert bereits und overwrite=False.")

            # Speicherung des Teils in eine neue CSV-Datei
            df_part.to_csv(part_path, index=False)
            print(f"Teil {i+1}: {part_path} ({len(df_part)} Zeilen)")

        print(f"Die Datei wurde erfolgreich in {len(output_paths)} Teile aufgeteilt.")
        return True

    except FileNotFoundError as e:
        print(f"Fehler: {str(e)}")
        return False
    except ValueError as e:
        print(f"Ungültiger Parameter: {str(e)}")
        return False
    except FileExistsError as e:
        print(f"Datei existiert bereits: {str(e)}")
        return False
    except OSError as e:
        print(f"Dateisystemfehler: {str(e)}")
        return False
    except Exception as e:
        print(f"Unerwarteter Fehler beim Aufteilen der Datei: {str(e)}")
        return False

if __name__ == "__main__":
    # Hauptausführung des Skripts
    success = split_csv_file()
    if success:
        print("Dateiaufteilung erfolgreich abgeschlossen.")
    else:
        print("Dateiaufteilung fehlgeschlagen.")
