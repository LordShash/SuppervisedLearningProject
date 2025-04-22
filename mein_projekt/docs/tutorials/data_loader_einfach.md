# Datenlademodul - Einfache Erklärung

## Zweck

Das Datenlademodul (`data_loader.py`) ist dafür zuständig, die Textdaten aus CSV-Dateien zu laden und für die Textklassifikation vorzubereiten. Es ist wie ein Bibliothekar, der die richtigen Bücher (Daten) findet und sie in einer Form bereitstellt, die für die Analyse nützlich ist.

## Ablauf

1. **Daten finden und laden**: 
   - Das Modul sucht nach Datendateien im `data`-Verzeichnis.
   - Es kann sowohl mit einer großen Datei als auch mit mehreren kleineren Teildateien arbeiten.
   - Wenn die Dateien gefunden wurden, werden sie eingelesen und zu einer Gesamtdatenmenge zusammengefügt.

2. **Daten aufbereiten**:
   - Die Textinhalte (in der Spalte "BODY") werden in ein Format umgewandelt, das für maschinelles Lernen geeignet ist.
   - Die gewünschte Zielspalte (z.B. "Fits_Topic_Code") wird extrahiert, um als Klassifikationsziel zu dienen.

3. **Informationen bereitstellen**:
   - Das Modul kann auch Informationen über die verfügbaren Zielspalten und deren Werte liefern.
   - Dies hilft bei der Entscheidung, welche Klassifikationsaufgabe durchgeführt werden soll.

## Beispiel-Workflow

Stellen Sie sich vor, Sie möchten Texte nach ihrer Dringlichkeit klassifizieren:

1. Sie rufen die Funktion `get_available_targets()` auf, um zu sehen, welche Zielspalten verfügbar sind.
2. Sie sehen, dass "Urgency_Code" eine mögliche Zielspalte ist und welche Werte sie enthält.
3. Sie rufen die Funktion `load_data(target_column='Urgency_Code')` auf.
4. Das Modul:
   - Lädt die Daten aus den CSV-Dateien
   - Wandelt die Texte in numerische Features um
   - Gibt Ihnen die Features (X) und die Dringlichkeitscodes (y) zurück
5. Jetzt können Sie diese Daten verwenden, um ein Modell zu trainieren, das die Dringlichkeit von Texten vorhersagen kann.

Das Datenlademodul erledigt die komplizierte Arbeit des Datenladens und der Vorverarbeitung, sodass Sie sich auf die Analyse und das Modelltraining konzentrieren können.