# Logistische Regression - Einfache Erklärung

## Zweck

Das Modul für logistische Regression (`train_logreg.py`) ist dafür zuständig, ein Textklassifikationsmodell zu trainieren, das auf dem mathematischen Verfahren der logistischen Regression basiert. Dieses Modell kann Texte automatisch in verschiedene Kategorien einordnen, nachdem es mit Beispieltexten trainiert wurde.

## Ablauf

1. **Daten vorbereiten**: 
   - Das Modul lädt die Textdaten mit Hilfe des Datenlademodul.
   - Die Daten werden in Trainings- und Testdaten aufgeteilt, damit später überprüft werden kann, wie gut das Modell funktioniert.

2. **Modell trainieren**:
   - Die Texte werden zunächst standardisiert (auf eine einheitliche Skala gebracht).
   - Das logistische Regressionsmodell wird mit den Trainingsdaten trainiert.
   - Das Modell lernt, welche Wortmuster mit welchen Kategorien zusammenhängen.

3. **Modell bewerten**:
   - Das trainierte Modell wird mit den Testdaten überprüft.
   - Es werden verschiedene Kennzahlen berechnet:
     - Genauigkeit (Accuracy): Wie viel Prozent der Texte wurden richtig klassifiziert?
     - Precision: Wie viele der als Kategorie X klassifizierten Texte gehören wirklich zu X?
     - Recall: Wie viele der tatsächlich zu Kategorie X gehörenden Texte wurden erkannt?
     - F1-Score: Ein Mittelwert aus Precision und Recall
   - Es wird ein detaillierter Bericht erstellt, der diese Werte für jede Kategorie zeigt.
   - Eine Konfusionsmatrix wird erstellt, die anzeigt, welche Kategorien wie oft verwechselt wurden.

4. **Modell speichern**:
   - Das trainierte Modell wird als Datei im `models`-Verzeichnis gespeichert.
   - Der Dateiname enthält die Information, für welche Zielvariable das Modell trainiert wurde.

## Beispiel-Workflow

Stellen Sie sich vor, Sie möchten Texte nach ihrer Dringlichkeit klassifizieren:

1. Sie rufen die Funktion `train_and_save_model(target_column='Urgency_Code')` auf.
2. Das Modul:
   - Lädt die Daten mit der Zielspalte "Urgency_Code"
   - Teilt die Daten in Trainings- und Testdaten auf
   - Trainiert ein logistisches Regressionsmodell
   - Bewertet, wie gut das Modell funktioniert
   - Speichert das Modell als `logreg_Urgency_Code_model.pkl`
3. Sie erhalten alle Bewertungskennzahlen: Genauigkeit (Accuracy), Precision, Recall, F1-Score, einen detaillierten Bericht und eine Konfusionsmatrix.

Die logistische Regression ist ein relativ einfaches, aber effektives Verfahren für die Textklassifikation. Es ist schneller zu trainieren als komplexere Modelle wie neuronale Netze und liefert oft gute Ergebnisse, besonders wenn die Datenmenge begrenzt ist.
