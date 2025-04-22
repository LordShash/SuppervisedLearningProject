# Neuronales Netzwerk - Einfache Erklärung

## Zweck

Das Modul für neuronale Netzwerke (`train_nn.py`) ist dafür zuständig, ein fortschrittliches Textklassifikationsmodell zu trainieren, das auf künstlichen neuronalen Netzwerken basiert. Diese Art von Modell kann komplexe Muster in Texten erkennen und sie in verschiedene Kategorien einordnen, nachdem es mit Beispieltexten trainiert wurde.

## Ablauf

1. **Daten vorbereiten**: 
   - Das Modul lädt die Textdaten mit Hilfe des Datenlademodul.
   - Die Daten werden in Trainings- und Testdaten aufgeteilt.
   - Die Texte werden in ein Format umgewandelt, das für neuronale Netzwerke geeignet ist.

2. **Netzwerk definieren**:
   - Es wird ein neuronales Netzwerk mit mehreren Schichten (Layern) erstellt.
   - Das Netzwerk enthält Eingabe-, versteckte und Ausgabeschichten.
   - Es werden spezielle Techniken wie Dropout verwendet, um Überanpassung zu vermeiden.

3. **Modell trainieren**:
   - Das neuronale Netzwerk wird über mehrere Durchläufe (Epochen) mit den Trainingsdaten trainiert.
   - In jeder Epoche lernt das Modell, seine Vorhersagen zu verbessern.
   - Der Fortschritt wird regelmäßig angezeigt, damit Sie den Lernprozess verfolgen können.

4. **Modell bewerten**:
   - Das trainierte Modell wird mit den Testdaten überprüft.
   - Es werden verschiedene Kennzahlen berechnet:
     - Genauigkeit (Accuracy): Wie viel Prozent der Texte wurden richtig klassifiziert?
     - Precision: Wie viele der als Kategorie X klassifizierten Texte gehören wirklich zu X?
     - Recall: Wie viele der tatsächlich zu Kategorie X gehörenden Texte wurden erkannt?
     - F1-Score: Ein Mittelwert aus Precision und Recall
   - Es wird ein detaillierter Bericht erstellt, der diese Werte für jede Kategorie zeigt.
   - Eine Konfusionsmatrix wird erstellt, die anzeigt, welche Kategorien wie oft verwechselt wurden.

5. **Modell speichern**:
   - Das trainierte Modell wird als Datei im `models`-Verzeichnis gespeichert.
   - Der Dateiname enthält die Information, für welche Zielvariable das Modell trainiert wurde.

## Beispiel-Workflow

Stellen Sie sich vor, Sie möchten Texte nach ihrer Kategorie klassifizieren:

1. Sie rufen die Funktion `train_and_save_model(target_column='Categorie_Code', epochs=30)` auf.
2. Das Modul:
   - Lädt die Daten mit der Zielspalte "Categorie_Code"
   - Bereitet die Daten für das neuronale Netzwerk vor
   - Erstellt und trainiert das Netzwerk über 30 Epochen
   - Zeigt den Fortschritt während des Trainings an
   - Bewertet, wie gut das Modell funktioniert
   - Speichert das Modell als `nn_Categorie_Code_model.pt`
3. Sie erhalten alle Bewertungskennzahlen: Genauigkeit (Accuracy), Precision, Recall, F1-Score, einen detaillierten Bericht und eine Konfusionsmatrix.

Neuronale Netzwerke sind komplexer und benötigen oft mehr Rechenleistung als einfachere Modelle wie die logistische Regression. Sie können jedoch bei ausreichender Datenmenge und richtigem Training bessere Ergebnisse liefern, besonders bei komplexen Textklassifikationsaufgaben.
