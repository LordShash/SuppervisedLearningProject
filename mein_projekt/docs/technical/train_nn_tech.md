# Neuronales Netzwerk - Technische Referenz

## Übersicht

Das Modul `train_nn.py` implementiert das Training und die Speicherung eines neuronalen Netzwerks für die Textklassifikation. Es verwendet PyTorch für die Modellimplementierung und Evaluierung und unterstützt sowohl binäre als auch Mehrklassen-Klassifikation.

## Klassen

### FeedForwardNN

**Beschreibung**: Feed-Forward neuronales Netzwerk mit zwei Hidden-Layern.

**Attribute**:
- `layer1`, `layer2`, `layer3`: Lineare Schichten des Netzwerks
- `relu`: ReLU-Aktivierungsfunktion
- `dropout`: Dropout-Layer zur Vermeidung von Overfitting
- `sigmoid`: Sigmoid-Aktivierungsfunktion für binäre Klassifikation
- `multi_class`: Flag für Multiclass-Klassifikation

**Methoden**:
- `__init__(input_dim, hidden_dim=128, output_dim=1, multi_class=False)`: Initialisiert das Netzwerk
- `forward(x)`: Führt den Forward-Pass durch das Netzwerk durch

## Funktionen

### prepare_data()

**Beschreibung**: Bereitet die Daten für das Training des neuronalen Netzes vor.

**Inputs**:
- `X` (np.ndarray): Feature-Matrix (kann eine Sparse-Matrix sein).
- `y` (np.ndarray): Zielvariable.
- `test_size` (float, optional): Anteil der Testdaten. Standard: 0.2.
- `random_state` (int, optional): Seed für die Reproduzierbarkeit. Standard: 42.

**Outputs**:
- `train_loader` (DataLoader): DataLoader für Trainingsdaten.
- `test_loader` (DataLoader): DataLoader für Testdaten.
- `input_dim` (int): Anzahl der Eingabefeatures.
- `num_classes` (int): Anzahl der Klassen.

**Verhalten**:
1. Teilt die Daten in Trainings- und Testsets auf.
2. Wandelt Sparse-Matrizen in Dense-Matrizen um, falls nötig.
3. Konvertiert die Daten in PyTorch-Tensoren.
4. Erstellt DataLoader für Trainings- und Testdaten.

### train_neural_network()

**Beschreibung**: Trainiert ein neuronales Netzwerk.

**Inputs**:
- `train_loader` (DataLoader): DataLoader für Trainingsdaten.
- `test_loader` (DataLoader): DataLoader für Testdaten.
- `input_dim` (int): Anzahl der Eingabefeatures.
- `num_classes` (int): Anzahl der Klassen.
- `target_name` (str, optional): Name der Zielvariable für Reporting. Standard: "Fits_Topic_Code".
- `epochs` (int, optional): Anzahl der Trainingsepochen. Standard: 50.

**Outputs**:
- `model` (nn.Module): Trainiertes Modell.
- `accuracy` (float): Genauigkeit auf dem Testset.
- `precision` (float): Precision-Score auf dem Testset.
- `recall` (float): Recall-Score auf dem Testset.
- `f1` (float): F1-Score auf dem Testset.
- `report` (str): Detaillierter Klassifikationsbericht.
- `conf_matrix` (np.ndarray): Konfusionsmatrix.

**Verhalten**:
1. Initialisiert ein neuronales Netzwerk basierend auf der Anzahl der Klassen.
2. Wählt die passende Loss-Funktion (CrossEntropyLoss für Multiclass, BCELoss für binäre Klassifikation).
3. Trainiert das Modell über die angegebene Anzahl von Epochen.
4. Evaluiert das Modell mit den Testdaten.
5. Berechnet Genauigkeit, Precision, Recall, F1-Score, Klassifikationsbericht und Konfusionsmatrix.

### save_model()

**Beschreibung**: Speichert das trainierte PyTorch-Modell.

**Inputs**:
- `model` (nn.Module): Trainiertes Modell.
- `target_column` (str, optional): Name der Zielspalte für den Dateinamen. Standard: "Fits_Topic_Code".

**Outputs**: Keine direkten Rückgabewerte.

**Verhalten**:
1. Ermittelt den Pfad zum models-Verzeichnis.
2. Erstellt das Verzeichnis, falls es nicht existiert.
3. Generiert einen Dateinamen basierend auf der Zielspalte.
4. Speichert das Modell mit torch.save().

### train_and_save_model()

**Beschreibung**: Lädt Daten, trainiert ein Modell und speichert es.

**Inputs**:
- `target_column` (str, optional): Name der Zielspalte. Standard: "Fits_Topic_Code".
- `epochs` (int, optional): Anzahl der Trainingsepochen. Standard: 50.

**Outputs**:
- `accuracy` (float): Genauigkeit des Modells.
- `precision` (float): Precision-Score des Modells.
- `recall` (float): Recall-Score des Modells.
- `f1` (float): F1-Score des Modells.
- `report` (str): Detaillierter Klassifikationsbericht.
- `conf_matrix` (np.ndarray): Konfusionsmatrix.

**Verhalten**:
1. Lädt die Daten mit der angegebenen Zielspalte.
2. Bereitet die Daten für das neuronale Netzwerk vor.
3. Trainiert das neuronale Netzwerk.
4. Speichert das trainierte Modell.
5. Gibt Genauigkeit, Precision, Recall, F1-Score, Klassifikationsbericht und Konfusionsmatrix zurück.

## Abhängigkeiten

- **Externe Bibliotheken**:
  - `os`: Für Pfadoperationen.
  - `sys`: Für Systemfunktionen.
  - `numpy`: Für numerische Operationen.
  - `torch`: Für die Implementierung und das Training des neuronalen Netzwerks.
  - `sklearn`: Für Datenteilung und Evaluierungsmetriken.
  - `scipy`: Für Sparse-Matrix-Operationen.

- **Interne Module**:
  - `data_loader`: Für das Laden der Daten.

## Modelldetails

- **Modelltyp**: Feed-Forward neuronales Netzwerk
- **Architektur**:
  - Eingabeschicht: Dimensionalität abhängig von den Features
  - Zwei versteckte Schichten mit je 128 Neuronen und ReLU-Aktivierung
  - Ausgabeschicht: 1 Neuron mit Sigmoid-Aktivierung (binär) oder n Neuronen (Multiclass)
- **Regularisierung**: Dropout (0.2) nach jeder versteckten Schicht
- **Optimizer**: Adam mit Lernrate 0.001

## Fehlerbehandlung

Das Modul verwendet Try-Except-Blöcke mit `sys.exit()` für verschiedene Fehlersituationen:
- Fehler beim Training des Modells
- Fehler beim Speichern des Modells
- Fehler beim Laden der Daten

## Ausgabedateien

- Modellspeicherung: `models/nn_<target_column>_model.pt`
