# Benutzeranleitung: Moderne Grafische Benutzeroberfläche

Diese Anleitung erklärt, wie Sie die moderne grafische Benutzeroberfläche (GUI) des Textklassifikationsprojekts verwenden können.

## Einführung

Die moderne grafische Benutzeroberfläche bietet einen benutzerfreundlichen Zugang zu den Funktionen des Textklassifikationsprojekts. Sie ermöglicht es Ihnen, Modelle zu trainieren, Parameter anzupassen und Ergebnisse zu visualisieren, ohne Kommandozeilenbefehle verwenden zu müssen. Die GUI basiert auf PyQt5 und bietet ein modernes Look-and-Feel mit verbesserten Visualisierungen und Benutzerinteraktionen.

## Starten der GUI

Um die grafische Benutzeroberfläche zu starten, führen Sie das Skript `run_gui.py` aus:

```bash
python run_gui.py
```

### Hinweis zu Abhängigkeiten

Die moderne GUI benötigt PyQt5. Wenn PyQt5 nicht installiert ist, wird das Skript Sie fragen, ob es automatisch installiert werden soll. Sie können PyQt5 auch manuell installieren mit:

```bash
pip install PyQt5>=5.15.0
```

Nach dem Start wird ein Fenster mit drei Tabs angezeigt:
- **Training**: Hier können Sie Modelle trainieren
- **Ergebnisse**: Hier werden die Trainingsergebnisse angezeigt
- **Über**: Hier finden Sie Informationen über die Anwendung

## Tab "Training"

Der Tab "Training" ist in drei Bereiche unterteilt:

### 1. Trainingsparameter

Hier können Sie verschiedene Parameter für das Training einstellen:

- **Zielvariable**: Wählen Sie die Zielvariable aus, für die das Modell trainiert werden soll (z.B. "Fits_Topic_Code", "Urgency_Code")
- **Modelltyp**: Wählen Sie zwischen logistischer Regression und neuronalem Netz
- **Max Features**: Maximale Anzahl der Features für die TF-IDF-Vektorisierung
- **Test Size**: Anteil der Daten, der für den Testdatensatz verwendet wird (0.0 bis 1.0)

Je nach gewähltem Modelltyp werden zusätzliche Parameter angezeigt:

#### Logistische Regression
- **Max Iterations**: Maximale Anzahl der Iterationen für den Optimierungsalgorithmus
- **C (Regularisierung)**: Regularisierungsparameter (kleinere Werte bedeuten stärkere Regularisierung)
- **Solver**: Algorithmus für die Optimierung

#### Neuronales Netz
- **Epochs**: Anzahl der Trainingsepochen
- **Patience**: Anzahl der Epochen ohne Verbesserung, bevor das Training frühzeitig beendet wird

### 2. Trainieren-Button

Klicken Sie auf den Button "Modell trainieren", um das Training mit den ausgewählten Parametern zu starten.

### 3. Ausgabebereich

Hier werden während des Trainings Informationen zum Fortschritt und nach Abschluss die Ergebnisse angezeigt.

## Tab "Ergebnisse"

Der Tab "Ergebnisse" wird automatisch angezeigt, nachdem ein Modell erfolgreich trainiert wurde. Er ist in drei Bereiche unterteilt:

### 1. Modellauswahl

Hier können Sie zwischen verschiedenen trainierten Modellen wählen, wenn Sie mehrere Modelle trainiert haben.

### 2. Metriken

Hier werden die wichtigsten Metriken des ausgewählten Modells angezeigt:
- **Accuracy**: Der Anteil der korrekt klassifizierten Beispiele
- **Precision**: Der Anteil der korrekt positiven Vorhersagen an allen positiven Vorhersagen
- **Recall**: Der Anteil der korrekt positiven Vorhersagen an allen tatsächlich positiven Beispielen
- **F1 Score**: Das harmonische Mittel aus Precision und Recall

### 3. Klassifikationsbericht und Konfusionsmatrix

Hier werden detaillierte Informationen zur Leistung des Modells angezeigt:
- Der **Klassifikationsbericht** zeigt Precision, Recall und F1-Score für jede Klasse
- Die **Konfusionsmatrix** visualisiert, wie viele Beispiele jeder Klasse korrekt oder falsch klassifiziert wurden

## Tab "Über"

Der Tab "Über" enthält Informationen über die Anwendung:
- Eine Beschreibung der Anwendung
- Eine Anleitung zur Verwendung
- Informationen zur Version und zum Copyright

## Tipps zur Verwendung

### Tooltips

Viele Elemente der Benutzeroberfläche verfügen über Tooltips, die zusätzliche Informationen anzeigen, wenn Sie mit der Maus darüber fahren. Achten Sie auf das Symbol ℹ️ neben den Parametern.

### Modellvergleich

Um verschiedene Modelle zu vergleichen:
1. Trainieren Sie ein Modell mit bestimmten Parametern
2. Notieren Sie sich die Ergebnisse oder lassen Sie das Fenster geöffnet
3. Trainieren Sie ein weiteres Modell mit anderen Parametern
4. Wechseln Sie im Tab "Ergebnisse" zwischen den Modellen, um die Leistung zu vergleichen

### Speicherung der Modelle

Alle trainierten Modelle werden automatisch im Verzeichnis `models/` gespeichert:
- Logistische Regression: `models/logreg_{target_column}_model.pkl`
- Neuronales Netz: `models/nn_{target_column}_model.pt`

Sie können diese Modelle später in eigenen Skripten verwenden. Das Projekt bietet dafür das Modul `model_loader.py`, das das Laden von Modellen vereinfacht:

```python
from src.model_loader import load_model

# Modell laden
model = load_model(model_type="logreg", target_column="Fits_Topic_Code")
```

Weitere Informationen zur Verwendung des Modelllademodul finden Sie im Tutorial `docs/tutorials/model_loader_einfach.md`.

## Fehlerbehebung

### GUI startet nicht

Wenn die GUI nicht startet, überprüfen Sie:
1. Ob PyQt5 installiert ist (`pip install PyQt5>=5.15.0`)
2. Ob alle anderen Abhängigkeiten installiert sind (`pip install -r requirements.txt`)
3. Ob Python 3.7 oder höher verwendet wird
4. Ob das Skript vom richtigen Verzeichnis aus aufgerufen wird

Wenn Sie Probleme mit PyQt5 haben, können Sie versuchen, es neu zu installieren:
```bash
pip uninstall PyQt5
pip install PyQt5>=5.15.0
```

### Training bricht ab

Wenn das Training abbricht, kann dies verschiedene Ursachen haben:
1. **Speicherprobleme**: Reduzieren Sie den Parameter "Max Features"
2. **Konvergenzprobleme**: Erhöhen Sie den Parameter "Max Iterations" (für logistische Regression)
3. **Datenfehler**: Stellen Sie sicher, dass die Datendateien korrekt sind und die erwarteten Spalten enthalten

### Langsames Training

Wenn das Training sehr langsam ist:
1. Verwenden Sie eine kleinere Anzahl von Features (Parameter "Max Features")
2. Verwenden Sie die logistische Regression statt des neuronalen Netzes
3. Stellen Sie sicher, dass Ihr System über ausreichend Ressourcen verfügt

## Nächste Schritte

Nach dem erfolgreichen Training von Modellen mit der GUI können Sie:
- Die Modelle in eigenen Skripten verwenden
- Die Kommandozeilenversion für Batch-Verarbeitung nutzen
- Die Modellparameter weiter optimieren, um bessere Ergebnisse zu erzielen
