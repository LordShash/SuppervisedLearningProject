# Anleitung zur Verwendung der grafischen Benutzeroberfläche

Diese Anleitung beschreibt, wie Sie die grafische Benutzeroberfläche (GUI) für die Textklassifikationsanwendung verwenden können.

## Starten der Anwendung

Um die grafische Benutzeroberfläche zu starten, führen Sie das Skript `run_gui.py` im Hauptverzeichnis des Projekts aus:

```bash
python run_gui.py
```

## Überblick über die Benutzeroberfläche

Die Benutzeroberfläche besteht aus drei Haupttabs:

1. **Training**: Hier können Sie Modelle konfigurieren und trainieren
2. **Ergebnisse**: Hier werden die Trainingsergebnisse visualisiert
3. **Über**: Hier finden Sie allgemeine Informationen zur Anwendung

### Training-Tab

Im Training-Tab können Sie folgende Einstellungen vornehmen:

#### Modellkonfiguration
- **Modelltyp**: Wählen Sie zwischen logistischer Regression, neuronalem Netz oder beiden
- **Zielvariable**: Wählen Sie die Zielvariable für das Training (Fits_Topic_Code, Urgency_Code oder Categorie_Code)

#### Hyperparameter
- **Max Features**: Maximale Anzahl der Features für TF-IDF (Standard: 1000)
- **Test Size**: Anteil der Testdaten (Standard: 0.2)

Für die logistische Regression:
- **Max Iterations**: Maximale Anzahl von Iterationen (Standard: 1000)
- **C (Regularisierung)**: Regularisierungsparameter (Standard: 1.0)
- **Solver**: Algorithmus für die Optimierung (Standard: lbfgs)

Für das neuronale Netz:
- **Epochen**: Anzahl der Trainingsepochen (Standard: 30)
- **Patience**: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird (Standard: 5)

#### Ausgabebereich
Im unteren Bereich des Training-Tabs wird der Fortschritt des Trainings angezeigt. Hier können Sie den Status des Trainings verfolgen und eventuelle Fehlermeldungen einsehen.

#### Aktionen
- **Modell trainieren**: Startet das Training mit den ausgewählten Einstellungen
- **Ausgabe löschen**: Löscht den Inhalt des Ausgabebereichs

### Ergebnisse-Tab

Im Ergebnisse-Tab können Sie die Ergebnisse des Trainings visualisieren:

- **Ergebnis auswählen**: Wählen Sie ein Trainingsergebnis aus der Dropdown-Liste
- **Metriken**: Zeigt Accuracy, Precision, Recall und F1-Score des Modells
- **Konfusionsmatrix**: Visualisiert die Konfusionsmatrix des Modells
- **Klassifikationsbericht**: Zeigt einen detaillierten Bericht mit Precision, Recall und F1-Score für jede Klasse

### Über-Tab

Der Über-Tab enthält allgemeine Informationen zur Anwendung, einschließlich einer Beschreibung der unterstützten Modelle und Zielvariablen.

## Typischer Arbeitsablauf

1. Starten Sie die Anwendung mit `python run_gui.py`
2. Im Training-Tab:
   - Wählen Sie den gewünschten Modelltyp
   - Wählen Sie die Zielvariable
   - Passen Sie die Hyperparameter nach Bedarf an
   - Klicken Sie auf "Modell trainieren"
3. Warten Sie, bis das Training abgeschlossen ist (der Fortschritt wird im Ausgabebereich angezeigt)
4. Nach Abschluss des Trainings wechselt die Anwendung automatisch zum Ergebnisse-Tab
5. Im Ergebnisse-Tab können Sie die Leistung des Modells anhand der Metriken, der Konfusionsmatrix und des Klassifikationsberichts bewerten

## Tipps

- Das Training kann je nach Datenmenge und gewählten Hyperparametern einige Zeit in Anspruch nehmen. Die GUI bleibt während des Trainings reaktionsfähig.
- Sie können mehrere Modelle nacheinander trainieren und die Ergebnisse später im Ergebnisse-Tab vergleichen.
- Wenn Sie beide Modelltypen (logistische Regression und neuronales Netz) auswählen, werden beide Modelle nacheinander trainiert.
- Die Ergebnisse werden nur für die aktuelle Sitzung gespeichert. Wenn Sie die Anwendung schließen, gehen die Visualisierungen verloren (die trainierten Modelle werden jedoch im `models`-Verzeichnis gespeichert).