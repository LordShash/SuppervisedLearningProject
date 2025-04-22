# Tutorial: Modelle trainieren mit dem Hauptmodul

In diesem Tutorial lernen Sie, wie Sie das Hauptmodul `main.py` verwenden, um verschiedene Textklassifikationsmodelle zu trainieren.

## Einführung

Das Hauptmodul `main.py` dient als zentraler Einstiegspunkt für das Training verschiedener Modelle. Es ermöglicht Ihnen, sowohl logistische Regressionsmodelle als auch neuronale Netze mit verschiedenen Parametern zu trainieren.

## Grundlegende Verwendung

Sie können das Hauptmodul direkt von der Kommandozeile aus aufrufen:

```bash
python -m suppervisedlearningproject.main
```

Ohne weitere Parameter wird ein logistisches Regressionsmodell und ein neuronales Netz für die Zielvariable "Fits_Topic_Code" mit Standardparametern trainiert.

## Parameter anpassen

Das Hauptmodul unterstützt verschiedene Kommandozeilenargumente, mit denen Sie das Training anpassen können:

### Modelltyp auswählen

```bash
# Nur logistische Regression trainieren
python -m suppervisedlearningproject.main --model logreg

# Nur neuronales Netz trainieren
python -m suppervisedlearningproject.main --model nn

# Beide Modelle trainieren (Standard)
python -m suppervisedlearningproject.main --model all
```

### Zielvariable auswählen

```bash
# Modell für "Urgency_Code" trainieren
python src/main.py --target Urgency_Code

# Modell für "Categorie_Code" trainieren
python src/main.py --target Categorie_Code

# Modelle für alle verfügbaren Zielvariablen trainieren
python src/main.py --target all
```

### Parameter für die Feature-Extraktion

```bash
# Maximale Anzahl der Features für TF-IDF anpassen
python src/main.py --max-features 2000

# Anteil der Testdaten anpassen
python src/main.py --test-size 0.3
```

### Parameter für logistische Regression

```bash
# Maximale Anzahl von Iterationen anpassen
python src/main.py --max-iter 2000

# Regularisierungsparameter anpassen
python src/main.py --C 0.5

# Optimierungsalgorithmus auswählen
python src/main.py --solver newton-cg
```

### Parameter für neuronales Netz

```bash
# Anzahl der Trainingsepochen anpassen
python src/main.py --epochs 50

# Patience für Early Stopping anpassen
python src/main.py --patience 10
```

## Kombinierte Parameter

Sie können mehrere Parameter kombinieren, um das Training genau anzupassen:

```bash
# Beispiel für eine komplexe Konfiguration
python src/main.py --model logreg --target Urgency_Code --max-features 2000 --test-size 0.25 --max-iter 1500 --C 0.8 --solver saga
```

## Ausgabe verstehen

Das Hauptmodul gibt während des Trainings Informationen zum Fortschritt aus und zeigt am Ende eine Zusammenfassung der Ergebnisse:

```
Zusammenfassung der Ergebnisse:
========================================

Modell: logreg_Fits_Topic_Code
Accuracy: 0.8245
Precision: 0.8312
Recall: 0.8245
F1 Score: 0.8278

Modell: nn_Fits_Topic_Code
Accuracy: 0.8356
Precision: 0.8401
Recall: 0.8356
F1 Score: 0.8378

Training abgeschlossen!
```

## Tipps und Tricks

1. **Modellvergleich**: Trainieren Sie beide Modelltypen mit den gleichen Parametern, um ihre Leistung zu vergleichen.

2. **Hyperparameter-Tuning**: Experimentieren Sie mit verschiedenen Parametern, um die beste Konfiguration für Ihre Daten zu finden.

3. **Speicherort der Modelle**: Die trainierten Modelle werden im Verzeichnis `models/` gespeichert und können später für Vorhersagen verwendet werden.

4. **Fehlerbehandlung**: Das Hauptmodul enthält robuste Fehlerbehandlung. Wenn ein Fehler auftritt, wird eine aussagekräftige Fehlermeldung angezeigt.

## Nächste Schritte

Nachdem Sie Modelle mit dem Hauptmodul trainiert haben, können Sie:

- Die grafische Benutzeroberfläche verwenden, um die Ergebnisse zu visualisieren (siehe Anleitung `gui_anleitung.md`)
- Die trainierten Modelle in eigenen Skripten verwenden
- Die Modelle mit anderen Parametern oder Zielvariablen trainieren, um die Ergebnisse zu verbessern
