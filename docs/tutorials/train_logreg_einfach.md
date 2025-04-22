# Tutorial: Logistische Regression für Textklassifikation

In diesem Tutorial lernen Sie, wie Sie das Modul `train_logreg.py` verwenden, um ein logistisches Regressionsmodell für die Textklassifikation zu trainieren.

## Einführung

Die logistische Regression ist ein leistungsfähiger Algorithmus für Klassifikationsaufgaben. Das Modul `train_logreg.py` bietet eine einfache Möglichkeit, ein solches Modell für die Textklassifikation zu trainieren und zu evaluieren.

## Grundlegende Verwendung

Sie können das Modul direkt in Ihrem eigenen Python-Code importieren und verwenden:

```python
from train_logreg import train_and_save_model

# Modell mit Standardparametern trainieren
accuracy, precision, recall, f1, report, conf_matrix = train_and_save_model(
    target_column="Fits_Topic_Code"
)

# Ausgabe der Ergebnisse
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nKlassifikationsbericht:")
print(report)
```

## Parameter anpassen

Sie können verschiedene Parameter anpassen, um die Leistung des Modells zu optimieren:

```python
# Modell mit angepassten Parametern trainieren
results = train_and_save_model(
    target_column="Urgency_Code",    # Andere Zielvariable wählen
    max_features=2000,               # Mehr Features verwenden
    test_size=0.3,                   # Größeren Testdatensatz verwenden
    max_iter=2000,                   # Mehr Iterationen für das Training
    C=0.5,                           # Stärkere Regularisierung
    solver="newton-cg"               # Anderen Optimierungsalgorithmus verwenden
)
```

### Wichtige Parameter

- **target_column**: Die Zielvariable für die Klassifikation (z.B. "Fits_Topic_Code", "Urgency_Code")
- **max_features**: Maximale Anzahl der Features für die TF-IDF-Vektorisierung
- **test_size**: Anteil der Daten, der für den Testdatensatz verwendet wird (0.0 bis 1.0)
- **max_iter**: Maximale Anzahl der Iterationen für den Optimierungsalgorithmus
- **C**: Regularisierungsparameter (kleinere Werte bedeuten stärkere Regularisierung)
- **solver**: Algorithmus für die Optimierung ("lbfgs", "newton-cg", "liblinear", "sag", "saga")

## Modellspeicherung und -verwendung

Das trainierte Modell wird automatisch im Verzeichnis `models/` gespeichert:

```
models/logreg_Fits_Topic_Code_model.pkl
```

Sie können das gespeicherte Modell später für Vorhersagen verwenden:

```python
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Modell laden
model_path = os.path.join("models", "logreg_Fits_Topic_Code_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Text vektorisieren (Sie müssen denselben Vektorisierer verwenden wie beim Training)
vectorizer = TfidfVectorizer(max_features=1000)
# Hier müssten Sie den Vektorisierer mit denselben Daten trainieren oder speichern/laden

# Vorhersage für einen neuen Text
text = "Dies ist ein Beispieltext für die Vorhersage."
X_new = vectorizer.transform([text])
prediction = model.predict(X_new)
print(f"Vorhersage: {prediction[0]}")
```

## Interpretation der Ergebnisse

Nach dem Training gibt das Modul verschiedene Metriken zurück:

1. **Accuracy**: Der Anteil der korrekt klassifizierten Beispiele (höher ist besser)
2. **Precision**: Der Anteil der korrekt positiven Vorhersagen an allen positiven Vorhersagen (höher ist besser)
3. **Recall**: Der Anteil der korrekt positiven Vorhersagen an allen tatsächlich positiven Beispielen (höher ist besser)
4. **F1-Score**: Das harmonische Mittel aus Precision und Recall (höher ist besser)
5. **Klassifikationsbericht**: Detaillierte Metriken für jede Klasse
6. **Konfusionsmatrix**: Eine Matrix, die zeigt, wie viele Beispiele jeder Klasse korrekt oder falsch klassifiziert wurden

## Tipps und Tricks

1. **Hyperparameter-Tuning**: Experimentieren Sie mit verschiedenen Werten für `max_features`, `C` und `solver`, um die beste Konfiguration zu finden.

2. **Datenvorverarbeitung**: Die Qualität der Textdaten hat einen großen Einfluss auf die Modellleistung. Stellen Sie sicher, dass Ihre Daten gut vorverarbeitet sind.

3. **Klassenungleichgewicht**: Wenn Ihre Daten ein Ungleichgewicht zwischen den Klassen aufweisen, können Sie den Parameter `class_weight='balanced'` hinzufügen, um dies auszugleichen.

4. **Modellvergleich**: Vergleichen Sie die Ergebnisse der logistischen Regression mit denen des neuronalen Netzes (siehe Tutorial `train_nn_einfach.md`), um zu sehen, welches Modell für Ihre Daten besser geeignet ist.

## Nächste Schritte

Nachdem Sie ein logistisches Regressionsmodell trainiert haben, können Sie:

- Ein neuronales Netz trainieren und die Ergebnisse vergleichen (siehe Tutorial `train_nn_einfach.md`)
- Das Hauptmodul verwenden, um verschiedene Modelle zu trainieren (siehe Tutorial `main_einfach.md`)
- Die grafische Benutzeroberfläche verwenden, um die Ergebnisse zu visualisieren (siehe Anleitung `gui_anleitung.md`)