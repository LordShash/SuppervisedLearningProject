# Tutorial: Neuronale Netze für Textklassifikation

In diesem Tutorial lernen Sie, wie Sie das Modul `train_nn.py` verwenden, um ein neuronales Netz für die Textklassifikation zu trainieren.

## Einführung

Neuronale Netze sind leistungsstarke Modelle für komplexe Klassifikationsaufgaben. Das Modul `train_nn.py` bietet eine einfache Möglichkeit, ein neuronales Netz für die Textklassifikation zu trainieren und zu evaluieren.

## Grundlegende Verwendung

Sie können das Modul direkt in Ihrem eigenen Python-Code importieren und verwenden:

```python
from train_nn import train_and_save_model

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
    epochs=50,                       # Mehr Trainingsepochen
    patience=10                      # Längere Geduld für Early Stopping
)
```

### Wichtige Parameter

- **target_column**: Die Zielvariable für die Klassifikation (z.B. "Fits_Topic_Code", "Urgency_Code")
- **max_features**: Maximale Anzahl der Features für die TF-IDF-Vektorisierung
- **test_size**: Anteil der Daten, der für den Testdatensatz verwendet wird (0.0 bis 1.0)
- **epochs**: Anzahl der Trainingsepochen
- **patience**: Anzahl der Epochen ohne Verbesserung, bevor das Training frühzeitig beendet wird

## Modellarchitektur

Das verwendete neuronale Netz hat eine einfache Architektur:

1. **Eingabeschicht**: Die Größe entspricht der Anzahl der Features (max_features)
2. **Versteckte Schicht**: 128 Neuronen mit ReLU-Aktivierungsfunktion und Dropout (0.2)
3. **Ausgabeschicht**: Die Größe entspricht der Anzahl der Klassen mit Softmax-Aktivierungsfunktion

Diese Architektur ist für viele Textklassifikationsaufgaben gut geeignet, kann aber bei Bedarf im Quellcode angepasst werden.

## Modellspeicherung und -verwendung

Das trainierte Modell wird automatisch im Verzeichnis `models/` gespeichert:

```
models/nn_Fits_Topic_Code_model.pt
```

Sie können das gespeicherte Modell später für Vorhersagen verwenden:

```python
import torch
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Modellklasse definieren (muss mit der Trainingsklasse übereinstimmen)
class TextClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TextClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(128, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Modell laden
model_path = os.path.join("models", "nn_Fits_Topic_Code_model.pt")
input_size = 1000  # Muss mit max_features beim Training übereinstimmen
output_size = 10   # Muss mit der Anzahl der Klassen übereinstimmen

model = TextClassifier(input_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()  # Modell in Evaluierungsmodus setzen

# Text vektorisieren (Sie müssen denselben Vektorisierer verwenden wie beim Training)
vectorizer = TfidfVectorizer(max_features=1000)
# Hier müssten Sie den Vektorisierer mit denselben Daten trainieren oder speichern/laden

# Vorhersage für einen neuen Text
text = "Dies ist ein Beispieltext für die Vorhersage."
X_new = vectorizer.transform([text]).toarray()
X_tensor = torch.FloatTensor(X_new)

with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)
    
print(f"Vorhersage: {predicted.item()}")
```

## Trainingsfortschritt

Während des Trainings gibt das Modul Informationen zum Fortschritt aus:

```
Epoch 1/30, Loss: 2.3026, Validation Accuracy: 0.2345
Epoch 2/30, Loss: 1.8721, Validation Accuracy: 0.4567
...
Epoch 15/30, Loss: 0.3456, Validation Accuracy: 0.8123
Early stopping nach 15 Epochen
```

Das Training wird automatisch beendet, wenn sich die Validierungsgenauigkeit für `patience` Epochen nicht verbessert (Early Stopping).

## Interpretation der Ergebnisse

Nach dem Training gibt das Modul verschiedene Metriken zurück:

1. **Accuracy**: Der Anteil der korrekt klassifizierten Beispiele (höher ist besser)
2. **Precision**: Der Anteil der korrekt positiven Vorhersagen an allen positiven Vorhersagen (höher ist besser)
3. **Recall**: Der Anteil der korrekt positiven Vorhersagen an allen tatsächlich positiven Beispielen (höher ist besser)
4. **F1-Score**: Das harmonische Mittel aus Precision und Recall (höher ist besser)
5. **Klassifikationsbericht**: Detaillierte Metriken für jede Klasse
6. **Konfusionsmatrix**: Eine Matrix, die zeigt, wie viele Beispiele jeder Klasse korrekt oder falsch klassifiziert wurden

## Tipps und Tricks

1. **Hyperparameter-Tuning**: Experimentieren Sie mit verschiedenen Werten für `max_features`, `epochs` und `patience`, um die beste Konfiguration zu finden.

2. **GPU-Beschleunigung**: Wenn verfügbar, verwendet das Modul automatisch eine GPU für schnelleres Training. Stellen Sie sicher, dass PyTorch mit CUDA-Unterstützung installiert ist, wenn Sie eine NVIDIA-GPU haben.

3. **Modellarchitektur**: Für komplexere Aufgaben können Sie die Modellarchitektur im Quellcode anpassen, z.B. durch Hinzufügen weiterer Schichten oder Ändern der Neuronenzahl.

4. **Modellvergleich**: Vergleichen Sie die Ergebnisse des neuronalen Netzes mit denen der logistischen Regression (siehe Tutorial `train_logreg_einfach.md`), um zu sehen, welches Modell für Ihre Daten besser geeignet ist.

## Nächste Schritte

Nachdem Sie ein neuronales Netz trainiert haben, können Sie:

- Ein logistisches Regressionsmodell trainieren und die Ergebnisse vergleichen (siehe Tutorial `train_logreg_einfach.md`)
- Das Hauptmodul verwenden, um verschiedene Modelle zu trainieren (siehe Tutorial `main_einfach.md`)
- Die grafische Benutzeroberfläche verwenden, um die Ergebnisse zu visualisieren (siehe Anleitung `gui_anleitung.md`)