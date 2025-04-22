# Moderne GUI für die Textklassifikation - Einfache Anleitung

Diese Anleitung erklärt, wie Sie die moderne grafische Benutzeroberfläche (GUI) für die Textklassifikationsanwendung verwenden können.

## Voraussetzungen

Die moderne GUI benötigt PyQt5. Das Startskript prüft automatisch, ob PyQt5 installiert ist und bietet Ihnen verschiedene Optionen:

### Option 1: Automatische Installation

Wenn Sie das Skript starten und PyQt5 nicht installiert ist, werden Sie gefragt, ob Sie es automatisch installieren möchten:

```
Möchten Sie PyQt5 jetzt installieren? (j/n):
```

Wenn Sie mit "j" oder "ja" antworten, wird PyQt5 automatisch installiert.

### Option 2: Manuelle Installation

Sie können PyQt5 auch manuell installieren:

```bash
pip install PyQt5
```

### Option 3: Standard-GUI verwenden

Wenn Sie PyQt5 nicht installieren möchten, können Sie stattdessen die Standard-GUI verwenden, die keine zusätzlichen Abhängigkeiten benötigt. Das Skript wird Sie fragen, ob Sie die Standard-GUI starten möchten:

```
Möchten Sie stattdessen die Standard-GUI starten? (j/n):
```

## Starten der modernen GUI

Um die moderne GUI zu starten, führen Sie das folgende Kommando im Hauptverzeichnis des Projekts aus:

```bash
python run_modern_gui.py
```

## Überblick über die Benutzeroberfläche

Die moderne GUI ist in drei Tabs organisiert:

1. **Training**: Hier können Sie Modelle trainieren
2. **Ergebnisse**: Hier werden die Trainingsergebnisse angezeigt
3. **Über**: Hier finden Sie Informationen über die Anwendung

### Training-Tab

Im Training-Tab können Sie:

- Eine **Zielvariable** aus der Dropdown-Liste auswählen
- Den **Modelltyp** wählen (Logistische Regression oder Neuronales Netz)
- Die **Parameter** für das Training anpassen:
  - **Max Features**: Maximale Anzahl der Features für die Vektorisierung
  - **Test Size**: Anteil der Daten, der für den Test verwendet wird

Je nach gewähltem Modelltyp werden unterschiedliche Parameter angezeigt:

- **Logistische Regression**:
  - **Max Iterations**: Maximale Anzahl der Iterationen
  - **C (Regularisierung)**: Regularisierungsparameter
  - **Solver**: Algorithmus für die Optimierung

- **Neuronales Netz**:
  - **Epochs**: Anzahl der Trainingszyklen
  - **Patience**: Anzahl der Epochen ohne Verbesserung, bevor das Training frühzeitig beendet wird

Klicken Sie auf den Button **"Modell trainieren"**, um das Training zu starten. Der Fortschritt und die Ergebnisse werden im Ausgabebereich angezeigt.

### Ergebnisse-Tab

Nach dem Training wechselt die Anwendung automatisch zum Ergebnisse-Tab. Hier können Sie:

- Verschiedene trainierte Modelle aus der Dropdown-Liste auswählen
- Die **Metriken** des ausgewählten Modells sehen (Accuracy, Precision, Recall, F1 Score)
- Den detaillierten **Klassifikationsbericht** einsehen
- Die **Konfusionsmatrix** visualisieren

Die Konfusionsmatrix zeigt, wie gut das Modell die verschiedenen Klassen klassifiziert hat. Die Zahlen in der Matrix geben an, wie viele Beispiele einer tatsächlichen Klasse (Zeilen) als eine bestimmte Klasse (Spalten) vorhergesagt wurden.

### Über-Tab

Der Über-Tab enthält allgemeine Informationen über die Anwendung, eine kurze Beschreibung und eine Anleitung zur Verwendung.

## Vorteile der modernen GUI

Die moderne GUI bietet im Vergleich zur Standard-GUI folgende Vorteile:

- **Modernes Design**: Flaches, ansprechendes Design mit modernen Farben und Schattierungen
- **Verbesserte Benutzerfreundlichkeit**: Intuitivere Bedienelemente und bessere Visualisierungen
- **Responsives Layout**: Passt sich an verschiedene Bildschirmgrößen an
- **Verbesserte Visualisierungen**: Interaktive Elemente für eine bessere Datenanalyse

## Tipps zur Verwendung

- **Tooltips**: Halten Sie den Mauszeiger über die Parameter, um Erklärungen zu erhalten
- **Modellvergleich**: Trainieren Sie mehrere Modelle mit unterschiedlichen Parametern und vergleichen Sie die Ergebnisse im Ergebnisse-Tab
- **Konfusionsmatrix**: Achten Sie auf die Diagonale der Konfusionsmatrix - hohe Werte auf der Diagonale bedeuten gute Klassifikationsergebnisse
