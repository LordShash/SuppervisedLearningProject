# Moderne GUI für die Textklassifikation - Technische Dokumentation

Diese technische Dokumentation beschreibt die Implementierung der modernen grafischen Benutzeroberfläche (GUI) für die Textklassifikationsanwendung.

## Architektur

Die moderne GUI ist in mehrere Module aufgeteilt, um die Wartbarkeit und Erweiterbarkeit zu verbessern:

1. **gui.py**: Enthält die Basisklassen und grundlegende Funktionalität
2. **gui_tabs.py**: Enthält die Implementierung der verschiedenen Tabs
3. **gui_complete.py**: Integriert die beiden Module zu einer vollständigen Anwendung
4. **run_gui.py**: Startskript für die Anwendung

### Klassendiagramm

```
ModernTextClassificationGUI (QMainWindow)
├── MatplotlibCanvas (FigureCanvas)
└── TrainingWorker (QThread)
```

## Technologien

Die moderne GUI verwendet folgende Technologien:

- **PyQt5**: Framework für die Erstellung der grafischen Benutzeroberfläche
- **Matplotlib**: Bibliothek für die Visualisierung von Daten
- **QThread**: Klasse für die Ausführung von Aufgaben in separaten Threads

## Modulbeschreibungen

### gui.py

Dieses Modul enthält die Basisklassen für die GUI:

- **MatplotlibCanvas**: Eine Klasse, die Matplotlib-Figuren in PyQt5 einbettet
- **TrainingWorker**: Ein Worker-Thread für die Ausführung des Trainings im Hintergrund
- **ModernTextClassificationGUI**: Die Hauptklasse für die GUI, die von QMainWindow erbt

Die ModernTextClassificationGUI-Klasse enthält:
- Initialisierungscode für die GUI
- Stylesheet-Definitionen für das moderne Aussehen
- Grundlegende UI-Struktur mit Tabs

### gui_tabs.py

Dieses Modul enthält die Implementierung der verschiedenen Tabs und der zugehörigen Funktionalität:

- **init_training_tab**: Initialisiert den Training-Tab mit Formularelementen
- **init_results_tab**: Initialisiert den Ergebnisse-Tab mit Visualisierungen
- **init_about_tab**: Initialisiert den Über-Tab mit Informationen
- Verschiedene Hilfsfunktionen für die Interaktion mit der GUI

### gui_complete.py

Dieses Modul integriert die Basisklassen aus gui.py mit den Tab-Implementierungen aus gui_tabs.py:

```python
# Erweitere die ModernTextClassificationGUI-Klasse um die Tab-Implementierungen
ModernTextClassificationGUI.init_training_tab = init_training_tab
ModernTextClassificationGUI.init_results_tab = init_results_tab
ModernTextClassificationGUI.init_about_tab = init_about_tab
# ... weitere Methoden
```

### run_gui.py

Dieses Skript dient als Einstiegspunkt für die Anwendung:

```python
def main():
    # Erstelle die QApplication
    app = QApplication(sys.argv)
    
    # Erstelle und zeige das Hauptfenster
    window = ModernTextClassificationGUI()
    window.show()
    
    # Starte die Anwendung
    sys.exit(app.exec_())
```

## Datenfluss

1. Der Benutzer gibt Parameter im Training-Tab ein und klickt auf "Modell trainieren"
2. Die `train_model`-Methode erstellt einen `TrainingWorker`-Thread und startet ihn
3. Der `TrainingWorker` führt das Training im Hintergrund aus und sendet Fortschrittsmeldungen an die GUI
4. Nach Abschluss des Trainings werden die Ergebnisse in der GUI angezeigt und im `training_results`-Dictionary gespeichert
5. Die Anwendung wechselt zum Ergebnisse-Tab, wo der Benutzer die Ergebnisse einsehen kann

## Signale und Slots

Die moderne GUI verwendet das Signal-Slot-System von PyQt5 für die Kommunikation zwischen Threads:

- **update_output**: Signal zum Aktualisieren der Ausgabe im Training-Tab
- **training_finished**: Signal, das gesendet wird, wenn das Training abgeschlossen ist
- **training_error**: Signal, das gesendet wird, wenn ein Fehler beim Training auftritt

## Styling

Das Styling der GUI wird über ein Stylesheet in der `set_stylesheet`-Methode definiert. Hier werden Farben, Schriftarten, Abstände und andere visuelle Eigenschaften festgelegt.

## Erweiterung der GUI

### Hinzufügen eines neuen Tabs

Um einen neuen Tab hinzuzufügen:

1. Erstellen Sie eine neue Methode `init_new_tab` in gui_tabs.py
2. Fügen Sie die Methode zur ModernTextClassificationGUI-Klasse in gui_complete.py hinzu
3. Rufen Sie die Methode in der `init_ui`-Methode auf

### Hinzufügen eines neuen Modelltyps

Um einen neuen Modelltyp hinzuzufügen:

1. Erweitern Sie die Radiobuttons im Training-Tab
2. Fügen Sie eine neue Gruppe für modellspezifische Parameter hinzu
3. Erweitern Sie die `toggle_model_options`-Methode
4. Erweitern Sie die `train_model`-Methode, um den neuen Modelltyp zu unterstützen

## Bekannte Einschränkungen

- Die GUI ist nicht für sehr kleine Bildschirme optimiert
- Die Konfusionsmatrix kann bei vielen Klassen unübersichtlich werden
- Das Training großer Modelle kann trotz Threading die GUI kurzzeitig blockieren

## Fehlerbehebung

- **ImportError: No module named 'PyQt5'**: PyQt5 ist nicht installiert. Installieren Sie es mit `pip install PyQt5`
- **RuntimeError: QPixmap: Must construct a QApplication before a QPaintDevice**: Die QApplication wurde nicht korrekt initialisiert
- **AttributeError: 'NoneType' object has no attribute 'count'**: Ein Layout wurde nicht korrekt initialisiert oder gelöscht