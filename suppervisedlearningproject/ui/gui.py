"""
Modernes GUI-Modul für die Textklassifikationsanwendung mit PyQt5.

Dieses Modul stellt eine moderne, ansprechende grafische Oberfläche bereit,
um verschiedene Modelle zu trainieren und die Ergebnisse zu visualisieren.
Es verwendet PyQt5 für ein modernes Look-and-Feel.
"""

import os
import sys
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import io
from contextlib import redirect_stdout
import traceback
import atexit
import json
import pickle

# PyQt5 Imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QPushButton, QRadioButton, QSpinBox, QDoubleSpinBox,
    QTabWidget, QGroupBox, QFormLayout, QTextEdit, QSplitter, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap

# Matplotlib für Visualisierungen
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Importiere die Funktionen aus den anderen Modulen mit der neuen Paketstruktur
from suppervisedlearningproject.core.data_loader import load_data, get_available_targets
from suppervisedlearningproject.models.train_logreg import train_and_save_model as train_logreg
from suppervisedlearningproject.models.train_nn import train_and_save_model as train_nn
from suppervisedlearningproject.utils import BASE_DIR, DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, setup_logging


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib Canvas für die Einbettung in PyQt5."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)


class TrainingWorker(QThread):
    """Worker-Thread für das Training im Hintergrund."""

    # Signale für die Kommunikation mit dem Hauptthread
    update_output = pyqtSignal(str)
    training_finished = pyqtSignal(dict)
    training_error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.setTerminationEnabled(True)

    def run(self):
        """Führt das Training im Hintergrund aus."""
        try:
            # Hole die gemeinsamen Parameter
            target_column = self.params['target_column']
            model_type = self.params['model_type']
            max_features = self.params['max_features']
            test_size = self.params['test_size']

            # Ausgabe der Parameter mit verbesserter Formatierung
            model_name = "logistischen Regressionsmodells" if model_type == "logreg" else "neuronalen Netzes"
            self.update_output.emit(f"=== Training eines {model_name} ===\n")
            self.update_output.emit(f"Allgemeine Parameter:")
            self.update_output.emit(f"  • Zielvariable:  {target_column}")
            self.update_output.emit(f"  • Max Features:  {max_features}")
            self.update_output.emit(f"  • Test Size:     {test_size}")

            # Modellspezifische Parameter und Training
            if model_type == "logreg":
                max_iter = self.params['max_iter']
                c_reg = self.params['c_reg']
                solver = self.params['solver']

                # Modellspezifische Parameter
                model_params = {
                    'max_iter': max_iter,
                    'C': c_reg,
                    'solver': solver
                }

                # Parameter-Beschreibungen für die Ausgabe mit verbesserter Formatierung
                self.update_output.emit(f"Modellspezifische Parameter:")
                self.update_output.emit(f"  • Max Iterations:    {max_iter}")
                self.update_output.emit(f"  • C (Regularisierung): {c_reg}")
                self.update_output.emit(f"  • Solver:           {solver}")
                self.update_output.emit("")  # Leerzeile

                # Training des Modells
                self.update_output.emit("=== Starte Training ===\n")

                # Umleitung der Standardausgabe
                f = io.StringIO()
                with redirect_stdout(f):
                    # Trainiere das Modell
                    results = train_logreg(
                        target_column=target_column,
                        max_features=max_features,
                        test_size=test_size,
                        **model_params
                    )

            else:  # Neuronales Netz
                epochs = self.params['epochs']
                patience = self.params['patience']

                # Modellspezifische Parameter
                model_params = {
                    'epochs': epochs,
                    'patience': patience
                }

                # Parameter-Beschreibungen für die Ausgabe mit verbesserter Formatierung
                self.update_output.emit(f"Modellspezifische Parameter:")
                self.update_output.emit(f"  • Epochs:           {epochs}")
                self.update_output.emit(f"  • Patience:         {patience}")
                self.update_output.emit("")  # Leerzeile

                # Training des Modells
                self.update_output.emit("=== Starte Training ===\n")

                # Umleitung der Standardausgabe
                f = io.StringIO()
                with redirect_stdout(f):
                    # Trainiere das Modell
                    results = train_nn(
                        target_column=target_column,
                        max_features=max_features,
                        test_size=test_size,
                        **model_params
                    )

            # Ausgabe der umgeleiteten Standardausgabe
            self.update_output.emit(f.getvalue())

            # Extrahiere die Ergebnisse
            accuracy = results.accuracy
            precision = results.precision
            recall = results.recall
            f1 = results.f1
            report = results.report
            conf_matrix = results.conf_matrix
            class_names = results.class_names
            roc_curve_data = results.roc_curve_data
            auc_scores = results.auc_scores

            # Erstelle Ergebnisdictionary
            results_dict = {
                'model_type': model_type,
                'target_column': target_column,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'report': report,
                'conf_matrix': conf_matrix,
                'class_names': class_names,
                'roc_curve_data': roc_curve_data,
                'auc_scores': auc_scores,
                'params': {
                    'target_column': target_column,
                    'max_features': max_features,
                    'test_size': test_size,
                    **model_params
                }
            }

            # Ausgabe der Ergebnisse mit verbesserter Formatierung
            self.update_output.emit("\n=== Training abgeschlossen ===")
            self.update_output.emit("\nErgebnismetriken:")
            self.update_output.emit(f"  • Accuracy:    {accuracy:.4f}")
            self.update_output.emit(f"  • Precision:   {precision:.4f}")
            self.update_output.emit(f"  • Recall:      {recall:.4f}")
            self.update_output.emit(f"  • F1 Score:    {f1:.4f}")

            # Sende die Ergebnisse an den Hauptthread
            self.training_finished.emit(results_dict)

        except Exception as e:
            # Fehlerbehandlung mit verbesserter Formatierung
            self.update_output.emit("\n=== Fehler beim Training ===")
            self.update_output.emit(f"Fehlermeldung: {str(e)}")
            self.update_output.emit("\nDetails:")
            self.update_output.emit(traceback.format_exc())
            self.training_error.emit(str(e))


class ModernTextClassificationGUI(QMainWindow):
    """Hauptklasse für die moderne grafische Benutzeroberfläche der Textklassifikationsanwendung."""

    def __init__(self):
        super().__init__()

        # Store reference to figures for cleanup
        self.figures = []

        # Tooltip-Erklärungen für Parameter mit detaillierten Informationen
        self.tooltips = {
            # Gemeinsame Parameter
            "max_features": """Maximale Anzahl der Features, die für die Vektorisierung verwendet werden.

Ein höherer Wert (z.B. 2000 statt 1000) kann die Genauigkeit verbessern, erhöht aber auch:
• Die Rechenzeit für das Training
• Den Speicherbedarf des Modells
• Das Risiko von Overfitting bei kleinen Datensätzen

Empfohlene Werte:
• Für kleine Datensätze: 500-1000
• Für mittlere Datensätze: 1000-2000
• Für große Datensätze: 2000-5000""",

            "test_size": """Anteil der Daten, der für den Testdatensatz verwendet wird (0.0 bis 1.0).

Ein Wert von 0.2 bedeutet, dass 20% der Daten für Tests und 80% für das Training verwendet werden.

Auswirkungen:
• Kleinerer Wert (z.B. 0.1): Mehr Trainingsdaten, aber weniger zuverlässige Evaluierung
• Größerer Wert (z.B. 0.3): Weniger Trainingsdaten, aber zuverlässigere Evaluierung

Empfohlene Werte:
• Standard: 0.2 (20%)
• Bei kleinen Datensätzen: 0.15-0.2
• Bei großen Datensätzen: 0.2-0.25""",

            # Logistische Regression Parameter
            "max_iter": """Maximale Anzahl der Iterationen für die logistische Regression.

Bestimmt, wie lange das Modell versucht, eine optimale Lösung zu finden.

Auswirkungen:
• Zu niedrig: Das Modell konvergiert möglicherweise nicht (Warnmeldung)
• Zu hoch: Längere Trainingszeit, aber bessere Chance auf Konvergenz

Empfohlene Werte:
• Standard: 1000
• Bei komplexen Problemen: 2000-5000
• Bei einfachen Problemen: 500-1000""",

            "c_reg": """Regularisierungsparameter C für die logistische Regression.

Kontrolliert die Stärke der Regularisierung (Vermeidung von Overfitting).

Auswirkungen:
• Kleinerer Wert (z.B. 0.1): Stärkere Regularisierung, einfacheres Modell
• Größerer Wert (z.B. 10.0): Schwächere Regularisierung, komplexeres Modell

Empfohlene Werte:
• Standard: 1.0
• Bei Overfitting: 0.1-0.5
• Bei Underfitting: 2.0-10.0""",

            "solver": """Algorithmus für die Optimierung der logistischen Regression.

Verschiedene Solver haben unterschiedliche Stärken und Schwächen:

• lbfgs: Schnell für kleine bis mittlere Datensätze, gut für Multiclass-Probleme
• newton-cg: Präzise, aber rechenintensiv
• liblinear: Gut für kleine Datensätze, nur für binäre Klassifikation oder One-vs-Rest
• sag: Schnell für große Datensätze
• saga: Schnell für große Datensätze, unterstützt L1-Regularisierung

Empfehlung:
• Für die meisten Fälle: lbfgs
• Für sehr große Datensätze: saga""",

            # Neuronales Netz Parameter
            "epochs": """Anzahl der Trainingszyklen für das neuronale Netz.

Ein Epoch bedeutet, dass das Modell einmal den gesamten Trainingsdatensatz durchlaufen hat.

Auswirkungen:
• Zu wenige Epochs: Underfitting (Modell lernt nicht genug)
• Zu viele Epochs: Overfitting und längere Trainingszeit

Empfohlene Werte:
• Standard: 30
• Bei komplexen Problemen: 50-100
• Bei einfachen Problemen: 10-20

Hinweis: Early Stopping (über 'Patience') kann helfen, die optimale Anzahl automatisch zu finden.""",

            "patience": """Anzahl der Epochen ohne Verbesserung, bevor das Training frühzeitig beendet wird.

Dies ist ein Early-Stopping-Mechanismus, der Overfitting verhindert.

Auswirkungen:
• Kleinerer Wert (z.B. 2): Schnelleres Training, aber Risiko von vorzeitigem Abbruch
• Größerer Wert (z.B. 10): Längeres Training, aber bessere Chance auf Konvergenz

Empfohlene Werte:
• Standard: 5
• Bei instabilem Training: 7-10
• Bei stabilem Training: 3-5"""
        }

        # Speichere die Trainingsergebnisse
        self.training_results = {}

        # Pfad zur Datei mit den gespeicherten Trainingsergebnissen
        self.results_file = os.path.join(DATA_DIR, 'training_results.pkl')

        # Lade gespeicherte Trainingsergebnisse, falls vorhanden
        self.load_training_results()

        # Initialisiere die UI
        self.init_ui()

        # Aktualisiere die Ergebnisse im Ergebnisse-Tab mit den geladenen Daten
        self.update_results_tab()

        # Lade die verfügbaren Zielvariablen
        try:
            self.targets = get_available_targets()
            self.target_combo.addItems(list(self.targets.keys()))
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden der Zielvariablen: {str(e)}")

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche."""
        # Setze Fenstertitel und Größe
        self.setWindowTitle("Textklassifikation - Moderne Benutzeroberfläche")
        self.setMinimumSize(1000, 700)

        # Empfohlene Größe für bessere Darstellung
        self.resize(1200, 800)

        # Erstelle ein zentrales Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Hauptlayout mit Margins für besseren Abstand zum Rand
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Erstelle ein Tab-Widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Erstelle die Tabs
        self.init_training_tab()
        self.init_results_tab()
        self.init_about_tab()

        # Setze das Stylesheet für ein modernes Aussehen
        self.set_stylesheet()

        # Verbinde das Tab-Changed-Signal mit einer Methode zur Anpassung der Layouts
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def init_training_tab(self):
        """Initialisiert den Training-Tab."""
        # Einfache Implementierung für den Test
        training_tab = QWidget()
        self.tab_widget.addTab(training_tab, "Training")

    def init_results_tab(self):
        """Initialisiert den Ergebnisse-Tab mit Visualisierungen."""
        # Erstelle den Tab
        results_tab = QWidget()
        self.tab_widget.addTab(results_tab, "Ergebnisse")

        # Layout für den Tab
        layout = QVBoxLayout(results_tab)

        # Modellauswahl
        model_select_group = QGroupBox("Modellauswahl")
        layout.addWidget(model_select_group)
        model_select_layout = QHBoxLayout(model_select_group)

        model_select_label = QLabel("Modell:")
        self.model_select_combo = QComboBox()

        # Initialisiere auch die Konfusionsmatrix-Modellauswahl
        # Diese wird in update_results_tab verwendet
        self.confusion_model_select_combo = QComboBox()

        model_select_layout.addWidget(model_select_label)
        model_select_layout.addWidget(self.model_select_combo)

    def init_about_tab(self):
        """Initialisiert den Über-Tab."""
        # Einfache Implementierung für den Test
        about_tab = QWidget()
        self.tab_widget.addTab(about_tab, "Über")

    def update_results_tab(self):
        """Aktualisiert die Modellauswahl in allen Tabs."""
        try:
            # Prüfe, ob training_results existiert und ein Dictionary ist
            if not hasattr(self, 'training_results') or not isinstance(self.training_results, dict):
                logger = setup_logging("gui")
                logger.warning("Warnung: training_results ist nicht verfügbar oder kein Dictionary")
                return

            # Blockiere Signale, um unerwünschte Aktualisierungen zu vermeiden
            if hasattr(self, 'model_select_combo'):
                self.model_select_combo.blockSignals(True)
            if hasattr(self, 'confusion_model_select_combo'):
                self.confusion_model_select_combo.blockSignals(True)

            # Entsperre die Signale wieder
            if hasattr(self, 'model_select_combo'):
                self.model_select_combo.blockSignals(False)
            if hasattr(self, 'confusion_model_select_combo'):
                self.confusion_model_select_combo.blockSignals(False)

        except Exception as e:
            # Fange alle Ausnahmen ab und protokolliere sie
            logger = setup_logging("gui")
            logger.error(f"Fehler in update_results_tab: {str(e)}")

    def set_stylesheet(self):
        """Setzt das Stylesheet für ein modernes Aussehen."""
        self.setStyleSheet("""
            /* Globale Schriftart-Einstellungen für Konsistenz */
            * {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 10pt;
            }

            QMainWindow {
                background-color: #f0f4f8;  /* Hellerer, moderner Blauton als Hintergrund */
            }

            /* Responsive Design für Tabs */
            QTabWidget::pane {
                border: 1px solid #d0d9e1;
                background-color: white;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtiler Schatten für Tiefe */
            }

            QTabBar::tab {
                background-color: #e8eef4;
                border: 1px solid #d0d9e1;
                border-bottom-color: #d0d9e1;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 8ex;
                padding: 8px 16px;
                margin-right: 2px;
                color: #455a64;  /* Dunklerer Blauton für Text */
                /* Verbesserte Textdarstellung */
                text-align: center;
            }

            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                color: #00796b;  /* Türkis für ausgewählten Tab */
                font-weight: bold;
                /* Etwas mehr horizontale Polsterung für die fette Schrift */
                padding: 8px 20px;
                /* Erlaube dem Tab, sich an die Textbreite anzupassen */
                min-width: 12ex;
                /* Verbesserte Textdarstellung */
                text-align: center;
            }

            QTabBar::tab:hover {
                background-color: #f5f9ff;
                color: #00796b;  /* Türkis beim Hover */
                /* Verbesserte Textdarstellung */
                text-align: center;
            }

            /* Buttons mit modernem Design */
            QPushButton {
                background-color: #009688;  /* Modernes Türkis statt Blau */
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtiler Schatten für Tiefe */
            }

            QPushButton:hover {
                background-color: #00897b;  /* Dunkleres Türkis beim Hover */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);  /* Stärkerer Schatten beim Hover */
            }

            QPushButton:pressed {
                background-color: #00695c;  /* Noch dunkleres Türkis beim Drücken */
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);  /* Flacherer Schatten beim Drücken */
            }

            QPushButton:disabled {
                background-color: #b2dfdb;  /* Helleres Türkis für deaktivierte Buttons */
                color: #e0f2f1;
                box-shadow: none;
            }

            /* Gruppierungen mit modernem Design */
            QGroupBox {
                border: 1px solid #d0d9e1;
                border-radius: 6px;
                margin-top: 1.5ex;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 0.7);  /* Leicht transparentes Weiß */
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);  /* Sehr subtiler Schatten */
                padding-top: 16px;  /* Mehr Platz für den Titel */
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #00796b;  /* Türkis für Gruppentitel */
                font-weight: bold;
            }

            /* Eingabefelder mit einheitlichem Design */
            QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #d0d9e1;
                border-radius: 6px;
                padding: 5px;
                background-color: white;
                selection-background-color: #e0f2f1;  /* Helles Türkis für Auswahl */
                min-height: 25px;  /* Mindesthöhe für bessere Bedienbarkeit */
            }

            /* Textfelder mit Monospace-Schrift für bessere Lesbarkeit von Code und Berichten */
            QTextEdit {
                border: 1px solid #d0d9e1;
                border-radius: 6px;
                background-color: white;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 10pt;
                line-height: 1.5;
                selection-background-color: #e0f2f1;  /* Helles Türkis für Textauswahl */
                padding: 8px;
                min-height: 100px;  /* Mindesthöhe für bessere Lesbarkeit */
            }

            /* Labels mit einheitlichem Design */
            QLabel {
                color: #37474f;  /* Dunklerer, moderner Blauton für Text */
                min-height: 20px;  /* Mindesthöhe für bessere Lesbarkeit */
            }

            /* Splitter für anpassbare Layouts */
            QSplitter::handle {
                background-color: #d0d9e1;
                height: 2px;
            }

            QSplitter::handle:hover {
                background-color: #009688;  /* Türkis beim Hover */
            }

            /* Tooltip-Stil für Informationsanzeigen */
            QToolTip {
                background-color: #f5f5f5;
                color: #333333;
                border: 1px solid #d0d9e1;
                border-radius: 4px;
                padding: 5px;
                opacity: 230;
                font-size: 10pt;
            }
        """)

    def embed_canvas(self, parent, fig):
        """
        Bettet eine Matplotlib-Figur in ein PyQt-Widget ein.

        Args:
            parent: Das Eltern-Widget, in das die Figur eingebettet werden soll
            fig: Die Matplotlib-Figur, die eingebettet werden soll

        Returns:
            canvas: Der erstellte MatplotlibCanvas
        """
        # räume altes Canvas auf
        layout = parent.layout()
        if layout is None:
            layout = QVBoxLayout(parent)
            parent.setLayout(layout)
        while layout.count():
            old = layout.takeAt(0).widget()
            if old:
                old.setParent(None)
        canvas = MatplotlibCanvas(parent)
        canvas.fig = fig
        canvas.axes = fig.axes[0] if fig.axes else fig.add_subplot(111)
        canvas.draw()
        layout.addWidget(canvas)
        self.figures.append(fig)  # zur Aufräum-Liste
        return canvas

    def cleanup_resources(self):
        """
        Bereinigt Ressourcen, die von der Anwendung verwendet werden.
        """
        # Close all matplotlib figures
        for fig in self.figures:
            plt.close(fig)
        self.figures = []
        plt.close('all')

    def on_tab_changed(self, index):
        """
        Wird aufgerufen, wenn der Tab gewechselt wird.
        Passt die Layouts an, um sicherzustellen, dass alle Inhalte korrekt angezeigt werden.

        Args:
            index: Index des neuen Tabs
        """
        # Aktualisiere das Layout des aktuellen Tabs
        current_tab = self.tab_widget.widget(index)
        if current_tab:
            # Erzwinge ein Update des Layouts
            current_tab.layout().activate()
            current_tab.layout().update()

            # Wenn es der Ergebnisse-Tab ist, aktualisiere die Textfelder und Visualisierungen
            if index == 1 and hasattr(self, 'report_text'):
                # Stelle sicher, dass der Text im Klassifikationsbericht korrekt angezeigt wird
                self.report_text.document().adjustSize()

                # Aktualisiere alle Visualisierungen, falls ein Modell ausgewählt ist
                if hasattr(self, 'model_select_combo') and self.model_select_combo.currentText():
                    model_key = self.model_select_combo.currentText()
                    if model_key in self.training_results:
                        # Hole die Ergebnisse für das ausgewählte Modell
                        results = self.training_results[model_key]

                        # Rufe on_model_selected auf, um alle Visualisierungen zu aktualisieren
                        logger = setup_logging("gui")
                        logger.info(f"Tab gewechselt zu Ergebnisse, aktualisiere Visualisierungen für Modell '{model_key}'")

                        # Finde den Index des Modells in der ComboBox
                        index = self.model_select_combo.findText(model_key)
                        if index >= 0:
                            # Rufe on_model_selected auf, um alle Visualisierungen zu aktualisieren
                            self.on_model_selected(index)
                        else:
                            logger.warning(f"Modell '{model_key}' nicht in ComboBox gefunden")
                    else:
                        logger = setup_logging("gui")
                        logger.warning(f"Modell '{model_key}' nicht in training_results gefunden")

            # Wenn es der Konfusionsmatrix-Tab ist, aktualisiere die Konfusionsmatrix
            elif index == 2 and hasattr(self, 'confusion_model_select_combo'):
                # Aktualisiere die Konfusionsmatrix, falls vorhanden
                if hasattr(self, 'confusion_model_select_combo') and self.confusion_model_select_combo.currentText():
                    model_key = self.confusion_model_select_combo.currentText()
                    if model_key in self.training_results:
                        # Prüfe, ob die Konfusionsmatrix ein gültiges numpy-Array ist
                        conf_matrix = self.training_results[model_key].get('conf_matrix')
                        if isinstance(conf_matrix, np.ndarray):
                            # Prüfe, ob conf_matrix ein leeres Array ist und konvertiere es ggf. in ein 2D-Array
                            if conf_matrix.size == 0:
                                logger = setup_logging("gui")
                                logger.warning(f"Konfusionsmatrix für Modell '{model_key}' ist leer, wird in 2x2-Matrix umgewandelt")
                                conf_matrix = np.zeros((2, 2), dtype=np.int32)
                                # Aktualisiere die Konfusionsmatrix im training_results Dictionary
                                self.training_results[model_key]['conf_matrix'] = conf_matrix

                            # Prüfe, ob conf_matrix die richtige Form hat
                            if len(conf_matrix.shape) != 2:
                                logger = setup_logging("gui")
                                logger.warning(f"Konfusionsmatrix für Modell '{model_key}' hat ein ungültiges Format, wird in 2x2-Matrix umgewandelt")
                                conf_matrix = np.zeros((2, 2), dtype=np.int32)
                                # Aktualisiere die Konfusionsmatrix im training_results Dictionary
                                self.training_results[model_key]['conf_matrix'] = conf_matrix

                            # Prüfe, ob die Dimensionen zu klein sind
                            if conf_matrix.shape[0] < 2 or conf_matrix.shape[1] < 2:
                                logger = setup_logging("gui")
                                logger.warning(f"Konfusionsmatrix für Modell '{model_key}' hat zu kleine Dimensionen, wird auf 2x2 erweitert")
                                new_conf_matrix = np.zeros((2, 2), dtype=np.int32)
                                rows = min(conf_matrix.shape[0], 2)
                                cols = min(conf_matrix.shape[1], 2)
                                new_conf_matrix[:rows, :cols] = conf_matrix[:rows, :cols]
                                conf_matrix = new_conf_matrix
                                # Aktualisiere die Konfusionsmatrix im training_results Dictionary
                                self.training_results[model_key]['conf_matrix'] = conf_matrix

                            # Aktualisiere die Konfusionsmatrix
                            self.update_confusion_matrix(conf_matrix)
                        else:
                            logger = setup_logging("gui")
                            logger.warning(f"Konfusionsmatrix für Modell '{model_key}' ist kein numpy-Array, wird in 2x2-Matrix umgewandelt")
                            conf_matrix = np.zeros((2, 2), dtype=np.int32)
                            # Aktualisiere die Konfusionsmatrix im training_results Dictionary
                            self.training_results[model_key]['conf_matrix'] = conf_matrix
                            # Aktualisiere die Konfusionsmatrix
                            self.update_confusion_matrix(conf_matrix)

    def closeEvent(self, event):
        """
        Wird aufgerufen, wenn das Fenster geschlossen wird.
        """
        # Speichere die Trainingsergebnisse
        self.save_training_results()

        # Bereinige Ressourcen
        self.cleanup_resources()
        event.accept()

    def load_training_results(self):
        """
        Lädt gespeicherte Trainingsergebnisse aus einer Datei.
        """
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

            # Prüfe, ob die Datei existiert
            if os.path.exists(self.results_file):
                try:
                    with open(self.results_file, 'rb') as f:
                        # Lade die Trainingsergebnisse
                        loaded_results = pickle.load(f)

                        # Validiere die geladenen Ergebnisse
                        if not isinstance(loaded_results, dict):
                            logger = setup_logging("gui")
                            logger.warning(f"Warnung: Geladene Trainingsergebnisse sind kein Dictionary, sondern {type(loaded_results)}")
                            self.training_results = {}
                        else:
                            # Validiere jedes Modell in den Ergebnissen
                            valid_results = {}
                            for key, model_data in loaded_results.items():
                                if not isinstance(model_data, dict):
                                    logger = setup_logging("gui")
                                    logger.warning(f"Warnung: Modell '{key}' hat ungültiges Format, wird übersprungen")
                                    continue

                                # Prüfe, ob alle erforderlichen Schlüssel vorhanden sind
                                required_keys = ['accuracy', 'precision', 'recall', 'f1', 'report', 'conf_matrix', 'model_type', 'target_column']
                                missing_keys = [k for k in required_keys if k not in model_data]
                                if missing_keys:
                                    logger = setup_logging("gui")
                                    logger.warning(f"Warnung: Modell '{key}' fehlen Schlüssel: {missing_keys}, wird übersprungen")
                                    continue

                                # Prüfe, ob die Modelldatei noch existiert
                                model_type = model_data['model_type']
                                target_column = model_data['target_column']

                                # Bestimme den Pfad zur Modelldatei basierend auf dem Modelltyp
                                model_exists = False
                                if model_type == 'logreg':
                                    # Suche nach Logistic Regression Modellen im MODELS_DIR
                                    model_pattern = f"logreg_{target_column}_*_model.pkl"
                                    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"logreg_{target_column}_") and f.endswith("_model.pkl")]
                                    model_exists = len(model_files) > 0
                                elif model_type == 'nn':
                                    # Suche nach Neural Network Modellen im CHECKPOINTS_DIR
                                    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{target_column}_best_model.pt")
                                    model_exists = os.path.exists(checkpoint_path)

                                if not model_exists:
                                    logger = setup_logging("gui")
                                    logger.warning(f"Warnung: Modelldatei für '{key}' existiert nicht mehr, wird übersprungen")
                                    continue

                                # Prüfe, ob conf_matrix ein gültiges numpy-Array ist
                                if not isinstance(model_data['conf_matrix'], np.ndarray):
                                    logger = setup_logging("gui")
                                    logger.warning(f"Warnung: Konfusionsmatrix für Modell '{key}' ist kein numpy-Array, wird übersprungen")
                                    continue

                                # Prüfe, ob conf_matrix ein leeres Array ist und konvertiere es ggf. in ein 2D-Array
                                if model_data['conf_matrix'].size == 0:
                                    logger = setup_logging("gui")
                                    logger.warning(f"Warnung: Konfusionsmatrix für Modell '{key}' ist leer, wird in 2x2-Matrix umgewandelt")
                                    model_data['conf_matrix'] = np.zeros((2, 2), dtype=np.int32)

                                # Modell ist gültig, füge es zu den validierten Ergebnissen hinzu
                                valid_results[key] = model_data

                            self.training_results = valid_results
                            logger = setup_logging("gui")
                            logger.info(f"Trainingsergebnisse geladen: {len(self.training_results)} gültige Modelle von {len(loaded_results)} gefunden.")
                except (pickle.UnpicklingError, EOFError) as pe:
                    logger = setup_logging("gui")
                    logger.error(f"Fehler beim Entpacken der Trainingsergebnisse: {str(pe)}")
                    logger.error("Die Datei mit den Trainingsergebnissen ist möglicherweise beschädigt.")
                    self.training_results = {}
            else:
                logger = setup_logging("gui")
                logger.info("Keine gespeicherten Trainingsergebnisse gefunden.")
                self.training_results = {}
        except Exception as e:
            logger = setup_logging("gui")
            logger.error(f"Fehler beim Laden der Trainingsergebnisse: {str(e)}")
            # Initialisiere mit leerem Dictionary im Fehlerfall
            self.training_results = {}

    def save_training_results(self):
        """
        Speichert die Trainingsergebnisse in einer Datei.
        """
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

            # Speichere die Trainingsergebnisse
            with open(self.results_file, 'wb') as f:
                pickle.dump(self.training_results, f)
                logger = setup_logging("gui")
                logger.info(f"Trainingsergebnisse gespeichert: {len(self.training_results)} Modelle.")
        except Exception as e:
            logger = setup_logging("gui")
            logger.error(f"Fehler beim Speichern der Trainingsergebnisse: {str(e)}")

# Hauptfunktion für die Standard-GUI
def main():
    """
    Hauptfunktion zum Starten der Standard-GUI.
    """
    # Erstelle die QApplication
    app = QApplication(sys.argv)

    # Erstelle und zeige das Hauptfenster
    window = ModernTextClassificationGUI()
    window.show()

    # Starte die Anwendung
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
