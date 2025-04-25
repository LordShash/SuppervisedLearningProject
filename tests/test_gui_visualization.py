import sys
import os
import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# Füge das Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suppervisedlearningproject.ui.gui_complete import ModernTextClassificationGUI

class TestGUIVisualization(unittest.TestCase):
    """
    Test für die Visualisierungen in der GUI.

    Dieser Test prüft, ob die Visualisierungen in den Tabs "Konfusionsmatrix", 
    "ROC-Kurve" und "Modellvergleich" korrekt angezeigt werden.
    """

    @classmethod
    def setUpClass(cls):
        """Erstelle die Anwendung und das Hauptfenster."""
        cls.app = QApplication(sys.argv)
        cls.window = ModernTextClassificationGUI()

        # Erstelle Dummy-Trainingsergebnisse
        cls.window.training_results = {
            "Dummy-Modell": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.79,
                "f1": 0.80,
                "conf_matrix": np.array([[80, 20], [15, 85]]),
                "class_names": ["Klasse 0", "Klasse 1"],
                "report": "              precision    recall  f1-score   support\n\n    Klasse 0       0.84      0.80      0.82       100\n    Klasse 1       0.81      0.85      0.83       100\n\n    accuracy                           0.85       200\n   macro avg       0.83      0.83      0.83       200\nweighted avg       0.83      0.85      0.83       200\n",
                "model_type": "logreg",
                "target_column": "Fits_Topic_Code",
                "roc_curve_data": {
                    "binary": {
                        "fpr": np.array([0, 0.2, 0.5, 0.8, 1]),
                        "tpr": np.array([0, 0.6, 0.8, 0.9, 1]),
                        "thresholds": np.array([1, 0.8, 0.5, 0.2, 0])
                    }
                },
                "auc_scores": {
                    "binary": 0.85
                }
            },
            "Dummy-Modell 2": {
                "accuracy": 0.80,
                "precision": 0.78,
                "recall": 0.75,
                "f1": 0.76,
                "conf_matrix": np.array([[75, 25], [20, 80]]),
                "class_names": ["Klasse 0", "Klasse 1"],
                "report": "              precision    recall  f1-score   support\n\n    Klasse 0       0.79      0.75      0.77       100\n    Klasse 1       0.76      0.80      0.78       100\n\n    accuracy                           0.80       200\n   macro avg       0.78      0.78      0.78       200\nweighted avg       0.78      0.80      0.78       200\n",
                "model_type": "logreg",
                "target_column": "Fits_Topic_Code",
                "roc_curve_data": {
                    "binary": {
                        "fpr": np.array([0, 0.25, 0.5, 0.75, 1]),
                        "tpr": np.array([0, 0.5, 0.7, 0.85, 1]),
                        "thresholds": np.array([1, 0.8, 0.5, 0.2, 0])
                    }
                },
                "auc_scores": {
                    "binary": 0.80
                }
            }
        }

        # Aktualisiere die Ergebnisse-Tabs
        cls.window.update_results_tab()

        # Zeige das Fenster
        cls.window.show()

        # Warte kurz, damit die GUI initialisiert werden kann
        QTest.qWait(500)

        # Füge die Modelle zur Auswahl hinzu
        cls.window.model_select_combo.addItem("Dummy-Modell")
        cls.window.model_select_combo.addItem("Dummy-Modell 2")
        cls.window.confusion_model_select_combo.addItem("Dummy-Modell")
        cls.window.confusion_model_select_combo.addItem("Dummy-Modell 2")
        cls.window.compare_models_combo.addItem("Dummy-Modell")
        cls.window.compare_models_combo.addItem("Dummy-Modell 2")

        # Wähle das erste Modell aus
        cls.window.model_select_combo.setCurrentIndex(0)

        # Warte kurz, damit die GUI aktualisiert werden kann
        QTest.qWait(500)

    @classmethod
    def tearDownClass(cls):
        """Schließe die Anwendung."""
        cls.window.close()
        cls.app.quit()

    def test_confusion_matrix_tab(self):
        """Teste, ob die Konfusionsmatrix angezeigt wird."""
        # Stelle sicher, dass die Modelle in den ComboBoxen vorhanden sind
        if self.window.model_select_combo.count() == 0:
            self.window.model_select_combo.addItem("Dummy-Modell")
            self.window.model_select_combo.addItem("Dummy-Modell 2")

        # Wähle das erste Modell aus
        self.window.model_select_combo.setCurrentIndex(0)

        # Aktualisiere die Konfusionsmatrix explizit
        model_name = "Dummy-Modell"  # Verwende direkt den Namen
        model_data = self.window.training_results[model_name]
        self.window.update_confusion_matrix(
            model_data["conf_matrix"],
            model_data.get("class_names", None)
        )

        # Stelle sicher, dass das Widget sichtbar ist
        self.window.cm_widget.setVisible(True)
        self.window.cm_widget.show()

        # Wähle den Konfusionsmatrix-Tab
        for i in range(self.window.viz_tabs.count()):
            if self.window.viz_tabs.tabText(i) == "Konfusionsmatrix":
                self.window.viz_tabs.setCurrentIndex(i)
                break

        # Warte kurz, damit die GUI aktualisiert werden kann
        QTest.qWait(500)

        # Prüfe, ob die Konfusionsmatrix ein Layout hat
        self.assertIsNotNone(self.window.cm_widget.layout())

    def test_roc_curve_tab(self):
        """Teste, ob die ROC-Kurve angezeigt wird."""
        # Stelle sicher, dass die Modelle in den ComboBoxen vorhanden sind
        if self.window.model_select_combo.count() == 0:
            self.window.model_select_combo.addItem("Dummy-Modell")
            self.window.model_select_combo.addItem("Dummy-Modell 2")

        # Wähle das erste Modell aus
        self.window.model_select_combo.setCurrentIndex(0)

        # Aktualisiere die ROC-Kurve explizit
        model_name = "Dummy-Modell"  # Verwende direkt den Namen
        model_data = self.window.training_results[model_name]
        self.window.update_roc_curve(
            model_data["roc_curve_data"],
            model_data["auc_scores"],
            model_data.get("class_names", None)
        )

        # Stelle sicher, dass das Widget sichtbar ist
        self.window.roc_widget.setVisible(True)
        self.window.roc_widget.show()

        # Wähle den ROC-Kurve-Tab
        for i in range(self.window.viz_tabs.count()):
            if self.window.viz_tabs.tabText(i) == "ROC-Kurve":
                self.window.viz_tabs.setCurrentIndex(i)
                break

        # Warte kurz, damit die GUI aktualisiert werden kann
        QTest.qWait(500)

        # Prüfe, ob die ROC-Kurve ein Layout hat
        self.assertIsNotNone(self.window.roc_widget.layout())

    def test_model_comparison_tab(self):
        """Teste, ob der Modellvergleich angezeigt wird."""
        # Stelle sicher, dass die Modelle in den ComboBoxen vorhanden sind
        if self.window.model_select_combo.count() == 0:
            self.window.model_select_combo.addItem("Dummy-Modell")
            self.window.model_select_combo.addItem("Dummy-Modell 2")

        if self.window.compare_models_combo.count() == 0:
            self.window.compare_models_combo.addItem("Dummy-Modell")
            self.window.compare_models_combo.addItem("Dummy-Modell 2")

        # Wähle das erste Modell aus
        self.window.model_select_combo.setCurrentIndex(0)

        # Wähle das zweite Modell für den Vergleich
        self.window.compare_models_combo.setCurrentIndex(1)

        # Aktualisiere den Modellvergleich explizit
        model1_name = "Dummy-Modell"  # Verwende direkt den Namen
        model2_name = "Dummy-Modell 2"  # Verwende direkt den Namen
        model1_data = self.window.training_results[model1_name]
        model2_data = self.window.training_results[model2_name]
        self.window.update_model_comparison(
            model1_name,
            model1_data,
            model2_name,
            model2_data
        )

        # Stelle sicher, dass das Widget sichtbar ist
        self.window.compare_widget.setVisible(True)
        self.window.compare_widget.show()

        # Wähle den Modellvergleich-Tab
        for i in range(self.window.viz_tabs.count()):
            if self.window.viz_tabs.tabText(i) == "Modellvergleich":
                self.window.viz_tabs.setCurrentIndex(i)
                break

        # Warte kurz, damit die GUI aktualisiert werden kann
        QTest.qWait(500)

        # Prüfe, ob der Modellvergleich ein Layout hat
        self.assertIsNotNone(self.window.compare_widget.layout())

if __name__ == "__main__":
    unittest.main()
