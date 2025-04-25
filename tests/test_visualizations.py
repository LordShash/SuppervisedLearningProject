import os
import sys
import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# Füge das Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suppervisedlearningproject.ui.gui_complete import ModernTextClassificationGUI
from suppervisedlearningproject.utils import setup_logging

# Konfiguriere das Logging für die Tests
logger = setup_logging("test_visualizations")

@pytest.fixture
def app():
    """Fixture für die QApplication."""
    app = QApplication([])
    yield app
    app.quit()

@pytest.fixture
def gui(app):
    """Fixture für die GUI."""
    gui = ModernTextClassificationGUI()
    # Füge Dummy-Trainingsergebnisse hinzu
    gui.training_results = {
        "logreg_dummy": create_dummy_results()
    }
    # Aktualisiere die Ergebnisse
    gui.update_results_tab()
    yield gui
    gui.close()

def create_dummy_results():
    """Erstellt Dummy-Trainingsergebnisse für Tests."""
    # Erstelle eine einfache 2x2 Konfusionsmatrix
    conf_matrix = np.array([[45, 5], [10, 40]])
    
    # Erstelle Dummy-ROC-Kurven-Daten
    fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr = np.array([0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0])
    thresholds = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    
    roc_curve_data = {
        "binary": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds
        }
    }
    
    # Erstelle Dummy-AUC-Scores
    auc_scores = {
        "binary": 0.85
    }
    
    # Erstelle einen Dummy-Klassifikationsbericht
    report = """              precision    recall  f1-score   support

           0       0.82      0.90      0.86        50
           1       0.89      0.80      0.84        50

    accuracy                           0.85       100
   macro avg       0.85      0.85      0.85       100
weighted avg       0.85      0.85      0.85       100
"""
    
    # Erstelle das Ergebnisdictionary
    results = {
        "model_type": "logreg",
        "target_column": "dummy",
        "accuracy": 0.85,
        "precision": 0.85,
        "recall": 0.85,
        "f1": 0.85,
        "report": report,
        "conf_matrix": conf_matrix,
        "class_names": ["Klasse 0", "Klasse 1"],
        "roc_curve_data": roc_curve_data,
        "auc_scores": auc_scores
    }
    
    return results

def test_confusion_matrix_tab(gui):
    """Testet, ob die Konfusionsmatrix korrekt angezeigt wird."""
    logger.info("Starte Test der Konfusionsmatrix-Visualisierung")
    
    # Wähle den ersten Eintrag in der Modellauswahl
    gui.model_select_combo.setCurrentIndex(0)
    
    # Wechsle zum Visualisierungen-Tab
    gui.viz_tabs.setCurrentIndex(0)  # Konfusionsmatrix-Tab
    
    # Prüfe, ob die Konfusionsmatrix-Widget ein Layout hat
    assert gui.cm_widget.layout() is not None, "Konfusionsmatrix-Widget hat kein Layout"
    
    # Prüfe, ob das Layout Widgets enthält
    assert gui.cm_widget.layout().count() > 0, "Konfusionsmatrix-Layout enthält keine Widgets"
    
    # Prüfe, ob eine QTableWidget im Layout vorhanden ist
    table_found = False
    for i in range(gui.cm_widget.layout().count()):
        widget = gui.cm_widget.layout().itemAt(i).widget()
        if widget and widget.__class__.__name__ == "QTableWidget":
            table_found = True
            # Prüfe, ob die Tabelle Zeilen und Spalten hat
            assert widget.rowCount() > 0, "Konfusionsmatrix-Tabelle hat keine Zeilen"
            assert widget.columnCount() > 0, "Konfusionsmatrix-Tabelle hat keine Spalten"
            break
    
    assert table_found, "Keine QTableWidget in der Konfusionsmatrix gefunden"
    logger.info("Test der Konfusionsmatrix-Visualisierung erfolgreich")

def test_roc_curve_tab(gui):
    """Testet, ob die ROC-Kurve korrekt angezeigt wird."""
    logger.info("Starte Test der ROC-Kurven-Visualisierung")
    
    # Wähle den ersten Eintrag in der Modellauswahl
    gui.model_select_combo.setCurrentIndex(0)
    
    # Wechsle zum ROC-Kurven-Tab
    gui.viz_tabs.setCurrentIndex(1)  # ROC-Kurven-Tab
    
    # Prüfe, ob das ROC-Widget ein Layout hat
    assert gui.roc_widget.layout() is not None, "ROC-Widget hat kein Layout"
    
    # Prüfe, ob das Layout Widgets enthält
    assert gui.roc_widget.layout().count() > 0, "ROC-Layout enthält keine Widgets"
    
    # Prüfe, ob ein QChartView im Layout vorhanden ist
    chart_view_found = False
    for i in range(gui.roc_widget.layout().count()):
        widget = gui.roc_widget.layout().itemAt(i).widget()
        if widget and widget.__class__.__name__ == "QChartView":
            chart_view_found = True
            break
    
    assert chart_view_found, "Kein QChartView in der ROC-Kurve gefunden"
    logger.info("Test der ROC-Kurven-Visualisierung erfolgreich")

def test_model_comparison_tab(gui):
    """Testet, ob der Modellvergleich korrekt funktioniert."""
    logger.info("Starte Test der Modellvergleichs-Visualisierung")
    
    # Füge ein zweites Dummy-Modell hinzu
    gui.training_results["nn_dummy"] = create_dummy_results()
    gui.training_results["nn_dummy"]["model_type"] = "nn"
    
    # Aktualisiere die Ergebnisse
    gui.update_results_tab()
    
    # Wähle den ersten Eintrag in der Modellauswahl
    gui.model_select_combo.setCurrentIndex(0)
    
    # Wähle den zweiten Eintrag in der Vergleichsauswahl
    gui.compare_models_combo.setCurrentIndex(0)
    
    # Klicke auf den Vergleichen-Button
    gui.compare_button.click()
    
    # Prüfe, ob das Vergleichs-Widget ein Layout hat
    assert gui.compare_widget.layout() is not None, "Vergleichs-Widget hat kein Layout"
    
    # Prüfe, ob das Layout Widgets enthält
    assert gui.compare_widget.layout().count() > 0, "Vergleichs-Layout enthält keine Widgets"
    
    # Prüfe, ob eine QTableWidget im Layout vorhanden ist
    table_found = False
    for i in range(gui.compare_widget.layout().count()):
        widget = gui.compare_widget.layout().itemAt(i).widget()
        if widget and widget.__class__.__name__ == "QTableWidget":
            table_found = True
            # Prüfe, ob die Tabelle Zeilen und Spalten hat
            assert widget.rowCount() > 0, "Vergleichs-Tabelle hat keine Zeilen"
            assert widget.columnCount() > 0, "Vergleichs-Tabelle hat keine Spalten"
            break
    
    assert table_found, "Keine QTableWidget im Modellvergleich gefunden"
    logger.info("Test der Modellvergleichs-Visualisierung erfolgreich")

if __name__ == "__main__":
    # Führe die Tests manuell aus, wenn das Skript direkt ausgeführt wird
    app = QApplication([])
    gui = ModernTextClassificationGUI()
    gui.training_results = {
        "logreg_dummy": create_dummy_results()
    }
    gui.update_results_tab()
    gui.show()
    
    # Führe die Tests aus
    test_confusion_matrix_tab(gui)
    test_roc_curve_tab(gui)
    
    # Füge ein zweites Dummy-Modell hinzu für den Vergleichstest
    gui.training_results["nn_dummy"] = create_dummy_results()
    gui.training_results["nn_dummy"]["model_type"] = "nn"
    gui.update_results_tab()
    test_model_comparison_tab(gui)
    
    print("Alle Tests erfolgreich!")
    
    # Starte die Event-Loop
    sys.exit(app.exec_())