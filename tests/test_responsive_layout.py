import sys
import os
import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSize

# Füge das Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suppervisedlearningproject.ui.gui_complete import ModernTextClassificationGUI

class TestResponsiveLayout(unittest.TestCase):
    """
    Test für die Responsivität des Layouts.
    
    Dieser Test prüft, ob alle wichtigen Widgets auch bei kleiner Fenstergröße
    (800x600 Pixel) sichtbar und zugänglich sind.
    """
    
    @classmethod
    def setUpClass(cls):
        """Erstelle die Anwendung und das Hauptfenster."""
        cls.app = QApplication(sys.argv)
        cls.window = ModernTextClassificationGUI()
        
        # Setze die Fenstergröße auf 800x600 Pixel
        cls.window.resize(800, 600)
        cls.window.show()
    
    @classmethod
    def tearDownClass(cls):
        """Schließe die Anwendung."""
        cls.window.close()
        cls.app.quit()
    
    def test_training_tab_widgets_visible(self):
        """Teste, ob alle wichtigen Widgets im Training-Tab sichtbar sind."""
        # Wähle den Training-Tab
        self.window.tab_widget.setCurrentIndex(0)
        
        # Prüfe, ob die wichtigsten Widgets sichtbar sind
        self.assertTrue(self.window.target_combo.isVisible())
        self.assertTrue(self.window.model_logreg_radio.isVisible())
        self.assertTrue(self.window.model_nn_radio.isVisible())
        self.assertTrue(self.window.max_features_spin.isVisible())
        self.assertTrue(self.window.test_size_spin.isVisible())
        self.assertTrue(self.window.train_button.isVisible())
        self.assertTrue(self.window.output_text.isVisible())
    
    def test_results_tab_widgets_visible(self):
        """Teste, ob alle wichtigen Widgets im Ergebnisse-Tab sichtbar sind."""
        # Wähle den Ergebnisse-Tab
        self.window.tab_widget.setCurrentIndex(1)
        
        # Prüfe, ob die wichtigsten Widgets sichtbar sind
        self.assertTrue(self.window.model_select_combo.isVisible())
        self.assertTrue(self.window.compare_models_combo.isVisible())
        self.assertTrue(self.window.compare_button.isVisible())
        self.assertTrue(self.window.viz_tabs.isVisible())
        self.assertTrue(self.window.accuracy_label.isVisible())
        self.assertTrue(self.window.report_text.isVisible())
    
    def test_about_tab_widgets_visible(self):
        """Teste, ob alle wichtigen Widgets im Über-Tab sichtbar sind."""
        # Wähle den Über-Tab
        self.window.tab_widget.setCurrentIndex(2)
        
        # Prüfe, ob die wichtigsten Widgets sichtbar sind
        # Hier können wir nur prüfen, ob der Tab selbst sichtbar ist,
        # da die Widgets dynamisch erstellt werden
        self.assertTrue(self.window.tab_widget.widget(2).isVisible())
    
    def test_compare_models_tab_accessible(self):
        """Teste, ob der Modellvergleich-Tab zugänglich ist."""
        # Wähle den Ergebnisse-Tab
        self.window.tab_widget.setCurrentIndex(1)
        
        # Prüfe, ob der Modellvergleich-Tab existiert und zugänglich ist
        for i in range(self.window.viz_tabs.count()):
            if self.window.viz_tabs.tabText(i) == "Modellvergleich":
                self.window.viz_tabs.setCurrentIndex(i)
                self.assertTrue(self.window.compare_widget.isVisible())
                break
        else:
            self.fail("Modellvergleich-Tab nicht gefunden")

if __name__ == "__main__":
    unittest.main()