"""
Implementierung der Tab-Funktionalität für die moderne GUI.

Dieses Modul enthält die Implementierung der verschiedenen Tabs
für die moderne PyQt5-basierte Benutzeroberfläche.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QRadioButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QTextEdit, QSplitter
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
import numpy as np

from suppervisedlearningproject.ui.modern_gui import MatplotlibCanvas


def init_training_tab(self):
    """Initialisiert den Training-Tab mit Formularelementen."""
    # Erstelle den Tab
    training_tab = QWidget()
    self.tab_widget.addTab(training_tab, "Training")

    # Layout für den Tab mit mehr Platz für Inhalte
    layout = QVBoxLayout(training_tab)
    layout.setSpacing(10)  # Erhöhe den Abstand zwischen Elementen
    layout.setContentsMargins(10, 10, 10, 10)  # Mehr Rand für bessere Lesbarkeit

    # Erstelle eine Gruppe für die Trainingsparameter
    params_group = QGroupBox("Trainingsparameter")
    layout.addWidget(params_group)

    # Layout für die Trainingsparameter mit mehr Platz
    params_layout = QFormLayout(params_group)
    params_layout.setSpacing(10)  # Erhöhe den Abstand zwischen Zeilen
    params_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # Erlaube Feldern zu wachsen

    # Zielvariable
    self.target_combo = QComboBox()
    self.target_combo.setMinimumWidth(200)  # Stelle sicher, dass die ComboBox breit genug ist
    params_layout.addRow(self.create_label_with_tooltip("Zielvariable:", ""), self.target_combo)

    # Modelltyp
    model_group = QWidget()
    model_layout = QHBoxLayout(model_group)
    model_layout.setContentsMargins(0, 0, 0, 0)
    model_layout.setSpacing(15)  # Mehr Platz zwischen den Radio-Buttons

    self.model_logreg_radio = QRadioButton("Logistische Regression")
    self.model_nn_radio = QRadioButton("Neuronales Netz")
    self.model_logreg_radio.setChecked(True)

    model_layout.addWidget(self.model_logreg_radio)
    model_layout.addWidget(self.model_nn_radio)
    model_layout.addStretch()  # Füge Stretch hinzu, damit die Buttons links ausgerichtet sind

    # Verbinde die Radiobuttons mit der Funktion zum Umschalten der Parameter
    self.model_logreg_radio.toggled.connect(self.toggle_model_options)

    params_layout.addRow("Modelltyp:", model_group)

    # Gemeinsame Parameter
    # Max Features
    self.max_features_spin = QSpinBox()
    self.max_features_spin.setRange(100, 10000)
    self.max_features_spin.setValue(1000)
    self.max_features_spin.setSingleStep(100)
    self.max_features_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    params_layout.addRow(self.create_label_with_tooltip("Max Features:", self.tooltips["max_features"]), 
                         self.max_features_spin)

    # Test Size
    self.test_size_spin = QDoubleSpinBox()
    self.test_size_spin.setRange(0.1, 0.5)
    self.test_size_spin.setValue(0.2)
    self.test_size_spin.setSingleStep(0.05)
    self.test_size_spin.setDecimals(2)
    self.test_size_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    params_layout.addRow(self.create_label_with_tooltip("Test Size:", self.tooltips["test_size"]), 
                         self.test_size_spin)

    # Parameter für logistische Regression
    self.logreg_group = QGroupBox("Parameter für logistische Regression")
    params_layout.addRow(self.logreg_group)
    logreg_layout = QFormLayout(self.logreg_group)
    logreg_layout.setSpacing(10)  # Erhöhe den Abstand zwischen Zeilen
    logreg_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # Erlaube Feldern zu wachsen

    # Max Iterations
    self.max_iter_spin = QSpinBox()
    self.max_iter_spin.setRange(100, 10000)
    self.max_iter_spin.setValue(1000)
    self.max_iter_spin.setSingleStep(100)
    self.max_iter_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    logreg_layout.addRow(self.create_label_with_tooltip("Max Iterations:", self.tooltips["max_iter"]), 
                         self.max_iter_spin)

    # C (Regularisierung)
    self.c_reg_spin = QDoubleSpinBox()
    self.c_reg_spin.setRange(0.1, 10.0)
    self.c_reg_spin.setValue(1.0)
    self.c_reg_spin.setSingleStep(0.1)
    self.c_reg_spin.setDecimals(2)
    self.c_reg_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    logreg_layout.addRow(self.create_label_with_tooltip("C (Regularisierung):", self.tooltips["c_reg"]), 
                         self.c_reg_spin)

    # Solver
    self.solver_combo = QComboBox()
    self.solver_combo.addItems(['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'])
    self.solver_combo.setMinimumWidth(150)  # Stelle sicher, dass die ComboBox breit genug ist
    logreg_layout.addRow(self.create_label_with_tooltip("Solver:", self.tooltips["solver"]), 
                         self.solver_combo)

    # Parameter für neuronales Netz
    self.nn_group = QGroupBox("Parameter für neuronales Netz")
    params_layout.addRow(self.nn_group)
    nn_layout = QFormLayout(self.nn_group)
    nn_layout.setSpacing(10)  # Erhöhe den Abstand zwischen Zeilen
    nn_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # Erlaube Feldern zu wachsen

    # Epochs
    self.epochs_spin = QSpinBox()
    self.epochs_spin.setRange(5, 100)
    self.epochs_spin.setValue(30)
    self.epochs_spin.setSingleStep(5)
    self.epochs_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    nn_layout.addRow(self.create_label_with_tooltip("Epochs:", self.tooltips["epochs"]), 
                     self.epochs_spin)

    # Patience
    self.patience_spin = QSpinBox()
    self.patience_spin.setRange(1, 20)
    self.patience_spin.setValue(5)
    self.patience_spin.setSingleStep(1)
    self.patience_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    nn_layout.addRow(self.create_label_with_tooltip("Patience:", self.tooltips["patience"]), 
                     self.patience_spin)

    # Verstecke die NN-Parameter initial
    self.nn_group.setVisible(False)

    # Button zum Trainieren
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 10, 0, 10)  # Mehr vertikalen Abstand
    self.train_button = QPushButton("Modell trainieren")
    self.train_button.clicked.connect(self.train_model)
    self.train_button.setMinimumWidth(150)  # Stelle sicher, dass der Button breit genug ist
    button_layout.addStretch()
    button_layout.addWidget(self.train_button)
    layout.addLayout(button_layout)

    # Ausgabebereich
    output_group = QGroupBox("Ausgabe")
    layout.addWidget(output_group, 1)  # 1 = stretch factor

    output_layout = QVBoxLayout(output_group)
    output_layout.setContentsMargins(10, 10, 10, 10)  # Mehr Rand für bessere Lesbarkeit
    self.output_text = QTextEdit()
    self.output_text.setReadOnly(True)
    self.output_text.setMinimumHeight(150)  # Stelle sicher, dass das Textfeld hoch genug ist
    output_layout.addWidget(self.output_text)


def init_results_tab(self):
    """Initialisiert den Ergebnisse-Tab."""
    # Erstelle den Tab
    results_tab = QWidget()
    self.tab_widget.addTab(results_tab, "Ergebnisse")

    # Layout für den Tab
    layout = QVBoxLayout(results_tab)

    # Modellauswahl
    model_group = QGroupBox("Modellauswahl")
    layout.addWidget(model_group)

    model_layout = QHBoxLayout(model_group)
    model_layout.addWidget(QLabel("Modell:"))
    self.model_select_combo = QComboBox()
    self.model_select_combo.currentIndexChanged.connect(self.on_model_selected)
    model_layout.addWidget(self.model_select_combo)

    # Metriken
    metrics_group = QGroupBox("Metriken")
    layout.addWidget(metrics_group)

    metrics_layout = QFormLayout(metrics_group)

    # Erstelle ein Grid für die Metriken
    metrics_widget = QWidget()
    metrics_grid = QFormLayout(metrics_widget)
    metrics_grid.setContentsMargins(0, 0, 0, 0)

    # Erste Zeile: Accuracy und Precision
    self.accuracy_label = QLabel("N/A")
    self.precision_label = QLabel("N/A")
    metrics_grid.addRow("Accuracy:", self.accuracy_label)
    metrics_grid.addRow("Precision:", self.precision_label)

    # Zweite Zeile: Recall und F1
    self.recall_label = QLabel("N/A")
    self.f1_label = QLabel("N/A")
    metrics_grid.addRow("Recall:", self.recall_label)
    metrics_grid.addRow("F1 Score:", self.f1_label)

    # Füge das Grid zum Layout hinzu
    metrics_layout.addRow(metrics_widget)

    # Klassifikationsbericht (volle Größe im Ergebnisse-Tab)
    report_group = QGroupBox("Klassifikationsbericht")
    layout.addWidget(report_group, 1)  # 1 = stretch factor
    report_layout = QVBoxLayout(report_group)
    self.report_text = QTextEdit()
    self.report_text.setReadOnly(True)
    # Entferne die Höhenbegrenzung, damit der Bericht den verfügbaren Platz nutzt
    report_layout.addWidget(self.report_text)

    # Erstelle einen separaten Tab für die Konfusionsmatrix
    confusion_tab = QWidget()
    self.tab_widget.addTab(confusion_tab, "Konfusionsmatrix")

    # Layout für den Konfusionsmatrix-Tab
    confusion_layout = QVBoxLayout(confusion_tab)

    # Füge die gleiche Modellauswahl zum Konfusionsmatrix-Tab hinzu
    confusion_model_group = QGroupBox("Modellauswahl")
    confusion_layout.addWidget(confusion_model_group)

    confusion_model_layout = QHBoxLayout(confusion_model_group)
    confusion_model_layout.addWidget(QLabel("Modell:"))
    # Erstelle eine neue ComboBox, die mit der im Ergebnisse-Tab synchronisiert wird
    self.confusion_model_select_combo = QComboBox()
    # Verbinde die beiden ComboBoxen, damit sie synchron bleiben
    self.model_select_combo.currentIndexChanged.connect(
        lambda idx: self.confusion_model_select_combo.setCurrentIndex(idx))
    self.confusion_model_select_combo.currentIndexChanged.connect(
        lambda idx: self.model_select_combo.setCurrentIndex(idx))
    confusion_model_layout.addWidget(self.confusion_model_select_combo)

    # Konfusionsmatrix (volle Größe im eigenen Tab)
    self.cm_group = QGroupBox("Konfusionsmatrix")
    confusion_layout.addWidget(self.cm_group, 1)  # 1 = stretch factor
    cm_layout = QVBoxLayout(self.cm_group)
    self.cm_widget = QWidget()
    cm_layout.addWidget(self.cm_widget)


def init_about_tab(self):
    """Initialisiert den Über-Tab mit Informationen zur Anwendung."""
    # Erstelle den Tab
    about_tab = QWidget()
    self.tab_widget.addTab(about_tab, "Über")

    # Layout für den Tab
    layout = QVBoxLayout(about_tab)

    # Titel
    title_label = QLabel("Textklassifikationsanwendung")
    title_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)

    # Beschreibung
    description_group = QGroupBox("Beschreibung")
    layout.addWidget(description_group)

    description_layout = QVBoxLayout(description_group)
    description_text = QLabel("""
    Diese Anwendung ermöglicht das Training und die Evaluation von Textklassifikationsmodellen.

    Unterstützte Modelle:
    - Logistische Regression
    - Neuronales Netz

    Die Anwendung bietet eine moderne Benutzeroberfläche zum Einstellen der Trainingsparameter
    und zur Visualisierung der Ergebnisse.
    """)
    description_text.setWordWrap(True)
    description_layout.addWidget(description_text)

    # Anleitung
    usage_group = QGroupBox("Anleitung")
    layout.addWidget(usage_group)

    usage_layout = QVBoxLayout(usage_group)
    usage_text = QLabel("""
    1. Wählen Sie im Tab "Training" die gewünschte Zielvariable und den Modelltyp aus.
    2. Passen Sie die Parameter nach Bedarf an.
    3. Klicken Sie auf "Modell trainieren", um das Training zu starten.
    4. Die Ergebnisse werden im Tab "Ergebnisse" angezeigt.
    5. Sie können zwischen verschiedenen trainierten Modellen wechseln, um die Ergebnisse zu vergleichen.
    """)
    usage_text.setWordWrap(True)
    usage_layout.addWidget(usage_text)

    # Version und Copyright
    footer_layout = QHBoxLayout()
    layout.addLayout(footer_layout)

    version_label = QLabel("Version 1.0")
    footer_layout.addWidget(version_label)

    footer_layout.addStretch()

    copyright_label = QLabel("© 2025 Textklassifikationsprojekt")
    footer_layout.addWidget(copyright_label)


def create_label_with_tooltip(self, text, tooltip):
    """
    Erstellt ein Label mit Tooltip und einem Fragezeichen-Icon für Hilfe.

    Args:
        text: Der anzuzeigende Text
        tooltip: Der Tooltip-Text, der angezeigt wird, wenn man mit der Maus über das Fragezeichen fährt

    Returns:
        Ein Widget, das das Label und optional ein Fragezeichen-Icon enthält
    """
    # Erstelle ein Container-Widget für Label und Fragezeichen
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)

    # Erstelle das Label
    label = QLabel(text)
    layout.addWidget(label)

    # Wenn ein Tooltip vorhanden ist, füge ein Fragezeichen-Icon hinzu
    if tooltip:
        # Erstelle ein Fragezeichen-Label mit einem besser integrierten Stil
        from PyQt5.QtWidgets import QToolTip

        class HelpLabel(QLabel):
            def __init__(self, text, tooltip):
                super().__init__(text)
                self.tooltip = tooltip
                self.setToolTip(tooltip)
                self.setCursor(Qt.WhatsThisCursor)

                # Angepasster Stil, der besser zum Rest der GUI passt
                self.setStyleSheet("""
                    QLabel {
                        color: #00796b;
                        font-size: 10pt;
                        font-weight: bold;
                        font-family: "Segoe UI", Arial, sans-serif;
                        background-color: #e0f2f1;
                        border-radius: 10px;
                        padding: 2px 6px;
                        margin: 0px 2px;
                    }
                    QLabel:hover {
                        background-color: #b2dfdb;
                    }
                """)

            def enterEvent(self, event):
                self.setStyleSheet("""
                    QLabel {
                        color: white;
                        font-size: 10pt;
                        font-weight: bold;
                        font-family: "Segoe UI", Arial, sans-serif;
                        background-color: #00796b;
                        border-radius: 10px;
                        padding: 2px 6px;
                        margin: 0px 2px;
                    }
                """)
                QToolTip.showText(event.globalPos(), self.tooltip, self)
                super().enterEvent(event)

            def leaveEvent(self, event):
                self.setStyleSheet("""
                    QLabel {
                        color: #00796b;
                        font-size: 10pt;
                        font-weight: bold;
                        font-family: "Segoe UI", Arial, sans-serif;
                        background-color: #e0f2f1;
                        border-radius: 10px;
                        padding: 2px 6px;
                        margin: 0px 2px;
                    }
                """)
                super().leaveEvent(event)

        # Erstelle ein Fragezeichen-Label mit der neuen Klasse
        help_icon = HelpLabel("?", tooltip)
        layout.addWidget(help_icon)

    # Füge einen Stretch hinzu, damit die Elemente links ausgerichtet sind
    layout.addStretch()

    return container


def toggle_model_options(self):
    """Zeigt oder versteckt die modellspezifischen Parameter je nach ausgewähltem Modelltyp."""
    is_logreg = self.model_logreg_radio.isChecked()
    self.logreg_group.setVisible(is_logreg)
    self.nn_group.setVisible(not is_logreg)


def train_model(self):
    """Trainiert das ausgewählte Modell mit den angegebenen Parametern."""
    # Deaktiviere den Trainieren-Button während des Trainings
    self.train_button.setEnabled(False)

    # Leere das Ausgabefeld
    self.output_text.clear()

    # Sammle die Parameter
    params = {
        'target_column': self.target_combo.currentText(),
        'model_type': "logreg" if self.model_logreg_radio.isChecked() else "nn",
        'max_features': self.max_features_spin.value(),
        'test_size': self.test_size_spin.value(),
        'max_iter': self.max_iter_spin.value(),
        'c_reg': self.c_reg_spin.value(),
        'solver': self.solver_combo.currentText(),
        'epochs': self.epochs_spin.value(),
        'patience': self.patience_spin.value()
    }

    # Erstelle und starte den Worker-Thread
    from suppervisedlearningproject.ui.modern_gui import TrainingWorker
    self.worker = TrainingWorker(params)
    self.worker.update_output.connect(self.append_to_output)
    self.worker.training_finished.connect(self.on_training_finished)
    self.worker.training_error.connect(self.on_training_error)
    self.worker.finished.connect(lambda: self.train_button.setEnabled(True))
    self.worker.start()


def append_to_output(self, text):
    """Fügt Text zum Ausgabefeld hinzu."""
    self.output_text.append(text)
    # Scrolle zum Ende
    cursor = self.output_text.textCursor()
    cursor.movePosition(cursor.End)
    self.output_text.setTextCursor(cursor)


def on_training_finished(self, results):
    """Wird aufgerufen, wenn das Training erfolgreich abgeschlossen wurde."""
    # Speichere die Ergebnisse
    model_key = f"{results['model_type']}_{results['target_column']}"
    self.training_results[model_key] = results

    # Speichere die Trainingsergebnisse auf der Festplatte
    self.save_training_results()

    # Aktualisiere die Ergebnisse im Ergebnisse-Tab
    self.update_results_tab()

    # Wechsle zum Ergebnisse-Tab
    self.tab_widget.setCurrentIndex(1)


def on_training_error(self, error_msg):
    """Wird aufgerufen, wenn ein Fehler beim Training auftritt."""
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.critical(self, "Trainingsfehler", f"Fehler beim Training: {error_msg}")


def update_results_tab(self):
    """Aktualisiert die Modellauswahl in allen Tabs."""
    try:
        # Prüfe, ob training_results existiert und ein Dictionary ist
        if not hasattr(self, 'training_results') or not isinstance(self.training_results, dict):
            print("Warnung: training_results ist nicht verfügbar oder kein Dictionary")
            return

        # Speichere den aktuell ausgewählten Wert
        current_text = self.model_select_combo.currentText()

        # Blockiere Signale, um unerwünschte Aktualisierungen zu vermeiden
        self.model_select_combo.blockSignals(True)
        self.confusion_model_select_combo.blockSignals(True)

        # Leere beide ComboBoxen
        self.model_select_combo.clear()
        self.confusion_model_select_combo.clear()

        # Hole die Modellnamen
        model_names = list(self.training_results.keys())
        if not model_names:
            print("Keine Modelle in training_results gefunden")
            # Entsperre die Signale wieder
            self.model_select_combo.blockSignals(False)
            self.confusion_model_select_combo.blockSignals(False)
            return

        # Füge die Modellnamen zu beiden ComboBoxen hinzu
        self.model_select_combo.addItems(model_names)
        self.confusion_model_select_combo.addItems(model_names)

        # Versuche, den vorherigen Wert wiederherzustellen
        index = self.model_select_combo.findText(current_text)
        if index >= 0:
            # Setze den Index in beiden ComboBoxen
            self.model_select_combo.setCurrentIndex(index)
            self.confusion_model_select_combo.setCurrentIndex(index)
        elif model_names:  # Wähle das neueste Modell aus, falls vorhanden
            # Setze den Index in beiden ComboBoxen
            self.model_select_combo.setCurrentIndex(len(model_names) - 1)
            self.confusion_model_select_combo.setCurrentIndex(len(model_names) - 1)

        # Entsperre die Signale wieder
        self.model_select_combo.blockSignals(False)
        self.confusion_model_select_combo.blockSignals(False)

        # Aktualisiere die Anzeige mit dem ausgewählten Modell
        if model_names:
            # Rufe on_model_selected manuell auf, um die Anzeige zu aktualisieren
            self.on_model_selected(self.model_select_combo.currentIndex())
    except Exception as e:
        # Fange alle Ausnahmen ab und protokolliere sie
        print(f"Fehler in update_results_tab: {str(e)}")


def on_model_selected(self, index):
    """Wird aufgerufen, wenn ein Modell im Ergebnisse-Tab oder Konfusionsmatrix-Tab ausgewählt wird."""
    try:
        if index < 0:
            return

        # Bestimme, welche ComboBox das Event ausgelöst hat und synchronisiere die andere
        sender = self.sender()
        if sender == self.model_select_combo:
            model_key = self.model_select_combo.currentText()
            # Synchronisiere die ComboBox im Konfusionsmatrix-Tab
            idx = self.confusion_model_select_combo.findText(model_key)
            if idx >= 0 and idx != self.confusion_model_select_combo.currentIndex():
                self.confusion_model_select_combo.blockSignals(True)
                self.confusion_model_select_combo.setCurrentIndex(idx)
                self.confusion_model_select_combo.blockSignals(False)
        elif sender == self.confusion_model_select_combo:
            model_key = self.confusion_model_select_combo.currentText()
            # Synchronisiere die ComboBox im Ergebnisse-Tab
            idx = self.model_select_combo.findText(model_key)
            if idx >= 0 and idx != self.model_select_combo.currentIndex():
                self.model_select_combo.blockSignals(True)
                self.model_select_combo.setCurrentIndex(idx)
                self.model_select_combo.blockSignals(False)
        else:
            # Fallback, falls der Sender nicht identifiziert werden kann
            model_key = self.model_select_combo.currentText()

        if not model_key or model_key not in self.training_results:
            return

        # Hole die Ergebnisse für das ausgewählte Modell
        results = self.training_results[model_key]

        # Prüfe, ob alle erforderlichen Schlüssel vorhanden sind
        required_keys = ['accuracy', 'precision', 'recall', 'f1', 'report', 'conf_matrix']
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Warnung", 
                               f"Fehlende Daten im Modell '{model_key}': {', '.join(missing_keys)}")
            return

        # Aktualisiere die Metriken im Ergebnisse-Tab
        self.accuracy_label.setText(f"{results['accuracy']:.4f}")
        self.precision_label.setText(f"{results['precision']:.4f}")
        self.recall_label.setText(f"{results['recall']:.4f}")
        self.f1_label.setText(f"{results['f1']:.4f}")

        # Aktualisiere den Klassifikationsbericht mit verbesserter Formatierung
        report_text = "=== Klassifikationsbericht ===\n\n" + results['report']
        # Verwende eine konsistente Schriftart, die in allen Tabs gleich ist
        self.report_text.setText(report_text)

        # Zeige die Konfusionsmatrix an
        self.update_confusion_matrix(results['conf_matrix'])
    except Exception as e:
        # Fange alle Ausnahmen ab und zeige eine Fehlermeldung an
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Fehler", f"Fehler beim Anzeigen der Modellergebnisse: {str(e)}")
        print(f"Fehler in on_model_selected: {str(e)}")


def update_confusion_matrix(self, conf_matrix):
    """Aktualisiert die Konfusionsmatrix-Visualisierung."""
    try:
        # Prüfe, ob conf_matrix ein gültiges numpy-Array ist
        if not isinstance(conf_matrix, np.ndarray):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Warnung", "Konfusionsmatrix ist kein gültiges numpy-Array.")
            return

        # Prüfe, ob conf_matrix die richtige Form hat
        if len(conf_matrix.shape) != 2 or conf_matrix.shape[0] == 0 or conf_matrix.shape[1] == 0:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Warnung", f"Konfusionsmatrix hat eine ungültige Form: {conf_matrix.shape}")
            return

        # Lösche das alte Layout
        if self.cm_widget.layout():
            # Lösche alle Widgets im Layout
            while self.cm_widget.layout().count():
                item = self.cm_widget.layout().takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            # Lösche das Layout selbst
            QWidget().setLayout(self.cm_widget.layout())

        # Erstelle ein neues Layout
        layout = QVBoxLayout(self.cm_widget)

        # Hole den aktuellen Modellnamen für den Titel
        model_name = self.model_select_combo.currentText() if hasattr(self, 'model_select_combo') else "Modell"

        # Erstelle eine neue Matplotlib-Figur mit größeren Dimensionen
        canvas = MatplotlibCanvas(self, width=8, height=7, dpi=100)  # Größere Figur für bessere Sichtbarkeit
        layout.addWidget(canvas)

        # Speichere die Figur für späteres Cleanup
        if hasattr(self, 'figures'):
            self.figures.append(canvas.fig)
        else:
            # Erstelle figures-Liste, falls sie nicht existiert
            self.figures = [canvas.fig]

        # Setze einen modernen Stil für die Figur
        canvas.fig.patch.set_facecolor('#f8f9fa')  # Heller Hintergrund für die Figur
        canvas.axes.set_facecolor('#ffffff')       # Weißer Hintergrund für den Plot

        # Zeichne die Konfusionsmatrix mit verbesserter Darstellung und modernem Farbschema
        im = canvas.axes.imshow(conf_matrix, interpolation='nearest', cmap='viridis', alpha=0.9)
        cbar = canvas.fig.colorbar(im, ax=canvas.axes, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Anzahl', rotation=270, fontsize=10, labelpad=15)

        # Beschriftungen für die Konfusionsmatrix mit verbessertem Styling
        canvas.axes.set_xlabel('Vorhergesagte Klasse', fontsize=11, fontweight='bold', labelpad=10)
        canvas.axes.set_ylabel('Tatsächliche Klasse', fontsize=11, fontweight='bold', labelpad=10)

        # Detaillierter Titel mit Modellname
        canvas.axes.set_title(f'Konfusionsmatrix: {model_name}', fontsize=14, fontweight='bold', pad=20)

        # Füge Gitterlinien hinzu für bessere Lesbarkeit
        canvas.axes.set_xticks(np.arange(conf_matrix.shape[1]))
        canvas.axes.set_yticks(np.arange(conf_matrix.shape[0]))
        canvas.axes.set_xticklabels(np.arange(conf_matrix.shape[1]), fontsize=9)
        canvas.axes.set_yticklabels(np.arange(conf_matrix.shape[0]), fontsize=9)

        # Füge Gitterlinien hinzu
        canvas.axes.grid(True, linestyle='--', alpha=0.3, color='gray')

        # Füge einen Rahmen um den Plot hinzu
        for spine in canvas.axes.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(0.8)

        # Füge die Werte in die Zellen ein mit verbesserter Formatierung
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                # Wähle weiße Textfarbe für dunkle Zellen und schwarze für helle Zellen
                color = "white" if conf_matrix[i, j] > thresh else "black"

                # Formatiere große Zahlen mit Tausendertrennzeichen
                value = f"{conf_matrix[i, j]:,}" if conf_matrix[i, j] >= 1000 else str(conf_matrix[i, j])

                # Passe die Schriftgröße basierend auf dem Wert an
                fontsize = 12 if conf_matrix[i, j] > 0 else 10

                # Füge den Text mit verbessertem Styling hinzu
                canvas.axes.text(j, i, value, ha="center", va="center", 
                                color=color, fontweight='bold', fontsize=fontsize)

        # Füge eine Beschriftung für die Diagonale hinzu (korrekte Vorhersagen)
        diag_sum = np.sum(np.diag(conf_matrix))
        total = np.sum(conf_matrix)
        accuracy = diag_sum / total if total > 0 else 0

        # Füge eine Textbox mit der Genauigkeit hinzu
        textstr = f'Genauigkeit: {accuracy:.2%}\n({diag_sum:,} von {total:,})'
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc')
        canvas.axes.text(0.05, 0.05, textstr, transform=canvas.axes.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=props)

        # Enge Layout-Anpassung für optimale Platznutzung
        canvas.fig.tight_layout(pad=2.0)
    except Exception as e:
        # Fange alle Ausnahmen ab und zeige eine Fehlermeldung an
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Fehler", f"Fehler beim Aktualisieren der Konfusionsmatrix: {str(e)}")
        print(f"Fehler in update_confusion_matrix: {str(e)}")
