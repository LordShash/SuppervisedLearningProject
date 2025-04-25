"""
Implementierung der Tab-Funktionalität für die moderne GUI.

Dieses Modul enthält die Implementierung der verschiedenen Tabs
für die moderne PyQt5-basierte Benutzeroberfläche.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QRadioButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QTextEdit, QSplitter, QToolTip,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPainter, QPen
from PyQt5.QtChart import QChartView, QPieSeries, QLineSeries, QChart, QValueAxis, QBarSeries, QBarSet
import matplotlib.pyplot as plt
import numpy as np

from suppervisedlearningproject.ui.gui import MatplotlibCanvas, TrainingWorker
from suppervisedlearningproject.utils import setup_logging


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



def init_training_tab(self):
    """Initialisiert den Training-Tab mit Formularelementen."""
    # Erstelle den Tab
    training_tab = QWidget()
    self.tab_widget.addTab(training_tab, "Training")

    # Layout für den Tab mit mehr Platz für Inhalte
    layout = QVBoxLayout(training_tab)
    layout.setSpacing(10)  # Erhöhe den Abstand zwischen Elementen
    layout.setContentsMargins(10, 10, 10, 10)  # Mehr Rand für bessere Lesbarkeit

    # Erstelle ein ScrollArea für bessere Responsivität bei kleinen Fenstern
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QScrollArea.NoFrame)

    # Container-Widget für den Scroll-Bereich
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    scroll_layout.setSpacing(10)

    # Füge das Scroll-Widget zum Layout hinzu
    layout.addWidget(scroll_area)
    scroll_area.setWidget(scroll_content)

    # Erstelle eine Gruppe für die Trainingsparameter
    params_group = QGroupBox("Trainingsparameter")
    # Setze eine flexible Größenpolitik
    params_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    scroll_layout.addWidget(params_group)

    # Layout für die Trainingsparameter mit mehr Platz
    params_layout = QFormLayout(params_group)
    params_layout.setSpacing(10)  # Erhöhe den Abstand zwischen Zeilen
    params_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # Erlaube Feldern zu wachsen

    # Zielvariable
    self.target_combo = QComboBox()
    self.target_combo.setMinimumWidth(200)  # Stelle sicher, dass die ComboBox breit genug ist
    self.target_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
    self.max_features_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    params_layout.addRow(self.create_label_with_tooltip("Max Features:", self.tooltips["max_features"]), 
                         self.max_features_spin)

    # Test Size
    self.test_size_spin = QDoubleSpinBox()
    self.test_size_spin.setRange(0.1, 0.5)
    self.test_size_spin.setValue(0.2)
    self.test_size_spin.setSingleStep(0.05)
    self.test_size_spin.setDecimals(2)
    self.test_size_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    self.test_size_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    params_layout.addRow(self.create_label_with_tooltip("Test Size:", self.tooltips["test_size"]), 
                         self.test_size_spin)

    # Parameter für logistische Regression
    self.logreg_group = QGroupBox("Parameter für logistische Regression")
    self.logreg_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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
    self.max_iter_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    logreg_layout.addRow(self.create_label_with_tooltip("Max Iterations:", self.tooltips["max_iter"]), 
                         self.max_iter_spin)

    # C (Regularisierung)
    self.c_reg_spin = QDoubleSpinBox()
    self.c_reg_spin.setRange(0.1, 10.0)
    self.c_reg_spin.setValue(1.0)
    self.c_reg_spin.setSingleStep(0.1)
    self.c_reg_spin.setDecimals(2)
    self.c_reg_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    self.c_reg_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    logreg_layout.addRow(self.create_label_with_tooltip("C (Regularisierung):", self.tooltips["c_reg"]), 
                         self.c_reg_spin)

    # Solver
    self.solver_combo = QComboBox()
    self.solver_combo.addItems(['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'])
    self.solver_combo.setMinimumWidth(150)  # Stelle sicher, dass die ComboBox breit genug ist
    self.solver_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    logreg_layout.addRow(self.create_label_with_tooltip("Solver:", self.tooltips["solver"]), 
                         self.solver_combo)

    # Parameter für neuronales Netz
    self.nn_group = QGroupBox("Parameter für neuronales Netz")
    self.nn_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    params_layout.addRow(self.nn_group)
    nn_layout = QFormLayout(self.nn_group)
    nn_layout.setSpacing(10)  # Erhöhe den Abstand zwischen Zeilen
    nn_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # Erlaube Feldern zu wachsen

    # Epochs
    self.epochs_spin = QSpinBox()
    self.epochs_spin.setRange(5, 200)
    self.epochs_spin.setValue(30)
    self.epochs_spin.setSingleStep(5)
    self.epochs_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    self.epochs_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    nn_layout.addRow(self.create_label_with_tooltip("Epochs:", self.tooltips["epochs"]), 
                     self.epochs_spin)

    # Patience
    self.patience_spin = QSpinBox()
    self.patience_spin.setRange(1, 20)
    self.patience_spin.setValue(5)
    self.patience_spin.setSingleStep(1)
    self.patience_spin.setMinimumWidth(100)  # Stelle sicher, dass das Feld breit genug ist
    self.patience_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    nn_layout.addRow(self.create_label_with_tooltip("Patience:", self.tooltips["patience"]), 
                     self.patience_spin)

    # Verstecke die NN-Parameter initial
    self.nn_group.setVisible(False)

    # Trainieren-Button
    self.train_button = QPushButton("Modell trainieren")
    self.train_button.setMinimumHeight(40)  # Höherer Button für bessere Sichtbarkeit
    self.train_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    self.train_button.clicked.connect(self.train_model)
    scroll_layout.addWidget(self.train_button)

    # Ausgabebereich
    output_group = QGroupBox("Ausgabe")
    output_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    scroll_layout.addWidget(output_group)
    output_layout = QVBoxLayout(output_group)

    self.output_text = QTextEdit()
    self.output_text.setReadOnly(True)
    self.output_text.setMinimumHeight(200)  # Stelle sicher, dass das Ausgabefeld groß genug ist
    self.output_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    output_layout.addWidget(self.output_text)


def init_results_tab(self):
    """Initialisiert den Ergebnisse-Tab mit Visualisierungen."""
    # Erstelle den Tab
    results_tab = QWidget()
    self.tab_widget.addTab(results_tab, "Ergebnisse")

    # Layout für den Tab
    layout = QVBoxLayout(results_tab)
    layout.setSpacing(10)
    layout.setContentsMargins(10, 10, 10, 10)

    # Erstelle ein ScrollArea für bessere Responsivität bei kleinen Fenstern
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QScrollArea.NoFrame)

    # Container-Widget für den Scroll-Bereich
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    scroll_layout.setSpacing(10)

    # Füge das Scroll-Widget zum Layout hinzu
    layout.addWidget(scroll_area)
    scroll_area.setWidget(scroll_content)

    # Modellauswahl und Vergleich
    model_select_group = QGroupBox("Modellauswahl und Vergleich")
    model_select_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    scroll_layout.addWidget(model_select_group)
    model_select_layout = QHBoxLayout(model_select_group)

    # Linke Seite: Modellauswahl
    model_select_widget = QWidget()
    model_select_inner_layout = QHBoxLayout(model_select_widget)
    model_select_inner_layout.setContentsMargins(0, 0, 0, 0)

    model_select_label = QLabel("Modell:")
    self.model_select_combo = QComboBox()
    self.model_select_combo.setMinimumWidth(300)
    self.model_select_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    self.model_select_combo.currentIndexChanged.connect(self.on_model_selected)

    # Initialisiere auch die Konfusionsmatrix-Modellauswahl
    # Diese wird in update_results_tab verwendet
    self.confusion_model_select_combo = QComboBox()
    self.confusion_model_select_combo.setMinimumWidth(300)
    self.confusion_model_select_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    self.confusion_model_select_combo.currentIndexChanged.connect(self.on_model_selected)

    model_select_inner_layout.addWidget(model_select_label)
    model_select_inner_layout.addWidget(self.model_select_combo)

    # Rechte Seite: Modellvergleich
    model_compare_widget = QWidget()
    model_compare_layout = QHBoxLayout(model_compare_widget)
    model_compare_layout.setContentsMargins(0, 0, 0, 0)

    compare_label = QLabel("Vergleichen mit:")
    self.compare_models_combo = QComboBox()
    self.compare_models_combo.setMinimumWidth(300)
    self.compare_models_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    self.compare_button = QPushButton("Vergleichen")
    self.compare_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    self.compare_button.clicked.connect(self.compare_models)

    model_compare_layout.addWidget(compare_label)
    model_compare_layout.addWidget(self.compare_models_combo)
    model_compare_layout.addWidget(self.compare_button)

    # Füge beide Widgets zum Layout hinzu
    model_select_layout.addWidget(model_select_widget)
    model_select_layout.addStretch()
    model_select_layout.addWidget(model_compare_widget)

    # Visualisierungen
    viz_group = QGroupBox("Visualisierungen")
    viz_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    scroll_layout.addWidget(viz_group)
    viz_layout = QVBoxLayout(viz_group)

    # Tabs für verschiedene Visualisierungen
    self.viz_tabs = QTabWidget()
    self.viz_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    viz_layout.addWidget(self.viz_tabs)

    # Tab für Konfusionsmatrix
    self.cm_tab = QWidget()
    self.viz_tabs.addTab(self.cm_tab, "Konfusionsmatrix")
    cm_layout = QVBoxLayout(self.cm_tab)

    # Widget für die Konfusionsmatrix
    self.cm_widget = QWidget()
    self.cm_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    cm_inner_layout = QVBoxLayout(self.cm_widget)
    cm_inner_layout.setContentsMargins(0, 0, 0, 0)
    cm_layout.addWidget(self.cm_widget)

    # Erstelle einen leeren Placeholder-Canvas für die Konfusionsmatrix
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_facecolor('#f0f4f8')
    ax.text(0.5, 0.5, "Konfusionsmatrix wird hier angezeigt", 
            ha='center', va='center', fontsize=12, color='#666666')
    ax.set_xticks([])
    ax.set_yticks([])
    self.embed_canvas(self.cm_widget, fig)

    # Tab für ROC-Kurve
    self.roc_tab = QWidget()
    self.viz_tabs.addTab(self.roc_tab, "ROC-Kurve")
    roc_layout = QVBoxLayout(self.roc_tab)

    # Widget für die ROC-Kurve
    self.roc_widget = QWidget()
    self.roc_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    roc_inner_layout = QVBoxLayout(self.roc_widget)
    roc_inner_layout.setContentsMargins(0, 0, 0, 0)
    roc_layout.addWidget(self.roc_widget)

    # Erstelle einen leeren Placeholder-Canvas für die ROC-Kurve
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_facecolor('#f0f4f8')
    ax.text(0.5, 0.5, "ROC-Kurve wird hier angezeigt", 
            ha='center', va='center', fontsize=12, color='#666666')
    ax.set_xticks([])
    ax.set_yticks([])
    self.embed_canvas(self.roc_widget, fig)

    # Tab für Modellvergleich
    self.compare_tab = QWidget()
    self.viz_tabs.addTab(self.compare_tab, "Modellvergleich")
    compare_layout = QVBoxLayout(self.compare_tab)

    # Widget für den Modellvergleich
    self.compare_widget = QWidget()
    self.compare_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    compare_inner_layout = QVBoxLayout(self.compare_widget)
    compare_inner_layout.setContentsMargins(0, 0, 0, 0)
    compare_layout.addWidget(self.compare_widget)

    # Erstelle einen leeren Placeholder-Canvas für den Modellvergleich
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_facecolor('#f0f4f8')
    ax.text(0.5, 0.5, "Modellvergleich wird hier angezeigt", 
            ha='center', va='center', fontsize=12, color='#666666')
    ax.set_xticks([])
    ax.set_yticks([])
    self.embed_canvas(self.compare_widget, fig)

    # Splitter für Metriken und Bericht
    splitter = QSplitter(Qt.Horizontal)
    splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    scroll_layout.addWidget(splitter)

    # Metriken
    metrics_group = QGroupBox("Metriken")
    metrics_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    metrics_layout = QFormLayout(metrics_group)
    metrics_layout.setSpacing(10)
    metrics_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    # Erstelle Labels für die Metriken mit einheitlichem Stil
    self.accuracy_label = QLabel("0.0000")
    self.precision_label = QLabel("0.0000")
    self.recall_label = QLabel("0.0000")
    self.f1_label = QLabel("0.0000")
    self.auc_label = QLabel("0.0000")  # Neu: AUC-Label

    # Setze Monospace-Schrift für alle Metrik-Labels
    monospace_font = QFont("Consolas", 10)
    self.accuracy_label.setFont(monospace_font)
    self.precision_label.setFont(monospace_font)
    self.recall_label.setFont(monospace_font)
    self.f1_label.setFont(monospace_font)
    self.auc_label.setFont(monospace_font)

    # Füge die Metriken zum Layout hinzu
    metrics_layout.addRow("Accuracy:", self.accuracy_label)
    metrics_layout.addRow("Precision:", self.precision_label)
    metrics_layout.addRow("Recall:", self.recall_label)
    metrics_layout.addRow("F1 Score:", self.f1_label)
    metrics_layout.addRow("AUC:", self.auc_label)  # Neu: AUC-Metrik

    # Klassifikationsbericht
    report_group = QGroupBox("Klassifikationsbericht")
    report_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    report_layout = QVBoxLayout(report_group)

    self.report_text = QTextEdit()
    self.report_text.setReadOnly(True)
    self.report_text.setMinimumHeight(200)
    self.report_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    report_layout.addWidget(self.report_text)

    # Füge die Widgets zum Splitter hinzu
    splitter.addWidget(metrics_group)
    splitter.addWidget(report_group)

    # Setze die Größenverhältnisse
    splitter.setSizes([200, 400])


def init_about_tab(self):
    """Initialisiert den Über-Tab mit Informationen zur Anwendung."""
    # Erstelle den Tab
    about_tab = QWidget()
    self.tab_widget.addTab(about_tab, "Über")

    # Layout für den Tab
    layout = QVBoxLayout(about_tab)
    layout.setSpacing(10)
    layout.setContentsMargins(10, 10, 10, 10)

    # Erstelle ein ScrollArea für bessere Responsivität bei kleinen Fenstern
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QScrollArea.NoFrame)

    # Container-Widget für den Scroll-Bereich
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    scroll_layout.setSpacing(20)
    scroll_layout.setContentsMargins(10, 10, 10, 10)

    # Füge das Scroll-Widget zum Layout hinzu
    layout.addWidget(scroll_area)
    scroll_area.setWidget(scroll_content)

    # Titel
    title_label = QLabel("Textklassifikationsanwendung")
    title_label.setAlignment(Qt.AlignCenter)
    title_font = QFont()
    title_font.setPointSize(16)
    title_font.setBold(True)
    title_label.setFont(title_font)
    title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    scroll_layout.addWidget(title_label)

    # Beschreibung
    description_text = QTextEdit()
    description_text.setReadOnly(True)
    description_text.setMinimumHeight(300)
    description_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    description_text.setHtml("""
    <h2>Über diese Anwendung</h2>
    <p>Diese Anwendung ermöglicht das Training und die Evaluation von Textklassifikationsmodellen 
    mit einer benutzerfreundlichen grafischen Oberfläche.</p>

    <h3>Funktionen:</h3>
    <ul>
        <li>Training von logistischen Regressionsmodellen</li>
        <li>Training von neuronalen Netzen</li>
        <li>Visualisierung der Ergebnisse</li>
        <li>Speicherung und Verwaltung von Modellen</li>
    </ul>

    <h3>Verwendung:</h3>
    <ol>
        <li>Wählen Sie eine Zielvariable aus</li>
        <li>Wählen Sie einen Modelltyp (logistische Regression oder neuronales Netz)</li>
        <li>Passen Sie die Parameter an Ihre Bedürfnisse an</li>
        <li>Klicken Sie auf "Modell trainieren"</li>
        <li>Analysieren Sie die Ergebnisse im Ergebnisse-Tab</li>
    </ol>

    <h3>Tipps:</h3>
    <ul>
        <li>Halten Sie den Mauszeiger über die Parameter, um Erklärungen zu erhalten</li>
        <li>Experimentieren Sie mit verschiedenen Parametern, um die Ergebnisse zu verbessern</li>
        <li>Vergleichen Sie die Ergebnisse verschiedener Modelle im Ergebnisse-Tab</li>
    </ul>
    """)
    layout.addWidget(description_text)

    # Version und Copyright
    version_label = QLabel("Version 1.0.0")
    version_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(version_label)

    copyright_label = QLabel("© 2023 Textklassifikationsprojekt")
    copyright_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(copyright_label)

    # Füge Stretch hinzu, damit die Elemente oben bleiben
    layout.addStretch()


def create_label_with_tooltip(self, text, tooltip):
    """
    Erstellt ein Label mit einem Tooltip-Icon.

    Args:
        text: Der Text des Labels
        tooltip: Der Tooltip-Text

    Returns:
        Ein Container-Widget mit dem Label und dem Tooltip-Icon
    """
    # Erstelle einen Container für das Label und das Tooltip-Icon
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)

    # Erstelle das Label
    label = QLabel(text)
    layout.addWidget(label)

    # Wenn kein Tooltip vorhanden ist, gib nur das Label zurück
    if not tooltip:
        return container

    # Erstelle eine benutzerdefinierte Label-Klasse für das Fragezeichen-Icon
    class HelpLabel(QLabel):
        def __init__(self, text, tooltip):
            super().__init__(text)
            self.tooltip = tooltip
            self.setToolTip(tooltip)
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
            self.setCursor(Qt.WhatsThisCursor)

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
            # Zeige den Tooltip an
            QToolTip.showText(event.globalPos(), self.tooltip)
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
    logger = setup_logging("gui_tabs")
    logger.info("on_training_finished aufgerufen")

    # Speichere die Ergebnisse
    model_key = f"{results['model_type']}_{results['target_column']}"
    logger.info(f"Speichere Ergebnisse für Modell '{model_key}'")
    self.training_results[model_key] = results

    # Speichere die Trainingsergebnisse auf der Festplatte
    self.save_training_results()

    # Speichere die Konfusionsmatrix als PNG
    if 'conf_matrix' in results and 'class_names' in results:
        logger.info("Rufe update_confusion_matrix auf, um Konfusionsmatrix zu speichern")
        self.update_confusion_matrix(
            conf_matrix=results['conf_matrix'],
            class_names=results['class_names'],
            save_to_file=True,
            target_column=results['target_column']
        )

    # Aktualisiere die Ergebnisse im Ergebnisse-Tab
    logger.info("Aktualisiere Ergebnisse im Ergebnisse-Tab")
    self.update_results_tab()

    # Wechsle zum Ergebnisse-Tab
    logger.info("Wechsle zum Ergebnisse-Tab")
    self.tab_widget.setCurrentIndex(1)


def on_training_error(self, error_msg):
    """Wird aufgerufen, wenn ein Fehler beim Training auftritt."""
    QMessageBox.critical(self, "Trainingsfehler", f"Fehler beim Training: {error_msg}")


def update_results_tab(self):
    """Aktualisiert die Modellauswahl in allen Tabs."""
    try:
        # Prüfe, ob training_results existiert und ein Dictionary ist
        if not hasattr(self, 'training_results') or not isinstance(self.training_results, dict):
            logger = setup_logging("gui_tabs")
            logger.warning("Warnung: training_results ist nicht verfügbar oder kein Dictionary")
            return

        # Prüfe, ob die erforderlichen Attribute existieren
        if not hasattr(self, 'model_select_combo'):
            logger = setup_logging("gui_tabs")
            logger.warning("Warnung: model_select_combo ist nicht verfügbar")
            return

        # Prüfe, ob confusion_model_select_combo existiert
        if not hasattr(self, 'confusion_model_select_combo'):
            logger = setup_logging("gui_tabs")
            logger.warning("Warnung: confusion_model_select_combo ist nicht verfügbar")
            # Erstelle das Attribut, wenn es nicht existiert
            self.confusion_model_select_combo = QComboBox()
            self.confusion_model_select_combo.setMinimumWidth(300)
            self.confusion_model_select_combo.currentIndexChanged.connect(self.on_model_selected)

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
            logger = setup_logging("gui_tabs")
            logger.warning("Keine Modelle in training_results gefunden")

            # Entsperre die Signale wieder
            self.model_select_combo.blockSignals(False)
            self.confusion_model_select_combo.blockSignals(False)

            # Zeige eine Meldung an, dass keine Modelle vorhanden sind
            if hasattr(self, 'report_text'):
                self.report_text.setText("Keine Modelle vorhanden. Bitte trainieren Sie zuerst ein Modell im Training-Tab.")

            # Setze die Metriken auf 0
            if hasattr(self, 'accuracy_label'):
                self.accuracy_label.setText("0.0000")
            if hasattr(self, 'precision_label'):
                self.precision_label.setText("0.0000")
            if hasattr(self, 'recall_label'):
                self.recall_label.setText("0.0000")
            if hasattr(self, 'f1_label'):
                self.f1_label.setText("0.0000")

            # Leere die Konfusionsmatrix
            if hasattr(self, 'cm_widget') and self.cm_widget.layout():
                # Lösche alle Widgets im Layout
                while self.cm_widget.layout().count():
                    item = self.cm_widget.layout().takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

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
        logger = setup_logging("gui_tabs")
        logger.error(f"Fehler in update_results_tab: {str(e)}")


def on_model_selected(self, index):
    """Wird aufgerufen, wenn ein Modell im Ergebnisse-Tab oder Konfusionsmatrix-Tab ausgewählt wird."""
    logger = setup_logging("gui_tabs")
    logger.info(f"on_model_selected aufgerufen mit index={index}")

    try:
        if index < 0:
            logger.warning("Ungültiger Index: index < 0")
            return

        # Prüfe, ob die erforderlichen Attribute existieren
        if not hasattr(self, 'model_select_combo'):
            logger = setup_logging("gui_tabs")
            logger.warning("Warnung: model_select_combo ist nicht verfügbar")
            return

        # Prüfe, ob confusion_model_select_combo existiert
        if not hasattr(self, 'confusion_model_select_combo'):
            logger = setup_logging("gui_tabs")
            logger.warning("Warnung: confusion_model_select_combo ist nicht verfügbar")
            # Erstelle das Attribut, wenn es nicht existiert
            self.confusion_model_select_combo = QComboBox()
            self.confusion_model_select_combo.setMinimumWidth(300)
            self.confusion_model_select_combo.currentIndexChanged.connect(self.on_model_selected)

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
            # Zeige eine Fehlermeldung an, anstatt Dummy-Werte hinzuzufügen

            logger = setup_logging("gui_tabs")
            error_msg = f"Fehlende Daten im Modell '{model_key}': {', '.join(missing_keys)}"
            logger.error(error_msg)

            QMessageBox.critical(self, "Fehler bei Modelldaten", 
                                f"Das Modell '{model_key}' enthält nicht alle erforderlichen Daten.\n\n"
                                f"Fehlende Daten: {', '.join(missing_keys)}\n\n"
                                "Bitte trainieren Sie das Modell erneut.")
            return

        # Aktualisiere die Metriken im Ergebnisse-Tab
        self.accuracy_label.setText(f"{results['accuracy']:.4f}")
        self.precision_label.setText(f"{results['precision']:.4f}")
        self.recall_label.setText(f"{results['recall']:.4f}")
        self.f1_label.setText(f"{results['f1']:.4f}")

        # Aktualisiere AUC-Wert, falls vorhanden
        if 'auc_scores' in results and results['auc_scores'] is not None:
            # Verwende den gewichteten AUC-Wert, falls vorhanden, sonst den ersten verfügbaren
            if 'weighted' in results['auc_scores']:
                auc_value = results['auc_scores']['weighted']
            elif 'binary' in results['auc_scores']:
                auc_value = results['auc_scores']['binary']
            elif len(results['auc_scores']) > 0:
                # Nehme den ersten verfügbaren AUC-Wert
                auc_value = next(iter(results['auc_scores'].values()))
            else:
                auc_value = 0.0

            self.auc_label.setText(f"{auc_value:.4f}")
        else:
            self.auc_label.setText("N/A")

        # Aktualisiere den Klassifikationsbericht mit verbesserter Formatierung
        # Formatiere den Klassifikationsbericht mit korrekter Einrückung
        report = results['report']

        # Verwende eine Monospace-Schriftart für den Bericht, um die Ausrichtung zu erhalten
        self.report_text.setFontFamily("Courier New")

        # Füge den formatierten Bericht hinzu
        report_text = "=== Klassifikationsbericht ===\n\n" + report
        self.report_text.setText(report_text)

        # Zeige die Konfusionsmatrix an
        conf_matrix = results.get('conf_matrix')

        # Prüfe, ob die Konfusionsmatrix gültig ist
        if not isinstance(conf_matrix, np.ndarray):
            logger = setup_logging("gui_tabs")
            error_msg = f"Konfusionsmatrix für Modell '{model_key}' ist kein gültiges numpy-Array"
            logger.error(error_msg)

            QMessageBox.warning(self, "Ungültige Konfusionsmatrix", 
                               f"Die Konfusionsmatrix für das Modell '{model_key}' ist ungültig.\n\n"
                               "Die Konfusionsmatrix kann nicht angezeigt werden.")
            return

        # Prüfe, ob die Konfusionsmatrix leer ist
        if conf_matrix.size == 0:
            logger = setup_logging("gui_tabs")
            error_msg = f"Konfusionsmatrix für Modell '{model_key}' ist leer"
            logger.error(error_msg)

            QMessageBox.warning(self, "Leere Konfusionsmatrix", 
                               f"Die Konfusionsmatrix für das Modell '{model_key}' ist leer.\n\n"
                               "Die Konfusionsmatrix kann nicht angezeigt werden.")
            return

        # Prüfe, ob die Konfusionsmatrix die richtige Form hat
        if len(conf_matrix.shape) != 2:
            logger = setup_logging("gui_tabs")
            error_msg = f"Konfusionsmatrix für Modell '{model_key}' hat ein ungültiges Format: {conf_matrix.shape}"
            logger.error(error_msg)

            QMessageBox.warning(self, "Ungültiges Format der Konfusionsmatrix", 
                               f"Die Konfusionsmatrix für das Modell '{model_key}' hat ein ungültiges Format.\n\n"
                               "Die Konfusionsmatrix kann nicht angezeigt werden.")
            return

        # Aktualisiere die Konfusionsmatrix
        logger.info(f"Rufe update_confusion_matrix für Modell '{model_key}' auf")
        self.update_confusion_matrix(conf_matrix, results.get('class_names'))

        # Aktualisiere die ROC-Kurve, falls Daten vorhanden sind
        if 'roc_curve_data' in results and results['roc_curve_data'] is not None:
            logger.info(f"Rufe update_roc_curve für Modell '{model_key}' auf")
            self.update_roc_curve(results['roc_curve_data'], results['auc_scores'], results.get('class_names'))
        else:
            logger.warning(f"Keine ROC-Kurven-Daten für Modell '{model_key}' vorhanden")
            # Leere die ROC-Kurve, wenn keine Daten vorhanden sind
            if hasattr(self, 'roc_widget'):
                if self.roc_widget.layout():
                    # Lösche alle Widgets im Layout
                    while self.roc_widget.layout().count():
                        item = self.roc_widget.layout().takeAt(0)
                        widget = item.widget()
                        if widget:
                            widget.deleteLater()

                    # Lösche das Layout selbst
                    old_layout = self.roc_widget.layout()
                    self.roc_widget.setLayout(None)
                    if old_layout:
                        old_layout.deleteLater()

                # Erstelle ein neues Layout
                layout = QVBoxLayout(self.roc_widget)

                # Zeige eine Meldung an, dass keine ROC-Kurve verfügbar ist
                no_data_label = QLabel("Keine ROC-Kurven-Daten für dieses Modell verfügbar.")
                no_data_label.setAlignment(Qt.AlignCenter)
                no_data_label.setStyleSheet("color: #666; font-style: italic; margin: 20px;")
                layout.addWidget(no_data_label)

                # Stelle sicher, dass das Widget sichtbar ist
                self.roc_widget.show()
                layout.update()

        # Aktualisiere auch die Vergleichsansicht
        self.update_compare_models_combo()

    except Exception as e:
        # Fange alle Ausnahmen ab und zeige eine Fehlermeldung an
        QMessageBox.critical(self, "Fehler", f"Fehler beim Anzeigen der Modellergebnisse: {str(e)}")
        logger = setup_logging("gui_tabs")
        logger.error(f"Fehler in on_model_selected: {str(e)}")


def update_roc_curve(self, roc_curve_data, auc_scores, class_names=None):
    """
    Aktualisiert die ROC-Kurven-Visualisierung mit QChartView und QLineSeries.

    Args:
        roc_curve_data: Dictionary mit ROC-Kurven-Daten (fpr, tpr, thresholds) für jede Klasse
        auc_scores: Dictionary mit AUC-Werten für jede Klasse
        class_names: Liste mit den Namen der Klassen (optional)

    # FIX visualization blank issue 2025-04-23:
    # - Verwende das bestehende Layout statt es zu löschen und neu zu erstellen
    # - Rufe explizit show() für das ChartView auf
    """
    logger = setup_logging("gui_tabs")
    logger.info("Starte update_roc_curve...")

    try:
        # Prüfe, ob roc_curve_data ein gültiges Dictionary ist
        if not isinstance(roc_curve_data, dict) or not roc_curve_data:
            logger.error("ROC-Kurven-Daten sind kein gültiges Dictionary oder leer.")

            QMessageBox.warning(self, "Ungültige ROC-Kurven-Daten", 
                               "Die ROC-Kurven-Daten sind ungültig oder leer.\n\n"
                               "Die ROC-Kurve kann nicht angezeigt werden.")
            return

        # Lösche alle Widgets im bestehenden Layout
        layout = self.roc_widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        else:
            # Falls kein Layout existiert, erstelle ein neues
            layout = QVBoxLayout(self.roc_widget)

        # Hole den aktuellen Modellnamen für den Titel
        model_name = self.model_select_combo.currentText() if hasattr(self, 'model_select_combo') else "Modell"

        # Erstelle einen Titel
        title_label = QLabel(f"ROC-Kurve: {model_name}")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00796b;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Erstelle ein QChart-Objekt
        chart = QChart()
        chart.setTitle("")  # Wir verwenden einen separaten QLabel als Titel
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        # Erstelle die Achsen
        axisX = QValueAxis()
        axisX.setTitleText("Falsch-Positiv-Rate")
        axisX.setRange(0, 1)
        axisX.setTickCount(6)
        axisX.setLabelFormat("%.1f")

        axisY = QValueAxis()
        axisY.setTitleText("Richtig-Positiv-Rate")
        axisY.setRange(0, 1)
        axisY.setTickCount(6)
        axisY.setLabelFormat("%.1f")

        chart.addAxis(axisX, Qt.AlignBottom)
        chart.addAxis(axisY, Qt.AlignLeft)

        # Definiere Farben für die Kurven
        colors = [
            QColor("#1f77b4"),  # Blau
            QColor("#ff7f0e"),  # Orange
            QColor("#2ca02c"),  # Grün
            QColor("#d62728"),  # Rot
            QColor("#9467bd"),  # Lila
            QColor("#8c564b"),  # Braun
            QColor("#e377c2"),  # Pink
            QColor("#7f7f7f"),  # Grau
            QColor("#bcbd22"),  # Olivgrün
            QColor("#17becf")   # Türkis
        ]

        # Erstelle eine Linie für den zufälligen Klassifikator (Diagonale)
        random_series = QLineSeries()
        random_series.setName("Zufälliger Klassifikator")
        random_series.append(0, 0)
        random_series.append(1, 1)
        random_series.setPen(QPen(QColor("gray"), 2, Qt.DashLine))

        chart.addSeries(random_series)
        random_series.attachAxis(axisX)
        random_series.attachAxis(axisY)

        # Zähler für die Farben
        color_idx = 0

        # Maximaler AUC-Wert für die Legende
        max_auc = 0.0
        max_auc_class = ""

        # Erstelle die ROC-Kurven
        for class_key, curve_data in roc_curve_data.items():
            # Hole die Daten
            fpr = curve_data.get('fpr')
            tpr = curve_data.get('tpr')

            if fpr is None or tpr is None or len(fpr) != len(tpr):
                logger = setup_logging("gui_tabs")
                logger.warning(f"Ungültige ROC-Kurven-Daten für Klasse '{class_key}'")
                continue

            # Hole den AUC-Wert
            auc_value = auc_scores.get(class_key, 0.0) if auc_scores else 0.0

            # Aktualisiere den maximalen AUC-Wert
            if auc_value > max_auc:
                max_auc = auc_value
                max_auc_class = class_key

            # Bestimme den Klassennamen
            if class_key == 'binary':
                class_label = 'Binäre Klassifikation'
            elif class_key == 'weighted':
                class_label = 'Gewichteter Durchschnitt'
            elif class_names and len(class_names) > int(class_key) if class_key.isdigit() else -1:
                class_label = class_names[int(class_key)]
            else:
                class_label = class_key

            # Erstelle eine QLineSeries für diese Klasse
            series = QLineSeries()
            series.setName(f"{class_label} (AUC = {auc_value:.4f})")

            # Füge die Datenpunkte hinzu
            for i in range(len(fpr)):
                series.append(fpr[i], tpr[i])

            # Setze die Farbe und Stärke der Linie
            color = colors[color_idx % len(colors)]
            series.setPen(QPen(color, 3))

            # Füge die Serie zum Chart hinzu
            chart.addSeries(series)
            series.attachAxis(axisX)
            series.attachAxis(axisY)

            # Erhöhe den Farbindex
            color_idx += 1

        # Erstelle einen QChartView und füge das Chart hinzu
        chartView = QChartView(chart)
        chartView.setRenderHint(QPainter.Antialiasing)
        chartView.setMinimumHeight(400)
        layout.addWidget(chartView)

        # Füge eine Beschriftung mit dem besten AUC-Wert hinzu
        if max_auc > 0:
            if max_auc_class == 'binary':
                best_class_label = 'Binäre Klassifikation'
            elif max_auc_class == 'weighted':
                best_class_label = 'Gewichteter Durchschnitt'
            elif class_names and len(class_names) > int(max_auc_class) if max_auc_class.isdigit() else -1:
                best_class_label = class_names[int(max_auc_class)]
            else:
                best_class_label = max_auc_class

            auc_label = QLabel(f"Bester AUC-Wert: {max_auc:.4f} ({best_class_label})")
            auc_label.setStyleSheet("font-size: 12pt; margin-top: 10px; color: #00796b;")
            auc_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(auc_label)

        # Füge eine Erklärung für die ROC-Kurve hinzu
        explanation = QLabel(
            "Die ROC-Kurve (Receiver Operating Characteristic) zeigt das Verhältnis zwischen "
            "der Richtig-Positiv-Rate (Sensitivität) und der Falsch-Positiv-Rate (1 - Spezifität) "
            "für verschiedene Schwellenwerte. Die Fläche unter der Kurve (AUC) ist ein Maß für die "
            "Qualität des Modells. Ein AUC-Wert von 1.0 bedeutet eine perfekte Klassifikation, "
            "während ein Wert von 0.5 (Diagonale) einem zufälligen Klassifikator entspricht."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #666; font-style: italic; margin-top: 10px;")
        layout.addWidget(explanation)

        # Setze eine Mindestgröße für das ChartView
        chartView.setMinimumSize(400, 300)

        # Stelle sicher, dass die Widgets sichtbar sind
        self.roc_widget.show()
        layout.update()
        chartView.show()

        logger.info("ROC-Kurve erfolgreich aktualisiert.")

    except Exception as e:
        # Fange alle Ausnahmen ab und zeige eine Fehlermeldung an
        QMessageBox.critical(self, "Fehler", f"Fehler beim Aktualisieren der ROC-Kurve: {str(e)}")
        logger.error(f"Fehler in update_roc_curve: {str(e)}")

        # Zeige eine Fehlermeldung im Widget an
        if self.roc_widget.layout() is None:
            layout = QVBoxLayout(self.roc_widget)
        else:
            layout = self.roc_widget.layout()
            # Lösche alle Widgets im Layout
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        error_label = QLabel(f"Fehler beim Anzeigen der ROC-Kurve:\n{str(e)}")
        error_label.setStyleSheet("color: red; font-weight: bold;")
        error_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(error_label)

        # Versuche, den Fehler zu diagnostizieren
        try:
            import importlib
            if importlib.util.find_spec("PyQt5.QtChart") is None:
                error_detail = QLabel("PyQt5.QtChart ist nicht installiert. Bitte installieren Sie das PyQtChart-Paket.")
                error_detail.setStyleSheet("color: red;")
                error_detail.setAlignment(Qt.AlignCenter)
                layout.addWidget(error_detail)
        except Exception as import_error:
            logger.error(f"Fehler beim Prüfen der PyQt5.QtChart-Installation: {str(import_error)}")

def update_compare_models_combo(self):
    """
    Aktualisiert die ComboBox für den Modellvergleich mit allen verfügbaren Modellen
    außer dem aktuell ausgewählten.
    """
    try:
        # Prüfe, ob die erforderlichen Attribute existieren
        if not hasattr(self, 'compare_models_combo') or not hasattr(self, 'model_select_combo'):
            return

        # Hole den aktuellen Modellnamen
        current_model = self.model_select_combo.currentText()

        # Blockiere Signale, um unerwünschte Aktualisierungen zu vermeiden
        self.compare_models_combo.blockSignals(True)

        # Leere die ComboBox
        self.compare_models_combo.clear()

        # Füge alle Modelle außer dem aktuell ausgewählten hinzu
        for model_name in self.training_results.keys():
            if model_name != current_model:
                self.compare_models_combo.addItem(model_name)

        # Entsperre die Signale wieder
        self.compare_models_combo.blockSignals(False)

        # Aktiviere/Deaktiviere den Vergleichen-Button je nach Anzahl der Modelle
        if hasattr(self, 'compare_button'):
            self.compare_button.setEnabled(self.compare_models_combo.count() > 0)

    except Exception as e:
        logger = setup_logging("gui_tabs")
        logger.error(f"Fehler in update_compare_models_combo: {str(e)}")

def compare_models(self):
    """
    Vergleicht das aktuell ausgewählte Modell mit dem in der Vergleichs-ComboBox ausgewählten Modell.
    """
    try:
        # Prüfe, ob die erforderlichen Attribute existieren
        if not hasattr(self, 'compare_models_combo') or not hasattr(self, 'model_select_combo'):
            return

        # Hole die Modellnamen
        current_model = self.model_select_combo.currentText()
        compare_model = self.compare_models_combo.currentText()

        if not current_model or not compare_model or current_model == compare_model:
            QMessageBox.warning(self, "Ungültiger Vergleich", 
                               "Bitte wählen Sie zwei verschiedene Modelle für den Vergleich aus.")
            return

        # Hole die Ergebnisse für beide Modelle
        current_results = self.training_results.get(current_model)
        compare_results = self.training_results.get(compare_model)

        if not current_results or not compare_results:
            QMessageBox.warning(self, "Fehlende Daten", 
                               "Für mindestens eines der ausgewählten Modelle sind keine Daten verfügbar.")
            return

        # Aktualisiere die Vergleichsansicht
        self.update_model_comparison(current_model, current_results, compare_model, compare_results)

        # Wechsle zum Vergleichs-Tab
        if hasattr(self, 'viz_tabs'):
            # Finde den Index des Vergleichs-Tabs
            for i in range(self.viz_tabs.count()):
                if self.viz_tabs.tabText(i) == "Modellvergleich":
                    self.viz_tabs.setCurrentIndex(i)
                    break

    except Exception as e:
        QMessageBox.critical(self, "Fehler", f"Fehler beim Vergleichen der Modelle: {str(e)}")
        logger = setup_logging("gui_tabs")
        logger.error(f"Fehler in compare_models: {str(e)}")

def update_model_comparison(self, model1_name, model1_results, model2_name, model2_results):
    """
    Aktualisiert die Modellvergleichsansicht mit einer Tabelle der Kernmetriken.

    Args:
        model1_name: Name des ersten Modells
        model1_results: Ergebnisse des ersten Modells
        model2_name: Name des zweiten Modells
        model2_results: Ergebnisse des zweiten Modells

    # FIX visualization blank issue 2025-04-23:
    # - Verwende das bestehende Layout statt es zu löschen und neu zu erstellen
    # - Passe die Spalten- und Zeilengrößen explizit an den Inhalt an
    # - Rufe explizit show() für die Tabelle auf
    # - Rufe explizit draw() für den Matplotlib-Canvas auf
    """
    logger = setup_logging("gui_tabs")
    logger.info("Starte update_model_comparison...")

    try:
        # Lösche alle Widgets im bestehenden Layout
        layout = self.compare_widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        else:
            # Falls kein Layout existiert, erstelle ein neues
            layout = QVBoxLayout(self.compare_widget)

        # Erstelle einen Titel
        title_label = QLabel(f"Vergleich: {model1_name} vs. {model2_name}")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00796b;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Erstelle eine Tabelle für den Vergleich
        table = QTableWidget()
        table.setColumnCount(3)  # Metrik, Modell 1, Modell 2

        # Setze die Spaltenüberschriften
        table.setHorizontalHeaderLabels(["Metrik", model1_name, model2_name])

        # Definiere die zu vergleichenden Metriken
        metrics = [
            ("Accuracy", "accuracy", 4),
            ("Precision", "precision", 4),
            ("Recall", "recall", 4),
            ("F1 Score", "f1", 4)
        ]

        # Füge AUC hinzu, falls verfügbar
        if ('auc_scores' in model1_results and model1_results['auc_scores'] is not None and
            'auc_scores' in model2_results and model2_results['auc_scores'] is not None):
            # Verwende den gewichteten AUC-Wert, falls vorhanden
            if ('weighted' in model1_results['auc_scores'] and 
                'weighted' in model2_results['auc_scores']):
                metrics.append(("AUC (gewichtet)", "auc_weighted", 4))
            # Oder den binären AUC-Wert
            elif ('binary' in model1_results['auc_scores'] and 
                  'binary' in model2_results['auc_scores']):
                metrics.append(("AUC (binär)", "auc_binary", 4))

        # Setze die Anzahl der Zeilen
        table.setRowCount(len(metrics))

        # Fülle die Tabelle
        for i, (metric_name, metric_key, decimals) in enumerate(metrics):
            # Setze den Metriknamen
            metric_item = QTableWidgetItem(metric_name)
            metric_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setItem(i, 0, metric_item)

            # Hole die Werte für beide Modelle
            if metric_key.startswith("auc_"):
                # Spezialfall für AUC
                auc_type = metric_key.split("_")[1]  # "weighted" oder "binary"
                value1 = model1_results['auc_scores'].get(auc_type, 0.0)
                value2 = model2_results['auc_scores'].get(auc_type, 0.0)
            else:
                # Normale Metriken
                value1 = model1_results.get(metric_key, 0.0)
                value2 = model2_results.get(metric_key, 0.0)

            # Formatiere die Werte
            value1_str = f"{value1:.{decimals}f}"
            value2_str = f"{value2:.{decimals}f}"

            # Erstelle die Tabelleneinträge
            value1_item = QTableWidgetItem(value1_str)
            value2_item = QTableWidgetItem(value2_str)

            # Setze die Ausrichtung
            value1_item.setTextAlignment(Qt.AlignCenter)
            value2_item.setTextAlignment(Qt.AlignCenter)

            # Hebe das bessere Ergebnis hervor
            if value1 > value2:
                value1_item.setBackground(QColor("#e0f2f1"))  # Helles Türkis
                value1_item.setForeground(QColor("#00796b"))  # Dunkles Türkis
                value1_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            elif value2 > value1:
                value2_item.setBackground(QColor("#e0f2f1"))  # Helles Türkis
                value2_item.setForeground(QColor("#00796b"))  # Dunkles Türkis
                value2_item.setFont(QFont("Segoe UI", 9, QFont.Bold))

            # Füge die Items zur Tabelle hinzu
            table.setItem(i, 1, value1_item)
            table.setItem(i, 2, value2_item)

        # Passe die Spaltenbreiten an
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        # Passe die Zeilenhöhen an
        for i in range(table.rowCount()):
            table.setRowHeight(i, 30)

        # Füge die Tabelle zum Layout hinzu
        layout.addWidget(table)

        # Erstelle eine Visualisierung für den Vergleich der ROC-Kurven, falls verfügbar
        if ('roc_curve_data' in model1_results and model1_results['roc_curve_data'] is not None and
            'roc_curve_data' in model2_results and model2_results['roc_curve_data'] is not None):

            # Erstelle eine Matplotlib-Figur für den ROC-Kurven-Vergleich
            roc_compare_group = QGroupBox("ROC-Kurven-Vergleich")
            roc_compare_layout = QVBoxLayout(roc_compare_group)

            # Erstelle eine neue Figure mit constrained_layout für bessere Darstellung
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100, constrained_layout=True)

            # Setze einen modernen Stil für die Figur
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')

            # Bette die Figur in das Widget ein
            canvas = self.embed_canvas(roc_compare_group, fig)

            # Zeichne die Diagonale (zufälliger Klassifikator)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8, 
                  label='Zufälliger Klassifikator', lw=2)

            # Zeichne die ROC-Kurven für beide Modelle
            colors1 = plt.cm.Blues(np.linspace(0.6, 1.0, 3))  # Blautöne für Modell 1
            colors2 = plt.cm.Oranges(np.linspace(0.6, 1.0, 3))  # Orangetöne für Modell 2

            # Funktion zum Zeichnen der ROC-Kurven eines Modells
            def plot_model_roc_curves(roc_data, auc_scores, model_name, colors):
                color_idx = 0
                for class_key, curve_data in roc_data.items():
                    # Überspringe, wenn keine gültigen Daten vorhanden sind
                    fpr = curve_data.get('fpr')
                    tpr = curve_data.get('tpr')
                    if fpr is None or tpr is None or len(fpr) != len(tpr):
                        continue

                    # Hole den AUC-Wert
                    auc_value = auc_scores.get(class_key, 0.0) if auc_scores else 0.0

                    # Bestimme den Klassennamen
                    if class_key == 'binary':
                        class_label = 'Binär'
                    elif class_key == 'weighted':
                        class_label = 'Gewichtet'
                    else:
                        class_label = class_key

                    # Zeichne die ROC-Kurve
                    ax.plot(fpr, tpr, lw=2, color=colors[color_idx % len(colors)],
                          label=f'{model_name} - {class_label} (AUC = {auc_value:.4f})')

                    color_idx += 1

            # Zeichne die ROC-Kurven für beide Modelle
            plot_model_roc_curves(model1_results['roc_curve_data'], model1_results['auc_scores'], 
                                model1_name, colors1)
            plot_model_roc_curves(model2_results['roc_curve_data'], model2_results['auc_scores'], 
                                model2_name, colors2)

            # Beschriftungen und Styling
            ax.set_xlabel('Falsch-Positiv-Rate', fontsize=11, fontweight='bold', labelpad=10)
            ax.set_ylabel('Richtig-Positiv-Rate', fontsize=11, fontweight='bold', labelpad=10)
            ax.set_title('ROC-Kurven-Vergleich', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='lower right', fontsize=9)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')

            # Rahmen
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.8)

            # Toolbar für Zoom/Pan
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            toolbar = NavigationToolbar(canvas, roc_compare_group)
            roc_compare_layout.addWidget(toolbar)

            # Layout-Anpassung wird durch constrained_layout=True automatisch durchgeführt

            # Zeichne den Canvas explizit
            canvas.draw()

            # Füge die ROC-Kurven-Vergleichsgruppe zum Layout hinzu
            layout.addWidget(roc_compare_group)

        # Füge einen Hinweis hinzu
        hint_label = QLabel("Hinweis: Die besseren Werte sind hervorgehoben.")
        hint_label.setStyleSheet("color: #666; font-style: italic;")
        hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint_label)

        # Setze eine Mindestgröße für die Tabelle
        table.setMinimumSize(400, 200)

        # Passe die Spalten- und Zeilengrößen an den Inhalt an
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        # Stelle sicher, dass die Widgets sichtbar sind
        self.compare_widget.show()
        layout.update()
        table.show()

        logger.info("Modellvergleich erfolgreich aktualisiert.")

    except Exception as e:
        QMessageBox.critical(self, "Fehler", f"Fehler beim Aktualisieren der Modellvergleichsansicht: {str(e)}")
        logger.error(f"Fehler in update_model_comparison: {str(e)}")

        # Zeige eine Fehlermeldung im Widget an
        if self.compare_widget.layout() is None:
            layout = QVBoxLayout(self.compare_widget)
        else:
            layout = self.compare_widget.layout()
            # Lösche alle Widgets im Layout
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        error_label = QLabel(f"Fehler beim Anzeigen des Modellvergleichs:\n{str(e)}")
        error_label.setStyleSheet("color: red; font-weight: bold;")
        error_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(error_label)

def update_confusion_matrix(self, conf_matrix, class_names=None, save_to_file=False, target_column=None):
    """
    Aktualisiert die Konfusionsmatrix-Visualisierung mit einem QTableWidget.

    Args:
        conf_matrix: Die Konfusionsmatrix als numpy-Array
        class_names: Liste mit den Namen der Klassen (optional)
        save_to_file: Wenn True, wird die Konfusionsmatrix als PNG gespeichert
        target_column: Name der Zielspalte für den Dateinamen (nur benötigt, wenn save_to_file=True)

    # FIX visualization blank issue 2025-04-23:
    # - Verwende das bestehende Layout statt es zu löschen und neu zu erstellen
    # - Passe die Spalten- und Zeilengrößen explizit an den Inhalt an
    # - Rufe explizit show() für die Tabelle auf
    """
    logger = setup_logging("gui_tabs")
    logger.info("Starte update_confusion_matrix...")

    try:
        # Prüfe, ob conf_matrix ein gültiges numpy-Array ist
        if not isinstance(conf_matrix, np.ndarray):
            logger.error("Konfusionsmatrix ist kein gültiges numpy-Array.")

            QMessageBox.warning(self, "Ungültige Konfusionsmatrix", 
                               "Die Konfusionsmatrix ist kein gültiges numpy-Array.\n\n"
                               "Die Konfusionsmatrix kann nicht angezeigt werden.")
            return

        # Prüfe, ob conf_matrix die richtige Form hat
        if len(conf_matrix.shape) != 2 or conf_matrix.shape[0] == 0 or conf_matrix.shape[1] == 0:
            logger.error(f"Konfusionsmatrix hat eine ungültige Form: {conf_matrix.shape}")

            QMessageBox.warning(self, "Ungültiges Format der Konfusionsmatrix", 
                               f"Die Konfusionsmatrix hat eine ungültige Form: {conf_matrix.shape}.\n\n"
                               "Die Konfusionsmatrix kann nicht angezeigt werden.")
            return

        # Lösche alle Widgets im bestehenden Layout
        layout = self.cm_widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        else:
            # Falls kein Layout existiert, erstelle ein neues
            layout = QVBoxLayout(self.cm_widget)

        # Hole den aktuellen Modellnamen für den Titel
        model_name = self.model_select_combo.currentText() if hasattr(self, 'model_select_combo') else "Modell"

        # Erstelle einen Titel
        title_label = QLabel(f"Konfusionsmatrix: {model_name}")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00796b;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Erstelle eine Tabelle für die Konfusionsmatrix
        table = QTableWidget()

        # Setze die Anzahl der Zeilen und Spalten
        num_classes = conf_matrix.shape[0]
        table.setRowCount(num_classes + 1)  # +1 für die Spaltenüberschriften
        table.setColumnCount(num_classes + 1)  # +1 für die Zeilenüberschriften

        # Setze die Spalten- und Zeilenüberschriften
        if class_names is not None and len(class_names) == num_classes:
            # Verwende die Klassennamen
            for i in range(num_classes):
                # Spaltenüberschriften (Vorhergesagte Klassen)
                header_item = QTableWidgetItem(class_names[i])
                header_item.setTextAlignment(Qt.AlignCenter)
                header_item.setBackground(QColor("#e0f2f1"))  # Helles Türkis
                table.setItem(0, i + 1, header_item)

                # Zeilenüberschriften (Tatsächliche Klassen)
                header_item = QTableWidgetItem(class_names[i])
                header_item.setTextAlignment(Qt.AlignCenter)
                header_item.setBackground(QColor("#e0f2f1"))  # Helles Türkis
                table.setItem(i + 1, 0, header_item)
        else:
            # Verwende numerische Indizes
            for i in range(num_classes):
                # Spaltenüberschriften (Vorhergesagte Klassen)
                header_item = QTableWidgetItem(f"Klasse {i}")
                header_item.setTextAlignment(Qt.AlignCenter)
                header_item.setBackground(QColor("#e0f2f1"))  # Helles Türkis
                table.setItem(0, i + 1, header_item)

                # Zeilenüberschriften (Tatsächliche Klassen)
                header_item = QTableWidgetItem(f"Klasse {i}")
                header_item.setTextAlignment(Qt.AlignCenter)
                header_item.setBackground(QColor("#e0f2f1"))  # Helles Türkis
                table.setItem(i + 1, 0, header_item)

        # Setze die Ecke oben links
        corner_item = QTableWidgetItem("Tats. \\ Vorh.")
        corner_item.setTextAlignment(Qt.AlignCenter)
        corner_item.setBackground(QColor("#b2dfdb"))  # Mittleres Türkis
        table.setItem(0, 0, corner_item)

        # Fülle die Tabelle mit den Werten aus der Konfusionsmatrix
        for i in range(num_classes):
            for j in range(num_classes):
                value = conf_matrix[i, j]

                # Erstelle das Tabellenelement
                value_item = QTableWidgetItem(str(value))
                value_item.setTextAlignment(Qt.AlignCenter)

                # Formatiere die Zellen basierend auf ihrem Wert
                if i == j:  # Diagonale (True Positives)
                    value_item.setBackground(QColor("#c8e6c9"))  # Helles Grün
                    value_item.setForeground(QColor("#2e7d32"))  # Dunkles Grün
                    value_item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                    # Füge Tooltip hinzu
                    if num_classes == 2 and i == 1:
                        value_item.setToolTip(f"True Positive (TP): {value}")
                    elif num_classes == 2 and i == 0:
                        value_item.setToolTip(f"True Negative (TN): {value}")
                    else:
                        value_item.setToolTip(f"Korrekte Vorhersage: {value}")
                else:
                    # Falsche Vorhersagen
                    intensity = min(1.0, value / (conf_matrix.max() / 2)) if conf_matrix.max() > 0 else 0
                    bg_color = QColor(255, int(255 * (1 - intensity * 0.5)), int(255 * (1 - intensity * 0.5)))
                    value_item.setBackground(bg_color)

                    # Füge Tooltip hinzu
                    if num_classes == 2:
                        if i == 1 and j == 0:
                            value_item.setToolTip(f"False Negative (FN): {value}")
                        elif i == 0 and j == 1:
                            value_item.setToolTip(f"False Positive (FP): {value}")
                    else:
                        value_item.setToolTip(f"Falsche Vorhersage: {value}")

                table.setItem(i + 1, j + 1, value_item)

        # Passe die Spaltenbreiten an
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # Passe die Zeilenhöhen an
        vertical_header = table.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.Stretch)

        # Verstecke die Standard-Header
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)

        # Füge die Tabelle zum Layout hinzu
        layout.addWidget(table)

        # Berechne die Genauigkeit
        diag_sum = np.sum(np.diag(conf_matrix))
        total = np.sum(conf_matrix)
        accuracy = diag_sum / total if total > 0 else 0

        # Füge eine Beschriftung mit der Genauigkeit hinzu
        accuracy_label = QLabel(f"Genauigkeit: {accuracy:.2%} ({diag_sum:,} von {total:,})")
        accuracy_label.setStyleSheet("font-size: 11pt; margin-top: 10px;")
        accuracy_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(accuracy_label)

        # Füge eine Legende hinzu für binäre Klassifikation
        if num_classes == 2:
            legend_widget = QWidget()
            legend_layout = QHBoxLayout(legend_widget)
            legend_layout.setContentsMargins(0, 10, 0, 0)

            # True Positive
            tp_label = QLabel("TP: True Positive")
            tp_label.setStyleSheet("background-color: #c8e6c9; padding: 5px; border-radius: 3px;")
            legend_layout.addWidget(tp_label)

            # True Negative
            tn_label = QLabel("TN: True Negative")
            tn_label.setStyleSheet("background-color: #c8e6c9; padding: 5px; border-radius: 3px;")
            legend_layout.addWidget(tn_label)

            # False Positive
            fp_label = QLabel("FP: False Positive")
            fp_label.setStyleSheet("background-color: #ffcdd2; padding: 5px; border-radius: 3px;")
            legend_layout.addWidget(fp_label)

            # False Negative
            fn_label = QLabel("FN: False Negative")
            fn_label.setStyleSheet("background-color: #ffcdd2; padding: 5px; border-radius: 3px;")
            legend_layout.addWidget(fn_label)

            layout.addWidget(legend_widget)

        # Speichere die Konfusionsmatrix als PNG, falls gewünscht
        if save_to_file and target_column:
            try:
                from datetime import datetime
                from suppervisedlearningproject.utils.config import get_plot_path

                # Generiere einen Zeitstempel für den Dateinamen
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Generiere den Pfad für die Datei
                plot_path = get_plot_path('confusion_matrix', target_column, timestamp)

                # Speichere die Tabelle als Bild
                pixmap = table.grab()
                pixmap.save(plot_path, "PNG")

                logger = setup_logging("gui_tabs")
                logger.info(f"Konfusionsmatrix gespeichert: {plot_path}")

                # Zeige eine Erfolgsmeldung an
                QMessageBox.information(self, "Konfusionsmatrix gespeichert", 
                                      f"Die Konfusionsmatrix wurde erfolgreich gespeichert:\n{plot_path}")
            except Exception as e:
                logger = setup_logging("gui_tabs")
                logger.error(f"Fehler beim Speichern der Konfusionsmatrix: {str(e)}")
                QMessageBox.warning(self, "Fehler beim Speichern", 
                                   f"Die Konfusionsmatrix konnte nicht gespeichert werden: {str(e)}")

        # Setze eine Mindestgröße für die Tabelle
        table.setMinimumSize(400, 300)

        # Passe die Spalten- und Zeilengrößen an den Inhalt an
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        # Stelle sicher, dass die Widgets sichtbar sind
        self.cm_widget.show()
        layout.update()
        table.show()

        logger.info("Konfusionsmatrix erfolgreich aktualisiert.")

    except Exception as e:
        # Fange alle Ausnahmen ab und zeige eine Fehlermeldung an
        QMessageBox.critical(self, "Fehler", f"Fehler beim Aktualisieren der Konfusionsmatrix: {str(e)}")
        logger.error(f"Fehler in update_confusion_matrix: {str(e)}")

        # Zeige eine Fehlermeldung im Widget an
        if self.cm_widget.layout() is None:
            layout = QVBoxLayout(self.cm_widget)
        else:
            layout = self.cm_widget.layout()
            # Lösche alle Widgets im Layout
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        error_label = QLabel(f"Fehler beim Anzeigen der Konfusionsmatrix:\n{str(e)}")
        error_label.setStyleSheet("color: red; font-weight: bold;")
        error_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(error_label)
