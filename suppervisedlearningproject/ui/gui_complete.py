"""
Vollständige Implementierung der modernen GUI für die Textklassifikationsanwendung.

Dieses Modul integriert die Basisklassen aus gui.py mit den Tab-Implementierungen
aus gui_tabs.py zu einer vollständigen GUI-Anwendung.
"""

from suppervisedlearningproject.ui.gui import ModernTextClassificationGUI
from suppervisedlearningproject.ui.gui_tabs import (
    init_training_tab, init_results_tab, init_about_tab,
    create_label_with_tooltip, toggle_model_options, train_model,
    append_to_output, on_training_finished, on_training_error,
    update_results_tab, on_model_selected, update_confusion_matrix,
    compare_models, update_compare_models_combo, update_roc_curve, update_model_comparison
)


# Erweitere die ModernTextClassificationGUI-Klasse um die Tab-Implementierungen
ModernTextClassificationGUI.init_training_tab = init_training_tab
ModernTextClassificationGUI.init_results_tab = init_results_tab
ModernTextClassificationGUI.init_about_tab = init_about_tab
ModernTextClassificationGUI.create_label_with_tooltip = create_label_with_tooltip
ModernTextClassificationGUI.toggle_model_options = toggle_model_options
ModernTextClassificationGUI.train_model = train_model
ModernTextClassificationGUI.append_to_output = append_to_output
ModernTextClassificationGUI.on_training_finished = on_training_finished
ModernTextClassificationGUI.on_training_error = on_training_error
ModernTextClassificationGUI.update_results_tab = update_results_tab
ModernTextClassificationGUI.on_model_selected = on_model_selected
ModernTextClassificationGUI.update_confusion_matrix = update_confusion_matrix
ModernTextClassificationGUI.compare_models = compare_models
ModernTextClassificationGUI.update_compare_models_combo = update_compare_models_combo
ModernTextClassificationGUI.update_roc_curve = update_roc_curve
ModernTextClassificationGUI.update_model_comparison = update_model_comparison
