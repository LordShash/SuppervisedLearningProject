#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI-Modul für die Textklassifikationsanwendung.

Dieses Modul stellt eine benutzerfreundliche grafische Oberfläche bereit,
um verschiedene Modelle zu trainieren und die Ergebnisse zu visualisieren.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from ttkthemes import ThemedTk, ThemedStyle
from PIL import Image, ImageTk
import matplotlib
import atexit
matplotlib.use("TkAgg")

# Importiere die Funktionen aus den anderen Modulen
from data_loader import load_data, get_available_targets
from train_logreg import train_and_save_model as train_logreg
from train_nn import train_and_save_model as train_nn


class ToolTip:
    """
    Erstellt einen Tooltip für ein Widget.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # Erstelle ein Toplevel-Fenster
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Erstelle ein Label mit dem Tooltip-Text
        label = ttk.Label(self.tooltip_window, text=self.text, wraplength=250,
                          background="#ffffe0", relief="solid", borderwidth=1,
                          padding=(5, 5))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class TextClassificationGUI:
    """
    Hauptklasse für die grafische Benutzeroberfläche der Textklassifikationsanwendung.
    """

    def __init__(self, root):
        # Store reference to figures for cleanup
        self.figures = []
        """
        Initialisiert die GUI.

        Args:
            root: Das Hauptfenster der Anwendung
        """
        # Tooltip-Erklärungen für Parameter
        self.tooltips = {
            "max_features": "Maximale Anzahl der Features, die für die Vektorisierung verwendet werden.\nBeispiel: Eine Erhöhung von 1000 auf 2000 kann die Genauigkeit verbessern, erhöht aber auch die Rechenzeit.",
            "test_size": "Anteil der Daten, der für den Testdatensatz verwendet wird (0.0 bis 1.0).\nBeispiel: 0.2 bedeutet 20% der Daten werden für Tests verwendet, 80% für das Training.",
            "max_iter": "Maximale Anzahl der Iterationen für die logistische Regression.\nBeispiel: Eine Erhöhung von 100 auf 1000 kann die Konvergenz verbessern, erhöht aber die Trainingszeit.",
            "c_reg": "Regularisierungsparameter C für die logistische Regression.\nBeispiel: Ein kleinerer Wert (z.B. 0.1) führt zu stärkerer Regularisierung und kann Overfitting reduzieren.",
            "solver": "Algorithmus für die Optimierung der logistischen Regression.\nBeispiel: 'lbfgs' ist gut für kleine Datensätze, 'saga' für große Datensätze mit vielen Features.",
            "epochs": "Anzahl der Trainingszyklen für das neuronale Netz.\nBeispiel: Eine Erhöhung von 10 auf 30 kann die Genauigkeit verbessern, erhöht aber die Trainingszeit.",
            "patience": "Anzahl der Epochen ohne Verbesserung, bevor das Training frühzeitig beendet wird.\nBeispiel: Ein Wert von 5 bedeutet, dass das Training stoppt, wenn sich die Validierungsgenauigkeit 5 Epochen lang nicht verbessert."
        }
        self.root = root
        self.root.title("Textklassifikation - Benutzeroberfläche")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # Setze Hintergrundfarbe für das Hauptfenster
        bg_color = root.cget('background')
        self.root.configure(background=bg_color)

        # Erstelle ein Hauptframe mit Padding
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Erstelle ein Notebook (Tab-Container) mit modernem Aussehen
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Erstelle die Tabs
        self.training_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.about_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.results_tab, text="Ergebnisse")
        self.notebook.add(self.about_tab, text="Über")

        # Initialisiere die Tabs
        self._init_training_tab()
        self._init_results_tab()
        self._init_about_tab()

        # Speichere die Trainingsergebnisse
        self.training_results = {}

        # Lade die verfügbaren Zielvariablen
        try:
            self.targets = get_available_targets()
            for target in self.targets.keys():
                self.target_var.set(list(self.targets.keys())[0])  # Setze die erste Zielvariable als Standard
                self.target_combobox['values'] = list(self.targets.keys())
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden der Zielvariablen: {str(e)}")

    def _init_training_tab(self):
        """
        Initialisiert den Training-Tab mit Formularelementen.
        """
        # Container für den gesamten Inhalt mit Padding
        content_frame = ttk.Frame(self.training_tab, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Erstelle ein Frame für die Eingabefelder
        input_frame = ttk.LabelFrame(content_frame, text="Trainingsparameter", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Erstelle ein Grid für die Eingabefelder
        row = 0

        # Zielvariable
        ttk.Label(input_frame, text="Zielvariable:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_var = tk.StringVar()
        self.target_combobox = ttk.Combobox(input_frame, textvariable=self.target_var, state="readonly", width=30)
        self.target_combobox.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1

        # Modelltyp
        ttk.Label(input_frame, text="Modelltyp:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="logreg")
        model_frame = ttk.Frame(input_frame)
        model_frame.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(model_frame, text="Logistische Regression", variable=self.model_var, value="logreg", command=self._toggle_model_options).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Neuronales Netz", variable=self.model_var, value="nn", command=self._toggle_model_options).pack(side=tk.LEFT, padx=5)
        row += 1

        # Gemeinsame Parameter
        # Max Features
        ttk.Label(input_frame, text="Max Features:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_features_var = tk.IntVar(value=1000)
        max_features_entry = ttk.Entry(input_frame, textvariable=self.max_features_var, width=10)
        max_features_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        max_features_info = ttk.Label(input_frame, text="ℹ️", cursor="hand2")
        max_features_info.grid(row=row, column=2, sticky=tk.W)
        ToolTip(max_features_info, self.tooltips["max_features"])
        row += 1

        # Test Size
        ttk.Label(input_frame, text="Test Size:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_entry = ttk.Entry(input_frame, textvariable=self.test_size_var, width=10)
        test_size_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        test_size_info = ttk.Label(input_frame, text="ℹ️", cursor="hand2")
        test_size_info.grid(row=row, column=2, sticky=tk.W)
        ToolTip(test_size_info, self.tooltips["test_size"])
        row += 1

        # Parameter für logistische Regression
        self.logreg_frame = ttk.LabelFrame(input_frame, text="Parameter für logistische Regression", padding="10")
        self.logreg_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        logreg_row = 0

        # Max Iterations
        ttk.Label(self.logreg_frame, text="Max Iterations:").grid(row=logreg_row, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_iter_var = tk.IntVar(value=1000)
        max_iter_entry = ttk.Entry(self.logreg_frame, textvariable=self.max_iter_var, width=10)
        max_iter_entry.grid(row=logreg_row, column=1, sticky=tk.W, padx=5, pady=5)
        max_iter_info = ttk.Label(self.logreg_frame, text="ℹ️", cursor="hand2")
        max_iter_info.grid(row=logreg_row, column=2, sticky=tk.W)
        ToolTip(max_iter_info, self.tooltips["max_iter"])
        logreg_row += 1

        # C (Regularisierung)
        ttk.Label(self.logreg_frame, text="C (Regularisierung):").grid(row=logreg_row, column=0, sticky=tk.W, padx=5, pady=5)
        self.c_var = tk.DoubleVar(value=1.0)
        c_entry = ttk.Entry(self.logreg_frame, textvariable=self.c_var, width=10)
        c_entry.grid(row=logreg_row, column=1, sticky=tk.W, padx=5, pady=5)
        c_info = ttk.Label(self.logreg_frame, text="ℹ️", cursor="hand2")
        c_info.grid(row=logreg_row, column=2, sticky=tk.W)
        ToolTip(c_info, self.tooltips["c_reg"])
        logreg_row += 1

        # Solver
        ttk.Label(self.logreg_frame, text="Solver:").grid(row=logreg_row, column=0, sticky=tk.W, padx=5, pady=5)
        self.solver_var = tk.StringVar(value="lbfgs")
        solver_combobox = ttk.Combobox(self.logreg_frame, textvariable=self.solver_var, state="readonly", width=10)
        solver_combobox['values'] = ('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga')
        solver_combobox.grid(row=logreg_row, column=1, sticky=tk.W, padx=5, pady=5)
        solver_info = ttk.Label(self.logreg_frame, text="ℹ️", cursor="hand2")
        solver_info.grid(row=logreg_row, column=2, sticky=tk.W)
        ToolTip(solver_info, self.tooltips["solver"])
        row += 1

        # Parameter für neuronales Netz
        self.nn_frame = ttk.LabelFrame(input_frame, text="Parameter für neuronales Netz", padding="10")
        self.nn_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        self.nn_frame.grid_remove()  # Verstecke das Frame initial
        nn_row = 0

        # Epochs
        ttk.Label(self.nn_frame, text="Epochs:").grid(row=nn_row, column=0, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=30)
        epochs_entry = ttk.Entry(self.nn_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=nn_row, column=1, sticky=tk.W, padx=5, pady=5)
        epochs_info = ttk.Label(self.nn_frame, text="ℹ️", cursor="hand2")
        epochs_info.grid(row=nn_row, column=2, sticky=tk.W)
        ToolTip(epochs_info, self.tooltips["epochs"])
        nn_row += 1

        # Patience
        ttk.Label(self.nn_frame, text="Patience:").grid(row=nn_row, column=0, sticky=tk.W, padx=5, pady=5)
        self.patience_var = tk.IntVar(value=5)
        patience_entry = ttk.Entry(self.nn_frame, textvariable=self.patience_var, width=10)
        patience_entry.grid(row=nn_row, column=1, sticky=tk.W, padx=5, pady=5)
        patience_info = ttk.Label(self.nn_frame, text="ℹ️", cursor="hand2")
        patience_info.grid(row=nn_row, column=2, sticky=tk.W)
        ToolTip(patience_info, self.tooltips["patience"])
        row += 1

        # Erstelle ein Frame für die Buttons
        button_frame = ttk.Frame(content_frame, padding="10")
        button_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Trainieren-Button
        self.train_button = ttk.Button(button_frame, text="Modell trainieren", command=self._train_model, style='Action.TButton')
        self.train_button.pack(side=tk.RIGHT, padx=5)

        # Erstelle ein Frame für die Ausgabe
        output_frame = ttk.LabelFrame(content_frame, text="Ausgabe", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Textfeld für die Ausgabe
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=80, height=15)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)

    def _toggle_model_options(self):
        """
        Zeigt oder versteckt die modellspezifischen Parameter je nach ausgewähltem Modelltyp.
        """
        if self.model_var.get() == "logreg":
            self.logreg_frame.grid()
            self.nn_frame.grid_remove()
        else:
            self.logreg_frame.grid_remove()
            self.nn_frame.grid()

    def _train_model(self):
        """
        Trainiert das ausgewählte Modell mit den angegebenen Parametern.
        """
        # Deaktiviere den Trainieren-Button während des Trainings
        self.train_button.config(state=tk.DISABLED)

        # Leere das Ausgabefeld
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

        # Starte das Training in einem separaten Thread
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        """
        Führt das Training im Hintergrund aus.
        """
        try:
            # Hole die Parameter aus den Eingabefeldern
            target_column = self.target_var.get()
            model_type = self.model_var.get()
            max_features = self.max_features_var.get()
            test_size = self.test_size_var.get()

            # Modellspezifische Parameter
            if model_type == "logreg":
                max_iter = self.max_iter_var.get()
                c_reg = self.c_var.get()
                solver = self.solver_var.get()

                # Ausgabe der Parameter
                self._append_to_output(f"Training eines logistischen Regressionsmodells mit folgenden Parametern:\n")
                self._append_to_output(f"Zielvariable: {target_column}")
                self._append_to_output(f"Max Features: {max_features}")
                self._append_to_output(f"Test Size: {test_size}")
                self._append_to_output(f"Max Iterations: {max_iter}")
                self._append_to_output(f"C (Regularisierung): {c_reg}")
                self._append_to_output(f"Solver: {solver}\n")

                # Training des Modells
                self._append_to_output("Starte Training...\n")

                # Umleitung der Standardausgabe
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    results = train_logreg(
                        target_column=target_column,
                        max_features=max_features,
                        test_size=test_size,
                        max_iter=max_iter,
                        C=c_reg,
                        solver=solver
                    )
                    accuracy = results.accuracy
                    precision = results.precision
                    recall = results.recall
                    f1 = results.f1
                    report = results.report
                    conf_matrix = results.conf_matrix

                # Ausgabe der umgeleiteten Standardausgabe
                self._append_to_output(f.getvalue())

                # Speichern der Ergebnisse
                model_key = f"logreg_{target_column}"
                self.training_results[model_key] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'report': report,
                    'conf_matrix': conf_matrix,
                    'params': {
                        'target_column': target_column,
                        'max_features': max_features,
                        'test_size': test_size,
                        'max_iter': max_iter,
                        'C': c_reg,
                        'solver': solver
                    }
                }

            else:  # Neuronales Netz
                epochs = self.epochs_var.get()
                patience = self.patience_var.get()

                # Ausgabe der Parameter
                self._append_to_output(f"Training eines neuronalen Netzes mit folgenden Parametern:\n")
                self._append_to_output(f"Zielvariable: {target_column}")
                self._append_to_output(f"Max Features: {max_features}")
                self._append_to_output(f"Test Size: {test_size}")
                self._append_to_output(f"Epochs: {epochs}")
                self._append_to_output(f"Patience: {patience}\n")

                # Training des Modells
                self._append_to_output("Starte Training...\n")

                # Umleitung der Standardausgabe
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    results = train_nn(
                        target_column=target_column,
                        max_features=max_features,
                        test_size=test_size,
                        epochs=epochs,
                        patience=patience
                    )
                    accuracy = results.accuracy
                    precision = results.precision
                    recall = results.recall
                    f1 = results.f1
                    report = results.report
                    conf_matrix = results.conf_matrix

                # Ausgabe der umgeleiteten Standardausgabe
                self._append_to_output(f.getvalue())

                # Speichern der Ergebnisse
                model_key = f"nn_{target_column}"
                self.training_results[model_key] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'report': report,
                    'conf_matrix': conf_matrix,
                    'params': {
                        'target_column': target_column,
                        'max_features': max_features,
                        'test_size': test_size,
                        'epochs': epochs,
                        'patience': patience
                    }
                }

            # Ausgabe der Ergebnisse
            self._append_to_output("\nTraining abgeschlossen!")
            self._append_to_output(f"Accuracy: {accuracy:.4f}")
            self._append_to_output(f"Precision: {precision:.4f}")
            self._append_to_output(f"Recall: {recall:.4f}")
            self._append_to_output(f"F1 Score: {f1:.4f}")

            # Aktualisiere die Ergebnisse im Ergebnisse-Tab
            self._update_results_tab()

            # Wechsle zum Ergebnisse-Tab
            self.root.after(0, lambda: self.notebook.select(1))

        except Exception as e:
            # Fehlerbehandlung
            self._append_to_output(f"\nFehler beim Training: {str(e)}")
            import traceback
            self._append_to_output(traceback.format_exc())
        finally:
            # Aktiviere den Trainieren-Button wieder
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))

    def _append_to_output(self, text):
        """
        Fügt Text zum Ausgabefeld hinzu.
        """
        self.root.after(0, lambda: self._append_to_output_direct(text))

    def _append_to_output_direct(self, text):
        """
        Fügt Text direkt zum Ausgabefeld hinzu (wird vom Hauptthread aufgerufen).
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

    def _init_results_tab(self):
        """
        Initialisiert den Ergebnisse-Tab.
        """
        # Container für den gesamten Inhalt mit Padding
        content_frame = ttk.Frame(self.results_tab, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Erstelle ein Frame für die Modellauswahl
        model_frame = ttk.LabelFrame(content_frame, text="Modellauswahl", padding="10")
        model_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Combobox für die Modellauswahl
        ttk.Label(model_frame, text="Modell:").pack(side=tk.LEFT, padx=5)
        self.model_select_var = tk.StringVar()
        self.model_select_combobox = ttk.Combobox(model_frame, textvariable=self.model_select_var, state="readonly", width=30)
        self.model_select_combobox.pack(side=tk.LEFT, padx=5)
        self.model_select_combobox.bind("<<ComboboxSelected>>", self._on_model_selected)

        # Erstelle ein Frame für die Metriken
        metrics_frame = ttk.LabelFrame(content_frame, text="Metriken", padding="10")
        metrics_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Grid für die Metriken
        self.accuracy_var = tk.StringVar(value="N/A")
        self.precision_var = tk.StringVar(value="N/A")
        self.recall_var = tk.StringVar(value="N/A")
        self.f1_var = tk.StringVar(value="N/A")

        ttk.Label(metrics_frame, text="Accuracy:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(metrics_frame, textvariable=self.accuracy_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(metrics_frame, text="Precision:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(metrics_frame, textvariable=self.precision_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(metrics_frame, text="Recall:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(metrics_frame, textvariable=self.recall_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(metrics_frame, text="F1 Score:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(metrics_frame, textvariable=self.f1_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Erstelle ein Frame für den Klassifikationsbericht
        report_frame = ttk.LabelFrame(content_frame, text="Klassifikationsbericht", padding="10")
        report_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Textfeld für den Klassifikationsbericht
        self.report_text = scrolledtext.ScrolledText(report_frame, wrap=tk.WORD, width=80, height=10)
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.report_text.config(state=tk.DISABLED)

        # Erstelle ein Frame für die Konfusionsmatrix
        self.cm_frame = ttk.LabelFrame(content_frame, text="Konfusionsmatrix", padding="10")
        self.cm_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _update_results_tab(self):
        """
        Aktualisiert die Modellauswahl im Ergebnisse-Tab.
        """
        # Aktualisiere die Modellauswahl
        model_names = list(self.training_results.keys())
        self.model_select_combobox['values'] = model_names

        # Wähle das neueste Modell aus, falls vorhanden
        if model_names:
            self.model_select_combobox.set(model_names[-1])
            self._on_model_selected()

    def _on_model_selected(self, event=None):
        """
        Wird aufgerufen, wenn ein Modell im Ergebnisse-Tab ausgewählt wird.
        """
        model_key = self.model_select_var.get()
        if not model_key or model_key not in self.training_results:
            return

        # Hole die Ergebnisse für das ausgewählte Modell
        results = self.training_results[model_key]

        # Aktualisiere die Metriken
        self.accuracy_var.set(f"{results['accuracy']:.4f}")
        self.precision_var.set(f"{results['precision']:.4f}")
        self.recall_var.set(f"{results['recall']:.4f}")
        self.f1_var.set(f"{results['f1']:.4f}")

        # Aktualisiere den Klassifikationsbericht
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, results['report'])
        self.report_text.config(state=tk.DISABLED)

        # Zeige die Konfusionsmatrix an
        for widget in self.cm_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))
        # Store reference to the figure for cleanup
        self.figures.append(fig)

        conf_matrix = results['conf_matrix']
        cax = ax.matshow(conf_matrix, cmap='Blues')
        fig.colorbar(cax)

        # Beschriftungen für die Konfusionsmatrix
        ax.set_xlabel('Vorhergesagte Klasse')
        ax.set_ylabel('Tatsächliche Klasse')
        ax.set_title('Konfusionsmatrix')

        # Füge die Werte in die Zellen ein
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center')

        # Erstelle ein Canvas für die Matplotlib-Figur
        canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _init_about_tab(self):
        """
        Initialisiert den Über-Tab mit Informationen zur Anwendung.
        """
        # Container für den gesamten Inhalt mit Padding
        content_frame = ttk.Frame(self.about_tab, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Titel
        title_label = ttk.Label(content_frame, text="Textklassifikationsanwendung", font=("Segoe UI", 16, "bold"))
        title_label.pack(pady=10)

        # Beschreibung
        description_frame = ttk.LabelFrame(content_frame, text="Beschreibung", padding="10")
        description_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        description_text = """
        Diese Anwendung ermöglicht das Training und die Evaluation von Textklassifikationsmodellen.

        Unterstützte Modelle:
        - Logistische Regression
        - Neuronales Netz

        Die Anwendung bietet eine benutzerfreundliche Oberfläche zum Einstellen der Trainingsparameter
        und zur Visualisierung der Ergebnisse.
        """

        description_label = ttk.Label(description_frame, text=description_text, wraplength=600, justify="left")
        description_label.pack(padx=5, pady=5)

        # Anleitung
        usage_frame = ttk.LabelFrame(content_frame, text="Anleitung", padding="10")
        usage_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        usage_text = """
        1. Wählen Sie im Tab "Training" die gewünschte Zielvariable und den Modelltyp aus.
        2. Passen Sie die Parameter nach Bedarf an.
        3. Klicken Sie auf "Modell trainieren", um das Training zu starten.
        4. Die Ergebnisse werden im Tab "Ergebnisse" angezeigt.
        5. Sie können zwischen verschiedenen trainierten Modellen wechseln, um die Ergebnisse zu vergleichen.
        """

        usage_label = ttk.Label(usage_frame, text=usage_text, wraplength=600, justify="left")
        usage_label.pack(padx=5, pady=5)

        # Version und Copyright
        footer_frame = ttk.Frame(content_frame)
        footer_frame.pack(fill=tk.X, expand=False, padx=5, pady=10)

        version_label = ttk.Label(footer_frame, text="Version 1.0")
        version_label.pack(side=tk.LEFT)

        copyright_label = ttk.Label(footer_frame, text="© 2023 Textklassifikationsprojekt")
        copyright_label.pack(side=tk.RIGHT)

    def cleanup_resources(self):
        """
        Bereinigt Ressourcen, die von der Anwendung verwendet werden.
        """
        # Close all matplotlib figures
        for fig in self.figures:
            plt.close(fig)
        self.figures = []
        plt.close('all')


# Register a global cleanup function to ensure matplotlib figures are closed
# This is a fallback in case the normal cleanup doesn't happen
def global_cleanup():
    plt.close('all')

# Register the global cleanup function to be called at exit
atexit.register(global_cleanup)

def main():
    """
    Hauptfunktion zum Starten der GUI.
    """
    try:
        # Verwende ThemedTk statt Tk für ein modernes Erscheinungbild
        root = ThemedTk(theme="arc")  # Moderne Themes: 'arc', 'equilux', 'breeze', etc.

        # Aktiviere DPI-Skalierung für bessere Darstellung auf hochauflösenden Displays
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass  # Ignoriere Fehler, falls nicht auf Windows oder DPI-Skalierung nicht unterstützt wird

        # Konfiguriere den Stil
        style = ThemedStyle(root)

        # Bestimme den Skalierungsfaktor basierend auf der Bildschirmauflösung
        scale_factor = root.winfo_fpixels('1i') / 96.0  # Standard-DPI ist 96
        font_size = int(10 * scale_factor)
        font_size = max(9, min(font_size, 14))  # Begrenze Schriftgröße zwischen 9 und 14

        # Grundlegende Schriftarten und Stile konfigurieren mit Skalierung
        style.configure('TButton', font=('Segoe UI', font_size))
        style.configure('TLabel', font=('Segoe UI', font_size))
        style.configure('TLabelframe.Label', font=('Segoe UI', font_size, 'bold'))
        style.configure('TNotebook.Tab', font=('Segoe UI', font_size))
        style.configure('TEntry', font=('Segoe UI', font_size))
        style.configure('TCombobox', font=('Segoe UI', font_size))

        # Spezielle Stile für Buttons
        style.configure('Action.TButton', font=('Segoe UI', font_size, 'bold'))
        style.map('Action.TButton', 
                  background=[('active', '#3498db'), ('pressed', '#2980b9')],
                  foreground=[('active', 'white'), ('pressed', 'white')])

        # Setze den Fenstertitel und Icon
        root.title("Textklassifikation - Moderne Benutzeroberfläche")

        # Erstelle die Anwendung
        app = TextClassificationGUI(root)

        # Register cleanup function for window close event
        def on_closing():
            app.cleanup_resources()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        # Zentriere das Fenster auf dem Bildschirm
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        # Starte die Hauptschleife
        root.mainloop()

    except Exception as e:
        # Fallback für den Fall, dass ttkthemes nicht verfügbar ist
        print(f"Fehler beim Starten der modernen UI: {str(e)}")
        print("Starte mit Standard-UI...")

        root = tk.Tk()
        root.title("Textklassifikation - Benutzeroberfläche")
        app = TextClassificationGUI(root)

        # Register cleanup function for window close event in fallback mode
        def on_closing():
            app.cleanup_resources()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        root.mainloop()


if __name__ == "__main__":
    main()
