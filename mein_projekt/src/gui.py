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

        # Erstelle einen Frame für die Formularelemente mit verbessertem Styling
        form_frame = ttk.LabelFrame(content_frame, text="Modellkonfiguration", padding="10")
        form_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Erstelle ein Grid-Layout für die Formularelemente
        form_frame.columnconfigure(0, weight=1)
        form_frame.columnconfigure(1, weight=3)

        # Modelltyp mit verbesserten Abständen
        ttk.Label(form_frame, text="Modelltyp:", font=('Segoe UI', 10)).grid(row=0, column=0, sticky=tk.W, padx=8, pady=8)
        self.model_var = tk.StringVar(value="logreg")
        model_frame = ttk.Frame(form_frame)
        model_frame.grid(row=0, column=1, sticky=tk.W, padx=8, pady=8)

        # Verbesserte Radiobuttons mit mehr Abstand
        ttk.Radiobutton(model_frame, text="Logistische Regression", variable=self.model_var, value="logreg").pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(model_frame, text="Neuronales Netz", variable=self.model_var, value="nn").pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(model_frame, text="Beide", variable=self.model_var, value="all").pack(side=tk.LEFT, padx=8)

        # Zielvariable mit verbesserten Abständen
        ttk.Label(form_frame, text="Zielvariable:", font=('Segoe UI', 10)).grid(row=1, column=0, sticky=tk.W, padx=8, pady=8)
        self.target_var = tk.StringVar()
        self.target_combobox = ttk.Combobox(form_frame, textvariable=self.target_var, state="readonly", width=30)
        self.target_combobox.grid(row=1, column=1, sticky=tk.W+tk.E, padx=8, pady=8)

        # Hyperparameter-Frame mit verbessertem Styling
        hyper_frame = ttk.LabelFrame(content_frame, text="Hyperparameter", padding="10")
        hyper_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=10)

        # Erstelle ein Grid-Layout für die Hyperparameter
        hyper_frame.columnconfigure(0, weight=1)
        hyper_frame.columnconfigure(1, weight=1)
        hyper_frame.columnconfigure(2, weight=1)
        hyper_frame.columnconfigure(3, weight=1)

        # Gemeinsame Hyperparameter mit verbesserten Abständen und Styling
        # Erweitere das Grid für die Hilfe-Buttons
        hyper_frame.columnconfigure(0, weight=1)  # Label
        hyper_frame.columnconfigure(1, weight=0)  # Hilfe-Button
        hyper_frame.columnconfigure(2, weight=1)  # Eingabefeld
        hyper_frame.columnconfigure(3, weight=1)  # Label
        hyper_frame.columnconfigure(4, weight=0)  # Hilfe-Button
        hyper_frame.columnconfigure(5, weight=1)  # Eingabefeld

        # Max Features
        ttk.Label(hyper_frame, text="Max Features:", font=('Segoe UI', 10)).grid(row=0, column=0, sticky=tk.W, padx=8, pady=8)
        self.create_help_button(hyper_frame, 0, 1, "max_features")
        self.max_features_var = tk.StringVar(value="1000")
        ttk.Entry(hyper_frame, textvariable=self.max_features_var, width=12).grid(row=0, column=2, sticky=tk.W, padx=8, pady=8)

        # Test Size
        ttk.Label(hyper_frame, text="Test Size:", font=('Segoe UI', 10)).grid(row=0, column=3, sticky=tk.W, padx=8, pady=8)
        self.create_help_button(hyper_frame, 0, 4, "test_size")
        self.test_size_var = tk.StringVar(value="0.2")
        ttk.Entry(hyper_frame, textvariable=self.test_size_var, width=12).grid(row=0, column=5, sticky=tk.W, padx=8, pady=8)

        # LogReg Hyperparameter mit verbesserten Abständen
        # Max Iterations
        ttk.Label(hyper_frame, text="Max Iterations:", font=('Segoe UI', 10)).grid(row=1, column=0, sticky=tk.W, padx=8, pady=8)
        self.create_help_button(hyper_frame, 1, 1, "max_iter")
        self.max_iter_var = tk.StringVar(value="1000")
        ttk.Entry(hyper_frame, textvariable=self.max_iter_var, width=12).grid(row=1, column=2, sticky=tk.W, padx=8, pady=8)

        # C (Regularisierung)
        ttk.Label(hyper_frame, text="C (Regularisierung):", font=('Segoe UI', 10)).grid(row=1, column=3, sticky=tk.W, padx=8, pady=8)
        self.create_help_button(hyper_frame, 1, 4, "c_reg")
        self.c_var = tk.StringVar(value="1.0")
        ttk.Entry(hyper_frame, textvariable=self.c_var, width=12).grid(row=1, column=5, sticky=tk.W, padx=8, pady=8)

        # Solver
        ttk.Label(hyper_frame, text="Solver:", font=('Segoe UI', 10)).grid(row=2, column=0, sticky=tk.W, padx=8, pady=8)
        self.create_help_button(hyper_frame, 2, 1, "solver")
        self.solver_var = tk.StringVar(value="lbfgs")
        solver_combobox = ttk.Combobox(hyper_frame, textvariable=self.solver_var, state="readonly", width=12)
        solver_combobox['values'] = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        solver_combobox.grid(row=2, column=2, sticky=tk.W, padx=8, pady=8)

        # NN Hyperparameter mit verbesserten Abständen
        # Epochen
        ttk.Label(hyper_frame, text="Epochen:", font=('Segoe UI', 10)).grid(row=2, column=3, sticky=tk.W, padx=8, pady=8)
        self.create_help_button(hyper_frame, 2, 4, "epochs")
        self.epochs_var = tk.StringVar(value="30")
        ttk.Entry(hyper_frame, textvariable=self.epochs_var, width=12).grid(row=2, column=5, sticky=tk.W, padx=8, pady=8)

        # Patience
        ttk.Label(hyper_frame, text="Patience:", font=('Segoe UI', 10)).grid(row=3, column=0, sticky=tk.W, padx=8, pady=8)
        self.create_help_button(hyper_frame, 3, 1, "patience")
        self.patience_var = tk.StringVar(value="5")
        ttk.Entry(hyper_frame, textvariable=self.patience_var, width=12).grid(row=3, column=2, sticky=tk.W, padx=8, pady=8)

        # Ausgabebereich mit verbessertem Styling
        output_frame = ttk.LabelFrame(content_frame, text="Ausgabe", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

        # Verbesserte Textausgabe mit angepasster Schriftart
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=80, height=10, font=('Consolas', 9))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)

        # Buttons mit verbessertem Styling
        button_frame = ttk.Frame(content_frame, padding="5")
        button_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Stil für die Buttons definieren
        style = ttk.Style()
        style.configure('Action.TButton', font=('Segoe UI', 10, 'bold'))

        # Verbesserte Buttons mit Icons oder Styling
        self.clear_button = ttk.Button(button_frame, text="Ausgabe löschen", command=self.clear_output, style='Action.TButton')
        self.clear_button.pack(side=tk.RIGHT, padx=8, pady=5)

        self.train_button = ttk.Button(button_frame, text="Modell trainieren", command=self.train_model, style='Action.TButton')
        self.train_button.pack(side=tk.RIGHT, padx=8, pady=5)

    def _init_results_tab(self):
        """
        Initialisiert den Ergebnisse-Tab für die Visualisierung.
        """
        # Container für den gesamten Inhalt mit Padding
        content_frame = ttk.Frame(self.results_tab, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Erstelle einen Frame für die Ergebnisauswahl mit verbessertem Styling
        selection_frame = ttk.LabelFrame(content_frame, text="Ergebnisauswahl", padding="10")
        selection_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Verbesserte Ergebnisauswahl mit mehr Abstand und besserer Schriftart
        ttk.Label(selection_frame, text="Trainiertes Modell:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=8, pady=8)
        self.result_var = tk.StringVar()
        self.result_combobox = ttk.Combobox(selection_frame, textvariable=self.result_var, state="readonly", width=50)
        self.result_combobox.pack(side=tk.LEFT, padx=8, pady=8, fill=tk.X, expand=True)
        self.result_combobox.bind("<<ComboboxSelected>>", self.display_selected_result)

        # Erstelle einen Frame für die Ergebnisanzeige
        self.results_display_frame = ttk.Frame(content_frame)
        self.results_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Erstelle einen Frame für die Metriken mit verbessertem Styling
        self.metrics_frame = ttk.LabelFrame(self.results_display_frame, text="Leistungsmetriken", padding="10")
        self.metrics_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Erstelle ein Grid für die Metriken
        for i in range(4):
            self.metrics_frame.columnconfigure(i, weight=1)

        # Verbesserte Metrik-Labels mit besserer Schriftart und Styling
        self.accuracy_label = ttk.Label(self.metrics_frame, text="Accuracy: -", font=('Segoe UI', 10))
        self.accuracy_label.grid(row=0, column=0, sticky=tk.W, padx=8, pady=8)

        self.precision_label = ttk.Label(self.metrics_frame, text="Precision: -", font=('Segoe UI', 10))
        self.precision_label.grid(row=0, column=1, sticky=tk.W, padx=8, pady=8)

        self.recall_label = ttk.Label(self.metrics_frame, text="Recall: -", font=('Segoe UI', 10))
        self.recall_label.grid(row=0, column=2, sticky=tk.W, padx=8, pady=8)

        self.f1_label = ttk.Label(self.metrics_frame, text="F1 Score: -", font=('Segoe UI', 10))
        self.f1_label.grid(row=0, column=3, sticky=tk.W, padx=8, pady=8)

        # Erstelle einen Frame für die Konfusionsmatrix und den Bericht mit verbessertem Layout
        self.viz_frame = ttk.Frame(self.results_display_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

        # Erstelle zwei Spalten für Konfusionsmatrix und Bericht
        self.viz_frame.columnconfigure(0, weight=1)
        self.viz_frame.columnconfigure(1, weight=1)
        self.viz_frame.rowconfigure(0, weight=1)

        # Frame für die Konfusionsmatrix mit verbessertem Styling
        self.cm_frame = ttk.LabelFrame(self.viz_frame, text="Konfusionsmatrix", padding="10")
        self.cm_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)

        # Frame für den Klassifikationsbericht mit verbessertem Styling
        self.report_frame = ttk.LabelFrame(self.viz_frame, text="Klassifikationsbericht", padding="10")
        self.report_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)

        # Verbesserte Textanzeige mit angepasster Schriftart
        self.report_text = scrolledtext.ScrolledText(self.report_frame, wrap=tk.WORD, width=40, height=15, font=('Consolas', 9))
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.report_text.config(state=tk.DISABLED)

    def _init_about_tab(self):
        """
        Initialisiert den Über-Tab mit Informationen zur Anwendung.
        """
        # Container für den gesamten Inhalt mit Padding
        content_frame = ttk.Frame(self.about_tab, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Titel mit verbessertem Styling
        title_frame = ttk.Frame(content_frame)
        title_frame.pack(fill=tk.X, expand=False, pady=15)

        title_label = ttk.Label(
            title_frame, 
            text="Textklassifikation mit maschinellem Lernen", 
            font=("Segoe UI", 18, "bold"),
            foreground="#2c3e50"  # Dunkelblau für den Titel
        )
        title_label.pack(pady=5)

        # Trennlinie unter dem Titel
        separator = ttk.Separator(content_frame, orient="horizontal")
        separator.pack(fill=tk.X, padx=20, pady=10)

        # Beschreibung in einem LabelFrame für besseres Styling
        desc_frame = ttk.LabelFrame(content_frame, text="Über die Anwendung", padding="15")
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        description = """
        Diese Anwendung ermöglicht das Training und die Evaluation von Textklassifikationsmodellen.

        Unterstützte Modelle:
        - Logistische Regression
        - Neuronales Netz

        Die Anwendung verwendet TF-IDF-Vektorisierung für die Textverarbeitung und bietet
        verschiedene Hyperparameter zur Optimierung der Modelle.

        Verfügbare Zielvariablen:
        - Fits_Topic_Code
        - Urgency_Code
        - Categorie_Code

        Entwickelt im Rahmen eines Supervised Learning Projekts.
        """
        desc_label = ttk.Label(
            desc_frame, 
            text=description, 
            wraplength=600, 
            justify=tk.LEFT,
            font=("Segoe UI", 10)
        )
        desc_label.pack(pady=10, fill=tk.X)

        # Footer-Bereich mit Version und Copyright
        footer_frame = ttk.Frame(content_frame)
        footer_frame.pack(fill=tk.X, expand=False, pady=10)

        version_label = ttk.Label(
            footer_frame, 
            text="Version 1.0", 
            font=("Segoe UI", 10, "italic"),
            foreground="#7f8c8d"  # Grau für die Version
        )
        version_label.pack(side=tk.LEFT, padx=20)

        copyright_label = ttk.Label(
            footer_frame, 
            text="© 2025 Supervised Learning Project", 
            font=("Segoe UI", 10),
            foreground="#7f8c8d"  # Grau für das Copyright
        )
        copyright_label.pack(side=tk.RIGHT, padx=20)

    def log_output(self, message):
        """
        Fügt eine Nachricht zum Ausgabebereich hinzu.

        Args:
            message: Die anzuzeigende Nachricht
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def create_help_button(self, parent, row, column, tooltip_key):
        """
        Erstellt einen Hilfe-Button (?) mit Tooltip.

        Args:
            parent: Das Eltern-Widget
            row: Zeilennummer im Grid
            column: Spaltennummer im Grid
            tooltip_key: Schlüssel für den Tooltip-Text
        """
        help_button = ttk.Label(parent, text="?", font=('Segoe UI', 9, 'bold'), 
                               foreground="blue", background="lightgray",
                               width=2, anchor="center")
        help_button.grid(row=row, column=column, padx=2, pady=8)

        # Füge Tooltip hinzu
        if tooltip_key in self.tooltips:
            ToolTip(help_button, self.tooltips[tooltip_key])

        return help_button

    def clear_output(self):
        """
        Löscht den Inhalt des Ausgabebereichs.
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

    def train_model(self):
        """
        Startet das Training des ausgewählten Modells mit den konfigurierten Parametern.
        """
        # Deaktiviere den Trainingsbutton während des Trainings
        self.train_button.config(state=tk.DISABLED)

        # Ändere den Text des Buttons, um den Trainingsstatus anzuzeigen
        self.train_button.config(text="Training läuft...")

        # Erstelle einen Fortschrittsbalken, wenn noch keiner existiert
        if not hasattr(self, 'progress_bar'):
            self.progress_frame = ttk.Frame(self.training_tab)
            self.progress_frame.pack(fill=tk.X, expand=False, padx=15, pady=5)

            self.progress_label = ttk.Label(self.progress_frame, text="Training:", font=('Segoe UI', 9))
            self.progress_label.pack(side=tk.LEFT, padx=5)

            self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=200)
            self.progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        else:
            # Zeige den Fortschrittsbalken an, falls er versteckt war
            self.progress_frame.pack(fill=tk.X, expand=False, padx=15, pady=5)

        # Starte den Fortschrittsbalken
        self.progress_bar.start(10)

        # Leere die Ausgabe und zeige eine Startmeldung an
        self.clear_output()
        self.log_output("Initialisiere Training...\n")

        # Starte das Training in einem separaten Thread
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        """
        Führt das Modelltraining in einem separaten Thread aus.
        """
        try:
            # Hole die Werte aus den Formularelementen
            model_type = self.model_var.get()
            target_column = self.target_var.get()

            # Parse die numerischen Werte
            try:
                max_features = int(self.max_features_var.get())
                test_size = float(self.test_size_var.get())
                max_iter = int(self.max_iter_var.get())
                C = float(self.c_var.get())
                epochs = int(self.epochs_var.get())
                patience = int(self.patience_var.get())
            except ValueError as e:
                self.log_output(f"Fehler: Ungültige Eingabe - {str(e)}")
                self.train_button.config(state=tk.NORMAL)
                return

            solver = self.solver_var.get()

            # Validiere die Eingaben
            if test_size <= 0 or test_size >= 1:
                self.log_output("Fehler: Test Size muss zwischen 0 und 1 liegen.")
                self.train_button.config(state=tk.NORMAL)
                return

            # Zeige die Konfiguration an
            self.log_output(f"Starte Training mit folgender Konfiguration:")
            self.log_output(f"- Modelltyp: {model_type}")
            self.log_output(f"- Zielvariable: {target_column}")
            self.log_output(f"- Max Features: {max_features}")
            self.log_output(f"- Test Size: {test_size}")

            if model_type in ["logreg", "all"]:
                self.log_output(f"- Max Iterations: {max_iter}")
                self.log_output(f"- C: {C}")
                self.log_output(f"- Solver: {solver}")

            if model_type in ["nn", "all"]:
                self.log_output(f"- Epochen: {epochs}")
                self.log_output(f"- Patience: {patience}")

            self.log_output("\nTraining läuft, bitte warten...")

            # Führe das Training durch
            results = {}

            if model_type in ["logreg", "all"]:
                self.log_output("\n--- Training des logistischen Regressionsmodells ---")
                accuracy, precision, recall, f1, report, conf_matrix = train_logreg(
                    target_column=target_column,
                    max_features=max_features,
                    test_size=test_size,
                    max_iter=max_iter,
                    C=C,
                    solver=solver
                )

                results["logreg"] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "report": report,
                    "conf_matrix": conf_matrix,
                    "target_column": target_column
                }

                self.log_output(f"Logistische Regression - Accuracy: {accuracy:.4f}")
                self.log_output(f"Logistische Regression - Precision: {precision:.4f}")
                self.log_output(f"Logistische Regression - Recall: {recall:.4f}")
                self.log_output(f"Logistische Regression - F1 Score: {f1:.4f}")

            if model_type in ["nn", "all"]:
                self.log_output("\n--- Training des neuronalen Netzes ---")
                accuracy, precision, recall, f1, report, conf_matrix = train_nn(
                    target_column=target_column,
                    epochs=epochs,
                    max_features=max_features,
                    test_size=test_size,
                    patience=patience
                )

                results["nn"] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "report": report,
                    "conf_matrix": conf_matrix,
                    "target_column": target_column
                }

                self.log_output(f"Neuronales Netz - Accuracy: {accuracy:.4f}")
                self.log_output(f"Neuronales Netz - Precision: {precision:.4f}")
                self.log_output(f"Neuronales Netz - Recall: {recall:.4f}")
                self.log_output(f"Neuronales Netz - F1 Score: {f1:.4f}")

            # Speichere die Ergebnisse
            result_key = f"{target_column} - {model_type} - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.training_results[result_key] = results

            # Aktualisiere die Ergebnisauswahl
            self.result_combobox['values'] = list(self.training_results.keys())
            self.result_combobox.current(len(self.training_results) - 1)

            # Zeige die Ergebnisse an
            self.display_selected_result(None)

            # Wechsle zum Ergebnisse-Tab
            self.notebook.select(1)  # Index 1 ist der Ergebnisse-Tab

            self.log_output("\nTraining abgeschlossen!")

        except Exception as e:
            self.log_output(f"Fehler beim Training: {str(e)}")
        finally:
            # Stoppe den Fortschrittsbalken
            if hasattr(self, 'progress_bar'):
                self.progress_bar.stop()
                # Verstecke den Fortschrittsbalken
                self.progress_frame.pack_forget()

            # Aktiviere den Trainingsbutton wieder und setze den Text zurück
            self.train_button.config(state=tk.NORMAL, text="Modell trainieren")

    def cleanup_resources(self):
        """
        Bereinigt alle Ressourcen, insbesondere matplotlib-Figuren.
        """
        # Close all matplotlib figures to prevent "main thread is not in main loop" error
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()

        # Close all remaining figures
        plt.close('all')

    def display_selected_result(self, event):
        """
        Zeigt das ausgewählte Ergebnis im Ergebnisse-Tab an.

        Args:
            event: Das Event-Objekt (kann None sein)
        """
        selected_key = self.result_var.get()
        if not selected_key or selected_key not in self.training_results:
            return

        results = self.training_results[selected_key]

        # Bestimme das anzuzeigende Modell (wenn "all" gewählt wurde, zeige logreg zuerst)
        model_key = list(results.keys())[0]

        # Zeige die Metriken an
        self.accuracy_label.config(text=f"Accuracy: {results[model_key]['accuracy']:.4f}")
        self.precision_label.config(text=f"Precision: {results[model_key]['precision']:.4f}")
        self.recall_label.config(text=f"Recall: {results[model_key]['recall']:.4f}")
        self.f1_label.config(text=f"F1 Score: {results[model_key]['f1']:.4f}")

        # Zeige den Klassifikationsbericht an
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, results[model_key]['report'])
        self.report_text.config(state=tk.DISABLED)

        # Zeige die Konfusionsmatrix an
        for widget in self.cm_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))
        # Store reference to the figure for cleanup
        self.figures.append(fig)

        conf_matrix = results[model_key]['conf_matrix']
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
