#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hauptmodul zur Orchestrierung des Trainings verschiedener Modelle.

Dieses Modul dient als Einstiegspunkt für das Projekt und ermöglicht es,
verschiedene Modelle mit verschiedenen Zielvariablen zu trainieren.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Tuple

# Importiere die Funktionen aus den anderen Modulen
from data_loader import load_data, get_available_targets
from train_logreg import train_and_save_model as train_logreg
from train_nn import train_and_save_model as train_nn


def parse_arguments() -> argparse.Namespace:
    """
    Parst die Kommandozeilenargumente.

    Returns:
        argparse.Namespace: Geparste Argumente
    """
    # Erstellung eines ArgumentParsers mit Beschreibung
    parser = argparse.ArgumentParser(description="Trainiert verschiedene Modelle für die Textklassifikation.")

    # Definition des Modelltyp-Arguments
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["logreg", "nn", "all"], 
        default="all",
        help="Zu trainierendes Modell (logreg, nn oder all)"
    )

    # Definition des Zielvariablen-Arguments
    parser.add_argument(
        "--target", 
        type=str, 
        choices=["Fits_Topic_Code", "Urgency_Code", "Categorie_Code", "all"], 
        default="Fits_Topic_Code",
        help="Zielvariable für das Training"
    )

    # Definition des Epochen-Arguments für neuronale Netze
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30,
        help="Anzahl der Trainingsepochen für neuronale Netze"
    )

    # Definition der Hyperparameter für die logistische Regression
    parser.add_argument(
        "--max-features", 
        type=int, 
        default=1000,
        help="Maximale Anzahl der Features für TF-IDF"
    )

    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Anteil der Testdaten (0.0-1.0)"
    )

    parser.add_argument(
        "--max-iter", 
        type=int, 
        default=1000,
        help="Maximale Anzahl von Iterationen für die logistische Regression"
    )

    parser.add_argument(
        "--C", 
        type=float, 
        default=1.0,
        help="Regularisierungsparameter für die logistische Regression (kleinere Werte bedeuten stärkere Regularisierung)"
    )

    parser.add_argument(
        "--solver", 
        type=str, 
        choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        default="lbfgs",
        help="Algorithmus für die Optimierung der logistischen Regression"
    )

    # Definition des Patience-Parameters für Early Stopping bei neuronalen Netzen
    parser.add_argument(
        "--patience", 
        type=int, 
        default=5,
        help="Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird (für neuronale Netze)"
    )

    # Rückgabe der geparsten Argumente
    return parser.parse_args()


def train_models(model_type: str, target_column: str, epochs: int = 30,
               max_features: int = 1000, test_size: float = 0.2, 
               max_iter: int = 1000, C: float = 1.0, solver: str = 'lbfgs',
               patience: int = 5) -> Dict[str, Any]:
    """
    Trainiert die angegebenen Modelle mit der angegebenen Zielvariable.

    Args:
        model_type: Typ des zu trainierenden Modells ("logreg", "nn" oder "all")
        target_column: Name der Zielspalte
        epochs: Anzahl der Trainingsepochen für neuronale Netze
        max_features: Maximale Anzahl der Features für TF-IDF
        test_size: Anteil der Testdaten
        max_iter: Maximale Anzahl von Iterationen für die logistische Regression
        C: Regularisierungsparameter für die logistische Regression
        solver: Algorithmus für die Optimierung der logistischen Regression
        patience: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird (für neuronale Netze)

    Returns:
        Dict[str, Any]: Dictionary mit Ergebnissen
    """
    # Initialisierung des Ergebnis-Dictionaries
    results = {}

    # Training des logistischen Regressionsmodells, falls ausgewählt
    if model_type in ["logreg", "all"]:
        print(f"\n--- Training des logistischen Regressionsmodells mit Zielspalte '{target_column}' ---")
        print(f"Hyperparameter: max_features={max_features}, test_size={test_size}, max_iter={max_iter}, C={C}, solver={solver}")

        # Aufruf der Trainingsfunktion für logistische Regression mit konfigurierbaren Hyperparametern
        accuracy, precision, recall, f1, report, conf_matrix = train_logreg(
            target_column=target_column,
            max_features=max_features,
            test_size=test_size,
            random_state=42,  # Fester Wert für Reproduzierbarkeit
            max_iter=max_iter,
            C=C,
            solver=solver
        )

        # Speicherung der Ergebnisse
        results["logreg"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": report,
            "conf_matrix": conf_matrix
        }

        # Ausgabe der Metriken
        print(f"Logistische Regression - Accuracy: {accuracy:.4f}")
        print(f"Logistische Regression - Precision: {precision:.4f}")
        print(f"Logistische Regression - Recall: {recall:.4f}")
        print(f"Logistische Regression - F1 Score: {f1:.4f}")

        # Ausgabe der Konfusionsmatrix
        print("\nLogistische Regression - Konfusionsmatrix:")
        print(conf_matrix)

    # Training des neuronalen Netzes, falls ausgewählt
    if model_type in ["nn", "all"]:
        print(f"\n--- Training des neuronalen Netzes mit Zielspalte '{target_column}' ---")
        print(f"Hyperparameter: epochs={epochs}, max_features={max_features}, test_size={test_size}")

        # Aufruf der Trainingsfunktion für neuronales Netz
        # Hier werden nur die relevanten Hyperparameter übergeben
        accuracy, precision, recall, f1, report, conf_matrix = train_nn(
            target_column=target_column, 
            epochs=epochs,
            max_features=max_features,
            test_size=test_size,
            patience=patience
        )

        # Speicherung der Ergebnisse
        results["nn"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": report,
            "conf_matrix": conf_matrix
        }

        # Ausgabe der Metriken
        print(f"Neuronales Netz - Accuracy: {accuracy:.4f}")
        print(f"Neuronales Netz - Precision: {precision:.4f}")
        print(f"Neuronales Netz - Recall: {recall:.4f}")
        print(f"Neuronales Netz - F1 Score: {f1:.4f}")

        # Ausgabe der Konfusionsmatrix
        print("\nNeuronales Netz - Konfusionsmatrix:")
        print(conf_matrix)

    # Rückgabe der gesammelten Ergebnisse
    return results


def main() -> None:
    """
    Hauptfunktion des Programms.
    """
    try:
        # Verarbeitung der Kommandozeilenargumente
        args = parse_arguments()

        # Abruf und Anzeige der verfügbaren Zielvariablen
        targets = get_available_targets()
        print("Verfügbare Zielvariablen:")
        for col, info in targets.items():
            print(f"- {col}: {info['unique_values']}")
            print(f"  Verteilung: {info['value_counts']}")

        # Hyperparameter aus den Kommandozeilenargumenten extrahieren
        hyperparams = {
            'epochs': args.epochs,
            'max_features': args.max_features,
            'test_size': args.test_size,
            'max_iter': args.max_iter,
            'C': args.C,
            'solver': args.solver,
            'patience': args.patience
        }

        print("\nVerwendete Hyperparameter:")
        for param, value in hyperparams.items():
            print(f"- {param}: {value}")

        # Modelltraining basierend auf den ausgewählten Optionen
        if args.target == "all":
            # Training mit allen verfügbaren Zielvariablen
            all_results = {}
            for target in targets.keys():
                print(f"\n=== Training mit Zielvariable '{target}' ===")
                all_results[target] = train_models(
                    args.model, 
                    target, 
                    **hyperparams
                )
        else:
            # Überprüfen, ob die angegebene Zielvariable verfügbar ist
            if args.target not in targets:
                raise ValueError(f"Die angegebene Zielvariable '{args.target}' ist nicht in den Daten verfügbar. "
                               f"Verfügbare Zielvariablen: {list(targets.keys())}")

            # Training mit einer spezifischen Zielvariable
            train_models(
                args.model, 
                args.target, 
                **hyperparams
            )

        # Abschlussmeldung nach erfolgreichem Training
        print("\nTraining abgeschlossen. Die Modelle wurden im Verzeichnis 'models' gespeichert.")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Fehler: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Ausführung der Hauptfunktion mit Fehlerbehandlung
    try:
        # Aufruf der Hauptfunktion
        main()
    except Exception as e:
        # Ausgabe von Fehlermeldungen und Beendigung mit Fehlercode
        print(f"Fehler: {str(e)}")
        sys.exit(1)
