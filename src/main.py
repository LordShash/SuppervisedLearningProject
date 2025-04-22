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
import logging
from typing import Dict, Any, List, Tuple, Optional

# Importiere die Funktionen aus den anderen Modulen
from data_loader import load_data, get_available_targets, clear_tfidf_cache
from train_logreg import train_and_save_model as train_logreg, ModelResults as LogRegResults
from train_nn import train_and_save_model as train_nn, NNModelResults as NNResults

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ausgabe in die Konsole
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'main.log'), 
                           mode='a', encoding='utf-8')  # Ausgabe in eine Datei
    ]
)
logger = logging.getLogger(__name__)

# Stelle sicher, dass das Logs-Verzeichnis existiert
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)


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

    # Parsen der Argumente
    return parser.parse_args()


def train_models(model_type: str, target_column: str, epochs: int = 30, max_features: int = 1000, 
                test_size: float = 0.2, max_iter: int = 1000, C: float = 1.0, 
                solver: str = 'lbfgs', patience: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Trainiert die angegebenen Modelle mit den angegebenen Parametern.

    Args:
        model_type: Zu trainierendes Modell ('logreg', 'nn' oder 'all')
        target_column: Zielvariable für das Training
        epochs: Anzahl der Trainingsepochen für neuronale Netze
        max_features: Maximale Anzahl der Features für TF-IDF
        test_size: Anteil der Testdaten
        max_iter: Maximale Anzahl von Iterationen für die logistische Regression
        C: Regularisierungsparameter für die logistische Regression
        solver: Algorithmus für die Optimierung der logistischen Regression
        patience: Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mit den Trainingsergebnissen
    """
    results = {}

    # Bestimme die zu trainierenden Modelle
    models_to_train = []
    if model_type == "all":
        models_to_train = ["logreg", "nn"]
    else:
        models_to_train = [model_type]

    # Bestimme die zu verwendenden Zielvariablen
    target_columns = []
    if target_column == "all":
        # Lade alle verfügbaren Zielvariablen
        try:
            targets = get_available_targets()
            target_columns = list(targets.keys())
        except Exception as e:
            print(f"Fehler beim Laden der verfügbaren Zielvariablen: {str(e)}")
            return results
    else:
        target_columns = [target_column]

    # Trainiere die ausgewählten Modelle mit den ausgewählten Zielvariablen
    for model in models_to_train:
        for target in target_columns:
            print(f"\n{'='*80}")
            print(f"Training des Modells '{model}' mit Zielvariable '{target}'")
            print(f"{'='*80}\n")

            try:
                if model == "logreg":
                    # Training des logistischen Regressionsmodells
                    print(f"Starte Training des logistischen Regressionsmodells mit folgenden Parametern:")
                    print(f"- Zielvariable: {target}")
                    print(f"- Max Features: {max_features}")
                    print(f"- Test Size: {test_size}")
                    print(f"- Max Iterations: {max_iter}")
                    print(f"- C (Regularisierung): {C}")
                    print(f"- Solver: {solver}\n")

                    # Aufruf der Trainingsfunktion
                    results_obj = train_logreg(
                        target_column=target,
                        max_features=max_features,
                        test_size=test_size,
                        max_iter=max_iter,
                        C=C,
                        solver=solver
                    )
                    accuracy = results_obj.accuracy
                    precision = results_obj.precision
                    recall = results_obj.recall
                    f1 = results_obj.f1
                    report = results_obj.report
                    conf_matrix = results_obj.conf_matrix

                    # Speichern der Ergebnisse
                    results[f"logreg_{target}"] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'report': report,
                        'conf_matrix': conf_matrix,
                        'params': {
                            'target_column': target,
                            'max_features': max_features,
                            'test_size': test_size,
                            'max_iter': max_iter,
                            'C': C,
                            'solver': solver
                        }
                    }

                elif model == "nn":
                    # Training des neuronalen Netzes
                    print(f"Starte Training des neuronalen Netzes mit folgenden Parametern:")
                    print(f"- Zielvariable: {target}")
                    print(f"- Max Features: {max_features}")
                    print(f"- Test Size: {test_size}")
                    print(f"- Epochs: {epochs}")
                    print(f"- Patience: {patience}\n")

                    # Aufruf der Trainingsfunktion
                    results_obj = train_nn(
                        target_column=target,
                        max_features=max_features,
                        test_size=test_size,
                        epochs=epochs,
                        patience=patience
                    )
                    accuracy = results_obj.accuracy
                    precision = results_obj.precision
                    recall = results_obj.recall
                    f1 = results_obj.f1
                    report = results_obj.report
                    conf_matrix = results_obj.conf_matrix

                    # Speichern der Ergebnisse
                    results[f"nn_{target}"] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'report': report,
                        'conf_matrix': conf_matrix,
                        'params': {
                            'target_column': target,
                            'max_features': max_features,
                            'test_size': test_size,
                            'epochs': epochs,
                            'patience': patience
                        }
                    }

            except Exception as e:
                print(f"Fehler beim Training des Modells '{model}' mit Zielvariable '{target}': {str(e)}")
                import traceback
                print(traceback.format_exc())

    return results


def main():
    """
    Hauptfunktion des Programms.
    """
    # Parse die Kommandozeilenargumente
    args = parse_arguments()

    print("\nTextklassifikation - Modelltraining")
    print("="*40)
    print(f"Modell: {args.model}")
    print(f"Zielvariable: {args.target}")
    print(f"Max Features: {args.max_features}")
    print(f"Test Size: {args.test_size}")

    if args.model in ["logreg", "all"]:
        print("\nParameter für logistische Regression:")
        print(f"- Max Iterations: {args.max_iter}")
        print(f"- C (Regularisierung): {args.C}")
        print(f"- Solver: {args.solver}")

    if args.model in ["nn", "all"]:
        print("\nParameter für neuronales Netz:")
        print(f"- Epochs: {args.epochs}")
        print(f"- Patience: {args.patience}")

    print("\nStarte Training...\n")

    # Trainiere die Modelle
    results = train_models(
        model_type=args.model,
        target_column=args.target,
        epochs=args.epochs,
        max_features=args.max_features,
        test_size=args.test_size,
        max_iter=args.max_iter,
        C=args.C,
        solver=args.solver,
        patience=args.patience
    )

    # Ausgabe der Ergebnisse
    print("\n\nZusammenfassung der Ergebnisse:")
    print("="*40)

    for model_key, model_results in results.items():
        print(f"\nModell: {model_key}")
        print(f"Accuracy: {model_results['accuracy']:.4f}")
        print(f"Precision: {model_results['precision']:.4f}")
        print(f"Recall: {model_results['recall']:.4f}")
        print(f"F1 Score: {model_results['f1']:.4f}")

    print("\nTraining abgeschlossen!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining wurde vom Benutzer abgebrochen.")
    except Exception as e:
        print(f"\nFehler: {str(e)}")
        import traceback
        print(traceback.format_exc())
