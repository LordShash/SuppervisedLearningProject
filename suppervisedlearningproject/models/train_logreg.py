#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul zum Training eines logistischen Regressionsmodells für die Textklassifikation.

Dieses Modul stellt Funktionen bereit, um ein logistisches Regressionsmodell
zu trainieren, zu evaluieren und zu speichern.
"""

import os
import sys
import datetime
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import joblib
import numpy as np
import matplotlib
# Use Agg backend for figure generation in background threads
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

# Importiere die Datenladefunktion aus dem core Paket
from suppervisedlearningproject.core.data_loader import load_data

# Importiere die Konfiguration und richte das Logging ein
from suppervisedlearningproject.utils import setup_logging, MODELS_DIR, LOGS_DIR, PLOTS_DIR, DEFAULT_LOGREG_CONFIG

# Konfiguration des Loggings mit dem zentralen Setup
logger = setup_logging(__name__)

@dataclass
class ModelResults:
    """Datenklasse zur Speicherung der Modellergebnisse."""
    model: Pipeline
    accuracy: float
    precision: float
    recall: float
    f1: float
    report: str
    conf_matrix: np.ndarray
    class_names: list
    cross_val_scores: Optional[np.ndarray] = None
    roc_curve_data: Optional[Dict[str, np.ndarray]] = None
    auc_scores: Optional[Dict[str, float]] = None


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: list, target_name: str, timestamp: str = None) -> str:
    """
    Erstellt und speichert eine Visualisierung der Konfusionsmatrix.

    Args:
        conf_matrix: Die Konfusionsmatrix als numpy-Array
        class_names: Liste mit den Namen der Klassen
        target_name: Name der Zielspalte für den Dateinamen
        timestamp: Optionaler Zeitstempel für den Dateinamen

    Returns:
        str: Pfad zur gespeicherten Visualisierung
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Visualisierung der Konfusionsmatrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Konfusionsmatrix für {target_name}')
    plt.colorbar()

    # Beschriftungen
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Werte in der Matrix anzeigen
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Vorhergesagte Klasse')
    plt.ylabel('Tatsächliche Klasse')

    # Speichere die Visualisierung
    plot_path = os.path.join(PLOTS_DIR, f'confusion_matrix_{target_name}_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Konfusionsmatrix-Visualisierung gespeichert: {plot_path}")
    return plot_path


def train_logistic_regression(
    X: np.ndarray, 
    y: np.ndarray, 
    target_name: str = "Fits_Topic_Code", 
    test_size: float = 0.2, 
    random_state: int = 42, 
    max_iter: int = 1000, 
    C: float = 1.0, 
    solver: str = 'lbfgs',
    cv_folds: int = 5,
    class_names: Optional[list] = None
) -> ModelResults:
    """
    Trainiert ein logistisches Regressionsmodell mit den gegebenen Daten.

    Args:
        X: Feature-Matrix
        y: Zielvariable
        target_name: Name der Zielspalte für Ausgabezwecke
        test_size: Anteil der Testdaten
        random_state: Seed für die Reproduzierbarkeit
        max_iter: Maximale Anzahl von Iterationen für die logistische Regression
        C: Regularisierungsparameter (kleinere Werte bedeuten stärkere Regularisierung)
        solver: Algorithmus für die Optimierung ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
        cv_folds: Anzahl der Folds für die Kreuzvalidierung
        class_names: Liste mit Namen für die Klassen (falls None, werden automatisch generiert)

    Returns:
        ModelResults: Objekt mit trainiertem Modell und allen Evaluationsmetriken

    Raises:
        ValueError: Wenn Probleme beim Training des Modells auftreten
    """
    try:
        logger.info(f"Starte Training des logistischen Regressionsmodells für Zielspalte '{target_name}'")

        # Überprüfe, ob X eine sparse Matrix ist
        is_sparse_matrix = issparse(X)
        logger.info(f"Eingabe-Matrix ist {'sparse' if is_sparse_matrix else 'dense'}")

        # Aufteilung der Daten in Trainings- und Testdaten
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Daten aufgeteilt: {X_train.shape[0]} Trainingsbeispiele, {X_test.shape[0]} Testbeispiele")

        # Erstellung einer Pipeline mit Standardisierung und logistischer Regression
        # Verwende with_mean=False nur für sparse Matrizen
        pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=not is_sparse_matrix)),
            ('classifier', LogisticRegression(
                max_iter=max_iter, 
                C=C, 
                solver=solver, 
                random_state=random_state, 
                n_jobs=-1,
                class_weight='balanced'  # Ausgleich für unbalancierte Klassen
            ))
        ])

        # Training des Modells mit den Trainingsdaten
        logger.info(f"Trainiere Modell mit Parametern: max_iter={max_iter}, C={C}, solver={solver}")
        pipeline.fit(X_train, y_train)
        logger.info("Modelltraining abgeschlossen")

        # Kreuzvalidierung
        logger.info(f"Führe {cv_folds}-fache Kreuzvalidierung durch")
        cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy')
        logger.info(f"Kreuzvalidierungs-Scores: {cv_scores}")
        logger.info(f"Mittlere CV-Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Vorhersagen auf den Testdaten
        y_pred = pipeline.predict(X_test)

        # Berechne Wahrscheinlichkeiten für ROC-Kurve
        y_pred_proba = pipeline.predict_proba(X_test)

        # Berechnung der Metriken
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Generiere Klassennamen, falls nicht angegeben
        if class_names is None:
            unique_classes = sorted(set(y))
            class_names = [f"{target_name}_{i}" for i in unique_classes]

        # Detaillierter Klassifikationsbericht
        report = classification_report(y_test, y_pred, target_names=class_names)

        # Erstellung der Konfusionsmatrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Ausgabe der Ergebnisse
        logger.info(f"\nErgebnisse für Zielspalte '{target_name}':")
        logger.info(f"Accuracy auf dem Test-Set: {accuracy:.4f}")
        logger.info(f"Precision auf dem Test-Set: {precision:.4f}")
        logger.info(f"Recall auf dem Test-Set: {recall:.4f}")
        logger.info(f"F1 Score auf dem Test-Set: {f1:.4f}")
        logger.info("\nKlassifikationsbericht:")
        logger.info(f"\n{report}")

        # Visualisierung der Konfusionsmatrix mit der dedizierten Funktion
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = plot_confusion_matrix(conf_matrix, class_names, target_name, timestamp)
            logger.info(f"Konfusionsmatrix-Visualisierung gespeichert: {plot_path}")
        except Exception as viz_error:
            logger.warning(f"Konnte Konfusionsmatrix nicht visualisieren: {str(viz_error)}")

        # Berechne ROC-Kurve und AUC für jede Klasse
        roc_curve_data = {}
        auc_scores = {}

        try:
            # Für binäre Klassifikation
            if len(class_names) == 2:
                # Verwende die Wahrscheinlichkeit der positiven Klasse (Index 1)
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
                roc_auc = auc(fpr, tpr)

                roc_curve_data["binary"] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds
                }
                auc_scores["binary"] = roc_auc

                logger.info(f"AUC für binäre Klassifikation: {roc_auc:.4f}")

            # Für Mehrklassen-Klassifikation (One-vs-Rest)
            else:
                # Berechne ROC-Kurve und AUC für jede Klasse
                for i, class_name in enumerate(class_names):
                    # One-vs-Rest: aktuelle Klasse vs. alle anderen
                    y_test_binary = (y_test == i).astype(int)
                    y_score = y_pred_proba[:, i]

                    fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
                    roc_auc = auc(fpr, tpr)

                    roc_curve_data[class_name] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds
                    }
                    auc_scores[class_name] = roc_auc

                    logger.info(f"AUC für Klasse '{class_name}': {roc_auc:.4f}")

                # Berechne auch den gewichteten Durchschnitt der AUC-Werte
                try:
                    weighted_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    auc_scores["weighted"] = weighted_auc
                    logger.info(f"Gewichteter AUC: {weighted_auc:.4f}")
                except Exception as auc_error:
                    logger.warning(f"Konnte gewichteten AUC nicht berechnen: {str(auc_error)}")

        except Exception as roc_error:
            logger.warning(f"Konnte ROC-Kurve nicht berechnen: {str(roc_error)}")
            roc_curve_data = None
            auc_scores = None

        # Erstelle und gib ModelResults-Objekt zurück
        results = ModelResults(
            model=pipeline,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            report=report,
            conf_matrix=conf_matrix,
            class_names=class_names,
            cross_val_scores=cv_scores,
            roc_curve_data=roc_curve_data,
            auc_scores=auc_scores
        )

        return results

    except Exception as e:
        # Fehlerbehandlung bei Problemen während des Trainings
        error_msg = f"Fehler beim Training des logistischen Regressionsmodells: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


def save_model(
    model: Pipeline, 
    target_column: str = "Fits_Topic_Code", 
    version: Optional[str] = None,
    overwrite: bool = True,
    include_timestamp: bool = True,
    include_metrics: bool = True,
    metrics: Optional[Dict[str, Any]] = None
) -> str:
    """
    Speichert das trainierte Modell als PKL-Datei mit Metadaten.

    Args:
        model: Trainiertes Modell
        target_column: Name der Zielspalte für den Dateinamen
        version: Optionale Versionsbezeichnung (z.B. "v1.0")
        overwrite: Wenn True, werden vorhandene Dateien überschrieben
        include_timestamp: Wenn True, wird ein Zeitstempel im Dateinamen verwendet
        include_metrics: Wenn True, werden Metriken mit dem Modell gespeichert
        metrics: Dictionary mit Metriken, die mit dem Modell gespeichert werden sollen

    Returns:
        str: Pfad zur gespeicherten Modelldatei

    Raises:
        ValueError: Wenn Probleme beim Speichern des Modells auftreten
        FileExistsError: Wenn die Datei bereits existiert und overwrite=False
    """
    # Generierung des Dateinamens basierend auf der Zielspalte und optionalen Parametern
    timestamp = ""
    if include_timestamp:
        timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    version_str = ""
    if version:
        version_str = f"_{version}"

    filename = f"logreg_{target_column}{version_str}{timestamp}_model.pkl"
    model_path = os.path.join(MODELS_DIR, filename)

    # Prüfen, ob die Datei bereits existiert
    if os.path.exists(model_path) and not overwrite:
        error_msg = f"Die Modelldatei '{model_path}' existiert bereits und overwrite=False."
        logger.error(error_msg)
        raise FileExistsError(error_msg)

    try:
        # Vorbereitung der Metadaten
        model_info = {
            'model_type': 'LogisticRegression',
            'target_column': target_column,
            'created_at': datetime.datetime.now().isoformat(),
            'python_version': sys.version,
            'sklearn_version': model.named_steps['classifier'].__class__.__module__.split('.')[1],
        }

        # Hinzufügen der Metriken, falls vorhanden
        if include_metrics and metrics:
            model_info['metrics'] = metrics

        # Speicherung des Modells mit Metadaten
        save_data = {
            'model': model,
            'metadata': model_info
        }

        # Speicherung des Modells im Pickle-Format mit joblib
        joblib.dump(save_data, model_path)

        # Ausgabe einer Erfolgsmeldung mit dem Speicherpfad
        logger.info(f"Modell erfolgreich gespeichert unter: {model_path}")
        logger.info(f"Modell-Metadaten: {model_info}")

        return model_path

    except Exception as e:
        # Fehlerbehandlung bei Problemen mit der Speicherung
        error_msg = f"Fehler beim Speichern des Modells: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


def train_and_save_model(
    target_column: str = "Fits_Topic_Code", 
    max_features: int = 1000,
    test_size: float = 0.2, 
    random_state: int = 42,
    max_iter: int = DEFAULT_LOGREG_CONFIG['max_iter'], 
    C: float = DEFAULT_LOGREG_CONFIG['C'], 
    solver: str = DEFAULT_LOGREG_CONFIG['solver'],
    cv_folds: int = 5,
    save_model_flag: bool = True,
    version: Optional[str] = None,
    overwrite: bool = True,
    class_names: Optional[list] = None
) -> ModelResults:
    """
    Lädt Daten, trainiert ein Modell und speichert es optional.

    Args:
        target_column: Name der Zielspalte
        max_features: Maximale Anzahl der Features für TF-IDF
        test_size: Anteil der Testdaten
        random_state: Seed für die Reproduzierbarkeit
        max_iter: Maximale Anzahl von Iterationen für die logistische Regression
        C: Regularisierungsparameter (kleinere Werte bedeuten stärkere Regularisierung)
        solver: Algorithmus für die Optimierung ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
        cv_folds: Anzahl der Folds für die Kreuzvalidierung
        save_model_flag: Wenn True, wird das Modell gespeichert
        version: Optionale Versionsbezeichnung für das gespeicherte Modell
        overwrite: Wenn True, werden vorhandene Modelldateien überschrieben
        class_names: Liste mit Namen für die Klassen (falls None, werden automatisch generiert)

    Returns:
        ModelResults: Objekt mit trainiertem Modell und allen Evaluationsmetriken

    Raises:
        FileNotFoundError: Wenn keine Datendateien gefunden wurden
        ValueError: Wenn Probleme beim Laden, Verarbeiten oder Speichern der Daten auftreten
        KeyError: Wenn die angegebene Zielspalte nicht im DataFrame vorhanden ist
        FileExistsError: Wenn die Modelldatei bereits existiert und overwrite=False
    """
    try:
        logger.info(f"Starte Training und Speicherung für Zielspalte '{target_column}'")

        # Laden der Daten mit der angegebenen Zielspalte und max_features
        logger.info(f"Lade Daten mit max_features={max_features}")
        X, y = load_data(target_column=target_column, max_features=max_features)
        logger.info(f"Daten geladen: X.shape={X.shape}, y.shape={y.shape}")

        # Training des logistischen Regressionsmodells mit den geladenen Daten und konfigurierbaren Hyperparametern
        logger.info("Starte Modelltraining")
        results = train_logistic_regression(
            X, y, 
            target_name=target_column,
            test_size=test_size,
            random_state=random_state,
            max_iter=max_iter,
            C=C,
            solver=solver,
            cv_folds=cv_folds,
            class_names=class_names
        )
        logger.info("Modelltraining abgeschlossen")

        # Speicherung des trainierten Modells für spätere Verwendung, falls gewünscht
        if save_model_flag:
            # Erstelle ein Dictionary mit den Metriken für die Speicherung
            metrics = {
                'accuracy': results.accuracy,
                'precision': results.precision,
                'recall': results.recall,
                'f1': results.f1,
                'cross_val_mean': float(results.cross_val_scores.mean()) if results.cross_val_scores is not None else None,
                'cross_val_std': float(results.cross_val_scores.std()) if results.cross_val_scores is not None else None,
                'test_size': test_size,
                'max_features': max_features,
                'hyperparameters': {
                    'max_iter': max_iter,
                    'C': C,
                    'solver': solver
                }
            }

            # Speichere das Modell mit den Metriken
            model_path = save_model(
                model=results.model, 
                target_column=target_column,
                version=version,
                overwrite=overwrite,
                include_timestamp=True,
                include_metrics=True,
                metrics=metrics
            )
            logger.info(f"Modell gespeichert unter: {model_path}")
        else:
            logger.info("Modell wurde nicht gespeichert (save_model_flag=False)")

        # Rückgabe der ModelResults
        return results

    except (FileNotFoundError, ValueError, KeyError, FileExistsError) as e:
        # Spezifische Fehler weiterleiten
        logger.error(f"Fehler beim Training und Speichern des Modells: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        # Unerwartete Fehler in ValueError umwandeln
        error_msg = f"Unerwarteter Fehler beim Training und Speichern des Modells: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


if __name__ == "__main__":
    # Ausführung des Skripts als eigenständiges Programm
    import argparse
    import logging

    # Kommandozeilenargumente parsen
    parser = argparse.ArgumentParser(description="Trainiert und speichert ein logistisches Regressionsmodell")
    parser.add_argument("--target", default="Fits_Topic_Code", 
                        help="Zielspalte für das Training (Standard: Fits_Topic_Code)")
    parser.add_argument("--max-features", type=int, default=1000, 
                        help="Maximale Anzahl der Features für TF-IDF (Standard: 1000)")
    parser.add_argument("--test-size", type=float, default=0.2, 
                        help="Anteil der Testdaten (Standard: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, 
                        help="Seed für die Reproduzierbarkeit (Standard: 42)")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_LOGREG_CONFIG['max_iter'], 
                        help=f"Maximale Anzahl von Iterationen (Standard: {DEFAULT_LOGREG_CONFIG['max_iter']})")
    parser.add_argument("--c", type=float, default=DEFAULT_LOGREG_CONFIG['C'], 
                        help=f"Regularisierungsparameter (Standard: {DEFAULT_LOGREG_CONFIG['C']})")
    parser.add_argument("--solver", default=DEFAULT_LOGREG_CONFIG['solver'], 
                        choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        help=f"Algorithmus für die Optimierung (Standard: {DEFAULT_LOGREG_CONFIG['solver']})")
    parser.add_argument("--cv-folds", type=int, default=5, 
                        help="Anzahl der Folds für die Kreuzvalidierung (Standard: 5)")
    parser.add_argument("--no-save", action="store_true", 
                        help="Modell nicht speichern")
    parser.add_argument("--version", 
                        help="Versionsbezeichnung für das Modell")
    parser.add_argument("--no-overwrite", action="store_true", 
                        help="Vorhandene Modelldateien nicht überschreiben")
    parser.add_argument("--multi-target", action="store_true", 
                        help="Trainiere Modelle für mehrere Zielspalten")
    parser.add_argument("--verbose", action="store_true", 
                        help="Ausführliche Ausgabe")

    args = parser.parse_args()

    # Logging-Level anpassen, falls verbose aktiviert ist
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug-Modus aktiviert")

    try:
        # Wenn multi-target aktiviert ist, trainiere Modelle für mehrere Zielspalten
        if args.multi_target:
            logger.info("Trainiere Modelle für mehrere Zielspalten")

            # Verfügbare Zielspalten ermitteln
            from suppervisedlearningproject.core.data_loader import get_available_targets
            targets = get_available_targets()

            # Ergebnisse für jede Zielspalte speichern
            results_dict = {}

            # Für jede verfügbare Zielspalte ein Modell trainieren
            for target_column in targets.keys():
                logger.info(f"\n\n{'=' * 50}")
                logger.info(f"Training für Zielspalte: {target_column}")
                logger.info(f"{'=' * 50}")

                try:
                    # Training des Modells mit den angegebenen Parametern
                    results = train_and_save_model(
                        target_column=target_column,
                        max_features=args.max_features,
                        test_size=args.test_size,
                        random_state=args.random_state,
                        max_iter=args.max_iter,
                        C=args.c,
                        solver=args.solver,
                        cv_folds=args.cv_folds,
                        save_model_flag=not args.no_save,
                        version=args.version,
                        overwrite=not args.no_overwrite
                    )

                    # Ergebnisse speichern
                    results_dict[target_column] = {
                        'accuracy': results.accuracy,
                        'precision': results.precision,
                        'recall': results.recall,
                        'f1': results.f1,
                        'cross_val_mean': float(results.cross_val_scores.mean()) if results.cross_val_scores is not None else None
                    }

                except Exception as target_error:
                    logger.error(f"Fehler beim Training für Zielspalte '{target_column}': {str(target_error)}")
                    results_dict[target_column] = {'error': str(target_error)}

            # Zusammenfassung der Ergebnisse
            logger.info("\n\n" + "=" * 50)
            logger.info("Zusammenfassung der Trainingsergebnisse:")
            logger.info("=" * 50)

            for target, result in results_dict.items():
                if 'error' in result:
                    logger.info(f"Zielspalte '{target}': Fehlgeschlagen - {result['error']}")
                else:
                    logger.info(f"Zielspalte '{target}':")
                    logger.info(f"  Accuracy: {result['accuracy']:.4f}")
                    logger.info(f"  F1 Score: {result['f1']:.4f}")
                    if result['cross_val_mean'] is not None:
                        logger.info(f"  CV Score: {result['cross_val_mean']:.4f}")

        else:
            # Training eines einzelnen Modells mit den angegebenen Parametern
            logger.info(f"Trainiere Modell für Zielspalte '{args.target}'")

            results = train_and_save_model(
                target_column=args.target,
                max_features=args.max_features,
                test_size=args.test_size,
                random_state=args.random_state,
                max_iter=args.max_iter,
                C=args.c,
                solver=args.solver,
                cv_folds=args.cv_folds,
                save_model_flag=not args.no_save,
                version=args.version,
                overwrite=not args.no_overwrite
            )

            # Ausgabe der Ergebnisse
            logger.info("\nZusammenfassung der Ergebnisse:")
            logger.info(f"Accuracy: {results.accuracy:.4f}")
            logger.info(f"Precision: {results.precision:.4f}")
            logger.info(f"Recall: {results.recall:.4f}")
            logger.info(f"F1 Score: {results.f1:.4f}")
            if results.cross_val_scores is not None:
                logger.info(f"CV Score: {results.cross_val_scores.mean():.4f} (±{results.cross_val_scores.std():.4f})")

            logger.info("\nTraining erfolgreich abgeschlossen!")

    except Exception as e:
        # Fehlerbehandlung bei Problemen während der Ausführung
        logger.error(f"Fehler: {str(e)}", exc_info=True)
        sys.exit(1)

    sys.exit(0)
