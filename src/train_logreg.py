#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul zum Training eines logistischen Regressionsmodells für die Textklassifikation.

Dieses Modul stellt Funktionen bereit, um ein logistisches Regressionsmodell
zu trainieren, zu evaluieren und zu speichern.
"""

import os
import sys
import logging
import datetime
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

# Importiere die Datenladefunktion aus dem data_loader Modul
from data_loader import load_data

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ausgabe in die Konsole
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'train_logreg.log'), 
                           mode='a', encoding='utf-8')  # Ausgabe in eine Datei
    ]
)
logger = logging.getLogger(__name__)

# Stelle sicher, dass das Logs-Verzeichnis existiert
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)

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

        # Visualisierung der Konfusionsmatrix (optional)
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Konfusionsmatrix für {target_name}')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            plt.xlabel('Vorhergesagte Klasse')
            plt.ylabel('Tatsächliche Klasse')
            plt.tight_layout()

            # Speichere die Visualisierung
            plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{target_name}_{timestamp}.png'))
            logger.info(f"Konfusionsmatrix-Visualisierung gespeichert")
        except Exception as viz_error:
            logger.warning(f"Konnte Konfusionsmatrix nicht visualisieren: {str(viz_error)}")

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
            cross_val_scores=cv_scores
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
    # Ermittlung des Pfads zum models-Verzeichnis relativ zum aktuellen Skript
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    # Erstellung des Verzeichnisses, falls es noch nicht existiert
    os.makedirs(models_dir, exist_ok=True)

    # Generierung des Dateinamens basierend auf der Zielspalte und optionalen Parametern
    timestamp = ""
    if include_timestamp:
        timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    version_str = ""
    if version:
        version_str = f"_{version}"

    filename = f"logreg_{target_column}{version_str}{timestamp}_model.pkl"
    model_path = os.path.join(models_dir, filename)

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
    max_iter: int = 1000, 
    C: float = 1.0, 
    solver: str = 'lbfgs',
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
    parser.add_argument("--max-iter", type=int, default=1000, 
                        help="Maximale Anzahl von Iterationen (Standard: 1000)")
    parser.add_argument("--c", type=float, default=1.0, 
                        help="Regularisierungsparameter (Standard: 1.0)")
    parser.add_argument("--solver", default="lbfgs", choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        help="Algorithmus für die Optimierung (Standard: lbfgs)")
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
            from data_loader import get_available_targets
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
