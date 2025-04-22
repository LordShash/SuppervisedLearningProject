#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul zum Training eines logistischen Regressionsmodells.

Dieses Modul stellt Funktionen bereit, um ein logistisches Regressionsmodell
zu trainieren, zu evaluieren und zu speichern.
"""

import os
import sys
from typing import Tuple, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Importiere die Datenladefunktion aus dem data_loader Modul
from data_loader import load_data


def train_logistic_regression(X: np.ndarray, y: np.ndarray, target_name: str = "Fits_Topic_Code",
                             test_size: float = 0.2, random_state: int = 42, 
                             max_iter: int = 1000, C: float = 1.0, solver: str = 'lbfgs') -> Tuple[Pipeline, float, float, float, float, str, np.ndarray]:
    """
    Trainiert ein logistisches Regressionsmodell mit Standardskalierung.

    Args:
        X: Feature-Matrix
        y: Zielvariable
        target_name: Name der Zielvariable für Reporting
        test_size: Anteil der Testdaten
        random_state: Seed für die Reproduzierbarkeit
        max_iter: Maximale Anzahl von Iterationen für die logistische Regression
        C: Regularisierungsparameter (kleinere Werte bedeuten stärkere Regularisierung)
        solver: Algorithmus für die Optimierung ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')

    Returns:
        Tuple[Pipeline, float, float, float, float, str, np.ndarray]: Trainiertes Modell, Genauigkeit, Precision, Recall, F1-Score auf dem Testset, Klassifikationsbericht und Konfusionsmatrix

    Raises:
        ValueError: Wenn Probleme beim Training des Modells auftreten
    """
    try:
        # Aufteilung der Daten in Trainings- und Testsets mit festgelegtem Zufallsseed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Erstellung einer Pipeline mit Skalierung und logistischer Regression
        # Hinweis: Bei Sparse-Matrizen (TF-IDF) muss with_mean=False gesetzt werden
        pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('logreg', LogisticRegression(
                random_state=random_state, 
                max_iter=max_iter,
                C=C,
                solver=solver
            ))
        ])

        # Training des Modells mit den Trainingsdaten
        pipeline.fit(X_train, y_train)

        # Generierung von Vorhersagen auf dem Testdatensatz
        y_pred = pipeline.predict(X_test)

        # Berechnung der Genauigkeit (Accuracy) des Modells
        accuracy = accuracy_score(y_test, y_pred)

        # Berechnung von Precision, Recall und F1-Score
        # Bei Multiclass-Klassifikation verwenden wir 'weighted' average
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Erstellung eines detaillierten Klassifikationsberichts mit Präzision, Recall und F1-Score
        report = classification_report(y_test, y_pred, target_names=[f"{target_name}_{i}" for i in sorted(set(y))])

        # Erstellung der Konfusionsmatrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Rückgabe des trainierten Modells, der Metriken, des Berichts und der Konfusionsmatrix
        return pipeline, accuracy, precision, recall, f1, report, conf_matrix

    except Exception as e:
        # Fehlerbehandlung bei Problemen während des Trainings
        raise ValueError(f"Fehler beim Training des logistischen Regressionsmodells: {str(e)}")


def save_model(model: Pipeline, target_column: str = "Fits_Topic_Code") -> str:
    """
    Speichert das trainierte Modell als PKL-Datei.

    Args:
        model: Trainiertes Modell
        target_column: Name der Zielspalte für den Dateinamen

    Returns:
        str: Pfad zur gespeicherten Modelldatei

    Raises:
        ValueError: Wenn Probleme beim Speichern des Modells auftreten
    """
    # Ermittlung des Pfads zum models-Verzeichnis relativ zum aktuellen Skript
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    # Erstellung des Verzeichnisses, falls es noch nicht existiert
    os.makedirs(models_dir, exist_ok=True)

    # Generierung des Dateinamens basierend auf der Zielspalte
    filename = f"logreg_{target_column}_model.pkl"
    model_path = os.path.join(models_dir, filename)

    try:
        # Speicherung des Modells im Pickle-Format mit joblib
        joblib.dump(model, model_path)
        # Ausgabe einer Erfolgsmeldung mit dem Speicherpfad
        print(f"Modell erfolgreich gespeichert unter: {model_path}")
        return model_path

    except Exception as e:
        # Fehlerbehandlung bei Problemen mit der Speicherung
        raise ValueError(f"Fehler beim Speichern des Modells: {str(e)}")


def train_and_save_model(target_column: str = "Fits_Topic_Code", 
                      max_features: int = 1000,
                      test_size: float = 0.2, 
                      random_state: int = 42,
                      max_iter: int = 1000, 
                      C: float = 1.0, 
                      solver: str = 'lbfgs') -> Tuple[float, float, float, float, str, np.ndarray]:
    """
    Lädt Daten, trainiert ein Modell und speichert es.

    Args:
        target_column: Name der Zielspalte
        max_features: Maximale Anzahl der Features für TF-IDF
        test_size: Anteil der Testdaten
        random_state: Seed für die Reproduzierbarkeit
        max_iter: Maximale Anzahl von Iterationen für die logistische Regression
        C: Regularisierungsparameter (kleinere Werte bedeuten stärkere Regularisierung)
        solver: Algorithmus für die Optimierung ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')

    Returns:
        Tuple[float, float, float, float, str, np.ndarray]: Genauigkeit, Precision, Recall, F1-Score, Klassifikationsbericht und Konfusionsmatrix

    Raises:
        FileNotFoundError: Wenn keine Datendateien gefunden wurden
        ValueError: Wenn Probleme beim Laden, Verarbeiten oder Speichern der Daten auftreten
        KeyError: Wenn die angegebene Zielspalte nicht im DataFrame vorhanden ist
    """
    try:
        # Laden der Daten mit der angegebenen Zielspalte und max_features
        X, y = load_data(target_column=target_column, max_features=max_features)

        # Training des logistischen Regressionsmodells mit den geladenen Daten und konfigurierbaren Hyperparametern
        model, accuracy, precision, recall, f1, report, conf_matrix = train_logistic_regression(
            X, y, 
            target_name=target_column,
            test_size=test_size,
            random_state=random_state,
            max_iter=max_iter,
            C=C,
            solver=solver
        )

        # Speicherung des trainierten Modells für spätere Verwendung
        model_path = save_model(model, target_column)
        print(f"Modell gespeichert unter: {model_path}")

        # Ausgabe der Konfusionsmatrix
        print("\nKonfusionsmatrix:")
        print(conf_matrix)

        # Rückgabe der Metriken, des Klassifikationsberichts und der Konfusionsmatrix
        return accuracy, precision, recall, f1, report, conf_matrix

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Spezifische Fehler weiterleiten
        raise
    except Exception as e:
        # Unerwartete Fehler in ValueError umwandeln
        raise ValueError(f"Unerwarteter Fehler beim Training und Speichern des Modells: {str(e)}")


if __name__ == "__main__":
    # Ausführung des Skripts als eigenständiges Programm
    try:
        # Training eines Modells mit der Standard-Zielspalte (Fits_Topic_Code)
        accuracy, precision, recall, f1, report, conf_matrix = train_and_save_model()

        # Ausgabe der Ergebnisse des ersten Modelltrainings
        print(f"Accuracy auf dem Test-Set: {accuracy:.4f}")
        print(f"Precision auf dem Test-Set: {precision:.4f}")
        print(f"Recall auf dem Test-Set: {recall:.4f}")
        print(f"F1 Score auf dem Test-Set: {f1:.4f}")
        print("\nKlassifikationsbericht:")
        print(report)

        # Demonstration der Flexibilität: Training mit einer alternativen Zielspalte
        print("\n--- Training mit Zielspalte 'Urgency_Code' ---")
        accuracy_urgency, precision_urgency, recall_urgency, f1_urgency, report_urgency, conf_matrix_urgency = train_and_save_model(target_column="Urgency_Code")

        # Ausgabe der Ergebnisse des zweiten Modelltrainings
        print(f"Accuracy auf dem Test-Set: {accuracy_urgency:.4f}")
        print(f"Precision auf dem Test-Set: {precision_urgency:.4f}")
        print(f"Recall auf dem Test-Set: {recall_urgency:.4f}")
        print(f"F1 Score auf dem Test-Set: {f1_urgency:.4f}")
        print("\nKlassifikationsbericht:")
        print(report_urgency)

    except SystemExit as e:
        # Ausgabe von Fehlermeldungen, die während des Trainings aufgetreten sind
        print(e)
