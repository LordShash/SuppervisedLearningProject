# Hauptmodul - Einfache Erklärung

## Zweck

Das Hauptmodul (`main.py`) ist das Steuerungszentrum des Projekts. Es ermöglicht Ihnen, verschiedene Textklassifikationsmodelle zu trainieren und zu vergleichen, ohne dass Sie sich mit den technischen Details der einzelnen Modelle befassen müssen.

## Ablauf

1. **Auswahl der Optionen**: 
   - Sie können wählen, welches Modell trainiert werden soll (logistische Regression, neuronales Netz oder beide).
   - Sie können die Zielvariable auswählen (z.B. Thema, Dringlichkeit oder Kategorie).
   - Sie können die Anzahl der Trainingsepochen für neuronale Netze festlegen.

2. **Datenanalyse**:
   - Das Modul zeigt Ihnen Informationen über die verfügbaren Zielvariablen und deren Werte an.
   - So bekommen Sie einen Überblick über die Daten, bevor das Training beginnt.

3. **Modelltraining**:
   - Die ausgewählten Modelle werden mit den Daten trainiert.
   - Während des Trainings werden Fortschrittsinformationen angezeigt.
   - Nach dem Training werden die Genauigkeit und andere Leistungsmetriken angezeigt.

4. **Speicherung der Modelle**:
   - Die trainierten Modelle werden automatisch im `models`-Verzeichnis gespeichert.
   - Sie können später auf diese Modelle zugreifen, um neue Texte zu klassifizieren.

## Beispiel-Workflow

Stellen Sie sich vor, Sie möchten ein Modell trainieren, das die Dringlichkeit von Texten vorhersagt:

1. Sie starten das Programm mit spezifischen Optionen:
   ```
   python src/main.py --model logreg --target Urgency_Code
   ```

2. Das Programm zeigt Ihnen Informationen über die verfügbaren Zielvariablen an, einschließlich der verschiedenen Dringlichkeitsstufen und ihrer Häufigkeit in den Daten.

3. Das logistische Regressionsmodell wird trainiert, und Sie sehen die Genauigkeit des Modells.

4. Das trainierte Modell wird im `models`-Verzeichnis gespeichert, sodass Sie es später verwenden können.

Wenn Sie kein bestimmtes Modell oder keine bestimmte Zielvariable angeben, verwendet das Programm Standardwerte und trainiert alle verfügbaren Modelle mit allen verfügbaren Zielvariablen.

Das Hauptmodul macht es einfach, verschiedene Modelle und Zielvariablen auszuprobieren, ohne den Code ändern zu müssen.