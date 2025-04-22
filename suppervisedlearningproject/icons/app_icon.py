"""
Modul zur Generierung eines KI-Icons für die Textklassifikationsanwendung.

Dieses Modul erstellt ein Icon mit einem KI/ML-Thema für die Anwendung.
"""

from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QPen, QFont, QLinearGradient
from PyQt5.QtCore import Qt, QSize, QRect, QPoint

def create_app_icon():
    """
    Erstellt ein KI-generiertes Icon für die Textklassifikationsanwendung.
    
    Returns:
        QIcon: Das erstellte Icon
    """
    # Erstelle ein Pixmap für das Icon
    size = 128
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    # Erstelle einen Painter für das Pixmap
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    
    # Hintergrund mit Farbverlauf
    gradient = QLinearGradient(0, 0, size, size)
    gradient.setColorAt(0, QColor(0, 150, 136, 240))  # Türkis (oben links)
    gradient.setColorAt(1, QColor(0, 105, 92, 240))   # Dunkleres Türkis (unten rechts)
    painter.setBrush(QBrush(gradient))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(0, 0, size, size, 20, 20)
    
    # Zeichne ein Gehirn-Symbol (stilisiert)
    painter.setPen(QPen(QColor(255, 255, 255, 200), 3))
    
    # Gehirn-Umriss
    brain_rect = QRect(20, 25, size - 40, size - 50)
    painter.drawEllipse(brain_rect)
    
    # Gehirn-Details (Windungen)
    painter.drawArc(30, 35, 40, 40, 0 * 16, 180 * 16)
    painter.drawArc(60, 35, 40, 40, 0 * 16, 180 * 16)
    painter.drawArc(30, 65, 40, 40, 180 * 16, 180 * 16)
    painter.drawArc(60, 65, 40, 40, 180 * 16, 180 * 16)
    
    # Verbindungslinien (neuronales Netz)
    painter.setPen(QPen(QColor(255, 255, 255, 150), 2))
    
    # Punkte für das neuronale Netz
    points = [
        QPoint(30, 40), QPoint(50, 60), QPoint(70, 40), QPoint(90, 60),
        QPoint(30, 80), QPoint(50, 100), QPoint(70, 80), QPoint(90, 100)
    ]
    
    # Verbinde die Punkte
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Nicht alle Punkte verbinden, um ein übersichtlicheres Bild zu erhalten
            if (i + j) % 3 == 0:
                painter.drawLine(points[i], points[j])
    
    # Zeichne die Punkte
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    for point in points:
        painter.drawEllipse(point, 3, 3)
    
    # Text "AI" in der Mitte
    painter.setFont(QFont("Arial", 24, QFont.Bold))
    painter.setPen(QPen(QColor(255, 255, 255), 2))
    painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, "AI")
    
    # Text "Text" am unteren Rand
    painter.setFont(QFont("Arial", 14, QFont.Bold))
    painter.drawText(QRect(0, size - 30, size, 20), Qt.AlignCenter, "Text")
    
    # Beende den Painter
    painter.end()
    
    # Erstelle ein QIcon aus dem Pixmap
    return QIcon(pixmap)

def get_app_icon():
    """
    Gibt das App-Icon zurück.
    
    Returns:
        QIcon: Das App-Icon
    """
    return create_app_icon()