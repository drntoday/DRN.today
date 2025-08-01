# interface/gui/components/glass_frame.py

import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QGraphicsDropShadowEffect, QSizePolicy, QSpacerItem, 
    QStyle, QStyleOption
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QTimer
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont, QFontMetrics, 
    QLinearGradient, QPainterPath, QPolygonF, QRegion, QBitmap
)


@dataclass
class GlassTheme:
    name: str
    background_color: QColor
    border_color: QColor
    text_color: QColor
    shadow_color: QColor
    blur_radius: int = 10
    opacity: float = 0.85
    corner_radius: int = 12
    border_width: int = 1


class GlassFrame(QFrame):
    """
    A customizable glassmorphism frame component for PyQt5 applications.
    Provides a modern glass-like appearance with blur, transparency, and shadow effects.
    """
    
    # Signals
    clicked = pyqtSignal()
    double_clicked = pyqtSignal()
    right_clicked = pyqtSignal(QPoint)
    
    def __init__(self, parent=None, theme: Optional[GlassTheme] = None):
        super().__init__(parent)
        
        # Set up logging
        self.logger = logging.getLogger("glass_frame")
        self.logger.setLevel(logging.INFO)
        
        # Default theme if none provided
        if theme is None:
            self.theme = self._get_default_theme()
        else:
            self.theme = theme
        
        # Frame properties
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFrameStyle(QFrame.NoFrame)
        
        # Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(10)
        
        # Header layout
        self.header_layout = QHBoxLayout()
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title label
        self.title_label = QLabel()
        self.title_label.setObjectName("glassFrameTitle")
        self.title_label.setStyleSheet(f"color: {self.theme.text_color.name()}; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Subtitle label
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("glassFrameSubtitle")
        self.subtitle_label.setStyleSheet(f"color: {self.theme.text_color.name()}; opacity: 180;")
        self.subtitle_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        
        # Footer layout
        self.footer_layout = QHBoxLayout()
        self.footer_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add components to layouts
        self.header_layout.addWidget(self.title_label)
        self.main_layout.addLayout(self.header_layout)
        self.main_layout.addWidget(self.subtitle_label)
        self.main_layout.addWidget(self.content_widget)
        self.main_layout.addLayout(self.footer_layout)
        
        # Apply initial styling
        self._apply_theme()
        
        # Set up shadow effect
        self._setup_shadow()
        
        # Set up animations
        self._setup_animations()
        
        # Track mouse position for hover effects
        self.mouse_pos = QPoint()
        self.hover_timer = QTimer(self)
        self.hover_timer.timeout.connect(self._update_hover_effect)
        self.hover_timer.start(50)
        
        # Initialize state
        self.is_hovered = False
        self.is_pressed = False
        
        # Minimum size
        self.setMinimumSize(300, 200)
        
        # Size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def _get_default_theme(self) -> GlassTheme:
        """Get the default glass theme"""
        return GlassTheme(
            name="Default Glass",
            background_color=QColor(255, 255, 255, 200),  # Semi-transparent white
            border_color=QColor(255, 255, 255, 100),
            text_color=QColor(50, 50, 50),
            shadow_color=QColor(0, 0, 0, 50),
            blur_radius=10,
            opacity=0.85,
            corner_radius=12,
            border_width=1
        )
    
    def _apply_theme(self):
        """Apply the current theme to the frame"""
        # Create stylesheet
        stylesheet = f"""
        GlassFrame {{
            background-color: rgba({self.theme.background_color.red()}, {self.theme.background_color.green()}, {self.theme.background_color.blue()}, {self.theme.background_color.alpha()});
            border: {self.theme.border_width}px solid rgba({self.theme.border_color.red()}, {self.theme.border_color.green()}, {self.theme.border_color.blue()}, {self.theme.border_color.alpha()});
            border-radius: {self.theme.corner_radius}px;
        }}
        
        QLabel#glassFrameTitle {{
            font-size: 16px;
            font-weight: bold;
            padding: 5px;
        }}
        
        QLabel#glassFrameSubtitle {{
            font-size: 12px;
            padding: 2px;
        }}
        """
        
        self.setStyleSheet(stylesheet)
    
    def _setup_shadow(self):
        """Set up the drop shadow effect"""
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(self.theme.shadow_color)
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)
    
    def _setup_animations(self):
        """Set up animations for hover and click effects"""
        # These will be implemented in the animation methods
        pass
    
    def set_title(self, title: str):
        """Set the title text"""
        self.title_label.setText(title)
        self.title_label.setVisible(bool(title))
    
    def set_subtitle(self, subtitle: str):
        """Set the subtitle text"""
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))
    
    def add_content(self, widget: QWidget):
        """Add a widget to the content area"""
        self.content_layout.addWidget(widget)
    
    def add_header_widget(self, widget: QWidget, alignment: Qt.AlignmentFlag = Qt.AlignRight):
        """Add a widget to the header area"""
        self.header_layout.addWidget(widget, alignment=alignment)
    
    def add_footer_widget(self, widget: QWidget, alignment: Qt.AlignmentFlag = Qt.AlignRight):
        """Add a widget to the footer area"""
        self.footer_layout.addWidget(widget, alignment=alignment)
    
    def add_spacer(self, layout_type: str = "content"):
        """Add a spacer to the specified layout"""
        if layout_type == "header":
            spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.header_layout.addItem(spacer)
        elif layout_type == "footer":
            spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.footer_layout.addItem(spacer)
        else:  # content
            spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
            self.content_layout.addItem(spacer)
    
    def clear_content(self):
        """Clear all widgets from the content area"""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def clear_header(self):
        """Clear all widgets from the header area except the title"""
        # Keep the title label, remove everything else
        while self.header_layout.count() > 1:
            item = self.header_layout.takeAt(1)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def clear_footer(self):
        """Clear all widgets from the footer area"""
        while self.footer_layout.count():
            item = self.footer_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def set_theme(self, theme: GlassTheme):
        """Set a new theme for the frame"""
        self.theme = theme
        self._apply_theme()
        self._setup_shadow()
    
    def paintEvent(self, event):
        """Custom paint event to create the glass effect"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get the rectangle of the widget
        rect = self.rect()
        
        # Create a path for the rounded rectangle
        path = QPainterPath()
        path.addRoundedRect(rect, self.theme.corner_radius, self.theme.corner_radius)
        
        # Set the clip region
        painter.setClipPath(path)
        
        # Create a semi-transparent background
        background_color = QColor(
            self.theme.background_color.red(),
            self.theme.background_color.green(),
            self.theme.background_color.blue(),
            int(255 * self.theme.opacity)
        )
        
        # Draw the background
        painter.fillRect(rect, background_color)
        
        # Draw the border
        if self.theme.border_width > 0:
            pen = QPen(
                QColor(
                    self.theme.border_color.red(),
                    self.theme.border_color.green(),
                    self.theme.border_color.blue(),
                    int(255 * self.theme.border_color.alpha())
                )
            )
            pen.setWidth(self.theme.border_width)
            painter.setPen(pen)
            painter.drawPath(path)
        
        # Add a subtle gradient overlay for depth
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor(255, 255, 255, 30))
        gradient.setColorAt(1, QColor(255, 255, 255, 0))
        painter.fillRect(rect, gradient)
        
        # Add a highlight effect on hover
        if self.is_hovered:
            highlight = QColor(255, 255, 255, 20)
            painter.fillRect(rect, highlight)
        
        # Add a pressed effect
        if self.is_pressed:
            pressed_overlay = QColor(0, 0, 0, 10)
            painter.fillRect(rect, pressed_overlay)
    
    def enterEvent(self, event):
        """Handle mouse enter event"""
        self.is_hovered = True
        self.setCursor(Qt.PointingHandCursor)
        self.update()
    
    def leaveEvent(self, event):
        """Handle mouse leave event"""
        self.is_hovered = False
        self.setCursor(Qt.ArrowCursor)
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press event"""
        if event.button() == Qt.LeftButton:
            self.is_pressed = True
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release event"""
        if event.button() == Qt.LeftButton:
            self.is_pressed = False
            self.update()
            self.clicked.emit()
    
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click event"""
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move event"""
        self.mouse_pos = event.pos()
    
    def contextMenuEvent(self, event):
        """Handle context menu event"""
        self.right_clicked.emit(event.pos())
    
    def _update_hover_effect(self):
        """Update the hover effect based on mouse position"""
        # This method is called by the hover timer
        # It can be used to create dynamic hover effects
        self.update()
    
    def resizeEvent(self, event):
        """Handle resize event"""
        super().resizeEvent(event)
        self.update()
    
    def set_corner_radius(self, radius: int):
        """Set the corner radius of the frame"""
        self.theme.corner_radius = radius
        self.update()
    
    def set_opacity(self, opacity: float):
        """Set the opacity of the frame"""
        self.theme.opacity = max(0.0, min(1.0, opacity))
        self.update()
    
    def set_blur_radius(self, radius: int):
        """Set the blur radius of the frame"""
        self.theme.blur_radius = radius
        self.update()
    
    def set_border_width(self, width: int):
        """Set the border width of the frame"""
        self.theme.border_width = width
        self.update()
    
    def enable_animation(self, enable: bool):
        """Enable or disable animations"""
        if enable:
            self.hover_timer.start()
        else:
            self.hover_timer.stop()
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the frame"""
        return {
            "theme": {
                "name": self.theme.name,
                "background_color": self.theme.background_color.name(),
                "border_color": self.theme.border_color.name(),
                "text_color": self.theme.text_color.name(),
                "shadow_color": self.theme.shadow_color.name(),
                "blur_radius": self.theme.blur_radius,
                "opacity": self.theme.opacity,
                "corner_radius": self.theme.corner_radius,
                "border_width": self.theme.border_width
            },
            "title": self.title_label.text(),
            "subtitle": self.subtitle_label.text(),
            "geometry": {
                "x": self.x(),
                "y": self.y(),
                "width": self.width(),
                "height": self.height()
            }
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load a previously saved state"""
        # Load theme
        if "theme" in state:
            theme_data = state["theme"]
            theme = GlassTheme(
                name=theme_data.get("name", "Custom"),
                background_color=QColor(theme_data.get("background_color", "#FFFFFF")),
                border_color=QColor(theme_data.get("border_color", "#FFFFFF")),
                text_color=QColor(theme_data.get("text_color", "#323232")),
                shadow_color=QColor(theme_data.get("shadow_color", "#000000")),
                blur_radius=theme_data.get("blur_radius", 10),
                opacity=theme_data.get("opacity", 0.85),
                corner_radius=theme_data.get("corner_radius", 12),
                border_width=theme_data.get("border_width", 1)
            )
            self.set_theme(theme)
        
        # Load text
        if "title" in state:
            self.set_title(state["title"])
        if "subtitle" in state:
            self.set_subtitle(state["subtitle"])
        
        # Load geometry
        if "geometry" in state:
            geometry = state["geometry"]
            self.setGeometry(
                geometry.get("x", self.x()),
                geometry.get("y", self.y()),
                geometry.get("width", self.width()),
                geometry.get("height", self.height())
            )
    
    def export_theme(self, file_path: str):
        """Export the current theme to a JSON file"""
        theme_data = {
            "name": self.theme.name,
            "background_color": self.theme.background_color.name(),
            "border_color": self.theme.border_color.name(),
            "text_color": self.theme.text_color.name(),
            "shadow_color": self.theme.shadow_color.name(),
            "blur_radius": self.theme.blur_radius,
            "opacity": self.theme.opacity,
            "corner_radius": self.theme.corner_radius,
            "border_width": self.theme.border_width
        }
        
        with open(file_path, 'w') as f:
            json.dump(theme_data, f, indent=2)
    
    def import_theme(self, file_path: str):
        """Import a theme from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                theme_data = json.load(f)
            
            theme = GlassTheme(
                name=theme_data.get("name", "Imported"),
                background_color=QColor(theme_data.get("background_color", "#FFFFFF")),
                border_color=QColor(theme_data.get("border_color", "#FFFFFF")),
                text_color=QColor(theme_data.get("text_color", "#323232")),
                shadow_color=QColor(theme_data.get("shadow_color", "#000000")),
                blur_radius=theme_data.get("blur_radius", 10),
                opacity=theme_data.get("opacity", 0.85),
                corner_radius=theme_data.get("corner_radius", 12),
                border_width=theme_data.get("border_width", 1)
            )
            
            self.set_theme(theme)
            return True
        except Exception as e:
            self.logger.error(f"Error importing theme: {str(e)}")
            return False
