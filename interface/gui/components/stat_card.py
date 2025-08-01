# interface/gui/components/stat_card.py

import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QGraphicsDropShadowEffect, QSizePolicy, QSpacerItem, 
    QStyle, QStyleOption, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont, QFontMetrics, 
    QLinearGradient, QPainterPath, QPolygonF, QRegion, QBitmap,
    QIcon, QPixmap, QPainterPath, QTransform
)

from interface.gui.components.glass_frame import GlassFrame, GlassTheme


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


@dataclass
class StatCardTheme:
    name: str
    background_color: QColor
    border_color: QColor
    title_color: QColor
    value_color: QColor
    subtitle_color: QColor
    positive_trend_color: QColor
    negative_trend_color: QColor
    flat_trend_color: QColor
    shadow_color: QColor
    blur_radius: int = 10
    opacity: float = 0.85
    corner_radius: int = 12
    border_width: int = 1


class StatCard(GlassFrame):
    """
    A customizable statistics card component for PyQt5 applications.
    Displays a statistic with title, value, and trend information.
    """
    
    # Signals
    clicked = pyqtSignal()
    value_changed = pyqtSignal(float, float)  # old_value, new_value
    
    def __init__(self, parent=None, theme: Optional[StatCardTheme] = None):
        # Initialize with a base glass frame
        super().__init__(parent)
        
        # Set up logging
        self.logger = logging.getLogger("stat_card")
        self.logger.setLevel(logging.INFO)
        
        # Default theme if none provided
        if theme is None:
            self.theme = self._get_default_theme()
        else:
            self.theme = theme
        
        # Card properties
        self._title = ""
        self._value = 0.0
        self._subtitle = ""
        self._trend_direction = TrendDirection.FLAT
        self._trend_percentage = 0.0
        self._icon_path = None
        self._format_string = "{:.0f}"
        self._show_trend = True
        self._animate_changes = True
        
        # Create UI elements
        self._setup_ui()
        
        # Apply theme
        self._apply_theme()
        
        # Set up animations
        self._setup_animations()
        
        # Initialize state
        self._previous_value = self._value
    
    def _get_default_theme(self) -> StatCardTheme:
        """Get the default stat card theme"""
        return StatCardTheme(
            name="Default Stat Card",
            background_color=QColor(255, 255, 255, 200),
            border_color=QColor(255, 255, 255, 100),
            title_color=QColor(100, 100, 100),
            value_color=QColor(50, 50, 50),
            subtitle_color=QColor(150, 150, 150),
            positive_trend_color=QColor(46, 204, 113),
            negative_trend_color=QColor(231, 76, 60),
            flat_trend_color=QColor(150, 150, 150),
            shadow_color=QColor(0, 0, 0, 50),
            blur_radius=10,
            opacity=0.85,
            corner_radius=12,
            border_width=1
        )
    
    def _setup_ui(self):
        """Set up the UI elements of the stat card"""
        # Main layout for content
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(5)
        
        # Top layout for title and icon
        self.top_layout = QHBoxLayout()
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title label
        self.title_label = QLabel()
        self.title_label.setObjectName("statCardTitle")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Icon label
        self.icon_label = QLabel()
        self.icon_label.setObjectName("statCardIcon")
        self.icon_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.icon_label.setFixedSize(24, 24)
        
        # Add to top layout
        self.top_layout.addWidget(self.title_label)
        self.top_layout.addStretch()
        self.top_layout.addWidget(self.icon_label)
        
        # Value label
        self.value_label = QLabel()
        self.value_label.setObjectName("statCardValue")
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.value_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        # Bottom layout for subtitle and trend
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Subtitle label
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("statCardSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Trend layout
        self.trend_layout = QHBoxLayout()
        self.trend_layout.setContentsMargins(0, 0, 0, 0)
        self.trend_layout.setSpacing(5)
        
        # Trend icon
        self.trend_icon = QLabel()
        self.trend_icon.setObjectName("statCardTrendIcon")
        self.trend_icon.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.trend_icon.setFixedSize(16, 16)
        
        # Trend percentage
        self.trend_percentage_label = QLabel()
        self.trend_percentage_label.setObjectName("statCardTrendPercentage")
        self.trend_percentage_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Add to trend layout
        self.trend_layout.addWidget(self.trend_icon)
        self.trend_layout.addWidget(self.trend_percentage_label)
        
        # Add to bottom layout
        self.bottom_layout.addWidget(self.subtitle_label)
        self.bottom_layout.addStretch()
        self.bottom_layout.addLayout(self.trend_layout)
        
        # Add all layouts to content
        self.content_layout.addLayout(self.top_layout)
        self.content_layout.addWidget(self.value_label)
        self.content_layout.addLayout(self.bottom_layout)
        
        # Add content to the frame
        self.add_content(self.content_widget)
        
        # Set initial visibility
        self.trend_layout.setVisible(self._show_trend)
    
    def _apply_theme(self):
        """Apply the current theme to the stat card"""
        # Update base glass frame theme
        base_theme = GlassTheme(
            name=self.theme.name,
            background_color=self.theme.background_color,
            border_color=self.theme.border_color,
            text_color=self.theme.title_color,
            shadow_color=self.theme.shadow_color,
            blur_radius=self.theme.blur_radius,
            opacity=self.theme.opacity,
            corner_radius=self.theme.corner_radius,
            border_width=self.theme.border_width
        )
        super().set_theme(base_theme)
        
        # Create stylesheet for stat card specific elements
        stylesheet = f"""
        QLabel#statCardTitle {{
            color: {self.theme.title_color.name()};
            font-size: 14px;
            font-weight: 500;
        }}
        
        QLabel#statCardValue {{
            color: {self.theme.value_color.name()};
            font-size: 24px;
            font-weight: bold;
        }}
        
        QLabel#statCardSubtitle {{
            color: {self.theme.subtitle_color.name()};
            font-size: 12px;
        }}
        
        QLabel#statCardTrendIcon {{
            color: {self.theme.flat_trend_color.name()};
        }}
        
        QLabel#statCardTrendPercentage {{
            color: {self.theme.flat_trend_color.name()};
            font-size: 12px;
        }}
        """
        
        self.setStyleSheet(stylesheet)
        
        # Update trend colors based on direction
        self._update_trend_colors()
    
    def _update_trend_colors(self):
        """Update the trend colors based on the current trend direction"""
        if self._trend_direction == TrendDirection.UP:
            trend_color = self.theme.positive_trend_color
        elif self._trend_direction == TrendDirection.DOWN:
            trend_color = self.theme.negative_trend_color
        else:  # FLAT
            trend_color = self.theme.flat_trend_color
        
        # Update trend icon and percentage colors
        self.trend_icon.setStyleSheet(f"color: {trend_color.name()};")
        self.trend_percentage_label.setStyleSheet(f"color: {trend_color.name()};")
    
    def _setup_animations(self):
        """Set up animations for the stat card"""
        # Value change animation
        self.value_animation = QPropertyAnimation(self, b"byteArray")
        self.value_animation.setDuration(500)
        self.value_animation.setEasingCurve(QEasingCurve.OutQuad)
    
    def set_title(self, title: str):
        """Set the title text"""
        self._title = title
        self.title_label.setText(title)
        self.title_label.setVisible(bool(title))
    
    def set_value(self, value: float, animate: bool = None):
        """Set the value and optionally animate the change"""
        if animate is None:
            animate = self._animate_changes
        
        if animate and self._value != value:
            # Store previous value for signal emission
            self._previous_value = self._value
            
            # Animate the change
            self._animate_value_change(self._value, value)
        else:
            # Set the value directly
            self._value = value
            self._update_value_display()
    
    def _animate_value_change(self, old_value: float, new_value: float):
        """Animate the value change from old_value to new_value"""
        # Create a custom property for animation
        self.setProperty("value", old_value)
        
        # Set up the animation
        self.value_animation.setStartValue(old_value)
        self.value_animation.setEndValue(new_value)
        
        # Connect the animation to update the display
        self.value_animation.valueChanged.connect(self._on_animation_value_changed)
        self.value_animation.finished.connect(self._on_animation_finished)
        
        # Start the animation
        self.value_animation.start()
    
    def _on_animation_value_changed(self, value):
        """Handle animation value change"""
        self.setProperty("value", value)
        self._update_value_display()
    
    def _on_animation_finished(self):
        """Handle animation finished"""
        # Emit value changed signal
        self.value_changed.emit(self._previous_value, self._value)
    
    def _update_value_display(self):
        """Update the value label display"""
        formatted_value = self._format_string.format(self._value)
        self.value_label.setText(formatted_value)
    
    def set_subtitle(self, subtitle: str):
        """Set the subtitle text"""
        self._subtitle = subtitle
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))
    
    def set_trend(self, direction: TrendDirection, percentage: float):
        """Set the trend direction and percentage"""
        self._trend_direction = direction
        self._trend_percentage = percentage
        self._update_trend_display()
    
    def _update_trend_display(self):
        """Update the trend display"""
        if not self._show_trend:
            return
        
        # Update trend colors
        self._update_trend_colors()
        
        # Set trend icon
        if self._trend_direction == TrendDirection.UP:
            self.trend_icon.setText("▲")
        elif self._trend_direction == TrendDirection.DOWN:
            self.trend_icon.setText("▼")
        else:  # FLAT
            self.trend_icon.setText("●")
        
        # Set trend percentage
        if self._trend_percentage >= 0:
            trend_text = f"+{self._trend_percentage:.1f}%"
        else:
            trend_text = f"{self._trend_percentage:.1f}%"
        
        self.trend_percentage_label.setText(trend_text)
    
    def set_icon(self, icon_path: str):
        """Set the icon from a file path"""
        self._icon_path = icon_path
        pixmap = QPixmap(icon_path)
        if not pixmap.isNull():
            # Scale pixmap to fit
            scaled_pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)
            self.icon_label.setVisible(True)
        else:
            self.icon_label.setVisible(False)
    
    def set_icon_from_resource(self, resource_name: str):
        """Set the icon from a Qt resource"""
        pixmap = QPixmap(f":/icons/{resource_name}")
        if not pixmap.isNull():
            # Scale pixmap to fit
            scaled_pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)
            self.icon_label.setVisible(True)
        else:
            self.icon_label.setVisible(False)
    
    def set_format_string(self, format_string: str):
        """Set the format string for the value display"""
        self._format_string = format_string
        self._update_value_display()
    
    def show_trend(self, show: bool):
        """Show or hide the trend display"""
        self._show_trend = show
        self.trend_layout.setVisible(show)
    
    def enable_animation(self, enable: bool):
        """Enable or disable value change animations"""
        self._animate_changes = enable
    
    def get_value(self) -> float:
        """Get the current value"""
        return self._value
    
    def get_title(self) -> str:
        """Get the current title"""
        return self._title
    
    def get_subtitle(self) -> str:
        """Get the current subtitle"""
        return self._subtitle
    
    def get_trend_direction(self) -> TrendDirection:
        """Get the current trend direction"""
        return self._trend_direction
    
    def get_trend_percentage(self) -> float:
        """Get the current trend percentage"""
        return self._trend_percentage
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the stat card"""
        return {
            "theme": {
                "name": self.theme.name,
                "background_color": self.theme.background_color.name(),
                "border_color": self.theme.border_color.name(),
                "title_color": self.theme.title_color.name(),
                "value_color": self.theme.value_color.name(),
                "subtitle_color": self.theme.subtitle_color.name(),
                "positive_trend_color": self.theme.positive_trend_color.name(),
                "negative_trend_color": self.theme.negative_trend_color.name(),
                "flat_trend_color": self.theme.flat_trend_color.name(),
                "shadow_color": self.theme.shadow_color.name(),
                "blur_radius": self.theme.blur_radius,
                "opacity": self.theme.opacity,
                "corner_radius": self.theme.corner_radius,
                "border_width": self.theme.border_width
            },
            "title": self._title,
            "value": self._value,
            "subtitle": self._subtitle,
            "trend_direction": self._trend_direction.value,
            "trend_percentage": self._trend_percentage,
            "icon_path": self._icon_path,
            "format_string": self._format_string,
            "show_trend": self._show_trend,
            "animate_changes": self._animate_changes,
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
            theme = StatCardTheme(
                name=theme_data.get("name", "Custom"),
                background_color=QColor(theme_data.get("background_color", "#FFFFFF")),
                border_color=QColor(theme_data.get("border_color", "#FFFFFF")),
                title_color=QColor(theme_data.get("title_color", "#646464")),
                value_color=QColor(theme_data.get("value_color", "#323232")),
                subtitle_color=QColor(theme_data.get("subtitle_color", "#969696")),
                positive_trend_color=QColor(theme_data.get("positive_trend_color", "#2ecc71")),
                negative_trend_color=QColor(theme_data.get("negative_trend_color", "#e74c3c")),
                flat_trend_color=QColor(theme_data.get("flat_trend_color", "#969696")),
                shadow_color=QColor(theme_data.get("shadow_color", "#000000")),
                blur_radius=theme_data.get("blur_radius", 10),
                opacity=theme_data.get("opacity", 0.85),
                corner_radius=theme_data.get("corner_radius", 12),
                border_width=theme_data.get("border_width", 1)
            )
            self.set_theme(theme)
        
        # Load data
        if "title" in state:
            self.set_title(state["title"])
        if "value" in state:
            self.set_value(state["value"], animate=False)
        if "subtitle" in state:
            self.set_subtitle(state["subtitle"])
        if "trend_direction" in state and "trend_percentage" in state:
            self.set_trend(
                TrendDirection(state["trend_direction"]),
                state["trend_percentage"]
            )
        if "icon_path" in state and state["icon_path"]:
            self.set_icon(state["icon_path"])
        if "format_string" in state:
            self.set_format_string(state["format_string"])
        if "show_trend" in state:
            self.show_trend(state["show_trend"])
        if "animate_changes" in state:
            self.enable_animation(state["animate_changes"])
        
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
            "title_color": self.theme.title_color.name(),
            "value_color": self.theme.value_color.name(),
            "subtitle_color": self.theme.subtitle_color.name(),
            "positive_trend_color": self.theme.positive_trend_color.name(),
            "negative_trend_color": self.theme.negative_trend_color.name(),
            "flat_trend_color": self.theme.flat_trend_color.name(),
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
            
            theme = StatCardTheme(
                name=theme_data.get("name", "Imported"),
                background_color=QColor(theme_data.get("background_color", "#FFFFFF")),
                border_color=QColor(theme_data.get("border_color", "#FFFFFF")),
                title_color=QColor(theme_data.get("title_color", "#646464")),
                value_color=QColor(theme_data.get("value_color", "#323232")),
                subtitle_color=QColor(theme_data.get("subtitle_color", "#969696")),
                positive_trend_color=QColor(theme_data.get("positive_trend_color", "#2ecc71")),
                negative_trend_color=QColor(theme_data.get("negative_trend_color", "#e74c3c")),
                flat_trend_color=QColor(theme_data.get("flat_trend_color", "#969696")),
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
    
    def enterEvent(self, event):
        """Handle mouse enter event"""
        super().enterEvent(event)
        # Add subtle hover effect
        self.set_opacity(min(1.0, self.theme.opacity + 0.1))
    
    def leaveEvent(self, event):
        """Handle mouse leave event"""
        super().leaveEvent(event)
        # Restore original opacity
        self.set_opacity(self.theme.opacity)
