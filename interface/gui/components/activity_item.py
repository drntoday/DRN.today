import json
import logging
from datetime import datetime, timedelta
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
    QIcon, QPixmap, QPainterPath, QTransform, QCursor
)

from interface.gui.components.glass_frame import GlassFrame, GlassTheme


class ActivityStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ActivityItemTheme:
    name: str
    background_color: QColor
    border_color: QColor
    title_color: QColor
    description_color: QColor
    timestamp_color: QColor
    status_colors: Dict[str, QColor]  # Mapping of status to color
    shadow_color: QColor
    blur_radius: int = 10
    opacity: float = 0.85
    corner_radius: int = 12
    border_width: int = 1


class ActivityItem(GlassFrame):
    """
    A customizable activity item component for PyQt5 applications.
    Displays an activity with icon, title, description, timestamp, and status.
    """
    
    # Signals
    clicked = pyqtSignal()
    double_clicked = pyqtSignal()
    right_clicked = pyqtSignal(QPoint)
    
    def __init__(self, parent=None, theme: Optional[ActivityItemTheme] = None):
        # Initialize with a base glass frame
        super().__init__(parent)
        
        # Set up logging
        self.logger = logging.getLogger("activity_item")
        self.logger.setLevel(logging.INFO)
        
        # Default theme if none provided
        if theme is None:
            self.theme = self._get_default_theme()
        else:
            self.theme = theme
        
        # Activity properties
        self._title = ""
        self._description = ""
        self._timestamp = datetime.now()
        self._status = ActivityStatus.PENDING
        self._icon_path = None
        self._show_timestamp = True
        self._show_status = True
        self._animate_changes = True
        self._is_unread = False
        
        # Create UI elements
        self._setup_ui()
        
        # Apply theme
        self._apply_theme()
        
        # Set up animations
        self._setup_animations()
        
        # Set cursor
        self.setCursor(Qt.PointingHandCursor)
    
    def _get_default_theme(self) -> ActivityItemTheme:
        """Get the default activity item theme"""
        return ActivityItemTheme(
            name="Default Activity Item",
            background_color=QColor(255, 255, 255, 200),
            border_color=QColor(255, 255, 255, 100),
            title_color=QColor(50, 50, 50),
            description_color=QColor(100, 100, 100),
            timestamp_color=QColor(150, 150, 150),
            status_colors={
                "pending": QColor(255, 193, 7),    # Orange
                "in_progress": QColor(41, 128, 185),  # Blue
                "completed": QColor(46, 204, 113),  # Green
                "failed": QColor(231, 76, 60),     # Red
                "cancelled": QColor(150, 150, 150)  # Gray
            },
            shadow_color=QColor(0, 0, 0, 50),
            blur_radius=10,
            opacity=0.85,
            corner_radius=12,
            border_width=1
        )
    
    def _setup_ui(self):
        """Set up the UI elements of the activity item"""
        # Main layout for content
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(15, 15, 15, 15)
        self.content_layout.setSpacing(15)
        
        # Icon label
        self.icon_label = QLabel()
        self.icon_label.setObjectName("activityItemIcon")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedSize(40, 40)
        
        # Text layout
        self.text_layout = QVBoxLayout()
        self.text_layout.setContentsMargins(0, 0, 0, 0)
        self.text_layout.setSpacing(5)
        
        # Title label
        self.title_label = QLabel()
        self.title_label.setObjectName("activityItemTitle")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setStyleSheet("font-weight: bold;")
        
        # Description label
        self.description_label = QLabel()
        self.description_label.setObjectName("activityItemDescription")
        self.description_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.description_label.setWordWrap(True)
        
        # Add to text layout
        self.text_layout.addWidget(self.title_label)
        self.text_layout.addWidget(self.description_label)
        
        # Right layout for timestamp and status
        self.right_layout = QVBoxLayout()
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(5)
        
        # Timestamp label
        self.timestamp_label = QLabel()
        self.timestamp_label.setObjectName("activityItemTimestamp")
        self.timestamp_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Status indicator
        self.status_indicator = QLabel()
        self.status_indicator.setObjectName("activityItemStatus")
        self.status_indicator.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_indicator.setFixedSize(12, 12)
        
        # Add to right layout
        self.right_layout.addWidget(self.timestamp_label)
        self.right_layout.addWidget(self.status_indicator)
        
        # Add all layouts to content
        self.content_layout.addWidget(self.icon_label)
        self.content_layout.addLayout(self.text_layout)
        self.content_layout.addStretch()
        self.content_layout.addLayout(self.right_layout)
        
        # Add content to the frame
        self.add_content(self.content_widget)
        
        # Set initial visibility
        self.timestamp_label.setVisible(self._show_timestamp)
        self.status_indicator.setVisible(self._show_status)
    
    def _apply_theme(self):
        """Apply the current theme to the activity item"""
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
        
        # Create stylesheet for activity item specific elements
        stylesheet = f"""
        QLabel#activityItemTitle {{
            color: {self.theme.title_color.name()};
            font-size: 14px;
            font-weight: bold;
        }}
        
        QLabel#activityItemDescription {{
            color: {self.theme.description_color.name()};
            font-size: 12px;
        }}
        
        QLabel#activityItemTimestamp {{
            color: {self.theme.timestamp_color.name()};
            font-size: 11px;
        }}
        """
        
        self.setStyleSheet(stylesheet)
        
        # Update status color
        self._update_status_color()
    
    def _update_status_color(self):
        """Update the status indicator color based on the current status"""
        status_color = self.theme.status_colors.get(self._status.value, QColor(150, 150, 150))
        
        # Create a colored circle for the status indicator
        self.status_indicator.setStyleSheet(f"""
            QLabel#activityItemStatus {{
                background-color: {status_color.name()};
                border-radius: 6px;
            }}
        """)
    
    def _setup_animations(self):
        """Set up animations for the activity item"""
        # Hover animation
        self.hover_animation = QPropertyAnimation(self, b"opacity")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.OutQuad)
        
        # Unread indicator animation
        self.unread_animation = QPropertyAnimation(self, b"byteArray")
        self.unread_animation.setDuration(300)
        self.unread_animation.setEasingCurve(QEasingCurve.OutQuad)
    
    def set_title(self, title: str):
        """Set the title text"""
        self._title = title
        self.title_label.setText(title)
        self.title_label.setVisible(bool(title))
    
    def set_description(self, description: str):
        """Set the description text"""
        self._description = description
        self.description_label.setText(description)
        self.description_label.setVisible(bool(description))
    
    def set_timestamp(self, timestamp: datetime):
        """Set the timestamp"""
        self._timestamp = timestamp
        self._update_timestamp_display()
    
    def _update_timestamp_display(self):
        """Update the timestamp display"""
        now = datetime.now()
        delta = now - self._timestamp
        
        if delta < timedelta(minutes=1):
            timestamp_str = "Just now"
        elif delta < timedelta(hours=1):
            timestamp_str = f"{delta.seconds // 60} minutes ago"
        elif delta < timedelta(days=1):
            timestamp_str = f"{delta.seconds // 3600} hours ago"
        elif delta < timedelta(days=7):
            timestamp_str = f"{delta.days} days ago"
        else:
            timestamp_str = self._timestamp.strftime("%Y-%m-%d")
        
        self.timestamp_label.setText(timestamp_str)
    
    def set_status(self, status: ActivityStatus):
        """Set the status"""
        self._status = status
        self._update_status_color()
    
    def set_icon(self, icon_path: str):
        """Set the icon from a file path"""
        self._icon_path = icon_path
        pixmap = QPixmap(icon_path)
        if not pixmap.isNull():
            # Scale pixmap to fit
            scaled_pixmap = pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)
            self.icon_label.setVisible(True)
        else:
            self.icon_label.setVisible(False)
    
    def set_icon_from_resource(self, resource_name: str):
        """Set the icon from a Qt resource"""
        pixmap = QPixmap(f":/icons/{resource_name}")
        if not pixmap.isNull():
            # Scale pixmap to fit
            scaled_pixmap = pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)
            self.icon_label.setVisible(True)
        else:
            self.icon_label.setVisible(False)
    
    def show_timestamp(self, show: bool):
        """Show or hide the timestamp"""
        self._show_timestamp = show
        self.timestamp_label.setVisible(show)
    
    def show_status(self, show: bool):
        """Show or hide the status indicator"""
        self._show_status = show
        self.status_indicator.setVisible(show)
    
    def mark_as_unread(self, unread: bool):
        """Mark the activity as unread or read"""
        self._is_unread = unread
        
        if unread:
            # Add a visual indicator for unread items
            self.setStyleSheet(self.styleSheet() + " ActivityItem { border-left: 4px solid #3498db; }")
        else:
            # Remove the unread indicator
            self.setStyleSheet(self.styleSheet().replace(" border-left: 4px solid #3498db;", ""))
    
    def enable_animation(self, enable: bool):
        """Enable or disable animations"""
        self._animate_changes = enable
    
    def get_title(self) -> str:
        """Get the current title"""
        return self._title
    
    def get_description(self) -> str:
        """Get the current description"""
        return self._description
    
    def get_timestamp(self) -> datetime:
        """Get the current timestamp"""
        return self._timestamp
    
    def get_status(self) -> ActivityStatus:
        """Get the current status"""
        return self._status
    
    def is_unread(self) -> bool:
        """Check if the activity is marked as unread"""
        return self._is_unread
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the activity item"""
        return {
            "theme": {
                "name": self.theme.name,
                "background_color": self.theme.background_color.name(),
                "border_color": self.theme.border_color.name(),
                "title_color": self.theme.title_color.name(),
                "description_color": self.theme.description_color.name(),
                "timestamp_color": self.theme.timestamp_color.name(),
                "status_colors": {
                    status: color.name() for status, color in self.theme.status_colors.items()
                },
                "shadow_color": self.theme.shadow_color.name(),
                "blur_radius": self.theme.blur_radius,
                "opacity": self.theme.opacity,
                "corner_radius": self.theme.corner_radius,
                "border_width": self.theme.border_width
            },
            "title": self._title,
            "description": self._description,
            "timestamp": self._timestamp.isoformat(),
            "status": self._status.value,
            "icon_path": self._icon_path,
            "show_timestamp": self._show_timestamp,
            "show_status": self._show_status,
            "is_unread": self._is_unread,
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
            status_colors = {}
            for status, color_name in theme_data.get("status_colors", {}).items():
                status_colors[status] = QColor(color_name)
            
            theme = ActivityItemTheme(
                name=theme_data.get("name", "Custom"),
                background_color=QColor(theme_data.get("background_color", "#FFFFFF")),
                border_color=QColor(theme_data.get("border_color", "#FFFFFF")),
                title_color=QColor(theme_data.get("title_color", "#323232")),
                description_color=QColor(theme_data.get("description_color", "#646464")),
                timestamp_color=QColor(theme_data.get("timestamp_color", "#969696")),
                status_colors=status_colors,
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
        if "description" in state:
            self.set_description(state["description"])
        if "timestamp" in state:
            self.set_timestamp(datetime.fromisoformat(state["timestamp"]))
        if "status" in state:
            self.set_status(ActivityStatus(state["status"]))
        if "icon_path" in state and state["icon_path"]:
            self.set_icon(state["icon_path"])
        if "show_timestamp" in state:
            self.show_timestamp(state["show_timestamp"])
        if "show_status" in state:
            self.show_status(state["show_status"])
        if "is_unread" in state:
            self.mark_as_unread(state["is_unread"])
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
            "description_color": self.theme.description_color.name(),
            "timestamp_color": self.theme.timestamp_color.name(),
            "status_colors": {
                status: color.name() for status, color in self.theme.status_colors.items()
            },
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
            
            status_colors = {}
            for status, color_name in theme_data.get("status_colors", {}).items():
                status_colors[status] = QColor(color_name)
            
            theme = ActivityItemTheme(
                name=theme_data.get("name", "Imported"),
                background_color=QColor(theme_data.get("background_color", "#FFFFFF")),
                border_color=QColor(theme_data.get("border_color", "#FFFFFF")),
                title_color=QColor(theme_data.get("title_color", "#323232")),
                description_color=QColor(theme_data.get("description_color", "#646464")),
                timestamp_color=QColor(theme_data.get("timestamp_color", "#969696")),
                status_colors=status_colors,
                shadow_color=QColor(theme_data.get("shadow_color", "#000000")),
                blur_radius=theme_data.get("blur_radius", 10),
                opacity=theme_data.get("opacity", 0.85),
                corner_radius=theme_data.get("corner_radius", 12),
                border_width=theme_data.get("border_width", 1)
            )
            self.set_theme(theme)

        except Exception as e:
            self.logger.error(f"Failed to import theme from {file_path}: {str(e)}")