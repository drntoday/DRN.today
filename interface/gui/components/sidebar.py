#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Sidebar Component Implementation
Production-Ready Implementation
"""

import logging
from typing import Dict, List, Optional, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFrame, QSizePolicy, QSpacerItem, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QColor, QBrush, QPen, QCursor
import os

# Initialize sidebar logger
logger = logging.getLogger(__name__)

class SidebarItem(QWidget):
    """Custom sidebar navigation item with hover effects"""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, icon_path: str, text: str, module_name: str, parent=None):
        super().__init__(parent)
        self.icon_path = icon_path
        self.text = text
        self.module_name = module_name
        self.is_active = False
        self.hover = False
        
        # Setup UI
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """Setup the item UI"""
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(12)
        
        # Icon
        self.icon_label = QLabel()
        if self.icon_path and os.path.exists(self.icon_path):
            pixmap = QPixmap(self.icon_path)
            self.icon_label.setPixmap(pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Use emoji as fallback
            self.icon_label.setText(self.get_emoji_fallback())
            self.icon_label.setStyleSheet("font-size: 20px;")
        
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)
        
        # Text
        self.text_label = QLabel(self.text)
        self.text_label.setFont(QFont("Arial", 12, QFont.Medium))
        layout.addWidget(self.text_label)
        
        # Add stretch
        layout.addStretch()
        
        # Set cursor
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
    def get_emoji_fallback(self) -> str:
        """Get emoji fallback for icon"""
        emoji_map = {
            "dashboard": "📊",
            "lead_generation": "🎯",
            "lead_enrichment": "💎",
            "conversation_mining": "💬",
            "web_crawlers": "🕷️",
            "lead_capture": "📩",
            "competitive_intel": "🔍",
            "email_system": "📧",
            "intent_engine": "📈",
            "template_engine": "📝",
            "integrations": "🔌",
            "marketplace": "🛒",
            "compliance": "🛡️"
        }
        return emoji_map.get(self.module_name, "📄")
    
    def setup_styles(self):
        """Setup item styles"""
        self.update_style()
    
    def update_style(self):
        """Update item style based on state"""
        base_style = """
            QWidget {
                background-color: transparent;
                border-radius: 8px;
                margin: 2px;
            }
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                background-color: transparent;
            }
        """
        
        hover_style = """
            QWidget {
                background-color: rgba(255, 255, 255, 0.1);
            }
            QLabel {
                color: #ffffff;
            }
        """
        
        active_style = """
            QWidget {
                background-color: rgba(76, 175, 80, 0.3);
                border-left: 3px solid #4CAF50;
            }
            QLabel {
                color: #ffffff;
                font-weight: bold;
            }
        """
        
        if self.is_active:
            self.setStyleSheet(base_style + active_style)
        elif self.hover:
            self.setStyleSheet(base_style + hover_style)
        else:
            self.setStyleSheet(base_style)
    
    def set_active(self, active: bool):
        """Set active state"""
        if self.is_active != active:
            self.is_active = active
            self.update_style()
    
    def enterEvent(self, event):
        """Handle mouse enter event"""
        self.hover = True
        self.update_style()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave event"""
        self.hover = False
        self.update_style()
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press event"""
        self.clicked.emit(self.module_name)
        super().mousePressEvent(event)

class UserProfileWidget(QWidget):
    """User profile widget for sidebar footer"""
    
    settings_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        """Setup the user profile UI"""
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(12)
        
        # Avatar
        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(40, 40)
        self.avatar_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                font-size: 18px;
            }
        """)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setText("👤")
        layout.addWidget(self.avatar_label)
        
        # User info
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)
        
        self.name_label = QLabel("Admin User")
        self.name_label.setFont(QFont("Arial", 12, QFont.Bold))
        info_layout.addWidget(self.name_label)
        
        self.license_label = QLabel("Admin License")
        self.license_label.setFont(QFont("Arial", 10))
        self.license_label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        info_layout.addWidget(self.license_label)
        
        layout.addLayout(info_layout)
        
        # Settings button
        self.settings_btn = QPushButton()
        self.settings_btn.setIcon(QIcon("resources/icons/settings.png"))
        self.settings_btn.setIconSize(QSize(20, 20))
        self.settings_btn.setFixedSize(36, 36)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 18px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        self.settings_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.settings_btn)
    
    def setup_styles(self):
        """Setup widget styles"""
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                margin: 2px;
            }
        """)
    
    def update_user_info(self, name: str, license_type: str):
        """Update user information"""
        self.name_label.setText(name)
        self.license_label.setText(f"{license_type.title()} License")

class Sidebar(QWidget):
    """Production-ready sidebar with navigation and user profile"""
    
    module_selected = pyqtSignal(str)
    settings_clicked = pyqtSignal()
    
    def __init__(self, orchestrator, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.current_module = None
        self.items: Dict[str, SidebarItem] = {}
        
        # Setup UI
        self.setup_ui()
        self.setup_styles()
        self.create_navigation_items()
        
    def setup_ui(self):
        """Setup the sidebar UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Logo section
        logo_section = self.create_logo_section()
        layout.addWidget(logo_section)
        
        # Navigation section
        self.navigation_section = self.create_navigation_section()
        layout.addWidget(self.navigation_section)
        
        # Add stretch
        layout.addStretch()
        
        # User profile section
        self.profile_section = UserProfileWidget()
        self.profile_section.settings_clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.profile_section)
        
        # Set fixed width
        self.setFixedWidth(250)
    
    def setup_styles(self):
        """Setup sidebar styles"""
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 0.8);
                border-right: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
    
    def create_logo_section(self) -> QWidget:
        """Create the logo section"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(15, 20, 15, 20)
        
        # Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap("resources/icons/app_icon.png")
        if logo_pixmap.isNull():
            # Fallback to text logo
            logo_label.setText("DRN")
            logo_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                    background-color: transparent;
                }
            """)
        else:
            logo_label.setPixmap(logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        layout.addWidget(logo_label)
        
        # App name
        name_label = QLabel("DRN.today")
        name_label.setFont(QFont("Arial", 16, QFont.Bold))
        name_label.setStyleSheet("color: #ffffff;")
        layout.addWidget(name_label)
        
        layout.addStretch()
        
        return widget
    
    def create_navigation_section(self) -> QWidget:
        """Create the navigation section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(5)
        
        # Section title
        title_label = QLabel("NAVIGATION")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.5);
                padding-left: 15px;
                padding-top: 10px;
                padding-bottom: 5px;
                background-color: transparent;
            }
        """)
        layout.addWidget(title_label)
        
        # Scroll area for navigation items
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: rgba(255, 255, 255, 0.1);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Container for scroll area
        container = QWidget()
        self.nav_layout = QVBoxLayout(container)
        self.nav_layout.setContentsMargins(0, 0, 0, 0)
        self.nav_layout.setSpacing(2)
        
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)
        
        return widget
    
    def create_navigation_items(self):
        """Create navigation items for all modules"""
        # Define navigation items
        nav_items = [
            {
                "module": "dashboard",
                "text": "Dashboard",
                "icon": "resources/icons/dashboard.png"
            },
            {
                "module": "lead_generation",
                "text": "Lead Generation",
                "icon": "resources/icons/lead_generation.png"
            },
            {
                "module": "lead_enrichment",
                "text": "Lead Enrichment",
                "icon": "resources/icons/lead_enrichment.png"
            },
            {
                "module": "conversation_mining",
                "text": "Conversation Mining",
                "icon": "resources/icons/conversation_mining.png"
            },
            {
                "module": "web_crawlers",
                "text": "Web Crawlers",
                "icon": "resources/icons/web_crawlers.png"
            },
            {
                "module": "lead_capture",
                "text": "Lead Capture",
                "icon": "resources/icons/lead_capture.png"
            },
            {
                "module": "competitive_intel",
                "text": "Competitive Intel",
                "icon": "resources/icons/competitive_intel.png"
            },
            {
                "module": "email_system",
                "text": "Email System",
                "icon": "resources/icons/email_system.png"
            },
            {
                "module": "intent_engine",
                "text": "Intent Engine",
                "icon": "resources/icons/intent_engine.png"
            },
            {
                "module": "template_engine",
                "text": "Template Engine",
                "icon": "resources/icons/template_engine.png"
            },
            {
                "module": "integrations",
                "text": "Integrations",
                "icon": "resources/icons/integrations.png"
            },
            {
                "module": "marketplace",
                "text": "Marketplace",
                "icon": "resources/icons/marketplace.png"
            },
            {
                "module": "compliance",
                "text": "Compliance",
                "icon": "resources/icons/compliance.png"
            }
        ]
        
        # Create items
        for item_data in nav_items:
            # Check if module is available in orchestrator
            if item_data["module"] == "dashboard" or item_data["module"] in self.orchestrator.modules:
                nav_item = SidebarItem(
                    icon_path=item_data["icon"],
                    text=item_data["text"],
                    module_name=item_data["module"]
                )
                nav_item.clicked.connect(self.on_item_clicked)
                self.nav_layout.addWidget(nav_item)
                self.items[item_data["module"]] = nav_item
        
        # Add stretch at the end
        self.nav_layout.addStretch()
    
    def on_item_clicked(self, module_name: str):
        """Handle navigation item click"""
        # Update active state
        if self.current_module != module_name:
            if self.current_module and self.current_module in self.items:
                self.items[self.current_module].set_active(False)
            
            self.current_module = module_name
            if module_name in self.items:
                self.items[module_name].set_active(True)
            
            # Emit signal
            self.module_selected.emit(module_name)
    
    def set_active_module(self, module_name: str):
        """Set the active module"""
        if self.current_module != module_name:
            if self.current_module and self.current_module in self.items:
                self.items[self.current_module].set_active(False)
            
            self.current_module = module_name
            if module_name in self.items:
                self.items[module_name].set_active(True)
    
    def update_user_profile(self, name: str, license_type: str):
        """Update user profile information"""
        self.profile_section.update_user_info(name, license_type)
    
    def paintEvent(self, event):
        """Custom paint event for glassmorphism effect"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create glass effect
        painter.fillRect(self.rect(), QColor(30, 30, 30, 204))  # Semi-transparent dark background
        
        # Add subtle border
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
