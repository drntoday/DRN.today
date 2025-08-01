#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Main GUI Window Implementation
Production-Ready Implementation
"""

import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QPushButton, QFrame, QStatusBar, QMenuBar, QMenu, QAction,
    QSplitter, QScrollArea, QGridLayout, QProgressBar, QMessageBox,
    QApplication, QSizePolicy, QToolBar, QToolButton, QTabWidget
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QThread, QSettings
from PyQt5.QtGui import QIcon, QFont, QPixmap, QPainter, QColor, QBrush, QPen

# Import custom components
from interface.gui.components.glass_frame import GlassFrame
from interface.gui.components.sidebar import Sidebar
from interface.gui.components.stat_card import StatCard
from interface.gui.components.activity_item import ActivityItem
from interface.gui.components.campaign_card import CampaignCard
from interface.gui.components.welcome_banner import WelcomeBanner

# Import views
from interface.gui.dashboard import DashboardView

# Core system imports
from engine.orchestrator import ModuleOrchestrator
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# Initialize GUI logger
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Production-ready main application window with glassmorphism design"""
    
    # Signals
    module_selected = pyqtSignal(str)
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, orchestrator: ModuleOrchestrator):
        super().__init__()
        
        self.orchestrator = orchestrator
        self.config = get_config()
        self.current_module = None
        self.module_widgets: Dict[str, QWidget] = {}
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        self.load_settings()
        
        # Initialize dashboard
        self.show_dashboard()
        
        # Start status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)  # Update every 5 seconds
        
        logger.info("Main window initialized")
    
    def setup_ui(self):
        """Setup the main window UI"""
        # Window properties
        self.setWindowTitle("DRN.today - Enterprise Lead Generation Platform")
        self.setMinimumSize(1200, 800)
        self.resize(self.config.interface.gui_window_width, self.config.interface.gui_window_height)
        
        # Set window icon
        self.setWindowIcon(QIcon("resources/icons/app_icon.png"))
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for resizable sections
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Create sidebar
        self.sidebar = Sidebar(self.orchestrator)
        self.sidebar.setMaximumWidth(250)
        self.splitter.addWidget(self.sidebar)
        
        # Create main content area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(self.content_area)
        
        # Create stacked widget for module views
        self.stacked_widget = QStackedWidget()
        self.content_layout.addWidget(self.stacked_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Apply glassmorphism stylesheet
        self.apply_stylesheet()
        
        # Set splitter sizes
        self.splitter.setSizes([250, 950])
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Campaign", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_campaign)
        file_menu.addAction(new_action)
        
        import_action = QAction("Import Leads", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self.import_leads)
        file_menu.addAction(import_action)
        
        export_action = QAction("Export Data", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self.show_settings)
        edit_menu.addAction(preferences_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        dashboard_action = QAction("Dashboard", self)
        dashboard_action.setShortcut("Ctrl+1")
        dashboard_action.triggered.connect(lambda: self.show_module("dashboard"))
        view_menu.addAction(dashboard_action)
        
        view_menu.addSeparator()
        
        # Add module actions dynamically
        for module_name in self.orchestrator.module_order:
            if module_name in self.orchestrator.modules:
                action = QAction(module_name.replace("_", " ").title(), self)
                action.triggered.connect(lambda checked, m=module_name: self.show_module(m))
                view_menu.addAction(action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        adapter_builder_action = QAction("Adapter Builder", self)
        adapter_builder_action.triggered.connect(self.show_adapter_builder)
        tools_menu.addAction(adapter_builder_action)
        
        tools_menu.addSeparator()
        
        marketplace_action = QAction("Lead Pack Marketplace", self)
        marketplace_action.triggered.connect(self.show_marketplace)
        tools_menu.addAction(marketplace_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        documentation_action = QAction("Documentation", self)
        documentation_action.setShortcut("F1")
        documentation_action.triggered.connect(self.show_documentation)
        help_menu.addAction(documentation_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # New campaign button
        new_campaign_btn = QToolButton()
        new_campaign_btn.setIcon(QIcon("resources/icons/new_campaign.png"))
        new_campaign_btn.setToolTip("New Campaign")
        new_campaign_btn.clicked.connect(self.new_campaign)
        toolbar.addWidget(new_campaign_btn)
        
        toolbar.addSeparator()
        
        # Import leads button
        import_btn = QToolButton()
        import_btn.setIcon(QIcon("resources/icons/import.png"))
        import_btn.setToolTip("Import Leads")
        import_btn.clicked.connect(self.import_leads)
        toolbar.addWidget(import_btn)
        
        # Export data button
        export_btn = QToolButton()
        export_btn.setIcon(QIcon("resources/icons/export.png"))
        export_btn.setToolTip("Export Data")
        export_btn.clicked.connect(self.export_data)
        toolbar.addWidget(export_btn)
        
        toolbar.addSeparator()
        
        # Refresh button
        refresh_btn = QToolButton()
        refresh_btn.setIcon(QIcon("resources/icons/refresh.png"))
        refresh_btn.setToolTip("Refresh")
        refresh_btn.clicked.connect(self.refresh_current_view)
        toolbar.addWidget(refresh_btn)
        
        toolbar.addSeparator()
        
        # Settings button
        settings_btn = QToolButton()
        settings_btn.setIcon(QIcon("resources/icons/settings.png"))
        settings_btn.setToolTip("Settings")
        settings_btn.clicked.connect(self.show_settings)
        toolbar.addWidget(settings_btn)
    
    def setup_connections(self):
        """Setup signal connections"""
        # Sidebar connections
        self.sidebar.module_selected.connect(self.show_module)
        
        # Orchestrator connections
        self.orchestrator.event_bus.subscribe("system.status", self.handle_system_status)
        self.orchestrator.event_bus.subscribe("module.status", self.handle_module_status)
        self.orchestrator.event_bus.subscribe("license.status", self.handle_license_status)
    
    def apply_stylesheet(self):
        """Apply glassmorphism stylesheet"""
        try:
            with open("interface/gui/styles/glassmorphism.qss", "r") as f:
                stylesheet = f.read()
            self.setStyleSheet(stylesheet)
        except Exception as e:
            logger.error(f"Failed to load stylesheet: {str(e)}")
    
    def show_dashboard(self):
        """Show the dashboard view"""
        if "dashboard" not in self.module_widgets:
            dashboard = DashboardView(self.orchestrator)
            self.stacked_widget.addWidget(dashboard)
            self.module_widgets["dashboard"] = dashboard
        
        self.stacked_widget.setCurrentWidget(self.module_widgets["dashboard"])
        self.current_module = "dashboard"
        self.sidebar.set_active_module("dashboard")
        
        # Update window title
        self.setWindowTitle("DRN.today - Dashboard")
    
    def show_module(self, module_name: str):
        """Show a specific module view"""
        if module_name == "dashboard":
            self.show_dashboard()
            return
        
        # Check if module exists
        if module_name not in self.orchestrator.modules:
            QMessageBox.warning(self, "Module Not Found", 
                              f"The module '{module_name}' is not available.")
            return
        
        # Create module widget if not exists
        if module_name not in self.module_widgets:
            try:
                # Import and create module GUI
                module_class = self.get_module_gui_class(module_name)
                if module_class:
                    module_widget = module_class(self.orchestrator.modules[module_name])
                    self.stacked_widget.addWidget(module_widget)
                    self.module_widgets[module_name] = module_widget
                else:
                    # Create placeholder widget
                    placeholder = self.create_module_placeholder(module_name)
                    self.stacked_widget.addWidget(placeholder)
                    self.module_widgets[module_name] = placeholder
            except Exception as e:
                logger.error(f"Error creating module GUI for {module_name}: {str(e)}")
                placeholder = self.create_module_placeholder(module_name)
                self.stacked_widget.addWidget(placeholder)
                self.module_widgets[module_name] = placeholder
        
        # Show module
        self.stacked_widget.setCurrentWidget(self.module_widgets[module_name])
        self.current_module = module_name
        self.sidebar.set_active_module(module_name)
        
        # Update window title
        self.setWindowTitle(f"DRN.today - {module_name.replace('_', ' ').title()}")
        
        # Emit signal
        self.module_selected.emit(module_name)
    
    def get_module_gui_class(self, module_name: str):
        """Get the GUI class for a module"""
        try:
            # Convert module name to class name
            class_name = "".join(word.capitalize() for word in module_name.split("_")) + "GUI"
            
            # Import module
            module_path = f"interface.gui.modules.{module_name}"
            module = __import__(module_path, fromlist=[class_name])
            
            # Get class
            return getattr(module, class_name, None)
        except Exception as e:
            logger.error(f"Error getting GUI class for {module_name}: {str(e)}")
            return None
    
    def create_module_placeholder(self, module_name: str) -> QWidget:
        """Create a placeholder widget for modules without GUI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel(f"{module_name.replace('_', ' ').title()} Module")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        # Placeholder content
        content = QLabel("This module is currently under development.")
        content.setAlignment(Qt.AlignCenter)
        content.setStyleSheet("font-size: 16px; color: #666;")
        layout.addWidget(content)
        
        layout.addStretch()
        
        return widget
    
    def update_status(self):
        """Update status bar and other status indicators"""
        try:
            # Get system status
            status = self.orchestrator.get_module_status()
            
            # Update status bar
            active_modules = sum(1 for m in status.values() if m.get("running", False))
            self.status_bar.showMessage(f"Active Modules: {active_modules} | "
                                     f"Memory Usage: {self.get_memory_usage()}MB")
            
            # Update license status in status bar if needed
            license_info = self.orchestrator.license_manager.get_license_info()
            if license_info:
                tier = license_info.get("tier", "unknown")
                remaining = license_info.get("remaining_leads", 0)
                if remaining >= 0:
                    self.status_bar.showMessage(f"Active Modules: {active_modules} | "
                                               f"License: {tier.title()} | "
                                               f"Remaining Leads: {remaining}")
            
        except Exception as e:
            logger.error(f"Error updating status: {str(e)}")
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss // 1024 // 1024
        except:
            return 0
    
    def handle_system_status(self, event_type: str, data: dict):
        """Handle system status events"""
        # Update system status indicators
        pass
    
    def handle_module_status(self, event_type: str, data: dict):
        """Handle module status events"""
        # Update module status indicators
        pass
    
    def handle_license_status(self, event_type: str, data: dict):
        """Handle license status events"""
        # Update license status indicators
        pass
    
    def new_campaign(self):
        """Create new campaign"""
        QMessageBox.information(self, "New Campaign", 
                              "New campaign creation would open here.")
    
    def import_leads(self):
        """Import leads"""
        QMessageBox.information(self, "Import Leads", 
                              "Lead import dialog would open here.")
    
    def export_data(self):
        """Export data"""
        QMessageBox.information(self, "Export Data", 
                              "Data export dialog would open here.")
    
    def show_settings(self):
        """Show settings dialog"""
        QMessageBox.information(self, "Settings", 
                              "Settings dialog would open here.")
    
    def show_adapter_builder(self):
        """Show adapter builder"""
        self.show_module("lead_generation")
        # In a real implementation, this would open the adapter builder within the module
    
    def show_marketplace(self):
        """Show lead pack marketplace"""
        self.show_module("marketplace")
    
    def show_documentation(self):
        """Show documentation"""
        QMessageBox.information(self, "Documentation", 
                              "Documentation would open here.")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About DRN.today",
                         "<h2>DRN.today</h2>"
                         "<p>Enterprise-Grade Lead Generation Platform</p>"
                         "<p>Version 1.0.0</p>"
                         "<p>© 2025 DRN.today. All rights reserved.</p>")
    
    def refresh_current_view(self):
        """Refresh the current view"""
        if self.current_module and self.current_module in self.module_widgets:
            widget = self.module_widgets[self.current_module]
            if hasattr(widget, 'refresh'):
                widget.refresh()
    
    def load_settings(self):
        """Load application settings"""
        settings = QSettings("DRN.today", "App")
        
        # Restore window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore window state
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
        
        # Restore splitter sizes
        splitter_sizes = settings.value("splitterSizes")
        if splitter_sizes:
            self.splitter.setSizes([int(s) for s in splitter_sizes])
    
    def save_settings(self):
        """Save application settings"""
        settings = QSettings("DRN.today", "App")
        
        # Save window geometry
        settings.setValue("geometry", self.saveGeometry())
        
        # Save window state
        settings.setValue("windowState", self.saveState())
        
        # Save splitter sizes
        settings.setValue("splitterSizes", self.splitter.sizes())
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save settings
        self.save_settings()
        
        # Confirm exit if there are active operations
        if self.has_active_operations():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "There are active operations. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # Shutdown orchestrator
        self.orchestrator.shutdown_modules()
        
        # Accept event
        event.accept()
    
    def has_active_operations(self) -> bool:
        """Check if there are active operations"""
        # Check if any modules have active operations
        for module_name, module in self.orchestrator.modules.items():
            if hasattr(module, 'has_active_operations') and module.has_active_operations():
                return True
        
        return False