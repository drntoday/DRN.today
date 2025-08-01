import os
from pathlib import Path

qss_path = Path(__file__).resolve().parents[3] / "interface" / "gui" / "styles" / "glassmorphism.qss"
GLASS_CARD_STYLE = qss_path.read_text(encoding="utf-8")

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QBrush

class CampaignCard(QWidget):
    clicked = pyqtSignal(str)
    paused = pyqtSignal(str)
    deleted = pyqtSignal(str)
    
    def __init__(self, campaign_id, name, status, leads, sent, opened, clicked, bounced, parent=None):
        super().__init__(parent)
        self.campaign_id = campaign_id
        self.name = name
        self.status = status
        self.leads = leads
        self.sent = sent
        self.opened = opened
        self.clicked = clicked
        self.bounced = bounced
        
        self.setFixedSize(320, 200)
        self.setStyleSheet(GLASS_CARD_STYLE)
        self.setCursor(Qt.PointingHandCursor)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Header
        header = QHBoxLayout()
        name_label = QLabel(self.name)
        name_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        name_label.setStyleSheet("color: #ffffff;")
        
        status_label = QLabel(self.status.upper())
        status_label.setFont(QFont("Segoe UI", 8))
        status_label.setStyleSheet(f"color: {self._get_status_color()}; background: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 10px;")
        
        header.addWidget(name_label)
        header.addStretch()
        header.addWidget(status_label)
        
        # Metrics
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(20)
        
        leads_label = self._create_metric_label("Leads", str(self.leads))
        sent_label = self._create_metric_label("Sent", str(self.sent))
        opened_label = self._create_metric_label("Opened", f"{self.opened}%")
        
        metrics_layout.addWidget(leads_label)
        metrics_layout.addWidget(sent_label)
        metrics_layout.addWidget(opened_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(self.opened)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background: rgba(255,255,255,0.1);
                border-radius: 5px;
                height: 6px;
            }
            QProgressBar::chunk {
                background: #4CAF50;
                border-radius: 5px;
            }
        """)
        
        # Additional metrics
        additional_metrics = QHBoxLayout()
        additional_metrics.setSpacing(20)
        
        clicked_label = self._create_metric_label("Clicked", f"{self.clicked}%")
        bounced_label = self._create_metric_label("Bounced", f"{self.bounced}%")
        
        additional_metrics.addWidget(clicked_label)
        additional_metrics.addWidget(bounced_label)
        additional_metrics.addStretch()
        
        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)
        
        view_btn = QPushButton("View")
        view_btn.setFixedSize(70, 30)
        view_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.2);
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255,255,255,0.3);
            }
        """)
        view_btn.clicked.connect(lambda: self.clicked.emit(self.campaign_id))
        
        pause_btn = QPushButton("Pause" if self.status == "running" else "Resume")
        pause_btn.setFixedSize(70, 30)
        pause_btn.setStyleSheet("""
            QPushButton {
                background: #FF9800;
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #F57C00;
            }
        """)
        pause_btn.clicked.connect(lambda: self.paused.emit(self.campaign_id))
        
        delete_btn = QPushButton("Delete")
        delete_btn.setFixedSize(70, 30)
        delete_btn.setStyleSheet("""
            QPushButton {
                background: #F44336;
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #D32F2F;
            }
        """)
        delete_btn.clicked.connect(lambda: self.deleted.emit(self.campaign_id))
        
        actions_layout.addWidget(view_btn)
        actions_layout.addWidget(pause_btn)
        actions_layout.addWidget(delete_btn)
        
        # Add all layouts
        main_layout.addLayout(header)
        main_layout.addLayout(metrics_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(additional_metrics)
        main_layout.addLayout(actions_layout)
        
    def _create_metric_label(self, title, value):
        container = QVBoxLayout()
        container.setContentsMargins(0, 0, 0, 0)
        container.setSpacing(2)
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        value_label.setStyleSheet("color: #ffffff;")
        value_label.setAlignment(Qt.AlignCenter)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 8))
        title_label.setStyleSheet("color: rgba(255,255,255,0.7);")
        title_label.setAlignment(Qt.AlignCenter)
        
        container.addWidget(value_label)
        container.addWidget(title_label)
        
        widget = QWidget()
        widget.setLayout(container)
        return widget
        
    def _get_status_color(self):
        return {
            "running": "#4CAF50",
            "paused": "#FF9800",
            "completed": "#2196F3",
            "failed": "#F44336"
        }.get(self.status.lower(), "#9E9E9E")
        
    def update_status(self, new_status):
        self.status = new_status
        # Would update UI in real implementation
        
    def update_metrics(self, leads, sent, opened, clicked, bounced):
        self.leads = leads
        self.sent = sent
        self.opened = opened
        self.clicked = clicked
        self.bounced = bounced
        # Would update UI in real implementation
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Glassmorphism background
        painter.fillRect(self.rect(), QColor(255, 255, 255, 20))
        
        # Subtle border
        painter.setPen(QColor(255, 255, 255, 50))
        painter.drawRoundedRect(self.rect(), 15, 15)
