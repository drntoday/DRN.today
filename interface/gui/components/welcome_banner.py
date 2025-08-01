from pathlib import Path

qss_path = Path(__file__).resolve().parents[3] / "interface" / "gui" / "styles" / "glassmorphism.qss"
GLASS_BANNER_STYLE = qss_path.read_text(encoding="utf-8")

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QBrush, QLinearGradient, QPen

class WelcomeBanner(QWidget):
    start_campaign = pyqtSignal()
    view_tutorial = pyqtSignal()
    import_leads = pyqtSignal()
    
    def __init__(self, username="User", parent=None):
        super().__init__(parent)
        self.username = username
        self.setFixedHeight(180)
        self.setStyleSheet(GLASS_BANNER_STYLE)
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 20, 25, 20)
        main_layout.setSpacing(15)
        
        # Header section
        header_layout = QHBoxLayout()
        
        # Welcome text
        welcome_layout = QVBoxLayout()
        welcome_label = QLabel(f"Welcome back, {self.username}!")
        welcome_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        welcome_label.setStyleSheet("color: #ffffff;")
        
        subtitle = QLabel("Ready to generate high-quality leads today?")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: rgba(255,255,255,0.8);")
        
        welcome_layout.addWidget(welcome_label)
        welcome_layout.addWidget(subtitle)
        
        # Stats section
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(8)
        
        leads_stat = self._create_stat("Leads Generated", "1,248")
        campaigns_stat = self._create_stat("Active Campaigns", "5")
        conversion_stat = self._create_stat("Conversion Rate", "24.7%")
        
        stats_layout.addWidget(leads_stat)
        stats_layout.addWidget(campaigns_stat)
        stats_layout.addWidget(conversion_stat)
        
        header_layout.addLayout(welcome_layout)
        header_layout.addStretch()
        header_layout.addLayout(stats_layout)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(15)
        
        campaign_btn = QPushButton("Start New Campaign")
        campaign_btn.setFixedSize(180, 40)
        campaign_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        campaign_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        campaign_btn.clicked.connect(self.start_campaign.emit)
        
        tutorial_btn = QPushButton("View Tutorial")
        tutorial_btn.setFixedSize(150, 40)
        tutorial_btn.setFont(QFont("Segoe UI", 10))
        tutorial_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background: rgba(255,255,255,0.3);
            }
        """)
        tutorial_btn.clicked.connect(self.view_tutorial.emit)
        
        import_btn = QPushButton("Import Leads")
        import_btn.setFixedSize(150, 40)
        import_btn.setFont(QFont("Segoe UI", 10))
        import_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background: rgba(255,255,255,0.3);
            }
        """)
        import_btn.clicked.connect(self.import_leads.emit)
        
        actions_layout.addWidget(campaign_btn)
        actions_layout.addWidget(tutorial_btn)
        actions_layout.addWidget(import_btn)
        actions_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        main_layout.addLayout(actions_layout)
        
    def _create_stat(self, label, value):
        container = QHBoxLayout()
        container.setSpacing(10)
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        value_label.setStyleSheet("color: #ffffff;")
        
        label_widget = QLabel(label)
        label_widget.setFont(QFont("Segoe UI", 9))
        label_widget.setStyleSheet("color: rgba(255,255,255,0.7);")
        
        container.addWidget(value_label)
        container.addWidget(label_widget)
        container.addStretch()
        
        widget = QWidget()
        widget.setLayout(container)
        return widget
        
    def update_username(self, username):
        self.username = username
        # Update UI in real implementation
        
    def update_stats(self, leads, campaigns, conversion):
        # Update stats in real implementation
        pass
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create gradient background
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(76, 175, 80, 40))
        gradient.setColorAt(1, QColor(33, 150, 243, 40))
        painter.fillRect(self.rect(), gradient)
        
        # Glassmorphism overlay
        painter.fillRect(self.rect(), QColor(255, 255, 255, 15))
        
        # Subtle border
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.drawRoundedRect(self.rect(), 15, 15)
        
        # Decorative elements
        painter.setPen(QPen(QColor(255, 255, 255, 10), 2))
        painter.drawLine(20, self.height() - 30, self.width() - 20, self.height() - 30)
