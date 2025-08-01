#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Global Configuration System
Production-Ready Implementation
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Initialize configuration logger
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """SQLite database configuration"""
    path: str = "data/drn.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_connections: int = 10
    timeout_seconds: int = 30
    encryption_key: Optional[str] = None

@dataclass
class AIConfig:
    """AI/ML model configuration"""
    tinybert_model_path: str = "ai/models/tinybert"
    scikit_model_path: str = "ai/models/scikit"
    gpt2_model_path: str = "ai/models/gpt2"
    max_text_length: int = 512
    batch_size: int = 32
    scoring_threshold: float = 0.75
    enable_gpu: bool = False
    model_cache_size: int = 100

@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DRN.today/1.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) DRN.today/1.0",
        "Mozilla/5.0 (X11; Linux x86_64) DRN.today/1.0"
    ])
    default_delay_seconds: float = 1.5
    max_retries: int = 3
    timeout_seconds: int = 30
    proxy_rotation: bool = True
    proxy_list_path: Optional[str] = None
    captcha_solver_enabled: bool = True
    respect_robots_txt: bool = True
    max_concurrent_requests: int = 5

@dataclass
class EmailConfig:
    """Email system configuration"""
    smtp_pool_size: int = 5
    imap_pool_size: int = 3
    max_attachment_size_mb: int = 10
    bounce_threshold: int = 3
    warming_enabled: bool = True
    warming_daily_limit: int = 50
    blacklist_check_enabled: bool = True
    tracking_pixel_enabled: bool = True
    unsubscribe_header_enabled: bool = True

@dataclass
class ComplianceConfig:
    """GDPR/CCPA compliance configuration"""
    geo_blocking_enabled: bool = True
    blocked_regions: List[str] = field(default_factory=lambda: ["EU", "CA"])
    data_retention_days: int = 90
    auto_opt_out_enabled: bool = True
    consent_required: bool = True
    do_not_track: bool = True
    source_blacklist_path: Optional[str] = None

@dataclass
class InterfaceConfig:
    """User interface configuration"""
    gui_theme: str = "glassmorphism"
    gui_font_size: int = 12
    gui_window_width: int = 1200
    gui_window_height: int = 800
    cli_output_format: str = "table"
    daemon_poll_interval: int = 60
    dashboard_refresh_interval: int = 30

@dataclass
class SecurityConfig:
    """Security configuration"""
    license_key_path: str = "license.key"
    encryption_algorithm: str = "AES-256-GCM"
    keyring_service: str = "DRN.today"
    session_timeout_minutes: int = 30
    audit_log_enabled: bool = True
    secure_storage_path: str = "secure"

@dataclass
class MarketplaceConfig:
    """Lead pack marketplace configuration"""
    marketplace_url: Optional[str] = None
    community_repo_url: str = "https://github.com/drn-today/community-packs"
    pack_validation_enabled: bool = True
    auto_update_enabled: bool = True
    contributor_verification: bool = True

class Config:
    """Global configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("DRN_CONFIG", "config.json")
        self._config_data = {}
        
        # Initialize configuration sections
        self.database = DatabaseConfig()
        self.ai = AIConfig()
        self.scraping = ScrapingConfig()
        self.email = EmailConfig()
        self.compliance = ComplianceConfig()
        self.interface = InterfaceConfig()
        self.security = SecurityConfig()
        self.marketplace = MarketplaceConfig()
        
        # Load configuration
        self._load_config()
        self._validate_config()
        
    def _load_config(self):
        """Load configuration from file and environment variables"""
        # Load from file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self._config_data = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config file: {str(e)}")
        
        # Override with environment variables
        self._load_env_vars()
        
        # Update configuration sections
        self._update_sections()
    
    def _load_env_vars(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "DRN_DB_PATH": ("database", "path"),
            "DRN_AI_GPU": ("ai", "enable_gpu"),
            "DRN_SCRAPING_DELAY": ("scraping", "default_delay_seconds"),
            "DRN_EMAIL_POOL_SIZE": ("email", "smtp_pool_size"),
            "DRN_GEO_BLOCKING": ("compliance", "geo_blocking_enabled"),
            "DRN_GUI_THEME": ("interface", "gui_theme"),
            "DRN_LICENSE_PATH": ("security", "license_key_path"),
            "DRN_MARKETPLACE_URL": ("marketplace", "marketplace_url")
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config_data:
                    self._config_data[section] = {}
                self._config_data[section][key] = self._convert_type(value, key)
    
    def _convert_type(self, value: str, key: str) -> any:
        """Convert string value to appropriate type based on key"""
        type_mappings = {
            "enable_gpu": lambda x: x.lower() == "true",
            "geo_blocking_enabled": lambda x: x.lower() == "true",
            "default_delay_seconds": float,
            "smtp_pool_size": int,
            "gui_theme": str
        }
        
        converter = type_mappings.get(key, str)
        try:
            return converter(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert {value} for {key}, using string")
            return value
    
    def _update_sections(self):
        """Update configuration sections with loaded data"""
        sections = {
            "database": self.database,
            "ai": self.ai,
            "scraping": self.scraping,
            "email": self.email,
            "compliance": self.compliance,
            "interface": self.interface,
            "security": self.security,
            "marketplace": self.marketplace
        }
        
        for section_name, section_obj in sections.items():
            if section_name in self._config_data:
                section_data = self._config_data[section_name]
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def _validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate paths
        if not self.database.path:
            errors.append("Database path cannot be empty")
        
        if not self.ai.tinybert_model_path:
            errors.append("TinyBERT model path cannot be empty")
        
        # Validate numeric ranges
        if self.scraping.default_delay_seconds < 0.1:
            errors.append("Scraping delay must be at least 0.1 seconds")
        
        if self.email.max_attachment_size_mb > 50:
            errors.append("Email attachment size cannot exceed 50MB")
        
        if self.compliance.data_retention_days < 1:
            errors.append("Data retention must be at least 1 day")
        
        # Validate security settings
        if not self.security.encryption_algorithm:
            errors.append("Encryption algorithm must be specified")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            logger.critical(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            "database": self.database.__dict__,
            "ai": self.ai.__dict__,
            "scraping": self.scraping.__dict__,
            "email": self.email.__dict__,
            "compliance": self.compliance.__dict__,
            "interface": self.interface.__dict__,
            "security": self.security.__dict__,
            "marketplace": self.marketplace.__dict__
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
            raise
    
    def get_module_config(self, module_name: str) -> Dict:
        """Get configuration for a specific module"""
        module_configs = {
            "lead_generation": {
                "scraping": self.scraping,
                "ai": self.ai
            },
            "lead_enrichment": {
                "ai": self.ai,
                "database": self.database
            },
            "email_system": {
                "email": self.email,
                "security": self.security
            },
            "compliance": {
                "compliance": self.compliance,
                "security": self.security
            }
        }
        
        return module_configs.get(module_name, {})
    
    def get_ai_model_paths(self) -> Dict[str, str]:
        """Get all AI model paths"""
        return {
            "tinybert": self.ai.tinybert_model_path,
            "scikit": self.ai.scikit_model_path,
            "gpt2": self.ai.gpt2_model_path
        }
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data storage paths"""
        return {
            "database": self.database.path,
            "secure_storage": self.security.secure_storage_path,
            "proxy_list": self.scraping.proxy_list_path,
            "blacklist": self.compliance.source_blacklist_path
        }

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config
