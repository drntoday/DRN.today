#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Secure Storage System
Production-Ready Implementation
"""

import os
import sqlite3
import threading
import logging
import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Initialize storage logger
logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """Storage configuration parameters"""
    db_path: str
    encryption_key: Optional[str] = None
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_connections: int = 10
    timeout_seconds: int = 30
    retention_days: int = 90
    audit_enabled: bool = True
    auto_vacuum: bool = True
    synchronous: str = "NORMAL"  # OFF, NORMAL, FULL
    journal_mode: str = "WAL"  # DELETE, TRUNCATE, PERSIST, WAL

class SecureStorage:
    """Production-ready secure storage system with encryption and maintenance"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self.backup_path = self.db_path.parent / "backups"
        self._encryption_key = None
        self._fernet = None
        self._connections: Dict[int, sqlite3.Connection] = {}
        self._connection_lock = threading.RLock()
        self._maintenance_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if config.backup_enabled:
            self.backup_path.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> bool:
        """Initialize storage with encryption and schema"""
        try:
            logger.info("Initializing secure storage...")
            
            # Setup encryption
            self._setup_encryption()
            
            # Initialize database
            if not self._initialize_database():
                logger.error("Database initialization failed")
                return False
            
            # Start maintenance if enabled
            if self.config.backup_enabled:
                self.start_maintenance(
                    backup_interval=self.config.backup_interval_hours,
                    max_connections=self.config.max_connections
                )
            
            logger.info("Secure storage initialized successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Storage initialization failed: {str(e)}", exc_info=True)
            return False
    
    def _setup_encryption(self):
        """Setup encryption key and Fernet instance"""
        try:
            if not self.config.encryption_key:
                # Generate a new key if not provided
                self._encryption_key = Fernet.generate_key()
                logger.warning("Generated new encryption key - store securely!")
            else:
                # Derive key from provided password
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'drn_salt',  # In production, use random salt
                    iterations=100000,
                )
                self._encryption_key = base64.urlsafe_b64encode(
                    kdf.derive(self.config.encryption_key.encode())
                )
            
            self._fernet = Fernet(self._encryption_key)
            logger.info("Encryption setup completed")
            
        except Exception as e:
            logger.error(f"Encryption setup failed: {str(e)}", exc_info=True)
            raise
    
    def _initialize_database(self) -> bool:
        """Initialize database schema and settings"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Configure database settings
            cursor.execute(f"PRAGMA synchronous = {self.config.synchronous}")
            cursor.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            cursor.execute(f"PRAGMA foreign_keys = ON")
            cursor.execute(f"PRAGMA temp_store = MEMORY")
            cursor.execute(f"PRAGMA mmap_size = 268435456")  # 256MB
            
            # Create tables
            self._create_tables(cursor)
            
            # Create indexes
            self._create_indexes(cursor)
            
            # Initialize metadata
            self._initialize_metadata(cursor)
            
            conn.commit()
            logger.info("Database schema initialized")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
            return False
    
    def _create_tables(self, cursor: sqlite3.Cursor):
        """Create all database tables"""
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Leads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                name TEXT,
                email TEXT,
                phone TEXT,
                company TEXT,
                website TEXT,
                location TEXT,
                social_links TEXT,
                source TEXT,
                category TEXT,
                tags TEXT,
                score REAL DEFAULT 0.0,
                persona TEXT,
                insights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retention_date TIMESTAMP,
                encrypted_data BLOB
            )
        """)
        
        # Campaigns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                template_id TEXT,
                status TEXT DEFAULT 'draft',
                settings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Email logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS email_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT,
                lead_id TEXT,
                email TEXT,
                subject TEXT,
                status TEXT,
                sent_at TIMESTAMP,
                opened_at TIMESTAMP,
                clicked_at TIMESTAMP,
                bounce_reason TEXT,
                tracking_data TEXT
            )
        """)
        
        # Compliance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id TEXT,
                type TEXT,
                action TEXT,
                ip_address TEXT,
                user_agent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """)
        
        # Audit log table
        if self.config.audit_enabled:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create database indexes for performance"""
        # Leads indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_email ON leads(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_company ON leads(company)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_source ON leads(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_score ON leads(score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_created ON leads(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_retention ON leads(retention_date)")
        
        # Campaigns indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_campaigns_created ON campaigns(created_at)")
        
        # Email logs indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_campaign ON email_logs(campaign_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_lead ON email_logs(lead_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_status ON email_logs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_sent ON email_logs(sent_at)")
        
        # Compliance indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_compliance_lead ON compliance(lead_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_compliance_type ON compliance(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_compliance_timestamp ON compliance(timestamp)")
        
        # Audit log indexes
        if self.config.audit_enabled:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
    
    def _initialize_metadata(self, cursor: sqlite3.Cursor):
        """Initialize metadata table with default values"""
        metadata_defaults = {
            "schema_version": "1.0",
            "encryption_enabled": "true",
            "created_at": datetime.utcnow().isoformat(),
            "last_backup": None,
            "total_leads": "0",
            "total_campaigns": "0"
        }
        
        for key, value in metadata_defaults.items():
            cursor.execute(
                "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
                (key, str(value))
            )
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection"""
        thread_id = threading.get_ident()
        
        with self._connection_lock:
            if thread_id not in self._connections:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.config.timeout_seconds,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                self._connections[thread_id] = conn
            
            return self._connections[thread_id]
    
    def _encrypt_data(self, data: Union[str, Dict, List]) -> bytes:
        """Encrypt sensitive data before storage"""
        try:
            if isinstance(data, (dict, list)):
                data = json.dumps(data)
            
            return self._fernet.encrypt(data.encode())
        except Exception as e:
            logger.error(f"Data encryption failed: {str(e)}", exc_info=True)
            raise
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data after retrieval"""
        try:
            return self._fernet.decrypt(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Data decryption failed: {str(e)}", exc_info=True)
            raise
    
    def start_maintenance(self, backup_interval: int, max_connections: int):
        """Start background maintenance thread"""
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            logger.warning("Maintenance thread already running")
            return
        
        self._running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            args=(backup_interval, max_connections),
            name="StorageMaintenance",
            daemon=True
        )
        self._maintenance_thread.start()
        logger.info("Storage maintenance started")
    
    def stop_maintenance(self):
        """Stop background maintenance thread"""
        self._running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)
            logger.info("Storage maintenance stopped")
    
    def _maintenance_loop(self, backup_interval: int, max_connections: int):
        """Background maintenance loop"""
        while self._running:
            try:
                # Perform backup if enabled
                if self.config.backup_enabled:
                    self._perform_backup()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Optimize database
                self._optimize_database()
                
                # Wait for next cycle
                time.sleep(backup_interval * 3600)  # Convert hours to seconds
                
            except Exception as e:
                logger.error(f"Maintenance error: {str(e)}", exc_info=True)
                time.sleep(60)  # Wait before retrying
    
    def _perform_backup(self):
        """Perform database backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_path / f"drn_backup_{timestamp}.db"
            
            # Create backup
            shutil.copy2(self.db_path, backup_file)
            
            # Update metadata
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE metadata SET value = ?, updated_at = ? WHERE key = 'last_backup'",
                (timestamp, datetime.utcnow().isoformat())
            )
            conn.commit()
            
            # Clean old backups (keep last 7)
            backups = sorted(self.backup_path.glob("drn_backup_*.db"))
            for old_backup in backups[:-7]:
                old_backup.unlink()
            
            logger.info(f"Backup completed: {backup_file}")
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}", exc_info=True)
    
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        try:
            retention_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Clean old leads
            cursor.execute(
                "DELETE FROM leads WHERE retention_date < ?",
                (retention_date.isoformat(),)
            )
            
            # Clean old email logs
            cursor.execute(
                "DELETE FROM email_logs WHERE sent_at < ?",
                (retention_date.isoformat(),)
            )
            
            # Clean old audit logs
            if self.config.audit_enabled:
                cursor.execute(
                    "DELETE FROM audit_log WHERE timestamp < ?",
                    (retention_date.isoformat(),)
                )
            
            conn.commit()
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}", exc_info=True)
    
    def _optimize_database(self):
        """Optimize database performance"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if self.config.auto_vacuum:
                cursor.execute("VACUUM")
            
            cursor.execute("ANALYZE")
            conn.commit()
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {str(e)}", exc_info=True)
    
    def save_lead(self, lead_data: Dict[str, Any]) -> bool:
        """Save lead data with encryption for sensitive fields"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Extract sensitive data for encryption
            sensitive_fields = ["email", "phone", "social_links", "insights"]
            encrypted_data = {}
            
            for field in sensitive_fields:
                if field in lead_data and lead_data[field]:
                    encrypted_data[field] = self._encrypt_data(lead_data[field])
                    lead_data[field] = None  # Clear from main record
            
            # Convert complex fields to JSON
            if "tags" in lead_data and isinstance(lead_data["tags"], list):
                lead_data["tags"] = json.dumps(lead_data["tags"])
            
            if "social_links" in lead_data and isinstance(lead_data["social_links"], dict):
                lead_data["social_links"] = json.dumps(lead_data["social_links"])
            
            # Set retention date
            if "retention_date" not in lead_data:
                lead_data["retention_date"] = (
                    datetime.utcnow() + timedelta(days=self.config.retention_days)
                ).isoformat()
            
            # Insert or replace lead
            cursor.execute("""
                INSERT OR REPLACE INTO leads (
                    uuid, name, email, phone, company, website, location, social_links,
                    source, category, tags, score, persona, insights, retention_date, encrypted_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lead_data.get("uuid"),
                lead_data.get("name"),
                lead_data.get("email"),
                lead_data.get("phone"),
                lead_data.get("company"),
                lead_data.get("website"),
                lead_data.get("location"),
                lead_data.get("social_links"),
                lead_data.get("source"),
                lead_data.get("category"),
                lead_data.get("tags"),
                lead_data.get("score", 0.0),
                lead_data.get("persona"),
                lead_data.get("insights"),
                lead_data.get("retention_date"),
                json.dumps(encrypted_data) if encrypted_data else None
            ))
            
            conn.commit()
            
            # Log audit event
            if self.config.audit_enabled:
                self._log_audit(
                    action="save_lead",
                    resource_type="lead",
                    resource_id=lead_data.get("uuid"),
                    details={"source": lead_data.get("source")}
                )
            
            logger.debug(f"Lead saved: {lead_data.get('uuid')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save lead: {str(e)}", exc_info=True)
            return False
    
    def get_lead(self, lead_uuid: str) -> Optional[Dict[str, Any]]:
        """Retrieve lead data with decryption"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM leads WHERE uuid = ?", (lead_uuid,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            lead_data = dict(row)
            
            # Decrypt sensitive data
            if lead_data.get("encrypted_data"):
                encrypted_data = json.loads(self._decrypt_data(lead_data["encrypted_data"]))
                for field, value in encrypted_data.items():
                    lead_data[field] = self._decrypt_data(value)
            
            # Parse JSON fields
            if lead_data.get("tags"):
                lead_data["tags"] = json.loads(lead_data["tags"])
            
            if lead_data.get("social_links"):
                lead_data["social_links"] = json.loads(lead_data["social_links"])
            
            # Remove internal fields
            lead_data.pop("encrypted_data", None)
            
            return lead_data
            
        except Exception as e:
            logger.error(f"Failed to get lead {lead_uuid}: {str(e)}", exc_info=True)
            return None
    
    def update_lead(self, lead_uuid: str, updates: Dict[str, Any]) -> bool:
        """Update lead data with encryption for sensitive fields"""
        try:
            # Get existing lead
            existing_lead = self.get_lead(lead_uuid)
            if not existing_lead:
                logger.error(f"Lead not found: {lead_uuid}")
                return False
            
            # Merge updates
            existing_lead.update(updates)
            
            # Save updated lead
            return self.save_lead(existing_lead)
            
        except Exception as e:
            logger.error(f"Failed to update lead {lead_uuid}: {str(e)}", exc_info=True)
            return False
    
    def delete_lead(self, lead_uuid: str) -> bool:
        """Delete lead data"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM leads WHERE uuid = ?", (lead_uuid,))
            conn.commit()
            
            # Log audit event
            if self.config.audit_enabled:
                self._log_audit(
                    action="delete_lead",
                    resource_type="lead",
                    resource_id=lead_uuid
                )
            
            logger.debug(f"Lead deleted: {lead_uuid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete lead {lead_uuid}: {str(e)}", exc_info=True)
            return False
    
    def query_leads(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Query leads with optional filters"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM leads"
            params = []
            
            if filters:
                conditions = []
                if "source" in filters:
                    conditions.append("source = ?")
                    params.append(filters["source"])
                if "category" in filters:
                    conditions.append("category = ?")
                    params.append(filters["category"])
                if "min_score" in filters:
                    conditions.append("score >= ?")
                    params.append(filters["min_score"])
                if "created_after" in filters:
                    conditions.append("created_at >= ?")
                    params.append(filters["created_after"])
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            leads = []
            for row in rows:
                lead_data = dict(row)
                
                # Decrypt sensitive data
                if lead_data.get("encrypted_data"):
                    encrypted_data = json.loads(self._decrypt_data(lead_data["encrypted_data"]))
                    for field, value in encrypted_data.items():
                        lead_data[field] = self._decrypt_data(value)
                
                # Parse JSON fields
                if lead_data.get("tags"):
                    lead_data["tags"] = json.loads(lead_data["tags"])
                
                if lead_data.get("social_links"):
                    lead_data["social_links"] = json.loads(lead_data["social_links"])
                
                # Remove internal fields
                lead_data.pop("encrypted_data", None)
                
                leads.append(lead_data)
            
            return leads
            
        except Exception as e:
            logger.error(f"Failed to query leads: {str(e)}", exc_info=True)
            return []
    
    def save_campaign(self, campaign_data: Dict[str, Any]) -> bool:
        """Save campaign data"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Convert complex fields to JSON
            if "settings" in campaign_data and isinstance(campaign_data["settings"], dict):
                campaign_data["settings"] = json.dumps(campaign_data["settings"])
            
            # Insert or replace campaign
            cursor.execute("""
                INSERT OR REPLACE INTO campaigns (
                    uuid, name, description, template_id, status, settings
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                campaign_data.get("uuid"),
                campaign_data.get("name"),
                campaign_data.get("description"),
                campaign_data.get("template_id"),
                campaign_data.get("status", "draft"),
                campaign_data.get("settings")
            ))
            
            conn.commit()
            
            # Log audit event
            if self.config.audit_enabled:
                self._log_audit(
                    action="save_campaign",
                    resource_type="campaign",
                    resource_id=campaign_data.get("uuid"),
                    details={"name": campaign_data.get("name")}
                )
            
            logger.debug(f"Campaign saved: {campaign_data.get('uuid')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save campaign: {str(e)}", exc_info=True)
            return False
    
    def get_campaign(self, campaign_uuid: str) -> Optional[Dict[str, Any]]:
        """Retrieve campaign data"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM campaigns WHERE uuid = ?", (campaign_uuid,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            campaign_data = dict(row)
            
            # Parse JSON fields
            if campaign_data.get("settings"):
                campaign_data["settings"] = json.loads(campaign_data["settings"])
            
            return campaign_data
            
        except Exception as e:
            logger.error(f"Failed to get campaign {campaign_uuid}: {str(e)}", exc_info=True)
            return None
    
    def _log_audit(self, action: str, resource_type: str, resource_id: str, 
                   details: Dict[str, Any] = None, user_id: str = None):
        """Log audit event"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_log (
                    user_id, action, resource_type, resource_id, details
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                action,
                resource_type,
                resource_id,
                json.dumps(details) if details else None
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}", exc_info=True)
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            
            return row["value"] if row else None
            
        except Exception as e:
            logger.error(f"Failed to get metadata {key}: {str(e)}", exc_info=True)
            return None
    
    def set_metadata(self, key: str, value: str) -> bool:
        """Set metadata value"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.utcnow().isoformat()))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set metadata {key}: {str(e)}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get table counts
            stats = {}
            
            cursor.execute("SELECT COUNT(*) FROM leads")
            stats["total_leads"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM campaigns")
            stats["total_campaigns"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM email_logs")
            stats["total_emails"] = cursor.fetchone()[0]
            
            # Get database size
            stats["db_size_bytes"] = self.db_path.stat().st_size
            
            # Get last backup time
            last_backup = self.get_metadata("last_backup")
            stats["last_backup"] = last_backup if last_backup else "Never"
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}", exc_info=True)
            return {}
    
    def close(self):
        """Close all database connections"""
        try:
            with self._connection_lock:
                for conn in self._connections.values():
                    conn.close()
                self._connections.clear()
            
            self.stop_maintenance()
            logger.info("Storage connections closed")
            
        except Exception as e:
            logger.error(f"Failed to close storage: {str(e)}", exc_info=True)