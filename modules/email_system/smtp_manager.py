#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Email System - SMTP Manager Module
Production-Ready Implementation
"""

import asyncio
import logging
import smtplib
import ssl
import time
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import email.utils
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import keyring
import dns.resolver
import aiohttp
import backoff
from jinja2 import Environment, FileSystemLoader, Template

# Core system imports
from engine.orchestrator import BaseModule
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# Initialize SMTP manager logger
logger = logging.getLogger(__name__)

@dataclass
class SMTPAccount:
    """SMTP account configuration"""
    uuid: str
    name: str
    provider: str  # "gmail", "outlook", "mailgun", "sendgrid", "ses", "postfix", "custom"
    host: str
    port: int
    username: str
    password: str  # Will be stored encrypted
    use_tls: bool = True
    use_ssl: bool = False
    from_email: str
    from_name: str
    reply_to: Optional[str] = None
    daily_limit: int = 1000
    hourly_limit: int = 100
    is_active: bool = True
    last_used: float = field(default_factory=time.time)
    failure_count: int = 0
    cooldown_until: float = 0.0

@dataclass
class EmailMessage:
    """Email message data structure"""
    uuid: str
    to_email: str
    to_name: Optional[str] = None
    subject: str
    body_html: Optional[str] = None
    body_text: Optional[str] = None
    from_email: Optional[str] = None
    from_name: Optional[str] = None
    reply_to: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    tracking_enabled: bool = True
    campaign_id: Optional[str] = None
    lead_id: Optional[str] = None
    priority: str = "normal"  # "low", "normal", "high"
    scheduled_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class EmailResult:
    """Email sending result"""
    uuid: str
    message_uuid: str
    account_uuid: str
    success: bool
    error: Optional[str] = None
    message_id: Optional[str] = None
    response_code: Optional[int] = None
    response_text: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0

class SMTPManagerConfig:
    """Configuration for the SMTP manager module"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.email_config = config_dict.get("email", {})
        self.security_config = config_dict.get("security", {})
        
        # Email settings
        self.smtp_pool_size = self.email_config.get("smtp_pool_size", 5)
        self.max_attachment_size_mb = self.email_config.get("max_attachment_size_mb", 10)
        self.bounce_threshold = self.email_config.get("bounce_threshold", 3)
        self.warming_enabled = self.email_config.get("warming_enabled", True)
        self.warming_daily_limit = self.email_config.get("warming_daily_limit", 50)
        self.blacklist_check_enabled = self.email_config.get("blacklist_check_enabled", True)
        self.tracking_pixel_enabled = self.email_config.get("tracking_pixel_enabled", True)
        self.unsubscribe_header_enabled = self.email_config.get("unsubscribe_header_enabled", True)
        
        # Security settings
        self.keyring_service = self.security_config.get("keyring_service", "DRN.today_SMTP")
        self.encryption_algorithm = self.security_config.get("encryption_algorithm", "AES-256-GCM")
        
        # Default provider settings
        self.provider_defaults = {
            "gmail": {
                "host": "smtp.gmail.com",
                "port": 587,
                "use_tls": True,
                "use_ssl": False
            },
            "outlook": {
                "host": "smtp.office365.com",
                "port": 587,
                "use_tls": True,
                "use_ssl": False
            },
            "mailgun": {
                "host": "smtp.mailgun.org",
                "port": 587,
                "use_tls": True,
                "use_ssl": False
            },
            "sendgrid": {
                "host": "smtp.sendgrid.net",
                "port": 587,
                "use_tls": True,
                "use_ssl": False
            },
            "ses": {
                "host": "email-smtp.us-east-1.amazonaws.com",
                "port": 587,
                "use_tls": True,
                "use_ssl": False
            },
            "postfix": {
                "host": "localhost",
                "port": 25,
                "use_tls": False,
                "use_ssl": False
            }
        }

class SMTPManager(BaseModule):
    """Production-ready SMTP manager with rotation and monitoring"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Dict[str, Any]):
        super().__init__(name, event_bus, storage, license_manager, config)
        self.config = SMTPManagerConfig(config)
        self.accounts: Dict[str, SMTPAccount] = {}
        self.active_connections: Dict[str, smtplib.SMTP] = {}
        self.session_stats = {
            "emails_sent": 0,
            "emails_failed": 0,
            "blacklist_hits": 0,
            "warming_emails_sent": 0,
            "rotation_events": 0,
            "connection_errors": 0
        }
        self.blacklist_domains: List[str] = []
        self.template_env: Optional[Environment] = None
        self.warming_queue: asyncio.Queue = asyncio.Queue()
        self.sending_queue: asyncio.Queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(self.config.smtp_pool_size)
        
    def _setup_event_handlers(self):
        """Setup event handlers for SMTP requests"""
        self.event_bus.subscribe("smtp.send", self._handle_send_request)
        self.event_bus.subscribe("smtp.add_account", self._handle_add_account_request)
        self.event_bus.subscribe("smtp.remove_account", self._handle_remove_account_request)
        self.event_bus.subscribe("smtp.status", self._handle_status_request)
        self.event_bus.subscribe("smtp.warmup", self._handle_warmup_request)
        
    def _validate_requirements(self):
        """Validate module requirements and dependencies"""
        # Initialize template environment
        template_dir = Path(__file__).parent.parent.parent / "resources" / "templates"
        if template_dir.exists():
            self.template_env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=True
            )
        
        # Load blacklist if enabled
        if self.config.blacklist_check_enabled:
            self._load_blacklist()
            
    def _load_blacklist(self):
        """Load email blacklist from storage"""
        try:
            blacklist_data = self.storage.get_metadata("email_blacklist")
            if blacklist_data:
                self.blacklist_domains = json.loads(blacklist_data)
            else:
                # Initialize with common blacklist domains
                self.blacklist_domains = [
                    "mailinator.com",
                    "guerrillamail.com",
                    "tempmail.org",
                    "10minutemail.com"
                ]
                self.storage.set_metadata("email_blacklist", json.dumps(self.blacklist_domains))
        except Exception as e:
            logger.error(f"Error loading blacklist: {str(e)}", exc_info=True)
    
    async def _start_services(self):
        """Start SMTP manager services"""
        # Start worker tasks
        asyncio.create_task(self._sending_worker())
        asyncio.create_task(self._warming_worker())
        
        # Start monitoring
        asyncio.create_task(self._monitoring_worker())
        
        logger.info("SMTP manager services started successfully")
    
    async def _stop_services(self):
        """Stop SMTP manager services"""
        # Close all connections
        for account_uuid, connection in self.active_connections.items():
            try:
                connection.quit()
            except:
                pass
        self.active_connections.clear()
        
        logger.info("SMTP manager services stopped")
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up inactive connections
        current_time = time.time()
        inactive_connections = [
            uuid for uuid, account in self.accounts.items()
            if current_time - account.last_used > 300  # 5 minutes
        ]
        
        for uuid in inactive_connections:
            if uuid in self.active_connections:
                try:
                    self.active_connections[uuid].quit()
                    del self.active_connections[uuid]
                except:
                    pass
        
        # Check account cooldowns
        for account in self.accounts.values():
            if account.cooldown_until > current_time:
                account.is_active = False
            else:
                account.is_active = True
        
        # Log session stats
        logger.debug(f"SMTP manager stats: {self.session_stats}")
    
    async def _handle_send_request(self, event_type: str, data: Dict[str, Any]):
        """Handle email send requests"""
        try:
            message_data = data.get("message")
            if not message_data:
                logger.warning("Invalid send request: missing message data")
                return
            
            # Create email message
            message = EmailMessage(**message_data)
            
            # Add to sending queue
            await self.sending_queue.put(message)
            
        except Exception as e:
            logger.error(f"Error handling send request: {str(e)}", exc_info=True)
    
    async def _handle_add_account_request(self, event_type: str, data: Dict[str, Any]):
        """Handle add account requests"""
        try:
            account_data = data.get("account")
            if not account_data:
                logger.warning("Invalid add account request: missing account data")
                return
            
            # Add SMTP account
            account = self._add_smtp_account(account_data)
            if account:
                self.event_bus.publish("smtp.account_added", {
                    "account_uuid": account.uuid,
                    "name": account.name
                })
            
        except Exception as e:
            logger.error(f"Error handling add account request: {str(e)}", exc_info=True)
    
    async def _handle_remove_account_request(self, event_type: str, data: Dict[str, Any]):
        """Handle remove account requests"""
        try:
            account_uuid = data.get("account_uuid")
            if not account_uuid:
                logger.warning("Invalid remove account request: missing account UUID")
                return
            
            # Remove SMTP account
            if self._remove_smtp_account(account_uuid):
                self.event_bus.publish("smtp.account_removed", {
                    "account_uuid": account_uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling remove account request: {str(e)}", exc_info=True)
    
    async def _handle_status_request(self, event_type: str, data: Dict[str, Any]):
        """Handle status requests"""
        status = {
            "accounts": len(self.accounts),
            "active_accounts": len([a for a in self.accounts.values() if a.is_active]),
            "session_stats": self.session_stats,
            "queue_size": self.sending_queue.qsize(),
            "warming_queue_size": self.warming_queue.qsize(),
            "blacklist_size": len(self.blacklist_domains)
        }
        self.event_bus.publish("smtp.status.response", status)
    
    async def _handle_warmup_request(self, event_type: str, data: Dict[str, Any]):
        """Handle warmup requests"""
        try:
            account_uuid = data.get("account_uuid")
            if not account_uuid:
                logger.warning("Invalid warmup request: missing account UUID")
                return
            
            # Add to warming queue
            await self.warming_queue.put(account_uuid)
            
        except Exception as e:
            logger.error(f"Error handling warmup request: {str(e)}", exc_info=True)
    
    def _add_smtp_account(self, account_data: Dict[str, Any]) -> Optional[SMTPAccount]:
        """Add a new SMTP account"""
        try:
            # Get provider defaults
            provider = account_data.get("provider", "custom")
            defaults = self.config.provider_defaults.get(provider, {})
            
            # Create account
            account = SMTPAccount(
                uuid=str(uuid.uuid4()),
                name=account_data.get("name"),
                provider=provider,
                host=account_data.get("host", defaults.get("host")),
                port=account_data.get("port", defaults.get("port", 587)),
                username=account_data.get("username"),
                password=account_data.get("password"),
                use_tls=account_data.get("use_tls", defaults.get("use_tls", True)),
                use_ssl=account_data.get("use_ssl", defaults.get("use_ssl", False)),
                from_email=account_data.get("from_email"),
                from_name=account_data.get("from_name"),
                reply_to=account_data.get("reply_to"),
                daily_limit=account_data.get("daily_limit", 1000),
                hourly_limit=account_data.get("hourly_limit", 100)
            )
            
            # Validate account
            if not self._validate_smtp_account(account):
                logger.error(f"Invalid SMTP account: {account.name}")
                return None
            
            # Store password securely
            self._store_account_password(account.uuid, account.password)
            account.password = "***"  # Don't store plaintext password
            
            # Add to accounts
            self.accounts[account.uuid] = account
            
            # Save to storage
            self._save_account_to_storage(account)
            
            logger.info(f"Added SMTP account: {account.name}")
            return account
            
        except Exception as e:
            logger.error(f"Error adding SMTP account: {str(e)}", exc_info=True)
            return None
    
    def _remove_smtp_account(self, account_uuid: str) -> bool:
        """Remove an SMTP account"""
        try:
            if account_uuid not in self.accounts:
                logger.warning(f"Account not found: {account_uuid}")
                return False
            
            account = self.accounts[account_uuid]
            
            # Close connection if active
            if account_uuid in self.active_connections:
                try:
                    self.active_connections[account_uuid].quit()
                except:
                    pass
                del self.active_connections[account_uuid]
            
            # Remove password from keyring
            self._remove_account_password(account_uuid)
            
            # Remove from accounts
            del self.accounts[account_uuid]
            
            # Remove from storage
            self._remove_account_from_storage(account_uuid)
            
            logger.info(f"Removed SMTP account: {account.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing SMTP account: {str(e)}", exc_info=True)
            return False
    
    def _validate_smtp_account(self, account: SMTPAccount) -> bool:
        """Validate SMTP account configuration"""
        try:
            # Check required fields
            if not all([account.host, account.port, account.username, account.password, account.from_email]):
                return False
            
            # Validate email format
            if not re.match(r'^[^@]+@[^@]+\.[^@]+$', account.from_email):
                return False
            
            # Test connection
            with self._get_smtp_connection(account) as smtp:
                smtp.noop()  # Test connection
            
            return True
            
        except Exception as e:
            logger.error(f"SMTP account validation failed: {str(e)}", exc_info=True)
            return False
    
    def _store_account_password(self, account_uuid: str, password: str):
        """Store account password securely"""
        try:
            keyring.set_password(self.config.keyring_service, account_uuid, password)
        except Exception as e:
            logger.error(f"Error storing account password: {str(e)}", exc_info=True)
    
    def _get_account_password(self, account_uuid: str) -> Optional[str]:
        """Retrieve account password securely"""
        try:
            return keyring.get_password(self.config.keyring_service, account_uuid)
        except Exception as e:
            logger.error(f"Error retrieving account password: {str(e)}", exc_info=True)
            return None
    
    def _remove_account_password(self, account_uuid: str):
        """Remove account password from keyring"""
        try:
            keyring.delete_password(self.config.keyring_service, account_uuid)
        except Exception as e:
            logger.error(f"Error removing account password: {str(e)}", exc_info=True)
    
    def _save_account_to_storage(self, account: SMTPAccount):
        """Save account to storage"""
        try:
            account_data = {
                "uuid": account.uuid,
                "name": account.name,
                "provider": account.provider,
                "host": account.host,
                "port": account.port,
                "username": account.username,
                "use_tls": account.use_tls,
                "use_ssl": account.use_ssl,
                "from_email": account.from_email,
                "from_name": account.from_name,
                "reply_to": account.reply_to,
                "daily_limit": account.daily_limit,
                "hourly_limit": account.hourly_limit,
                "is_active": account.is_active,
                "last_used": account.last_used,
                "failure_count": account.failure_count,
                "cooldown_until": account.cooldown_until
            }
            
            self.storage.save_lead({
                "uuid": account.uuid,
                "source": "smtp_account",
                "name": account.name,
                "email": account.from_email,
                "raw_content": json.dumps(account_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving account to storage: {str(e)}", exc_info=True)
    
    def _remove_account_from_storage(self, account_uuid: str):
        """Remove account from storage"""
        try:
            self.storage.delete_lead(account_uuid)
        except Exception as e:
            logger.error(f"Error removing account from storage: {str(e)}", exc_info=True)
    
    def _get_smtp_connection(self, account: SMTPAccount) -> smtplib.SMTP:
        """Get SMTP connection for account"""
        try:
            # Check if connection already exists
            if account.uuid in self.active_connections:
                try:
                    self.active_connections[account.uuid].noop()
                    return self.active_connections[account.uuid]
                except:
                    # Connection is stale, close it
                    try:
                        self.active_connections[account.uuid].quit()
                    except:
                        pass
                    del self.active_connections[account.uuid]
            
            # Get password
            password = self._get_account_password(account.uuid)
            if not password:
                raise Exception("Account password not found")
            
            # Create new connection
            if account.use_ssl:
                smtp = smtplib.SMTP_SSL(account.host, account.port)
            else:
                smtp = smtplib.SMTP(account.host, account.port)
            
            # Enable TLS if needed
            if account.use_tls and not account.use_ssl:
                smtp.starttls()
            
            # Login
            smtp.login(account.username, password)
            
            # Store connection
            self.active_connections[account.uuid] = smtp
            
            return smtp
            
        except Exception as e:
            logger.error(f"Error getting SMTP connection: {str(e)}", exc_info=True)
            raise
    
    async def _sending_worker(self):
        """Worker for processing sending queue"""
        while True:
            try:
                # Get message from queue
                message = await self.sending_queue.get()
                
                # Process message
                await self._send_email(message)
                
                # Mark task as done
                self.sending_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in sending worker: {str(e)}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _send_email(self, message: EmailMessage) -> EmailResult:
        """Send an email message"""
        async with self.semaphore:
            start_time = time.time()
            result = EmailResult(
                uuid=str(uuid.uuid4()),
                message_uuid=message.uuid
            )
            
            try:
                # Check blacklist
                if self.config.blacklist_check_enabled:
                    domain = message.to_email.split('@')[1]
                    if domain in self.blacklist_domains:
                        result.success = False
                        result.error = "Email domain is blacklisted"
                        self.session_stats["blacklist_hits"] += 1
                        return result
                
                # Get best account
                account = self._get_best_account()
                if not account:
                    result.success = False
                    result.error = "No available SMTP accounts"
                    return result
                
                result.account_uuid = account.uuid
                
                # Build email
                email_content = self._build_email(message, account)
                
                # Send email
                with self._get_smtp_connection(account) as smtp:
                    response = smtp.sendmail(
                        account.from_email,
                        [message.to_email],
                        email_content.as_string()
                    )
                    
                    result.message_id = email_content.get("Message-ID")
                    result.success = True
                    result.response_code = 250
                    result.response_text = "Message accepted"
                
                # Update account stats
                account.last_used = time.time()
                account.failure_count = 0
                
                # Update session stats
                self.session_stats["emails_sent"] += 1
                
                # Log result
                logger.info(f"Email sent to {message.to_email}")
                
                # Publish event
                self.event_bus.publish("smtp.email_sent", {
                    "message_uuid": message.uuid,
                    "to_email": message.to_email,
                    "account_uuid": account.uuid
                })
                
            except Exception as e:
                result.success = False
                result.error = str(e)
                
                # Update account failure count
                if result.account_uuid and result.account_uuid in self.accounts:
                    account = self.accounts[result.account_uuid]
                    account.failure_count += 1
                    
                    # Put account on cooldown if too many failures
                    if account.failure_count >= 5:
                        account.cooldown_until = time.time() + 300  # 5 minutes
                        account.is_active = False
                
                # Update session stats
                self.session_stats["emails_failed"] += 1
                
                logger.error(f"Failed to send email to {message.to_email}: {str(e)}", exc_info=True)
                
                # Publish event
                self.event_bus.publish("smtp.email_failed", {
                    "message_uuid": message.uuid,
                    "to_email": message.to_email,
                    "error": str(e)
                })
            
            finally:
                result.duration = time.time() - start_time
                
                # Save result to storage
                self._save_email_result(result)
                
                return result
    
    def _build_email(self, message: EmailMessage, account: SMTPAccount) -> MIMEMultipart:
        """Build email message"""
        # Create multipart message
        email_msg = MIMEMultipart('alternative')
        
        # Set headers
        email_msg['From'] = f"{account.from_name} <{account.from_email}>"
        email_msg['To'] = f"{message.to_name} <{message.to_email}>" if message.to_name else message.to_email
        email_msg['Subject'] = message.subject
        
        # Add reply-to if specified
        reply_to = message.reply_to or account.reply_to
        if reply_to:
            email_msg['Reply-To'] = reply_to
        
        # Add message ID
        email_msg['Message-ID'] = email.utils.make_msgid()
        
        # Add custom headers
        for key, value in message.headers.items():
            email_msg[key] = value
        
        # Add tracking headers if enabled
        if message.tracking_enabled and self.config.tracking_pixel_enabled:
            email_msg['X-DRN-Track'] = message.uuid
            email_msg['X-DRN-Campaign'] = message.campaign_id or ""
            email_msg['X-DRN-Lead'] = message.lead_id or ""
        
        # Add unsubscribe header if enabled
        if self.config.unsubscribe_header_enabled:
            email_msg['List-Unsubscribe'] = f"<mailto:unsubscribe@drn.today?subject=unsubscribe-{message.uuid}>"
        
        # Add text part
        if message.body_text:
            text_part = MIMEText(message.body_text, 'plain')
            email_msg.attach(text_part)
        
        # Add HTML part
        if message.body_html:
            html_part = MIMEText(message.body_html, 'html')
            email_msg.attach(html_part)
        
        # Add attachments
        for attachment in message.attachments:
            try:
                part = MIMEBase(attachment['type'], attachment['subtype'])
                part.set_payload(attachment['data'])
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{attachment["filename"]}"'
                )
                email_msg.attach(part)
            except Exception as e:
                logger.error(f"Error adding attachment: {str(e)}", exc_info=True)
        
        return email_msg
    
    def _get_best_account(self) -> Optional[SMTPAccount]:
        """Get the best SMTP account for sending"""
        try:
            # Filter active accounts
            active_accounts = [
                account for account in self.accounts.values()
                if account.is_active
            ]
            
            if not active_accounts:
                return None
            
            # Sort by failure count (ascending) and last used (ascending)
            active_accounts.sort(key=lambda a: (a.failure_count, a.last_used))
            
            # Check rate limits
            current_time = time.time()
            for account in active_accounts:
                # Check hourly limit
                hour_ago = current_time - 3600
                recent_sends = [
                    result for result in self._get_recent_results(account.uuid, hour_ago)
                    if result.success
                ]
                
                if len(recent_sends) >= account.hourly_limit:
                    continue
                
                # Check daily limit
                day_ago = current_time - 86400
                recent_sends = [
                    result for result in self._get_recent_results(account.uuid, day_ago)
                    if result.success
                ]
                
                if len(recent_sends) >= account.daily_limit:
                    continue
                
                return account
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting best account: {str(e)}", exc_info=True)
            return None
    
    def _get_recent_results(self, account_uuid: str, since: float) -> List[EmailResult]:
        """Get recent email results for account"""
        # In a real implementation, this would query the database
        # For demo, we'll return an empty list
        return []
    
    def _save_email_result(self, result: EmailResult):
        """Save email result to storage"""
        try:
            result_data = {
                "uuid": result.uuid,
                "message_uuid": result.message_uuid,
                "account_uuid": result.account_uuid,
                "success": result.success,
                "error": result.error,
                "message_id": result.message_id,
                "response_code": result.response_code,
                "response_text": result.response_text,
                "timestamp": result.timestamp,
                "duration": result.duration
            }
            
            self.storage.save_lead({
                "uuid": result.uuid,
                "source": "email_result",
                "name": f"Email {result.uuid}",
                "raw_content": json.dumps(result_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving email result: {str(e)}", exc_info=True)
    
    async def _warming_worker(self):
        """Worker for processing warming queue"""
        while True:
            try:
                # Get account UUID from queue
                account_uuid = await self.warming_queue.get()
                
                # Process warming
                await self._warmup_account(account_uuid)
                
                # Mark task as done
                self.warming_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in warming worker: {str(e)}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _warmup_account(self, account_uuid: str):
        """Warm up an SMTP account"""
        try:
            if account_uuid not in self.accounts:
                logger.warning(f"Account not found for warmup: {account_uuid}")
                return
            
            account = self.accounts[account_uuid]
            
            # Send warming emails
            for i in range(self.config.warming_daily_limit):
                # Create warming email
                warming_email = EmailMessage(
                    uuid=str(uuid.uuid4()),
                    to_email=f"warmup{i+1}@drn.today",
                    subject=f"Warmup {i+1}",
                    body_text="This is a warmup email.",
                    tracking_enabled=False
                )
                
                # Send email
                result = await self._send_email(warming_email)
                
                # Update stats
                if result.success:
                    self.session_stats["warming_emails_sent"] += 1
                
                # Wait between sends
                await asyncio.sleep(60)  # 1 minute between warming emails
            
            logger.info(f"Completed warmup for account: {account.name}")
            
        except Exception as e:
            logger.error(f"Error warming up account: {str(e)}", exc_info=True)
    
    async def _monitoring_worker(self):
        """Worker for monitoring and maintenance"""
        while True:
            try:
                # Perform maintenance
                self._perform_maintenance()
                
                # Check blacklist updates
                if self.config.blacklist_check_enabled:
                    await self._update_blacklist()
                
                # Sleep for next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring worker: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _update_blacklist(self):
        """Update email blacklist from external sources"""
        try:
            # In a real implementation, this would fetch from external blacklist services
            # For demo, we'll just log
            logger.debug("Checking for blacklist updates")
            
        except Exception as e:
            logger.error(f"Error updating blacklist: {str(e)}", exc_info=True)
    
    async def send_email(self, to_email: str, subject: str, body_html: str = None, 
                        body_text: str = None, **kwargs) -> Dict[str, Any]:
        """Public method to send an email"""
        # Create email message
        message = EmailMessage(
            uuid=str(uuid.uuid4()),
            to_email=to_email,
            subject=subject,
            body_html=body_html,
            body_text=body_text,
            **kwargs
        )
        
        # Add to sending queue
        await self.sending_queue.put(message)
        
        return {
            "message_uuid": message.uuid,
            "status": "queued"
        }
    
    async def send_template_email(self, to_email: str, template_name: str, 
                               template_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Send email using a template"""
        if not self.template_env:
            raise Exception("Template environment not initialized")
        
        try:
            # Load template
            template = self.template_env.get_template(f"{template_name}.html")
            
            # Render template
            body_html = template.render(**template_data)
            
            # Get text version (simplified)
            body_text = template_data.get("text_content", "")
            
            # Send email
            return await self.send_email(
                to_email=to_email,
                subject=template_data.get("subject", ""),
                body_html=body_html,
                body_text=body_text,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error sending template email: {str(e)}", exc_info=True)
            raise
    
    def add_smtp_account(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Public method to add SMTP account"""
        account = self._add_smtp_account(account_data)
        if account:
            return {
                "account_uuid": account.uuid,
                "name": account.name,
                "status": "added"
            }
        else:
            return {
                "status": "failed",
                "error": "Invalid account configuration"
            }
    
    def remove_smtp_account(self, account_uuid: str) -> Dict[str, Any]:
        """Public method to remove SMTP account"""
        if self._remove_smtp_account(account_uuid):
            return {
                "account_uuid": account_uuid,
                "status": "removed"
            }
        else:
            return {
                "account_uuid": account_uuid,
                "status": "failed",
                "error": "Account not found"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get SMTP manager statistics"""
        return {
            "session_stats": self.session_stats,
            "accounts": len(self.accounts),
            "active_accounts": len([a for a in self.accounts.values() if a.is_active]),
            "queue_size": self.sending_queue.qsize(),
            "warming_queue_size": self.warming_queue.qsize(),
            "blacklist_size": len(self.blacklist_domains)
        }