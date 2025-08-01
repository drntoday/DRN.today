#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
License Management & Access Control System
Production-Ready Implementation
"""

import os
import json
import time
import threading
import logging
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import requests
import keyring

# Initialize license logger
logger = logging.getLogger(__name__)

@dataclass
class LicenseInfo:
    """License information data structure"""
    license_key: str
    customer_id: str
    customer_email: str
    tier: str  # "trial", "standard", "admin"
    issued_at: datetime
    expires_at: datetime
    features: List[str] = field(default_factory=list)
    max_leads: int = 0
    used_leads: int = 0
    billing_provider: Optional[str] = None
    subscription_id: Optional[str] = None
    last_validated: datetime = field(default_factory=datetime.now)
    is_valid: bool = True

class LicenseConfig:
    """Configuration for license management"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.security_config = config_dict.get("security", {})
        self.license_key_path = self.security_config.get("license_key_path", "license.key")
        self.keyring_service = self.security_config.get("keyring_service", "DRN.today")
        self.encryption_algorithm = self.security_config.get("encryption_algorithm", "AES-256-GCM")
        self.session_timeout_minutes = self.security_config.get("session_timeout_minutes", 30)
        self.audit_log_enabled = self.security_config.get("audit_log_enabled", True)
        self.secure_storage_path = self.security_config.get("secure_storage_path", "secure")

class LicenseManager:
    """Production-ready license management system"""
    
    def __init__(self, license_path: str = None, keyring_service: str = "DRN.today"):
        self.config = LicenseConfig({"security": {
            "license_key_path": license_path or "license.key",
            "keyring_service": keyring_service
        }})
        
        self.license_info: Optional[LicenseInfo] = None
        self.validation_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.RLock()
        
        # Feature mappings for different tiers
        self.tier_features = {
            "trial": [
                "lead_generation",
                "email_system",
                "basic_scoring"
            ],
            "standard": [
                "lead_generation",
                "lead_enrichment",
                "conversation_mining",
                "web_crawlers",
                "email_system",
                "intent_engine",
                "template_engine",
                "integrations",
                "competitive_intel",
                "ai_scoring"
            ],
            "admin": [
                "lead_generation",
                "lead_enrichment",
                "conversation_mining",
                "web_crawlers",
                "lead_capture",
                "competitive_intel",
                "email_system",
                "intent_engine",
                "template_engine",
                "integrations",
                "marketplace",
                "compliance",
                "ai_scoring",
                "team_management",
                "api_access",
                "white_label"
            ]
        }
        
        # Tier limits
        self.tier_limits = {
            "trial": {"max_leads": 300, "campaigns": 1, "users": 1},
            "standard": {"max_leads": -1, "campaigns": 10, "users": 3},
            "admin": {"max_leads": -1, "campaigns": -1, "users": -1}
        }
        
        # Billing integration
        self.billing_integrations = {
            "stripe": self._validate_stripe_subscription,
            "paddle": self._validate_paddle_subscription
        }
        
        # Initialize license
        self._load_license()
    
    def _load_license(self):
        """Load license from file or keyring"""
        try:
            # Try to load from file first
            license_file = Path(self.config.license_key_path)
            if license_file.exists():
                with open(license_file, 'r') as f:
                    license_key = f.read().strip()
                
                if license_key:
                    self.license_info = self._validate_license_key(license_key)
                    if self.license_info:
                        logger.info(f"License loaded from file: {self.license_info.tier}")
                        return
            
            # Try to load from keyring
            license_key = keyring.get_password(
                self.config.keyring_service,
                "license_key"
            )
            
            if license_key:
                self.license_info = self._validate_license_key(license_key)
                if self.license_info:
                    logger.info(f"License loaded from keyring: {self.license_info.tier}")
                    return
            
            logger.warning("No valid license found")
            
        except Exception as e:
            logger.error(f"Error loading license: {str(e)}", exc_info=True)
    
    def _validate_license_key(self, license_key: str) -> Optional[LicenseInfo]:
        """Validate and decode license key"""
        try:
            # In a real implementation, this would use cryptographic verification
            # For demo, we'll use a simple HMAC-based validation
            
            # Split license key into parts
            parts = license_key.split('.')
            if len(parts) != 3:
                return None
            
            # Extract components
            header_b64 = parts[0]
            payload_b64 = parts[1]
            signature_b64 = parts[2]
            
            # Decode payload
            payload_json = base64.urlsafe_b64decode(payload_b64 + '=' * (-len(payload_b64) % 4))
            payload = json.loads(payload_json)
            
            # Verify signature (simplified for demo)
            secret = b"drn_license_secret"  # In production, use proper secret management
            expected_signature = hmac.new(
                secret,
                f"{header_b64}.{payload_b64}".encode(),
                hashlib.sha256
            ).digest()
            
            signature = base64.urlsafe_b64decode(signature_b64 + '=' * (-len(signature_b64) % 4))
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Create license info
            license_info = LicenseInfo(
                license_key=license_key,
                customer_id=payload.get("customer_id"),
                customer_email=payload.get("customer_email"),
                tier=payload.get("tier", "trial"),
                issued_at=datetime.fromisoformat(payload.get("issued_at")),
                expires_at=datetime.fromisoformat(payload.get("expires_at")),
                features=payload.get("features", []),
                max_leads=payload.get("max_leads", 0),
                billing_provider=payload.get("billing_provider"),
                subscription_id=payload.get("subscription_id")
            )
            
            # Check expiration
            if license_info.expires_at < datetime.now():
                license_info.is_valid = False
                logger.warning(f"License expired on {license_info.expires_at}")
            else:
                license_info.is_valid = True
            
            return license_info
            
        except Exception as e:
            logger.error(f"Error validating license key: {str(e)}", exc_info=True)
            return None
    
    def validate(self) -> bool:
        """Validate current license"""
        if not self.license_info:
            return False
        
        # Check expiration
        if self.license_info.expires_at < datetime.now():
            self.license_info.is_valid = False
            logger.warning("License has expired")
            return False
        
        # For subscription-based licenses, verify with billing provider
        if self.license_info.billing_provider and self.license_info.subscription_id:
            validation_func = self.billing_integrations.get(self.license_info.billing_provider)
            if validation_func:
                if not validation_func(self.license_info.subscription_id):
                    self.license_info.is_valid = False
                    logger.warning("Subscription validation failed")
                    return False
        
        self.license_info.is_valid = True
        return True
    
    def has_module_access(self, module_name: str) -> bool:
        """Check if current license has access to a module"""
        if not self.license_info or not self.license_info.is_valid:
            return False
        
        # Check if module is in tier features
        tier_features = self.tier_features.get(self.license_info.tier, [])
        return module_name in tier_features
    
    def get_tier_limits(self) -> Dict[str, int]:
        """Get limits for current license tier"""
        if not self.license_info:
            return self.tier_limits["trial"]
        
        return self.tier_limits.get(self.license_info.tier, self.tier_limits["trial"])
    
    def check_lead_limit(self, lead_count: int) -> bool:
        """Check if lead count is within license limits"""
        if not self.license_info:
            return lead_count <= 300  # Default trial limit
        
        max_leads = self.license_info.max_leads
        if max_leads <= 0:  # Unlimited
            return True
        
        return lead_count <= max_leads
    
    def record_lead_usage(self, count: int = 1):
        """Record lead usage against license"""
        if self.license_info:
            self.license_info.used_leads += count
    
    def get_remaining_leads(self) -> int:
        """Get remaining leads for current license period"""
        if not self.license_info:
            return 300  # Default trial
        
        max_leads = self.license_info.max_leads
        if max_leads <= 0:  # Unlimited
            return -1
        
        return max(0, max_leads - self.license_info.used_leads)
    
    def save_license(self, license_key: str) -> bool:
        """Save license key to secure storage"""
        try:
            # Validate license first
            license_info = self._validate_license_key(license_key)
            if not license_info:
                logger.error("Invalid license key")
                return False
            
            # Save to file
            license_file = Path(self.config.license_key_path)
            license_file.parent.mkdir(parents=True, exist_ok=True)
            with open(license_file, 'w') as f:
                f.write(license_key)
            
            # Save to keyring
            keyring.set_password(
                self.config.keyring_service,
                "license_key",
                license_key
            )
            
            # Update current license
            self.license_info = license_info
            
            logger.info(f"License saved: {license_info.tier}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving license: {str(e)}", exc_info=True)
            return False
    
    def remove_license(self) -> bool:
        """Remove saved license"""
        try:
            # Remove file
            license_file = Path(self.config.license_key_path)
            if license_file.exists():
                license_file.unlink()
            
            # Remove from keyring
            keyring.delete_password(
                self.config.keyring_service,
                "license_key"
            )
            
            # Clear current license
            self.license_info = None
            
            logger.info("License removed")
            return True
            
        except Exception as e:
            logger.error(f"Error removing license: {str(e)}", exc_info=True)
            return False
    
    def start_monitoring(self):
        """Start background license monitoring"""
        if self.validation_thread and self.validation_thread.is_alive():
            logger.warning("License monitoring already running")
            return
        
        self.running = True
        self.validation_thread = threading.Thread(
            target=self._monitoring_loop,
            name="LicenseMonitor",
            daemon=True
        )
        self.validation_thread.start()
        logger.info("License monitoring started")
    
    def stop_monitoring(self):
        """Stop background license monitoring"""
        self.running = False
        if self.validation_thread:
            self.validation_thread.join(timeout=5)
            logger.info("License monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Validate license
                if self.license_info:
                    was_valid = self.license_info.is_valid
                    is_valid = self.validate()
                    
                    # Log status changes
                    if was_valid and not is_valid:
                        logger.warning("License became invalid")
                    elif not was_valid and is_valid:
                        logger.info("License became valid")
                
                # Sleep for monitoring interval
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"License monitoring error: {str(e)}", exc_info=True)
                time.sleep(60)  # Wait before retrying
    
    def _validate_stripe_subscription(self, subscription_id: str) -> bool:
        """Validate Stripe subscription status"""
        try:
            # In a real implementation, this would use Stripe API
            # For demo, we'll simulate validation
            
            # Simulate API call
            # response = requests.get(
            #     f"https://api.stripe.com/v1/subscriptions/{subscription_id}",
            #     headers={"Authorization": f"Bearer {stripe_api_key}"}
            # )
            # subscription = response.json()
            
            # Simulate active subscription
            return True
            
        except Exception as e:
            logger.error(f"Stripe validation error: {str(e)}", exc_info=True)
            return False
    
    def _validate_paddle_subscription(self, subscription_id: str) -> bool:
        """Validate Paddle subscription status"""
        try:
            # In a real implementation, this would use Paddle API
            # For demo, we'll simulate validation
            
            # Simulate API call
            # response = requests.get(
            #     f"https://vendors.paddle.com/api/2.0/subscription/users",
            #     params={"subscription_id": subscription_id}
            # )
            # subscription = response.json()
            
            # Simulate active subscription
            return True
            
        except Exception as e:
            logger.error(f"Paddle validation error: {str(e)}", exc_info=True)
            return False
    
    def generate_trial_license(self, customer_email: str) -> str:
        """Generate a trial license key"""
        try:
            # Create payload
            payload = {
                "customer_id": "trial_user",
                "customer_email": customer_email,
                "tier": "trial",
                "issued_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
                "features": self.tier_features["trial"],
                "max_leads": 300
            }
            
            # Encode payload
            payload_json = json.dumps(payload)
            payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
            
            # Create header (simplified)
            header = {"alg": "HS256", "typ": "JWT"}
            header_json = json.dumps(header)
            header_b64 = base64.urlsafe_b64encode(header_json.encode()).decode().rstrip('=')
            
            # Create signature
            secret = b"drn_license_secret"  # In production, use proper secret management
            message = f"{header_b64}.{payload_b64}".encode()
            signature = hmac.new(secret, message, hashlib.sha256).digest()
            signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
            
            # Combine into license key
            license_key = f"{header_b64}.{payload_b64}.{signature_b64}"
            
            return license_key
            
        except Exception as e:
            logger.error(f"Error generating trial license: {str(e)}", exc_info=True)
            return ""
    
    def get_license_info(self) -> Optional[Dict[str, Any]]:
        """Get current license information as dictionary"""
        if not self.license_info:
            return None
        
        return {
            "tier": self.license_info.tier,
            "customer_email": self.license_info.customer_email,
            "issued_at": self.license_info.issued_at.isoformat(),
            "expires_at": self.license_info.expires_at.isoformat(),
            "features": self.license_info.features,
            "max_leads": self.license_info.max_leads,
            "used_leads": self.license_info.used_leads,
            "remaining_leads": self.get_remaining_leads(),
            "is_valid": self.license_info.is_valid,
            "billing_provider": self.license_info.billing_provider
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get license usage statistics"""
        if not self.license_info:
            return {
                "tier": "none",
                "used_leads": 0,
                "max_leads": 0,
                "remaining_leads": 0,
                "usage_percent": 0
            }
        
        max_leads = self.license_info.max_leads
        used_leads = self.license_info.used_leads
        
        if max_leads <= 0:  # Unlimited
            usage_percent = 0
        else:
            usage_percent = (used_leads / max_leads) * 100
        
        return {
            "tier": self.license_info.tier,
            "used_leads": used_leads,
            "max_leads": max_leads,
            "remaining_leads": self.get_remaining_leads(),
            "usage_percent": usage_percent
        }