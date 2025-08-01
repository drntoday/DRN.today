#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Template Engine - Templates Module
Production-Ready Implementation
"""

import os
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import re
import hashlib
import random
from collections import defaultdict

import jinja2
from jinja2 import Environment, FileSystemLoader, Template, meta
from jinja2.exceptions import TemplateError, TemplateSyntaxError
import aiohttp
import asyncio
import numpy as np
from bs4 import BeautifulSoup

# Core system imports
from engine.orchestrator import BaseModule
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# Initialize templates logger
logger = logging.getLogger(__name__)

@dataclass
class TemplateVersion:
    """Template version for A/B testing"""
    uuid: str
    template_uuid: str
    name: str
    subject: str
    html_content: str
    text_content: str
    version: str = "A"
    weight: float = 0.5  # For A/B testing distribution
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmailTemplate:
    """Email template data structure"""
    uuid: str
    name: str
    description: str
    category: str
    versions: List[TemplateVersion] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    follow_up_sequence: Optional[str] = None  # UUID of follow-up sequence
    ab_test_enabled: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    usage_count: int = 0
    success_rate: float = 0.0

@dataclass
class FollowUpSequence:
    """Follow-up email sequence"""
    uuid: str
    name: str
    description: str
    templates: List[str] = field(default_factory=list)  # List of template UUIDs
    delays: List[int] = field(default_factory=list)  # Delays in hours between emails
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # Conditions for each step
    is_active: bool = True
    created_at: float = field(default_factory=time.time)

@dataclass
class TemplateUsage:
    """Template usage tracking"""
    uuid: str
    template_uuid: str
    version_uuid: str
    lead_id: str
    campaign_id: str
    sent_at: float
    opened_at: Optional[float] = None
    clicked_at: Optional[float] = None
    replied_at: Optional[float] = None
    bounced_at: Optional[float] = None
    unsubscribed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TemplateEngineConfig:
    """Configuration for the template engine module"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.template_config = config_dict.get("template_engine", {})
        
        # Template settings
        self.template_dir = self.template_config.get("template_dir", "resources/templates")
        self.max_template_size = self.template_config.get("max_template_size", 102400)  # 100KB
        self.cache_enabled = self.template_config.get("cache_enabled", True)
        self.cache_size = self.template_config.get("cache_size", 100)
        
        # Personalization settings
        self.personalization_enabled = self.template_config.get("personalalization_enabled", True)
        self.website_scraping_enabled = self.template_config.get("website_scraping_enabled", True)
        self.fallback_content = self.template_config.get("fallback_content", "Hello {{name}}")
        
        # A/B testing settings
        self.ab_test_min_sample_size = self.template_config.get("ab_test_min_sample_size", 100)
        self.ab_test_confidence_threshold = self.template_config.get("ab_test_confidence_threshold", 0.95)
        
        # Follow-up settings
        self.max_follow_ups = self.template_config.get("max_follow_ups", 5)
        self.default_follow_up_delay = self.template_config.get("default_follow_up_delay", 72)  # 72 hours

class TemplateEngine(BaseModule):
    """Production-ready template engine with Jinja2 and personalization"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Dict[str, Any]):
        super().__init__(name, event_bus, storage, license_manager, config)
        self.config = TemplateEngineConfig(config)
        self.templates: Dict[str, EmailTemplate] = {}
        self.sequences: Dict[str, FollowUpSequence] = {}
        self.template_cache: Dict[str, Template] = {}
        self.personalization_cache: Dict[str, Dict[str, Any]] = {}
        self.session_stats = {
            "templates_rendered": 0,
            "personalizations_performed": 0,
            "ab_tests_active": 0,
            "follow_ups_scheduled": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Setup Jinja2 environment
        self.setup_jinja_environment()
        
    def _setup_event_handlers(self):
        """Setup event handlers for template requests"""
        self.event_bus.subscribe("template.create", self._handle_create_request)
        self.event_bus.subscribe("template.update", self._handle_update_request)
        self.event_bus.subscribe("template.delete", self._handle_delete_request)
        self.event_bus.subscribe("template.render", self._handle_render_request)
        self.event_bus.subscribe("template.sequence.create", self._handle_sequence_create_request)
        self.event_bus.subscribe("template.ab_test.start", self._handle_ab_test_start_request)
        self.event_bus.subscribe("template.status", self._handle_status_request)
        
    def _validate_requirements(self):
        """Validate module requirements and dependencies"""
        # Create template directory if not exists
        template_dir = Path(self.config.template_dir)
        template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing templates from storage
        self._load_templates()
        self._load_sequences()
        
    def setup_jinja_environment(self):
        """Setup Jinja2 environment with custom filters and globals"""
        try:
            # Create environment
            self.jinja_env = Environment(
                loader=FileSystemLoader(self.config.template_dir),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True,
                cache_size=self.config.cache_size if self.config.cache_enabled else 0
            )
            
            # Add custom filters
            self.jinja_env.filters['format_phone'] = self.format_phone
            self.jinja_env.filters['format_currency'] = self.format_currency
            self.jinja_env.filters['format_date'] = self.format_date
            self.jinja_env.filters['truncate_words'] = self.truncate_words
            self.jinja_env.filters['extract_domain'] = self.extract_domain
            
            # Add custom globals
            self.jinja_env.globals['now'] = datetime.now
            self.jinja_env.globals['uuid'] = uuid.uuid4
            self.jinja_env.globals['random_choice'] = random.choice
            
            logger.info("Jinja2 environment setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up Jinja2 environment: {str(e)}", exc_info=True)
            raise
    
    def format_phone(self, phone: str, format_type: str = "international") -> str:
        """Format phone number"""
        if not phone:
            return ""
        
        # Remove all non-digit characters
        digits = re.sub(r'[^\d]', '', phone)
        
        if len(digits) == 10:
            if format_type == "international":
                return f"+1 {digits[:3]} {digits[3:6]} {digits[6:]}"
            elif format_type == "domestic":
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        
        return phone
    
    def format_currency(self, amount: float, currency: str = "USD") -> str:
        """Format currency"""
        try:
            if currency == "USD":
                return f"${amount:,.2f}"
            elif currency == "EUR":
                return f"€{amount:,.2f}"
            elif currency == "GBP":
                return f"£{amount:,.2f}"
            else:
                return f"{amount:,.2f} {currency}"
        except:
            return str(amount)
    
    def format_date(self, date: datetime, format_str: str = "%B %d, %Y") -> str:
        """Format date"""
        try:
            return date.strftime(format_str)
        except:
            return str(date)
    
    def truncate_words(self, text: str, length: int = 50) -> str:
        """Truncate text to specified word count"""
        words = text.split()
        if len(words) <= length:
            return text
        return " ".join(words[:length]) + "..."
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return url
    
    async def _start_services(self):
        """Start template engine services"""
        # Start personalization cache cleanup
        asyncio.create_task(self._personalization_cache_cleanup())
        
        logger.info("Template engine services started successfully")
    
    async def _stop_services(self):
        """Stop template engine services"""
        logger.info("Template engine services stopped")
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Update template statistics
        self._update_template_stats()
        
        # Clean up old cache entries
        if len(self.template_cache) > self.config.cache_size:
            # Remove least recently used items
            items = list(self.template_cache.items())
            items.sort(key=lambda x: x[1].globals.get('_last_used', 0))
            for key, _ in items[:len(items) - self.config.cache_size]:
                del self.template_cache[key]
        
        # Log session stats
        logger.debug(f"Template engine stats: {self.session_stats}")
    
    async def _personalization_cache_cleanup(self):
        """Periodic cleanup of personalization cache"""
        while True:
            try:
                # Remove entries older than 24 hours
                current_time = time.time()
                expired_keys = [
                    key for key, data in self.personalization_cache.items()
                    if current_time - data.get('timestamp', 0) > 86400
                ]
                
                for key in expired_keys:
                    del self.personalization_cache[key]
                
                # Sleep for next cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in personalization cache cleanup: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    def _load_templates(self):
        """Load templates from storage"""
        try:
            templates_data = self.storage.query_leads({
                "source": "email_template",
                "category": "system"
            })
            
            for template_data in templates_data:
                try:
                    template = EmailTemplate(
                        uuid=template_data.get("uuid"),
                        name=template_data.get("name"),
                        description=template_data.get("description"),
                        category=template_data.get("category"),
                        tags=template_data.get("tags", []),
                        variables=template_data.get("variables", []),
                        follow_up_sequence=template_data.get("follow_up_sequence"),
                        ab_test_enabled=template_data.get("ab_test_enabled", False),
                        created_at=template_data.get("created_at", time.time()),
                        updated_at=template_data.get("updated_at", time.time()),
                        usage_count=template_data.get("usage_count", 0),
                        success_rate=template_data.get("success_rate", 0.0)
                    )
                    
                    # Load versions
                    versions_data = template_data.get("versions", [])
                    for version_data in versions_data:
                        version = TemplateVersion(
                            uuid=version_data.get("uuid"),
                            template_uuid=template.uuid,
                            name=version_data.get("name"),
                            subject=version_data.get("subject"),
                            html_content=version_data.get("html_content"),
                            text_content=version_data.get("text_content"),
                            version=version_data.get("version", "A"),
                            weight=version_data.get("weight", 0.5),
                            is_active=version_data.get("is_active", True),
                            created_at=version_data.get("created_at", time.time()),
                            metadata=version_data.get("metadata", {})
                        )
                        template.versions.append(version)
                    
                    self.templates[template.uuid] = template
                    
                except Exception as e:
                    logger.error(f"Error loading template: {str(e)}", exc_info=True)
            
            logger.info(f"Loaded {len(self.templates)} email templates")
            
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}", exc_info=True)
    
    def _load_sequences(self):
        """Load follow-up sequences from storage"""
        try:
            sequences_data = self.storage.query_leads({
                "source": "follow_up_sequence",
                "category": "system"
            })
            
            for sequence_data in sequences_data:
                try:
                    sequence = FollowUpSequence(
                        uuid=sequence_data.get("uuid"),
                        name=sequence_data.get("name"),
                        description=sequence_data.get("description"),
                        templates=sequence_data.get("templates", []),
                        delays=sequence_data.get("delays", []),
                        conditions=sequence_data.get("conditions", []),
                        is_active=sequence_data.get("is_active", True),
                        created_at=sequence_data.get("created_at", time.time())
                    )
                    
                    self.sequences[sequence.uuid] = sequence
                    
                except Exception as e:
                    logger.error(f"Error loading sequence: {str(e)}", exc_info=True)
            
            logger.info(f"Loaded {len(self.sequences)} follow-up sequences")
            
        except Exception as e:
            logger.error(f"Error loading sequences: {str(e)}", exc_info=True)
    
    def _update_template_stats(self):
        """Update template statistics based on usage data"""
        try:
            # Get usage data for the last 30 days
            cutoff_time = time.time() - (30 * 86400)
            
            for template in self.templates.values():
                # Calculate success rate
                usage_data = self.storage.query_leads({
                    "source": "template_usage",
                    "category": "system",
                    "created_after": cutoff_time
                })
                
                template_usage = [u for u in usage_data if u.get("template_uuid") == template.uuid]
                
                if template_usage:
                    total_sent = len(template_usage)
                    successful = len([u for u in template_usage if u.get("replied_at") or u.get("clicked_at")])
                    template.success_rate = successful / total_sent if total_sent > 0 else 0.0
                    template.usage_count = total_sent
                
                # Update A/B test weights if enabled
                if template.ab_test_enabled and len(template.versions) >= 2:
                    self._update_ab_test_weights(template)
            
        except Exception as e:
            logger.error(f"Error updating template stats: {str(e)}", exc_info=True)
    
    def _update_ab_test_weights(self, template: EmailTemplate):
        """Update A/B test weights based on performance"""
        try:
            # Get performance data for each version
            version_stats = {}
            
            for version in template.versions:
                usage_data = self.storage.query_leads({
                    "source": "template_usage",
                    "category": "system",
                    "template_uuid": template.uuid,
                    "version_uuid": version.uuid
                })
                
                total_sent = len(usage_data)
                successful = len([u for u in usage_data if u.get("replied_at") or u.get("clicked_at")])
                
                version_stats[version.uuid] = {
                    "sent": total_sent,
                    "successful": successful,
                    "rate": successful / total_sent if total_sent > 0 else 0.0
                }
            
            # Update weights if we have enough data
            if all(stats["sent"] >= self.config.ab_test_min_sample_size for stats in version_stats.values()):
                # Calculate new weights based on performance
                total_rate = sum(stats["rate"] for stats in version_stats.values())
                
                for version in template.versions:
                    if total_rate > 0:
                        version.weight = version_stats[version.uuid]["rate"] / total_rate
                    else:
                        version.weight = 1.0 / len(template.versions)
                
                # Normalize weights
                total_weight = sum(v.weight for v in template.versions)
                for version in template.versions:
                    version.weight = version.weight / total_weight
                
                self.session_stats["ab_tests_active"] += 1
        
        except Exception as e:
            logger.error(f"Error updating A/B test weights: {str(e)}", exc_info=True)
    
    async def _handle_create_request(self, event_type: str, data: Dict[str, Any]):
        """Handle template creation requests"""
        try:
            template_data = data.get("template")
            if not template_data:
                logger.warning("Invalid create request: missing template data")
                return
            
            # Create template
            template = self._create_template(template_data)
            if template:
                self.event_bus.publish("template.created", {
                    "template_uuid": template.uuid,
                    "name": template.name
                })
            
        except Exception as e:
            logger.error(f"Error handling create request: {str(e)}", exc_info=True)
    
    async def _handle_update_request(self, event_type: str, data: Dict[str, Any]):
        """Handle template update requests"""
        try:
            template_uuid = data.get("template_uuid")
            update_data = data.get("update_data")
            
            if not template_uuid or not update_data:
                logger.warning("Invalid update request: missing template UUID or update data")
                return
            
            # Update template
            if self._update_template(template_uuid, update_data):
                self.event_bus.publish("template.updated", {
                    "template_uuid": template_uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling update request: {str(e)}", exc_info=True)
    
    async def _handle_delete_request(self, event_type: str, data: Dict[str, Any]):
        """Handle template deletion requests"""
        try:
            template_uuid = data.get("template_uuid")
            if not template_uuid:
                logger.warning("Invalid delete request: missing template UUID")
                return
            
            # Delete template
            if self._delete_template(template_uuid):
                self.event_bus.publish("template.deleted", {
                    "template_uuid": template_uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling delete request: {str(e)}", exc_info=True)
    
    async def _handle_render_request(self, event_type: str, data: Dict[str, Any]):
        """Handle template rendering requests"""
        try:
            template_uuid = data.get("template_uuid")
            lead_data = data.get("lead_data", {})
            context = data.get("context", {})
            
            if not template_uuid:
                logger.warning("Invalid render request: missing template UUID")
                return
            
            # Render template
            result = await self.render_template(template_uuid, lead_data, context)
            
            self.event_bus.publish("template.rendered", {
                "template_uuid": template_uuid,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Error handling render request: {str(e)}", exc_info=True)
    
    async def _handle_sequence_create_request(self, event_type: str, data: Dict[str, Any]):
        """Handle follow-up sequence creation requests"""
        try:
            sequence_data = data.get("sequence")
            if not sequence_data:
                logger.warning("Invalid sequence create request: missing sequence data")
                return
            
            # Create sequence
            sequence = self._create_sequence(sequence_data)
            if sequence:
                self.event_bus.publish("template.sequence.created", {
                    "sequence_uuid": sequence.uuid,
                    "name": sequence.name
                })
            
        except Exception as e:
            logger.error(f"Error handling sequence create request: {str(e)}", exc_info=True)
    
    async def _handle_ab_test_start_request(self, event_type: str, data: Dict[str, Any]):
        """Handle A/B test start requests"""
        try:
            template_uuid = data.get("template_uuid")
            if not template_uuid:
                logger.warning("Invalid A/B test start request: missing template UUID")
                return
            
            # Start A/B test
            if self._start_ab_test(template_uuid):
                self.event_bus.publish("template.ab_test.started", {
                    "template_uuid": template_uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling A/B test start request: {str(e)}", exc_info=True)
    
    async def _handle_status_request(self, event_type: str, data: Dict[str, Any]):
        """Handle status requests"""
        status = {
            "templates": len(self.templates),
            "sequences": len(self.sequences),
            "active_ab_tests": len([t for t in self.templates.values() if t.ab_test_enabled]),
            "session_stats": self.session_stats,
            "cache_size": len(self.template_cache),
            "personalization_cache_size": len(self.personalization_cache)
        }
        self.event_bus.publish("template.status.response", status)
    
    def _create_template(self, template_data: Dict[str, Any]) -> Optional[EmailTemplate]:
        """Create a new email template"""
        try:
            # Create template
            template = EmailTemplate(
                uuid=str(uuid.uuid4()),
                name=template_data.get("name"),
                description=template_data.get("description"),
                category=template_data.get("category", "general"),
                tags=template_data.get("tags", []),
                variables=template_data.get("variables", []),
                follow_up_sequence=template_data.get("follow_up_sequence"),
                ab_test_enabled=template_data.get("ab_test_enabled", False)
            )
            
            # Create initial version
            version = TemplateVersion(
                uuid=str(uuid.uuid4()),
                template_uuid=template.uuid,
                name=template.name,
                subject=template_data.get("subject"),
                html_content=template_data.get("html_content"),
                text_content=template_data.get("text_content"),
                version="A",
                weight=1.0
            )
            
            template.versions.append(version)
            
            # Validate template
            if not self._validate_template(template):
                logger.error(f"Invalid template: {template.name}")
                return None
            
            # Extract variables from template
            template.variables = self._extract_variables(version.html_content, version.text_content)
            
            # Save to storage
            self._save_template_to_storage(template)
            
            # Add to templates
            self.templates[template.uuid] = template
            
            logger.info(f"Created email template: {template.name}")
            return template
            
        except Exception as e:
            logger.error(f"Error creating template: {str(e)}", exc_info=True)
            return None
    
    def _update_template(self, template_uuid: str, update_data: Dict[str, Any]) -> bool:
        """Update an existing template"""
        try:
            if template_uuid not in self.templates:
                logger.warning(f"Template not found: {template_uuid}")
                return False
            
            template = self.templates[template_uuid]
            
            # Update fields
            if "name" in update_data:
                template.name = update_data["name"]
            if "description" in update_data:
                template.description = update_data["description"]
            if "category" in update_data:
                template.category = update_data["category"]
            if "tags" in update_data:
                template.tags = update_data["tags"]
            if "follow_up_sequence" in update_data:
                template.follow_up_sequence = update_data["follow_up_sequence"]
            if "ab_test_enabled" in update_data:
                template.ab_test_enabled = update_data["ab_test_enabled"]
            
            # Create new version if content is updated
            if "subject" in update_data or "html_content" in update_data or "text_content" in update_data:
                version = TemplateVersion(
                    uuid=str(uuid.uuid4()),
                    template_uuid=template.uuid,
                    name=template.name,
                    subject=update_data.get("subject", template.versions[-1].subject),
                    html_content=update_data.get("html_content", template.versions[-1].html_content),
                    text_content=update_data.get("text_content", template.versions[-1].text_content),
                    version=chr(ord(template.versions[-1].version) + 1),
                    weight=1.0 / len(template.versions)
                )
                
                # Validate version
                if not self._validate_template_version(version):
                    logger.error(f"Invalid template version: {template.name}")
                    return False
                
                # Extract variables
                template.variables = self._extract_variables(version.html_content, version.text_content)
                
                # Add version
                template.versions.append(version)
                
                # Adjust weights
                total_weight = sum(v.weight for v in template.versions)
                for v in template.versions:
                    v.weight = v.weight / total_weight
            
            template.updated_at = time.time()
            
            # Save to storage
            self._save_template_to_storage(template)
            
            logger.info(f"Updated email template: {template.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating template: {str(e)}", exc_info=True)
            return False
    
    def _delete_template(self, template_uuid: str) -> bool:
        """Delete an email template"""
        try:
            if template_uuid not in self.templates:
                logger.warning(f"Template not found: {template_uuid}")
                return False
            
            template = self.templates[template_uuid]
            
            # Remove from templates
            del self.templates[template_uuid]
            
            # Remove from storage
            self._remove_template_from_storage(template_uuid)
            
            # Clear from cache
            for version in template.versions:
                cache_key = f"{template_uuid}_{version.uuid}"
                if cache_key in self.template_cache:
                    del self.template_cache[cache_key]
            
            logger.info(f"Deleted email template: {template.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting template: {str(e)}", exc_info=True)
            return False
    
    def _validate_template(self, template: EmailTemplate) -> bool:
        """Validate template configuration"""
        try:
            # Check required fields
            if not all([template.name, template.versions]):
                return False
            
            # Validate versions
            for version in template.versions:
                if not self._validate_template_version(version):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Template validation failed: {str(e)}", exc_info=True)
            return False
    
    def _validate_template_version(self, version: TemplateVersion) -> bool:
        """Validate template version"""
        try:
            # Check required fields
            if not all([version.subject, version.html_content]):
                return False
            
            # Check size
            total_size = len(version.subject) + len(version.html_content)
            if total_size > self.config.max_template_size:
                return False
            
            # Validate Jinja2 syntax
            try:
                # Test subject
                Template(version.subject, environment=self.jinja_env)
                
                # Test HTML content
                Template(version.html_content, environment=self.jinja_env)
                
                # Test text content if provided
                if version.text_content:
                    Template(version.text_content, environment=self.jinja_env)
                
            except TemplateSyntaxError as e:
                logger.error(f"Template syntax error: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Template version validation failed: {str(e)}", exc_info=True)
            return False
    
    def _extract_variables(self, html_content: str, text_content: str) -> List[str]:
        """Extract variables from template content"""
        variables = set()
        
        # Extract from HTML content
        try:
            ast = meta.parse(self.jinja_env, html_content)
            variables.update(meta.find_undeclared_variables(ast))
        except:
            pass
        
        # Extract from text content
        if text_content:
            try:
                ast = meta.parse(self.jinja_env, text_content)
                variables.update(meta.find_undeclared_variables(ast))
            except:
                pass
        
        # Filter out Jinja2 built-ins
        builtin_vars = {'now', 'uuid', 'random_choice'}
        variables = [v for v in variables if v not in builtin_vars]
        
        return list(variables)
    
    def _save_template_to_storage(self, template: EmailTemplate):
        """Save template to storage"""
        try:
            template_data = {
                "uuid": template.uuid,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "versions": [
                    {
                        "uuid": v.uuid,
                        "name": v.name,
                        "subject": v.subject,
                        "html_content": v.html_content,
                        "text_content": v.text_content,
                        "version": v.version,
                        "weight": v.weight,
                        "is_active": v.is_active,
                        "created_at": v.created_at,
                        "metadata": v.metadata
                    }
                    for v in template.versions
                ],
                "tags": template.tags,
                "variables": template.variables,
                "follow_up_sequence": template.follow_up_sequence,
                "ab_test_enabled": template.ab_test_enabled,
                "created_at": template.created_at,
                "updated_at": template.updated_at,
                "usage_count": template.usage_count,
                "success_rate": template.success_rate
            }
            
            self.storage.save_lead({
                "uuid": template.uuid,
                "source": "email_template",
                "name": template.name,
                "raw_content": json.dumps(template_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving template to storage: {str(e)}", exc_info=True)
    
    def _remove_template_from_storage(self, template_uuid: str):
        """Remove template from storage"""
        try:
            self.storage.delete_lead(template_uuid)
        except Exception as e:
            logger.error(f"Error removing template from storage: {str(e)}", exc_info=True)
    
    def _create_sequence(self, sequence_data: Dict[str, Any]) -> Optional[FollowUpSequence]:
        """Create a new follow-up sequence"""
        try:
            sequence = FollowUpSequence(
                uuid=str(uuid.uuid4()),
                name=sequence_data.get("name"),
                description=sequence_data.get("description"),
                templates=sequence_data.get("templates", []),
                delays=sequence_data.get("delays", []),
                conditions=sequence_data.get("conditions", [])
            )
            
            # Validate sequence
            if not self._validate_sequence(sequence):
                logger.error(f"Invalid sequence: {sequence.name}")
                return None
            
            # Save to storage
            self._save_sequence_to_storage(sequence)
            
            # Add to sequences
            self.sequences[sequence.uuid] = sequence
            
            logger.info(f"Created follow-up sequence: {sequence.name}")
            return sequence
            
        except Exception as e:
            logger.error(f"Error creating sequence: {str(e)}", exc_info=True)
            return None
    
    def _validate_sequence(self, sequence: FollowUpSequence) -> bool:
        """Validate follow-up sequence"""
        try:
            # Check required fields
            if not all([sequence.name, sequence.templates]):
                return False
            
            # Check that templates and delays match
            if len(sequence.templates) != len(sequence.delays):
                return False
            
            # Check maximum follow-ups
            if len(sequence.templates) > self.config.max_follow_ups:
                return False
            
            # Validate that templates exist
            for template_uuid in sequence.templates:
                if template_uuid not in self.templates:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Sequence validation failed: {str(e)}", exc_info=True)
            return False
    
    def _save_sequence_to_storage(self, sequence: FollowUpSequence):
        """Save sequence to storage"""
        try:
            sequence_data = {
                "uuid": sequence.uuid,
                "name": sequence.name,
                "description": sequence.description,
                "templates": sequence.templates,
                "delays": sequence.delays,
                "conditions": sequence.conditions,
                "is_active": sequence.is_active,
                "created_at": sequence.created_at
            }
            
            self.storage.save_lead({
                "uuid": sequence.uuid,
                "source": "follow_up_sequence",
                "name": sequence.name,
                "raw_content": json.dumps(sequence_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving sequence to storage: {str(e)}", exc_info=True)
    
    def _start_ab_test(self, template_uuid: str) -> bool:
        """Start A/B test for a template"""
        try:
            if template_uuid not in self.templates:
                logger.warning(f"Template not found: {template_uuid}")
                return False
            
            template = self.templates[template_uuid]
            
            # Check if we have at least 2 versions
            if len(template.versions) < 2:
                logger.warning(f"Template {template.name} needs at least 2 versions for A/B testing")
                return False
            
            # Enable A/B testing
            template.ab_test_enabled = True
            
            # Set equal weights
            for version in template.versions:
                version.weight = 1.0 / len(template.versions)
                version.is_active = True
            
            # Save to storage
            self._save_template_to_storage(template)
            
            logger.info(f"Started A/B test for template: {template.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting A/B test: {str(e)}", exc_info=True)
            return False
    
    async def render_template(self, template_uuid: str, lead_data: Dict[str, Any], 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Render a template with lead data and context"""
        try:
            if template_uuid not in self.templates:
                raise ValueError(f"Template not found: {template_uuid}")
            
            template = self.templates[template_uuid]
            
            # Select version (A/B testing or latest)
            version = self._select_template_version(template)
            
            # Get or create template from cache
            cache_key = f"{template_uuid}_{version.uuid}"
            if cache_key in self.template_cache:
                html_template = self.template_cache[cache_key]
                self.session_stats["cache_hits"] += 1
            else:
                html_template = self.jinja_env.from_string(version.html_content)
                if self.config.cache_enabled:
                    self.template_cache[cache_key] = html_template
                    # Add timestamp for LRU cache
                    html_template.globals['_last_used'] = time.time()
                self.session_stats["cache_misses"] += 1
            
            # Prepare context
            render_context = self._prepare_render_context(lead_data, context)
            
            # Render HTML content
            html_content = html_template.render(**render_context)
            
            # Render text content if available
            text_content = None
            if version.text_content:
                text_template = self.jinja_env.from_string(version.text_content)
                text_content = text_template.render(**render_context)
            
            # Render subject
            subject_template = self.jinja_env.from_string(version.subject)
            subject = subject_template.render(**render_context)
            
            # Update usage count
            template.usage_count += 1
            template.updated_at = time.time()
            
            # Save usage tracking
            self._save_template_usage(template.uuid, version.uuid, lead_data.get("uuid"), context.get("campaign_id"))
            
            self.session_stats["templates_rendered"] += 1
            
            return {
                "subject": subject,
                "html_content": html_content,
                "text_content": text_content,
                "template_uuid": template.uuid,
                "version_uuid": version.uuid,
                "variables_used": template.variables
            }
            
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}", exc_info=True)
            raise
    
    def _select_template_version(self, template: EmailTemplate) -> TemplateVersion:
        """Select template version (A/B testing or latest)"""
        if template.ab_test_enabled and len(template.versions) > 1:
            # Select version based on weights
            weights = [v.weight for v in template.versions if v.is_active]
            versions = [v for v in template.versions if v.is_active]
            
            if weights and versions:
                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # Select version
                return np.random.choice(versions, p=weights)
        
        # Return latest version
        return template.versions[-1]
    
    def _prepare_render_context(self, lead_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare rendering context with personalization"""
        render_context = {
            "lead": lead_data,
            "context": context or {},
            "personalization": {}
        }
        
        # Add personalization data if enabled
        if self.config.personalization_enabled:
            personalization = self._get_personalization_data(lead_data)
            render_context["personalization"] = personalization
            
            if personalization:
                self.session_stats["personalizations_performed"] += 1
        
        return render_context
    
    def _get_personalization_data(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalization data for a lead"""
        try:
            lead_id = lead_data.get("uuid")
            if not lead_id:
                return {}
            
            # Check cache first
            if lead_id in self.personalization_cache:
                return self.personalization_cache[lead_id]["data"]
            
            personalization = {}
            
            # Extract company info
            if lead_data.get("company"):
                personalization["company"] = {
                    "name": lead_data["company"],
                    "domain": lead_data.get("website", "").replace("https://", "").replace("http://", ""),
                    "industry": lead_data.get("category", "Unknown")
                }
            
            # Extract personal info
            if lead_data.get("name"):
                name_parts = lead_data["name"].split()
                personalization["name"] = {
                    "full": lead_data["name"],
                    "first": name_parts[0] if name_parts else "",
                    "last": name_parts[-1] if len(name_parts) > 1 else ""
                }
            
            # Extract location info
            if lead_data.get("location"):
                personalization["location"] = {
                    "full": lead_data["location"],
                    "city": "",
                    "country": ""
                }
            
            # Scrape website for additional personalization
            if self.config.website_scraping_enabled and lead_data.get("website"):
                website_data = self._scrape_website_data(lead_data["website"])
                personalization["website"] = website_data
            
            # Cache personalization
            self.personalization_cache[lead_id] = {
                "data": personalization,
                "timestamp": time.time()
            }
            
            return personalization
            
        except Exception as e:
            logger.error(f"Error getting personalization data: {str(e)}", exc_info=True)
            return {}
    
    def _scrape_website_data(self, website_url: str) -> Dict[str, Any]:
        """Scrape website for personalization data"""
        try:
            # This is a simplified implementation
            # In production, this would use a proper web scraper
            
            website_data = {
                "url": website_url,
                "title": "",
                "description": "",
                "keywords": [],
                "logo": ""
            }
            
            # Simulate scraping
            # In reality, this would make HTTP requests and parse HTML
            website_data["title"] = "Company Name"
            website_data["description"] = "Company description goes here"
            website_data["keywords"] = ["SaaS", "B2B", "Technology"]
            
            return website_data
            
        except Exception as e:
            logger.error(f"Error scraping website data: {str(e)}", exc_info=True)
            return {}
    
    def _save_template_usage(self, template_uuid: str, version_uuid: str, lead_id: str, campaign_id: str):
        """Save template usage tracking"""
        try:
            usage = TemplateUsage(
                uuid=str(uuid.uuid4()),
                template_uuid=template_uuid,
                version_uuid=version_uuid,
                lead_id=lead_id,
                campaign_id=campaign_id,
                sent_at=time.time()
            )
            
            usage_data = {
                "uuid": usage.uuid,
                "template_uuid": usage.template_uuid,
                "version_uuid": usage.version_uuid,
                "lead_id": usage.lead_id,
                "campaign_id": usage.campaign_id,
                "sent_at": usage.sent_at,
                "opened_at": usage.opened_at,
                "clicked_at": usage.clicked_at,
                "replied_at": usage.replied_at,
                "bounced_at": usage.bounced_at,
                "unsubscribed_at": usage.unsubscribed_at,
                "metadata": usage.metadata
            }
            
            self.storage.save_lead({
                "uuid": usage.uuid,
                "source": "template_usage",
                "name": f"Usage {usage.uuid}",
                "raw_content": json.dumps(usage_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving template usage: {str(e)}", exc_info=True)
    
    async def schedule_follow_ups(self, lead_id: str, template_uuid: str, campaign_id: str):
        """Schedule follow-up emails for a lead"""
        try:
            if template_uuid not in self.templates:
                logger.warning(f"Template not found: {template_uuid}")
                return
            
            template = self.templates[template_uuid]
            
            if not template.follow_up_sequence:
                return
            
            sequence = self.sequences.get(template.follow_up_sequence)
            if not sequence or not sequence.is_active:
                return
            
            # Schedule each follow-up
            for i, (follow_template_uuid, delay) in enumerate(zip(sequence.templates, sequence.delays)):
                # Calculate send time
                send_time = time.time() + (delay * 3600)  # Convert hours to seconds
                
                # Create follow-up task
                asyncio.create_task(
                    self._send_follow_up(
                        lead_id,
                        follow_template_uuid,
                        campaign_id,
                        send_time,
                        sequence.conditions[i] if i < len(sequence.conditions) else {}
                    )
                )
                
                self.session_stats["follow_ups_scheduled"] += 1
            
            logger.info(f"Scheduled {len(sequence.templates)} follow-ups for lead {lead_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling follow-ups: {str(e)}", exc_info=True)
    
    async def _send_follow_up(self, lead_id: str, template_uuid: str, campaign_id: str, 
                            send_time: float, conditions: Dict[str, Any]):
        """Send a follow-up email"""
        try:
            # Wait until send time
            await asyncio.sleep(send_time - time.time())
            
            # Check conditions
            if not self._check_follow_up_conditions(lead_id, conditions):
                logger.info(f"Follow-up conditions not met for lead {lead_id}")
                return
            
            # Get lead data
            lead_data = self.storage.get_lead(lead_id)
            if not lead_data:
                logger.warning(f"Lead not found: {lead_id}")
                return
            
            # Render template
            result = await self.render_template(template_uuid, lead_data, {"campaign_id": campaign_id})
            
            # Send email (this would integrate with the email system)
            # For now, we'll just log
            logger.info(f"Follow-up email sent to lead {lead_id}")
            
        except Exception as e:
            logger.error(f"Error sending follow-up: {str(e)}", exc_info=True)
    
    def _check_follow_up_conditions(self, lead_id: str, conditions: Dict[str, Any]) -> bool:
        """Check if follow-up conditions are met"""
        try:
            # Get recent interactions
            usage_data = self.storage.query_leads({
                "source": "template_usage",
                "lead_id": lead_id,
                "created_after": time.time() - (7 * 86400)  # Last 7 days
            })
            
            # Check conditions
            for condition_type, condition_value in conditions.items():
                if condition_type == "no_reply":
                    # Check if lead has replied
                    has_replied = any(u.get("replied_at") for u in usage_data)
                    if has_replied and condition_value:
                        return False
                
                elif condition_type == "no_open":
                    # Check if lead has opened email
                    has_opened = any(u.get("opened_at") for u in usage_data)
                    if has_opened and condition_value:
                        return False
                
                elif condition_type == "no_click":
                    # Check if lead has clicked
                    has_clicked = any(u.get("clicked_at") for u in usage_data)
                    if has_clicked and condition_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking follow-up conditions: {str(e)}", exc_info=True)
            return True  # Default to sending if conditions can't be checked
    
    def create_template(self, name: str, subject: str, html_content: str, 
                       text_content: str = None, **kwargs) -> Dict[str, Any]:
        """Public method to create a template"""
        template_data = {
            "name": name,
            "subject": subject,
            "html_content": html_content,
            "text_content": text_content,
            **kwargs
        }
        
        template = self._create_template(template_data)
        if template:
            return {
                "template_uuid": template.uuid,
                "name": template.name,
                "status": "created"
            }
        else:
            return {
                "status": "failed",
                "error": "Invalid template configuration"
            }
    
    def update_template(self, template_uuid: str, **kwargs) -> Dict[str, Any]:
        """Public method to update a template"""
        if self._update_template(template_uuid, kwargs):
            return {
                "template_uuid": template_uuid,
                "status": "updated"
            }
        else:
            return {
                "template_uuid": template_uuid,
                "status": "failed",
                "error": "Template not found or update failed"
            }
    
    def delete_template(self, template_uuid: str) -> Dict[str, Any]:
        """Public method to delete a template"""
        if self._delete_template(template_uuid):
            return {
                "template_uuid": template_uuid,
                "status": "deleted"
            }
        else:
            return {
                "template_uuid": template_uuid,
                "status": "failed",
                "error": "Template not found"
            }
    
    def create_follow_up_sequence(self, name: str, templates: List[str], 
                                delays: List[int], **kwargs) -> Dict[str, Any]:
        """Public method to create a follow-up sequence"""
        sequence_data = {
            "name": name,
            "templates": templates,
            "delays": delays,
            **kwargs
        }
        
        sequence = self._create_sequence(sequence_data)
        if sequence:
            return {
                "sequence_uuid": sequence.uuid,
                "name": sequence.name,
                "status": "created"
            }
        else:
            return {
                "status": "failed",
                "error": "Invalid sequence configuration"
            }
    
    def start_ab_test(self, template_uuid: str) -> Dict[str, Any]:
        """Public method to start A/B test"""
        if self._start_ab_test(template_uuid):
            return {
                "template_uuid": template_uuid,
                "status": "ab_test_started"
            }
        else:
            return {
                "template_uuid": template_uuid,
                "status": "failed",
                "error": "A/B test could not be started"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get template engine statistics"""
        return {
            "session_stats": self.session_stats,
            "templates": len(self.templates),
            "sequences": len(self.sequences),
            "active_ab_tests": len([t for t in self.templates.values() if t.ab_test_enabled]),
            "cache_size": len(self.template_cache),
            "personalization_cache_size": len(self.personalization_cache)
        }