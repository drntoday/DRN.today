#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Lead Generation - Adapter Builder Module
Production-Ready Implementation
"""

import os
import logging
import json
import uuid
import time
import ast
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import yaml
import jinja2

# Core system imports
from engine.orchestrator import BaseModule
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# Initialize adapter builder logger
logger = logging.getLogger(__name__)

@dataclass
class AdapterConfig:
    """Adapter configuration data structure"""
    uuid: str
    name: str
    description: str
    source_type: str  # "website", "api", "rss", "sitemap", etc.
    base_url: str
    selectors: Dict[str, str] = field(default_factory=dict)
    pagination: Dict[str, Any] = field(default_factory=dict)
    data_extraction: Dict[str, Any] = field(default_factory=dict)
    request_config: Dict[str, Any] = field(default_factory=dict)
    post_processing: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    is_active: bool = True
    version: str = "1.0"

@dataclass
class GeneratedAdapter:
    """Generated adapter data structure"""
    uuid: str
    config_uuid: str
    python_code: str
    generated_at: float = field(default_factory=time.time)
    last_tested: Optional[float] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = False

class AdapterBuilderConfig:
    """Configuration for the adapter builder module"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.scraping_config = config_dict.get("scraping", {})
        self.ai_config = config_dict.get("ai", {})
        
        # Builder settings
        self.template_dir = self.scraping_config.get("adapter_template_dir", "resources/adapter_templates")
        self.output_dir = self.scraping_config.get("adapter_output_dir", "modules/lead_generation/generated_adapters")
        self.max_selectors_per_page = self.scraping_config.get("max_selectors_per_page", 50)
        self.enable_code_validation = self.scraping_config.get("enable_code_validation", True)
        
        # Code generation settings
        self.code_style = self.scraping_config.get("code_style", "async")
        self.include_error_handling = self.scraping_config.get("include_error_handling", True)
        self.include_logging = self.scraping_config.get("include_logging", True)
        self.include_rate_limiting = self.scraping_config.get("include_rate_limiting", True)

class AdapterBuilder(BaseModule):
    """Production-ready adapter builder for creating custom crawlers"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Dict[str, Any]):
        super().__init__(name, event_bus, storage, license_manager, config)
        self.config = AdapterBuilderConfig(config)
        self.adapters: Dict[str, AdapterConfig] = {}
        self.generated_adapters: Dict[str, GeneratedAdapter] = {}
        self.template_env: Optional[jinja2.Environment] = None
        
        # Setup template environment
        self._setup_template_environment()
        
        # Load existing adapters
        self._load_adapters()
        
    def _setup_event_handlers(self):
        """Setup event handlers for adapter builder requests"""
        self.event_bus.subscribe("adapter_builder.create", self._handle_create_request)
        self.event_bus.subscribe("adapter_builder.update", self._handle_update_request)
        self.event_bus.subscribe("adapter_builder.delete", self._handle_delete_request)
        self.event_bus.subscribe("adapter_builder.generate", self._handle_generate_request)
        self.event_bus.subscribe("adapter_builder.test", self._handle_test_request)
        self.event_bus.subscribe("adapter_builder.export", self._handle_export_request)
        self.event_bus.subscribe("adapter_builder.status", self._handle_status_request)
        
    def _validate_requirements(self):
        """Validate module requirements and dependencies"""
        # Create output directory if not exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    async def _start_services(self):
        """Start adapter builder services"""
        logger.info("Adapter builder services started successfully")
    
    async def _stop_services(self):
        """Stop adapter builder services"""
        logger.info("Adapter builder services stopped")
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up old generated adapters
        self._cleanup_old_adapters()
        
        # Log session stats
        logger.debug("Adapter builder maintenance completed")
    
    async def _handle_create_request(self, event_type: str, data: Dict[str, Any]):
        """Handle adapter creation requests"""
        try:
            adapter_data = data.get("adapter")
            if not adapter_data:
                logger.warning("Invalid create request: missing adapter data")
                return
            
            # Create adapter
            adapter = self._create_adapter(adapter_data)
            if adapter:
                self.event_bus.publish("adapter_builder.created", {
                    "adapter_uuid": adapter.uuid,
                    "name": adapter.name
                })
            
        except Exception as e:
            logger.error(f"Error handling create request: {str(e)}", exc_info=True)
    
    async def _handle_update_request(self, event_type: str, data: Dict[str, Any]):
        """Handle adapter update requests"""
        try:
            adapter_uuid = data.get("adapter_uuid")
            update_data = data.get("update_data")
            
            if not adapter_uuid or not update_data:
                logger.warning("Invalid update request: missing adapter UUID or update data")
                return
            
            # Update adapter
            if self._update_adapter(adapter_uuid, update_data):
                self.event_bus.publish("adapter_builder.updated", {
                    "adapter_uuid": adapter_uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling update request: {str(e)}", exc_info=True)
    
    async def _handle_delete_request(self, event_type: str, data: Dict[str, Any]):
        """Handle adapter deletion requests"""
        try:
            adapter_uuid = data.get("adapter_uuid")
            if not adapter_uuid:
                logger.warning("Invalid delete request: missing adapter UUID")
                return
            
            # Delete adapter
            if self._delete_adapter(adapter_uuid):
                self.event_bus.publish("adapter_builder.deleted", {
                    "adapter_uuid": adapter_uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling delete request: {str(e)}", exc_info=True)
    
    async def _handle_generate_request(self, event_type: str, data: Dict[str, Any]):
        """Handle adapter generation requests"""
        try:
            adapter_uuid = data.get("adapter_uuid")
            if not adapter_uuid:
                logger.warning("Invalid generate request: missing adapter UUID")
                return
            
            # Generate adapter
            generated = self._generate_adapter(adapter_uuid)
            if generated:
                self.event_bus.publish("adapter_builder.generated", {
                    "adapter_uuid": adapter_uuid,
                    "generated_uuid": generated.uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling generate request: {str(e)}", exc_info=True)
    
    async def _handle_test_request(self, event_type: str, data: Dict[str, Any]):
        """Handle adapter testing requests"""
        try:
            generated_uuid = data.get("generated_uuid")
            test_url = data.get("test_url")
            
            if not generated_uuid or not test_url:
                logger.warning("Invalid test request: missing generated UUID or test URL")
                return
            
            # Test adapter
            results = await self._test_adapter(generated_uuid, test_url)
            
            self.event_bus.publish("adapter_builder.tested", {
                "generated_uuid": generated_uuid,
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Error handling test request: {str(e)}", exc_info=True)
    
    async def _handle_export_request(self, event_type: str, data: Dict[str, Any]):
        """Handle adapter export requests"""
        try:
            adapter_uuid = data.get("adapter_uuid")
            export_format = data.get("format", "python")
            
            if not adapter_uuid:
                logger.warning("Invalid export request: missing adapter UUID")
                return
            
            # Export adapter
            exported_path = self._export_adapter(adapter_uuid, export_format)
            
            self.event_bus.publish("adapter_builder.exported", {
                "adapter_uuid": adapter_uuid,
                "export_path": exported_path
            })
            
        except Exception as e:
            logger.error(f"Error handling export request: {str(e)}", exc_info=True)
    
    async def _handle_status_request(self, event_type: str, data: Dict[str, Any]):
        """Handle status requests"""
        status = {
            "adapters": len(self.adapters),
            "active_adapters": len([a for a in self.adapters.values() if a.is_active]),
            "generated_adapters": len(self.generated_adapters),
            "valid_adapters": len([g for g in self.generated_adapters.values() if g.is_valid]),
            "template_env_available": self.template_env is not None
        }
        self.event_bus.publish("adapter_builder.status.response", status)
    
    def _setup_template_environment(self):
        """Setup Jinja2 template environment"""
        try:
            template_dir = Path(self.config.template_dir)
            if template_dir.exists():
                self.template_env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(str(template_dir)),
                    autoescape=True,
                    trim_blocks=True,
                    lstrip_blocks=True
                )
                logger.info("Template environment setup completed")
            else:
                logger.warning(f"Template directory not found: {template_dir}")
                
        except Exception as e:
            logger.error(f"Error setting up template environment: {str(e)}", exc_info=True)
    
    def _load_adapters(self):
        """Load existing adapters from storage"""
        try:
            adapters_data = self.storage.query_leads({
                "source": "adapter_config",
                "category": "system"
            })
            
            for adapter_data in adapters_data:
                try:
                    adapter_config = json.loads(adapter_data.get("raw_content", "{}"))
                    adapter = AdapterConfig(
                        uuid=adapter_config.get("uuid"),
                        name=adapter_config.get("name"),
                        description=adapter_config.get("description"),
                        source_type=adapter_config.get("source_type"),
                        base_url=adapter_config.get("base_url"),
                        selectors=adapter_config.get("selectors", {}),
                        pagination=adapter_config.get("pagination", {}),
                        data_extraction=adapter_config.get("data_extraction", {}),
                        request_config=adapter_config.get("request_config", {}),
                        post_processing=adapter_config.get("post_processing", {}),
                        created_at=adapter_config.get("created_at", time.time()),
                        updated_at=adapter_config.get("updated_at", time.time()),
                        is_active=adapter_config.get("is_active", True),
                        version=adapter_config.get("version", "1.0")
                    )
                    
                    self.adapters[adapter.uuid] = adapter
                    
                except Exception as e:
                    logger.error(f"Error loading adapter: {str(e)}", exc_info=True)
            
            logger.info(f"Loaded {len(self.adapters)} adapters")
            
        except Exception as e:
            logger.error(f"Error loading adapters: {str(e)}", exc_info=True)
    
    def _create_adapter(self, adapter_data: Dict[str, Any]) -> Optional[AdapterConfig]:
        """Create a new adapter configuration"""
        try:
            # Create adapter
            adapter = AdapterConfig(
                uuid=str(uuid.uuid4()),
                name=adapter_data.get("name"),
                description=adapter_data.get("description"),
                source_type=adapter_data.get("source_type", "website"),
                base_url=adapter_data.get("base_url"),
                selectors=adapter_data.get("selectors", {}),
                pagination=adapter_data.get("pagination", {}),
                data_extraction=adapter_data.get("data_extraction", {}),
                request_config=adapter_data.get("request_config", {}),
                post_processing=adapter_data.get("post_processing", {})
            )
            
            # Validate adapter
            if not self._validate_adapter(adapter):
                logger.error(f"Invalid adapter: {adapter.name}")
                return None
            
            # Save to storage
            self._save_adapter_to_storage(adapter)
            
            # Add to adapters
            self.adapters[adapter.uuid] = adapter
            
            logger.info(f"Created adapter: {adapter.name}")
            return adapter
            
        except Exception as e:
            logger.error(f"Error creating adapter: {str(e)}", exc_info=True)
            return None
    
    def _update_adapter(self, adapter_uuid: str, update_data: Dict[str, Any]) -> bool:
        """Update an existing adapter configuration"""
        try:
            if adapter_uuid not in self.adapters:
                logger.warning(f"Adapter not found: {adapter_uuid}")
                return False
            
            adapter = self.adapters[adapter_uuid]
            
            # Update fields
            if "name" in update_data:
                adapter.name = update_data["name"]
            if "description" in update_data:
                adapter.description = update_data["description"]
            if "source_type" in update_data:
                adapter.source_type = update_data["source_type"]
            if "base_url" in update_data:
                adapter.base_url = update_data["base_url"]
            if "selectors" in update_data:
                adapter.selectors.update(update_data["selectors"])
            if "pagination" in update_data:
                adapter.pagination.update(update_data["pagination"])
            if "data_extraction" in update_data:
                adapter.data_extraction.update(update_data["data_extraction"])
            if "request_config" in update_data:
                adapter.request_config.update(update_data["request_config"])
            if "post_processing" in update_data:
                adapter.post_processing.update(update_data["post_processing"])
            
            adapter.updated_at = time.time()
            
            # Validate adapter
            if not self._validate_adapter(adapter):
                logger.error(f"Invalid adapter after update: {adapter.name}")
                return False
            
            # Save to storage
            self._save_adapter_to_storage(adapter)
            
            logger.info(f"Updated adapter: {adapter.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating adapter: {str(e)}", exc_info=True)
            return False
    
    def _delete_adapter(self, adapter_uuid: str) -> bool:
        """Delete an adapter configuration"""
        try:
            if adapter_uuid not in self.adapters:
                logger.warning(f"Adapter not found: {adapter_uuid}")
                return False
            
            adapter = self.adapters[adapter_uuid]
            
            # Remove from adapters
            del self.adapters[adapter_uuid]
            
            # Remove from storage
            self._remove_adapter_from_storage(adapter_uuid)
            
            # Remove generated adapters
            generated_to_remove = [
                uuid for uuid, gen in self.generated_adapters.items()
                if gen.config_uuid == adapter_uuid
            ]
            for uuid in generated_to_remove:
                del self.generated_adapters[uuid]
            
            logger.info(f"Deleted adapter: {adapter.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting adapter: {str(e)}", exc_info=True)
            return False
    
    def _validate_adapter(self, adapter: AdapterConfig) -> bool:
        """Validate adapter configuration"""
        try:
            # Check required fields
            if not all([adapter.name, adapter.source_type, adapter.base_url]):
                return False
            
            # Validate URL format
            if not re.match(r'^https?://', adapter.base_url):
                return False
            
            # Validate selectors
            if not isinstance(adapter.selectors, dict):
                return False
            
            # Validate source type
            valid_source_types = ["website", "api", "rss", "sitemap", "json"]
            if adapter.source_type not in valid_source_types:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Adapter validation failed: {str(e)}", exc_info=True)
            return False
    
    def _save_adapter_to_storage(self, adapter: AdapterConfig):
        """Save adapter configuration to storage"""
        try:
            adapter_data = {
                "uuid": adapter.uuid,
                "name": adapter.name,
                "description": adapter.description,
                "source_type": adapter.source_type,
                "base_url": adapter.base_url,
                "selectors": adapter.selectors,
                "pagination": adapter.pagination,
                "data_extraction": adapter.data_extraction,
                "request_config": adapter.request_config,
                "post_processing": adapter.post_processing,
                "created_at": adapter.created_at,
                "updated_at": adapter.updated_at,
                "is_active": adapter.is_active,
                "version": adapter.version
            }
            
            self.storage.save_lead({
                "uuid": adapter.uuid,
                "source": "adapter_config",
                "name": adapter.name,
                "raw_content": json.dumps(adapter_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving adapter to storage: {str(e)}", exc_info=True)
    
    def _remove_adapter_from_storage(self, adapter_uuid: str):
        """Remove adapter configuration from storage"""
        try:
            self.storage.delete_lead(adapter_uuid)
        except Exception as e:
            logger.error(f"Error removing adapter from storage: {str(e)}", exc_info=True)
    
    def _generate_adapter(self, adapter_uuid: str) -> Optional[GeneratedAdapter]:
        """Generate Python code from adapter configuration"""
        try:
            if adapter_uuid not in self.adapters:
                logger.warning(f"Adapter not found: {adapter_uuid}")
                return None
            
            adapter = self.adapters[adapter_uuid]
            
            # Generate code based on source type
            if adapter.source_type == "website":
                code = self._generate_website_adapter(adapter)
            elif adapter.source_type == "api":
                code = self._generate_api_adapter(adapter)
            elif adapter.source_type == "rss":
                code = self._generate_rss_adapter(adapter)
            elif adapter.source_type == "sitemap":
                code = self._generate_sitemap_adapter(adapter)
            elif adapter.source_type == "json":
                code = self._generate_json_adapter(adapter)
            else:
                logger.error(f"Unsupported source type: {adapter.source_type}")
                return None
            
            # Create generated adapter
            generated = GeneratedAdapter(
                uuid=str(uuid.uuid4()),
                config_uuid=adapter_uuid,
                python_code=code
            )
            
            # Validate generated code
            if self.config.enable_code_validation:
                generated.is_valid = self._validate_generated_code(code)
            else:
                generated.is_valid = True
            
            # Save generated adapter
            self._save_generated_adapter(generated)
            
            logger.info(f"Generated adapter for: {adapter.name}")
            return generated
            
        except Exception as e:
            logger.error(f"Error generating adapter: {str(e)}", exc_info=True)
            return None
    
    def _generate_website_adapter(self, adapter: AdapterConfig) -> str:
        """Generate website scraper adapter code"""
        try:
            if not self.template_env:
                return self._generate_fallback_website_adapter(adapter)
            
            template = self.template_env.get_template("website_adapter.py.j2")
            
            # Prepare template context
            context = {
                "adapter": adapter,
                "config": self.config,
                "imports": self._get_website_imports(),
                "helper_functions": self._get_website_helper_functions()
            }
            
            # Render template
            code = template.render(**context)
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating website adapter: {str(e)}", exc_info=True)
            return self._generate_fallback_website_adapter(adapter)
    
    def _generate_api_adapter(self, adapter: AdapterConfig) -> str:
        """Generate API adapter code"""
        try:
            if not self.template_env:
                return self._generate_fallback_api_adapter(adapter)
            
            template = self.template_env.get_template("api_adapter.py.j2")
            
            # Prepare template context
            context = {
                "adapter": adapter,
                "config": self.config,
                "imports": self._get_api_imports(),
                "helper_functions": self._get_api_helper_functions()
            }
            
            # Render template
            code = template.render(**context)
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating API adapter: {str(e)}", exc_info=True)
            return self._generate_fallback_api_adapter(adapter)
    
    def _generate_rss_adapter(self, adapter: AdapterConfig) -> str:
        """Generate RSS adapter code"""
        try:
            if not self.template_env:
                return self._generate_fallback_rss_adapter(adapter)
            
            template = self.template_env.get_template("rss_adapter.py.j2")
            
            # Prepare template context
            context = {
                "adapter": adapter,
                "config": self.config,
                "imports": self._get_rss_imports(),
                "helper_functions": self._get_rss_helper_functions()
            }
            
            # Render template
            code = template.render(**context)
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating RSS adapter: {str(e)}", exc_info=True)
            return self._generate_fallback_rss_adapter(adapter)
    
    def _generate_sitemap_adapter(self, adapter: AdapterConfig) -> str:
        """Generate sitemap adapter code"""
        try:
            if not self.template_env:
                return self._generate_fallback_sitemap_adapter(adapter)
            
            template = self.template_env.get_template("sitemap_adapter.py.j2")
            
            # Prepare template context
            context = {
                "adapter": adapter,
                "config": self.config,
                "imports": self._get_sitemap_imports(),
                "helper_functions": self._get_sitemap_helper_functions()
            }
            
            # Render template
            code = template.render(**context)
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating sitemap adapter: {str(e)}", exc_info=True)
            return self._generate_fallback_sitemap_adapter(adapter)
    
    def _generate_json_adapter(self, adapter: AdapterConfig) -> str:
        """Generate JSON adapter code"""
        try:
            if not self.template_env:
                return self._generate_fallback_json_adapter(adapter)
            
            template = self.template_env.get_template("json_adapter.py.j2")
            
            # Prepare template context
            context = {
                "adapter": adapter,
                "config": self.config,
                "imports": self._get_json_imports(),
                "helper_functions": self._get_json_helper_functions()
            }
            
            # Render template
            code = template.render(**context)
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating JSON adapter: {str(e)}", exc_info=True)
            return self._generate_fallback_json_adapter(adapter)
    
    def _get_website_imports(self) -> List[str]:
        """Get imports for website adapter"""
        imports = [
            "import asyncio",
            "import logging",
            "from typing import Dict, List, Optional, Any",
            "from urllib.parse import urljoin, urlparse",
            "from pathlib import Path",
            "import aiohttp",
            "from bs4 import BeautifulSoup",
            "from dataclasses import dataclass, field"
        ]
        
        if self.config.include_rate_limiting:
            imports.append("import time")
            imports.append("from asyncio import Semaphore")
        
        return imports
    
    def _get_api_imports(self) -> List[str]:
        """Get imports for API adapter"""
        imports = [
            "import asyncio",
            "import logging",
            "from typing import Dict, List, Optional, Any",
            "import aiohttp",
            "from dataclasses import dataclass, field"
        ]
        
        if self.config.include_rate_limiting:
            imports.append("import time")
            imports.append("from asyncio import Semaphore")
        
        return imports
    
    def _get_rss_imports(self) -> List[str]:
        """Get imports for RSS adapter"""
        imports = [
            "import asyncio",
            "import logging",
            "from typing import Dict, List, Optional, Any",
            "import aiohttp",
            "import feedparser",
            "from dataclasses import dataclass, field"
        ]
        
        return imports
    
    def _get_sitemap_imports(self) -> List[str]:
        """Get imports for sitemap adapter"""
        imports = [
            "import asyncio",
            "import logging",
            "from typing import Dict, List, Optional, Any",
            "from urllib.parse import urljoin",
            "import aiohttp",
            "from bs4 import BeautifulSoup",
            "from dataclasses import dataclass, field"
        ]
        
        return imports
    
    def _get_json_imports(self) -> List[str]:
        """Get imports for JSON adapter"""
        imports = [
            "import asyncio",
            "import logging",
            "from typing import Dict, List, Optional, Any",
            "import aiohttp",
            "from dataclasses import dataclass, field"
        ]
        
        return imports
    
    def _get_website_helper_functions(self) -> str:
        """Get helper functions for website adapter"""
        functions = """
@dataclass
class ScrapingResult:
    url: str
    data: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None

async def scrape_page(session, url, selectors, config):
    \"\"\"Scrape a single page\"\"\"
    try:
        async with session.get(url, **config) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                data = {}
                for field, selector in selectors.items():
                    element = soup.select_one(selector)
                    if element:
                        data[field] = element.get_text(strip=True)
                
                # Extract links for pagination
                links = [a.get('href') for a in soup.find_all('a', href=True)]
                
                return ScrapingResult(url=url, data=data, links=links, success=True)
            else:
                return ScrapingResult(url=url, success=False, error=f"HTTP {response.status}")
    except Exception as e:
        return ScrapingResult(url=url, success=False, error=str(e))
"""
        
        if self.config.include_rate_limiting:
            functions += """
async def rate_limit(semaphore, func):
    \"\"\"Rate limiting decorator\"\"\"
    async def wrapper(*args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)
    return wrapper
"""
        
        return functions
    
    def _get_api_helper_functions(self) -> str:
        """Get helper functions for API adapter"""
        functions = """
@dataclass
class APIResult:
    url: str
    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None

async def fetch_api_data(session, url, config):
    \"\"\"Fetch data from API endpoint\"\"\"
    try:
        async with session.get(url, **config) as response:
            if response.status == 200:
                data = await response.json()
                return APIResult(url=url, data=data, success=True)
            else:
                return APIResult(url=url, success=False, error=f"HTTP {response.status}")
    except Exception as e:
        return APIResult(url=url, success=False, error=str(e))
"""
        
        if self.config.include_rate_limiting:
            functions += """
async def rate_limit(semaphore, func):
    \"\"\"Rate limiting decorator\"\"\"
    async def wrapper(*args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)
    return wrapper
"""
        
        return functions
    
    def _get_rss_helper_functions(self) -> str:
        """Get helper functions for RSS adapter"""
        functions = """
@dataclass
class RSSResult:
    url: str
    entries: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None

async def fetch_rss_feed(url):
    \"\"\"Fetch and parse RSS feed\"\"\"
    try:
        feed = feedparser.parse(url)
        entries = []
        
        for entry in feed.entries:
            entries.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "description": entry.get("description", ""),
                "published": entry.get("published", "")
            })
        
        return RSSResult(url=url, entries=entries, success=True)
    except Exception as e:
        return RSSResult(url=url, success=False, error=str(e))
"""
        
        return functions
    
    def _get_sitemap_helper_functions(self) -> str:
        """Get helper functions for sitemap adapter"""
        functions = """
@dataclass
class SitemapResult:
    url: str
    urls: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None

async def parse_sitemap(session, url):
    \"\"\"Parse sitemap XML\"\"\"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                soup = BeautifulSoup(await response.text(), 'xml')
                urls = [loc.text for loc in soup.find_all('loc')]
                return SitemapResult(url=url, urls=urls, success=True)
            else:
                return SitemapResult(url=url, success=False, error=f"HTTP {response.status}")
    except Exception as e:
        return SitemapResult(url=url, success=False, error=str(e))
"""
        
        return functions
    
    def _get_json_helper_functions(self) -> str:
        """Get helper functions for JSON adapter"""
        functions = """
@dataclass
class JSONResult:
    url: str
    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None

async def fetch_json_data(session, url, config):
    \"\"\"Fetch JSON data\"\"\"
    try:
        async with session.get(url, **config) as response:
            if response.status == 200:
                data = await response.json()
                return JSONResult(url=url, data=data, success=True)
            else:
                return JSONResult(url=url, success=False, error=f"HTTP {response.status}")
    except Exception as e:
        return JSONResult(url=url, success=False, error=str(e))
"""
        
        return functions
    
    def _generate_fallback_website_adapter(self, adapter: AdapterConfig) -> str:
        """Generate fallback website adapter code"""
        code = f'''import asyncio
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import aiohttp
from bs4 import BeautifulSoup
from dataclasses import dataclass, field

@dataclass
class ScrapingResult:
    url: str
    data: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None

class {adapter.name}Adapter:
    def __init__(self):
        self.base_url = "{adapter.base_url}"
        self.selectors = {adapter.selectors}
        self.request_config = {adapter.request_config}
        
    async def scrape(self, session, url):
        try:
            async with session.get(url, **self.request_config) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    data = {{}}
                    for field, selector in self.selectors.items():
                        element = soup.select_one(selector)
                        if element:
                            data[field] = element.get_text(strip=True)
                    
                    links = [a.get('href') for a in soup.find_all('a', href=True)]
                    
                    return ScrapingResult(url=url, data=data, links=links, success=True)
                else:
                    return ScrapingResult(url=url, success=False, error=f"HTTP {{response.status}}")
        except Exception as e:
            return ScrapingResult(url=url, success=False, error=str(e))
    
    async def run(self):
        results = []
        async with aiohttp.ClientSession() as session:
            result = await self.scrape(session, self.base_url)
            results.append(result)
            
            # Handle pagination if configured
            if {adapter.pagination}:
                # Pagination logic would go here
                pass
                
        return results
'''
        return code
    
    def _generate_fallback_api_adapter(self, adapter: AdapterConfig) -> str:
        """Generate fallback API adapter code"""
        code = f'''import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
from dataclasses import dataclass, field

@dataclass
class APIResult:
    url: str
    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None

class {adapter.name}Adapter:
    def __init__(self):
        self.base_url = "{adapter.base_url}"
        self.request_config = {adapter.request_config}
        
    async def fetch(self, session, url):
        try:
            async with session.get(url, **self.request_config) as response:
                if response.status == 200:
                    data = await response.json()
                    return APIResult(url=url, data=data, success=True)
                else:
                    return APIResult(url=url, success=False, error=f"HTTP {{response.status}}")
        except Exception as e:
            return APIResult(url=url, success=False, error=str(e))
    
    async def run(self):
        async with aiohttp.ClientSession() as session:
            result = await self.fetch(session, self.base_url)
            return result
'''
        return code
    
    def _generate_fallback_rss_adapter(self, adapter: AdapterConfig) -> str:
        """Generate fallback RSS adapter code"""
        code = f'''import asyncio
import logging
from typing import Dict, List, Optional, Any
import feedparser
from dataclasses import dataclass, field

@dataclass
class RSSResult:
    url: str
    entries: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None

class {adapter.name}Adapter:
    def __init__(self):
        self.base_url = "{adapter.base_url}"
        
    async def fetch(self):
        try:
            feed = feedparser.parse(self.base_url)
            entries = []
            
            for entry in feed.entries:
                entries.append({{
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "description": entry.get("description", ""),
                    "published": entry.get("published", "")
                }})
            
            return RSSResult(url=self.base_url, entries=entries, success=True)
        except Exception as e:
            return RSSResult(url=self.base_url, success=False, error=str(e))
    
    async def run(self):
        return await self.fetch()
'''
        return code
    
    def _generate_fallback_sitemap_adapter(self, adapter: AdapterConfig) -> str:
        """Generate fallback sitemap adapter code"""
        code = f'''import asyncio
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import aiohttp
from bs4 import BeautifulSoup
from dataclasses import dataclass, field

@dataclass
class SitemapResult:
    url: str
    urls: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None

class {adapter.name}Adapter:
    def __init__(self):
        self.base_url = "{adapter.base_url}"
        
    async def parse(self, session, url):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    soup = BeautifulSoup(await response.text(), 'xml')
                    urls = [loc.text for loc in soup.find_all('loc')]
                    return SitemapResult(url=url, urls=urls, success=True)
                else:
                    return SitemapResult(url=url, success=False, error=f"HTTP {{response.status}}")
        except Exception as e:
            return SitemapResult(url=url, success=False, error=str(e))
    
    async def run(self):
        async with aiohttp.ClientSession() as session:
            result = await self.parse(session, self.base_url)
            return result
'''
        return code
    
    def _generate_fallback_json_adapter(self, adapter: AdapterConfig) -> str:
        """Generate fallback JSON adapter code"""
        code = f'''import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
from dataclasses import dataclass, field

@dataclass
class JSONResult:
    url: str
    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None

class {adapter.name}Adapter:
    def __init__(self):
        self.base_url = "{adapter.base_url}"
        self.request_config = {adapter.request_config}
        
    async def fetch(self, session, url):
        try:
            async with session.get(url, **self.request_config) as response:
                if response.status == 200:
                    data = await response.json()
                    return JSONResult(url=url, data=data, success=True)
                else:
                    return JSONResult(url=url, success=False, error=f"HTTP {{response.status}}")
        except Exception as e:
            return JSONResult(url=url, success=False, error=str(e))
    
    async def run(self):
        async with aiohttp.ClientSession() as session:
            result = await self.fetch(session, self.base_url)
            return result
'''
        return code
    
    def _validate_generated_code(self, code: str) -> bool:
        """Validate generated Python code"""
        try:
            # Try to compile the code
            ast.parse(code)
            
            # Check for syntax errors
            compile(code, "<string>", "exec")
            
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error validating generated code: {str(e)}")
            return False
    
    def _save_generated_adapter(self, generated: GeneratedAdapter):
        """Save generated adapter to storage"""
        try:
            # Save to file
            output_path = Path(self.config.output_dir) / f"{generated.uuid}.py"
            with open(output_path, 'w') as f:
                f.write(generated.python_code)
            
            # Save to storage
            generated_data = {
                "uuid": generated.uuid,
                "config_uuid": generated.config_uuid,
                "python_code": generated.python_code,
                "generated_at": generated.generated_at,
                "last_tested": generated.last_tested,
                "test_results": generated.test_results,
                "is_valid": generated.is_valid
            }
            
            self.storage.save_lead({
                "uuid": generated.uuid,
                "source": "generated_adapter",
                "name": f"Generated Adapter {generated.uuid}",
                "raw_content": json.dumps(generated_data),
                "category": "system"
            })
            
            # Add to generated adapters
            self.generated_adapters[generated.uuid] = generated
            
        except Exception as e:
            logger.error(f"Error saving generated adapter: {str(e)}", exc_info=True)
    
    async def _test_adapter(self, generated_uuid: str, test_url: str) -> Dict[str, Any]:
        """Test a generated adapter"""
        try:
            if generated_uuid not in self.generated_adapters:
                logger.warning(f"Generated adapter not found: {generated_uuid}")
                return {"success": False, "error": "Adapter not found"}
            
            generated = self.generated_adapters[generated_uuid]
            
            # Execute the generated code
            namespace = {}
            try:
                exec(generated.python_code, namespace)
            except Exception as e:
                error_msg = f"Code execution error: {str(e)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Get the adapter class
            adapter_class_name = None
            for name, obj in namespace.items():
                if hasattr(obj, 'run') and hasattr(obj, '__class__'):
                    adapter_class_name = name
                    break
            
            if not adapter_class_name:
                return {"success": False, "error": "Adapter class not found"}
            
            # Create adapter instance and test
            adapter_class = namespace[adapter_class_name]
            adapter = adapter_class()
            
            # Test the adapter
            if hasattr(adapter, 'run'):
                if asyncio.iscoroutinefunction(adapter.run):
                    # Async adapter
                    try:
                        result = await adapter.run()
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                else:
                    # Sync adapter
                    try:
                        result = adapter.run()
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
            else:
                return {"success": False, "error": "Adapter has no run method"}
            
            # Update test results
            generated.last_tested = time.time()
            generated.test_results = {
                "test_url": test_url,
                "result": result,
                "tested_at": generated.last_tested
            }
            
            # Update storage
            self._save_generated_adapter(generated)
            
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Error testing adapter: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _export_adapter(self, adapter_uuid: str, export_format: str) -> str:
        """Export adapter configuration"""
        try:
            if adapter_uuid not in self.adapters:
                logger.warning(f"Adapter not found: {adapter_uuid}")
                return ""
            
            adapter = self.adapters[adapter_uuid]
            
            # Create export data
            export_data = {
                "name": adapter.name,
                "description": adapter.description,
                "source_type": adapter.source_type,
                "base_url": adapter.base_url,
                "selectors": adapter.selectors,
                "pagination": adapter.pagination,
                "data_extraction": adapter.data_extraction,
                "request_config": adapter.request_config,
                "post_processing": adapter.post_processing,
                "version": adapter.version
            }
            
            # Export in requested format
            if export_format.lower() == "json":
                export_path = Path(self.config.output_dir) / f"{adapter.name}.json"
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif export_format.lower() == "yaml":
                export_path = Path(self.config.output_dir) / f"{adapter.name}.yaml"
                with open(export_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            elif export_format.lower() == "python":
                # Get generated adapter
                generated = next(
                    (g for g in self.generated_adapters.values() 
                     if g.config_uuid == adapter_uuid),
                    None
                )
                if generated:
                    export_path = Path(self.config.output_dir) / f"{adapter.name}_adapter.py"
                    with open(export_path, 'w') as f:
                        f.write(generated.python_code)
            
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting adapter: {str(e)}", exc_info=True)
            return ""
    
    def _cleanup_old_adapters(self):
        """Clean up old generated adapters"""
        try:
            cutoff_time = time.time() - (7 * 86400)  # 7 days
            
            # Remove old generated adapters from memory
            old_generated = [
                uuid for uuid, gen in self.generated_adapters.items()
                if gen.generated_at < cutoff_time
            ]
            for uuid in old_generated:
                del self.generated_adapters[uuid]
            
            # Remove old generated adapter files
            output_dir = Path(self.config.output_dir)
            if output_dir.exists():
                for file_path in output_dir.glob("*.py"):
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                        except:
                            pass
            
        except Exception as e:
            logger.error(f"Error cleaning up old adapters: {str(e)}", exc_info=True)
    
    def create_adapter(self, name: str, source_type: str, base_url: str, **kwargs) -> Dict[str, Any]:
        """Public method to create an adapter"""
        adapter_data = {
            "name": name,
            "source_type": source_type,
            "base_url": base_url,
            **kwargs
        }
        
        adapter = self._create_adapter(adapter_data)
        if adapter:
            return {
                "adapter_uuid": adapter.uuid,
                "name": adapter.name,
                "status": "created"
            }
        else:
            return {
                "status": "failed",
                "error": "Invalid adapter configuration"
            }
    
    def update_adapter(self, adapter_uuid: str, **kwargs) -> Dict[str, Any]:
        """Public method to update an adapter"""
        if self._update_adapter(adapter_uuid, kwargs):
            return {
                "adapter_uuid": adapter_uuid,
                "status": "updated"
            }
        else:
            return {
                "adapter_uuid": adapter_uuid,
                "status": "failed",
                "error": "Adapter not found or update failed"
            }
    
    def delete_adapter(self, adapter_uuid: str) -> Dict[str, Any]:
        """Public method to delete an adapter"""
        if self._delete_adapter(adapter_uuid):
            return {
                "adapter_uuid": adapter_uuid,
                "status": "deleted"
            }
        else:
            return {
                "adapter_uuid": adapter_uuid,
                "status": "failed",
                "error": "Adapter not found"
            }
    
    def generate_adapter_code(self, adapter_uuid: str) -> Dict[str, Any]:
        """Public method to generate adapter code"""
        generated = self._generate_adapter(adapter_uuid)
        if generated:
            return {
                "adapter_uuid": adapter_uuid,
                "generated_uuid": generated.uuid,
                "python_code": generated.python_code,
                "is_valid": generated.is_valid,
                "status": "generated"
            }
        else:
            return {
                "adapter_uuid": adapter_uuid,
                "status": "failed",
                "error": "Adapter not found or generation failed"
            }
    
    async def test_adapter(self, generated_uuid: str, test_url: str) -> Dict[str, Any]:
        """Public method to test an adapter"""
        results = await self._test_adapter(generated_uuid, test_url)
        return {
            "generated_uuid": generated_uuid,
            "results": results
        }
    
    def export_adapter(self, adapter_uuid: str, export_format: str = "json") -> Dict[str, Any]:
        """Public method to export an adapter"""
        export_path = self._export_adapter(adapter_uuid, export_format)
        if export_path:
            return {
                "adapter_uuid": adapter_uuid,
                "export_path": export_path,
                "status": "exported"
            }
        else:
            return {
                "adapter_uuid": adapter_uuid,
                "status": "failed",
                "error": "Export failed"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter builder statistics"""
        return {
            "adapters": len(self.adapters),
            "active_adapters": len([a for a in self.adapters.values() if a.is_active]),
            "generated_adapters": len(self.generated_adapters),
            "valid_adapters": len([g for g in self.generated_adapters.values() if g.is_valid]),
            "template_env_available": self.template_env is not None
        }