import asyncio
import logging
import re
import time
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import aiohttp
import backoff
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse, parse_qs
import hashlib

# Core system imports
from engine.orchestrator import BaseModule
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# Initialize ad tracker logger
logger = logging.getLogger(__name__)

@dataclass
class CompetitorAd:
    """Competitor advertisement data structure"""
    uuid: str
    competitor_id: str
    ad_platform: str  # "google_ads", "bing_ads", "facebook_ads", etc.
    ad_id: Optional[str] = None
    headline: str = ""
    description: str = ""
    display_url: str = ""
    final_url: str = ""
    landing_page: str = ""
    contact_info: Dict[str, Any] = field(default_factory=dict)
    pricing_info: Optional[str] = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompetitorProfile:
    """Competitor profile data structure"""
    uuid: str
    name: str
    domain: str
    industry: Optional[str] = None
    website: str = ""
    ads: List[CompetitorAd] = field(default_factory=list)
    social_profiles: Dict[str, str] = field(default_factory=dict)
    pricing_history: List[Dict[str, Any]] = field(default_factory=list)
    job_postings: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    monitoring_enabled: bool = True

@dataclass
class AdMonitoringTask:
    """Ad monitoring task configuration"""
    uuid: str
    competitor_id: str
    platforms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    frequency_hours: int = 24
    is_active: bool = True
    last_run: float = 0.0
    next_run: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

class AdTrackerConfig:
    """Configuration for the ad tracker module"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.competitive_config = config_dict.get("competitive_intel", {})
        self.scraping_config = config_dict.get("scraping", {})
        
        # Monitoring settings
        self.monitoring_enabled = self.competitive_config.get("monitoring_enabled", True)
        self.max_competitors = self.competitive_config.get("max_competitors", 50)
        self.default_monitoring_frequency = self.competitive_config.get("default_monitoring_frequency", 24)
        
        # Scraping settings
        self.user_agents = self.scraping_config.get("user_agents", [])
        self.default_delay = self.scraping_config.get("default_delay_seconds", 2.0)
        self.max_retries = self.scraping_config.get("max_retries", 3)
        self.timeout = self.scraping_config.get("timeout_seconds", 30)
        self.proxy_rotation = self.scraping_config.get("proxy_rotation", True)
        self.proxy_list = self.scraping_config.get("proxy_list_path")
        
        # Platform settings
        self.platforms = {
            "google_ads": {
                "search_url": "https://www.google.com/search",
                "ad_selectors": [".ads-ad", ".commercial-unit", ".commercial-unit-desktop-rhs"],
                "contact_extractors": [
                    r"tel:(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
                    r"mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
                ]
            },
            "bing_ads": {
                "search_url": "https://www.bing.com/search",
                "ad_selectors": [".b_ad", ".b_adTitle"],
                "contact_extractors": [
                    r"tel:(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
                    r"mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
                ]
            }
        }
        
        # Data retention
        self.ad_retention_days = self.competitive_config.get("ad_retention_days", 90)
        self.history_retention_days = self.competitive_config.get("history_retention_days", 365)

class AdTracker(BaseModule):
    """Production-ready ad tracker for competitive intelligence"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Dict[str, Any]):
        super().__init__(name, event_bus, storage, license_manager, config)
        self.config = AdTrackerConfig(config)
        self.competitors: Dict[str, CompetitorProfile] = {}
        self.monitoring_tasks: Dict[str, AdMonitoringTask] = {}
        self.proxy_pool: List[str] = []
        self.current_proxy_index = 0
        self.session_stats = {
            "competitors_monitored": 0,
            "ads_discovered": 0,
            "contacts_extracted": 0,
            "pricing_changes_detected": 0,
            "monitoring_runs": 0,
            "scraping_errors": 0
        }
        
    def _setup_event_handlers(self):
        """Setup event handlers for ad tracking requests"""
        self.event_bus.subscribe("ad_tracker.add_competitor", self._handle_add_competitor_request)
        self.event_bus.subscribe("ad_tracker.remove_competitor", self._handle_remove_competitor_request)
        self.event_bus.subscribe("ad_tracker.start_monitoring", self._handle_start_monitoring_request)
        self.event_bus.subscribe("ad_tracker.stop_monitoring", self._handle_stop_monitoring_request)
        self.event_bus.subscribe("ad_tracker.scan_ads", self._handle_scan_ads_request)
        self.event_bus.subscribe("ad_tracker.status", self._handle_status_request)
        
    def _validate_requirements(self):
        """Validate module requirements and dependencies"""
        # Load competitors from storage
        self._load_competitors()
        
        # Load monitoring tasks
        self._load_monitoring_tasks()
        
        # Initialize proxy pool if enabled
        if self.config.proxy_rotation and self.config.proxy_list:
            self._load_proxy_pool()
            
    def _load_proxy_pool(self):
        """Load proxy list from file"""
        try:
            with open(self.config.proxy_list, 'r') as f:
                self.proxy_pool = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.proxy_pool)} proxies")
        except Exception as e:
            logger.error(f"Failed to load proxy pool: {str(e)}", exc_info=True)
            self.config.proxy_rotation = False
    
    def _get_next_proxy(self) -> Optional[str]:
        """Get next proxy from rotation pool"""
        if not self.proxy_pool:
            return None
            
        proxy = self.proxy_pool[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_pool)
        return proxy
    
    async def _start_services(self):
        """Start ad tracker services"""
        # Start monitoring worker
        asyncio.create_task(self._monitoring_worker())
        
        # Start cleanup worker
        asyncio.create_task(self._cleanup_worker())
        
        logger.info("Ad tracker services started successfully")
    
    async def _stop_services(self):
        """Stop ad tracker services"""
        logger.info("Ad tracker services stopped")
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up old ads and history
        self._cleanup_old_data()
        
        # Rotate proxies if needed
        if self.config.proxy_rotation and self.proxy_pool:
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_pool)
        
        # Log session stats
        logger.debug(f"Ad tracker stats: {self.session_stats}")
    
    async def _handle_add_competitor_request(self, event_type: str, data: Dict[str, Any]):
        """Handle add competitor requests"""
        try:
            competitor_data = data.get("competitor")
            if not competitor_data:
                logger.warning("Invalid add competitor request: missing competitor data")
                return
            
            # Add competitor
            competitor = self._add_competitor(competitor_data)
            if competitor:
                self.event_bus.publish("ad_tracker.competitor_added", {
                    "competitor_id": competitor.uuid,
                    "name": competitor.name
                })
            
        except Exception as e:
            logger.error(f"Error handling add competitor request: {str(e)}", exc_info=True)
    
    async def _handle_remove_competitor_request(self, event_type: str, data: Dict[str, Any]):
        """Handle remove competitor requests"""
        try:
            competitor_id = data.get("competitor_id")
            if not competitor_id:
                logger.warning("Invalid remove competitor request: missing competitor ID")
                return
            
            # Remove competitor
            if self._remove_competitor(competitor_id):
                self.event_bus.publish("ad_tracker.competitor_removed", {
                    "competitor_id": competitor_id
                })
            
        except Exception as e:
            logger.error(f"Error handling remove competitor request: {str(e)}", exc_info=True)
    
    async def _handle_start_monitoring_request(self, event_type: str, data: Dict[str, Any]):
        """Handle start monitoring requests"""
        try:
            competitor_id = data.get("competitor_id")
            platforms = data.get("platforms", [])
            keywords = data.get("keywords", [])
            frequency = data.get("frequency", self.config.default_monitoring_frequency)
            
            if not competitor_id:
                logger.warning("Invalid start monitoring request: missing competitor ID")
                return
            
            # Start monitoring
            task = self._start_monitoring(competitor_id, platforms, keywords, frequency)
            if task:
                self.event_bus.publish("ad_tracker.monitoring_started", {
                    "competitor_id": competitor_id,
                    "task_id": task.uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling start monitoring request: {str(e)}", exc_info=True)
    
    async def _handle_stop_monitoring_request(self, event_type: str, data: Dict[str, Any]):
        """Handle stop monitoring requests"""
        try:
            competitor_id = data.get("competitor_id")
            if not competitor_id:
                logger.warning("Invalid stop monitoring request: missing competitor ID")
                return
            
            # Stop monitoring
            if self._stop_monitoring(competitor_id):
                self.event_bus.publish("ad_tracker.monitoring_stopped", {
                    "competitor_id": competitor_id
                })
            
        except Exception as e:
            logger.error(f"Error handling stop monitoring request: {str(e)}", exc_info=True)
    
    async def _handle_scan_ads_request(self, event_type: str, data: Dict[str, Any]):
        """Handle scan ads requests"""
        try:
            competitor_id = data.get("competitor_id")
            platform = data.get("platform")
            keywords = data.get("keywords", [])
            
            if not competitor_id or not platform:
                logger.warning("Invalid scan ads request: missing competitor ID or platform")
                return
            
            # Scan ads
            ads = await self._scan_competitor_ads(competitor_id, platform, keywords)
            
            self.event_bus.publish("ad_tracker.ads_scanned", {
                "competitor_id": competitor_id,
                "platform": platform,
                "ads_found": len(ads)
            })
            
        except Exception as e:
            logger.error(f"Error handling scan ads request: {str(e)}", exc_info=True)
    
    async def _handle_status_request(self, event_type: str, data: Dict[str, Any]):
        """Handle status requests"""
        status = {
            "competitors": len(self.competitors),
            "active_competitors": len([c for c in self.competitors.values() if c.monitoring_enabled]),
            "monitoring_tasks": len(self.monitoring_tasks),
            "active_tasks": len([t for t in self.monitoring_tasks.values() if t.is_active]),
            "session_stats": self.session_stats,
            "proxy_pool_size": len(self.proxy_pool)
        }
        self.event_bus.publish("ad_tracker.status.response", status)
    
    def _load_competitors(self):
        """Load competitors from storage"""
        try:
            competitors_data = self.storage.query_leads({
                "source": "competitor_profile",
                "category": "system"
            })
            
            for competitor_data in competitors_data:
                try:
                    competitor = CompetitorProfile(
                        uuid=competitor_data.get("uuid"),
                        name=competitor_data.get("name"),
                        domain=competitor_data.get("domain"),
                        industry=competitor_data.get("industry"),
                        website=competitor_data.get("website"),
                        social_profiles=competitor_data.get("social_profiles", {}),
                        pricing_history=competitor_data.get("pricing_history", []),
                        job_postings=competitor_data.get("job_postings", []),
                        created_at=competitor_data.get("created_at", time.time()),
                        updated_at=competitor_data.get("updated_at", time.time()),
                        monitoring_enabled=competitor_data.get("monitoring_enabled", True)
                    )
                    
                    # Load ads
                    ads_data = competitor_data.get("ads", [])
                    for ad_data in ads_data:
                        ad = CompetitorAd(
                            uuid=ad_data.get("uuid"),
                            competitor_id=competitor.uuid,
                            ad_platform=ad_data.get("ad_platform"),
                            ad_id=ad_data.get("ad_id"),
                            headline=ad_data.get("headline"),
                            description=ad_data.get("description"),
                            display_url=ad_data.get("display_url"),
                            final_url=ad_data.get("final_url"),
                            landing_page=ad_data.get("landing_page"),
                            contact_info=ad_data.get("contact_info", {}),
                            pricing_info=ad_data.get("pricing_info"),
                            first_seen=ad_data.get("first_seen", time.time()),
                            last_seen=ad_data.get("last_seen", time.time()),
                            is_active=ad_data.get("is_active", True),
                            metadata=ad_data.get("metadata", {})
                        )
                        competitor.ads.append(ad)
                    
                    self.competitors[competitor.uuid] = competitor
                    
                except Exception as e:
                    logger.error(f"Error loading competitor: {str(e)}", exc_info=True)
            
            logger.info(f"Loaded {len(self.competitors)} competitors")
            
        except Exception as e:
            logger.error(f"Error loading competitors: {str(e)}", exc_info=True)
    
    def _load_monitoring_tasks(self):
        """Load monitoring tasks from storage"""
        try:
            tasks_data = self.storage.query_leads({
                "source": "monitoring_task",
                "category": "system"
            })
            
            for task_data in tasks_data:
                try:
                    task = AdMonitoringTask(
                        uuid=task_data.get("uuid"),
                        competitor_id=task_data.get("competitor_id"),
                        platforms=task_data.get("platforms", []),
                        keywords=task_data.get("keywords", []),
                        frequency_hours=task_data.get("frequency_hours", 24),
                        is_active=task_data.get("is_active", True),
                        last_run=task_data.get("last_run", 0.0),
                        next_run=task_data.get("next_run", time.time()),
                        created_at=task_data.get("created_at", time.time())
                    )
                    
                    self.monitoring_tasks[task.uuid] = task
                    
                except Exception as e:
                    logger.error(f"Error loading monitoring task: {str(e)}", exc_info=True)
            
            logger.info(f"Loaded {len(self.monitoring_tasks)} monitoring tasks")
            
        except Exception as e:
            logger.error(f"Error loading monitoring tasks: {str(e)}", exc_info=True)
    
    def _add_competitor(self, competitor_data: Dict[str, Any]) -> Optional[CompetitorProfile]:
        """Add a new competitor"""
        try:
            # Check competitor limit
            if len(self.competitors) >= self.config.max_competitors:
                logger.warning(f"Competitor limit reached: {self.config.max_competitors}")
                return None
            
            # Create competitor
            competitor = CompetitorProfile(
                uuid=str(uuid.uuid4()),
                name=competitor_data.get("name"),
                domain=competitor_data.get("domain"),
                industry=competitor_data.get("industry"),
                website=competitor_data.get("website", f"https://{competitor_data.get('domain')}"),
                monitoring_enabled=competitor_data.get("monitoring_enabled", True)
            )
            
            # Validate competitor
            if not self._validate_competitor(competitor):
                logger.error(f"Invalid competitor: {competitor.name}")
                return None
            
            # Save to storage
            self._save_competitor_to_storage(competitor)
            
            # Add to competitors
            self.competitors[competitor.uuid] = competitor
            
            logger.info(f"Added competitor: {competitor.name}")
            return competitor
            
        except Exception as e:
            logger.error(f"Error adding competitor: {str(e)}", exc_info=True)
            return None
    
    def _remove_competitor(self, competitor_id: str) -> bool:
        """Remove a competitor"""
        try:
            if competitor_id not in self.competitors:
                logger.warning(f"Competitor not found: {competitor_id}")
                return False
            
            competitor = self.competitors[competitor_id]
            
            # Stop monitoring
            self._stop_monitoring(competitor_id)
            
            # Remove from competitors
            del self.competitors[competitor_id]
            
            # Remove from storage
            self._remove_competitor_from_storage(competitor_id)
            
            logger.info(f"Removed competitor: {competitor.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing competitor: {str(e)}", exc_info=True)
            return False
    
    def _validate_competitor(self, competitor: CompetitorProfile) -> bool:
        """Validate competitor configuration"""
        try:
            # Check required fields
            if not all([competitor.name, competitor.domain]):
                return False
            
            # Validate domain format
            if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', competitor.domain):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Competitor validation failed: {str(e)}", exc_info=True)
            return False
    
    def _save_competitor_to_storage(self, competitor: CompetitorProfile):
        """Save competitor to storage"""
        try:
            competitor_data = {
                "uuid": competitor.uuid,
                "name": competitor.name,
                "domain": competitor.domain,
                "industry": competitor.industry,
                "website": competitor.website,
                "social_profiles": competitor.social_profiles,
                "pricing_history": competitor.pricing_history,
                "job_postings": competitor.job_postings,
                "ads": [
                    {
                        "uuid": ad.uuid,
                        "ad_platform": ad.ad_platform,
                        "ad_id": ad.ad_id,
                        "headline": ad.headline,
                        "description": ad.description,
                        "display_url": ad.display_url,
                        "final_url": ad.final_url,
                        "landing_page": ad.landing_page,
                        "contact_info": ad.contact_info,
                        "pricing_info": ad.pricing_info,
                        "first_seen": ad.first_seen,
                        "last_seen": ad.last_seen,
                        "is_active": ad.is_active,
                        "metadata": ad.metadata
                    }
                    for ad in competitor.ads
                ],
                "created_at": competitor.created_at,
                "updated_at": competitor.updated_at,
                "monitoring_enabled": competitor.monitoring_enabled
            }
            
            self.storage.save_lead({
                "uuid": competitor.uuid,
                "source": "competitor_profile",
                "name": competitor.name,
                "raw_content": json.dumps(competitor_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving competitor to storage: {str(e)}", exc_info=True)
    
    def _remove_competitor_from_storage(self, competitor_id: str):
        """Remove competitor from storage"""
        try:
            self.storage.delete_lead(competitor_id)
        except Exception as e:
            logger.error(f"Error removing competitor from storage: {str(e)}", exc_info=True)
    
    def _start_monitoring(self, competitor_id: str, platforms: List[str], keywords: List[str], 
                         frequency_hours: int) -> Optional[AdMonitoringTask]:
        """Start monitoring a competitor"""
        try:
            if competitor_id not in self.competitors:
                logger.warning(f"Competitor not found: {competitor_id}")
                return None
            
            # Check if monitoring already exists
            for task in self.monitoring_tasks.values():
                if task.competitor_id == competitor_id and task.is_active:
                    logger.warning(f"Monitoring already active for competitor: {competitor_id}")
                    return task
            
            # Create monitoring task
            task = AdMonitoringTask(
                uuid=str(uuid.uuid4()),
                competitor_id=competitor_id,
                platforms=platforms or list(self.config.platforms.keys()),
                keywords=keywords,
                frequency_hours=frequency_hours,
                next_run=time.time()
            )
            
            # Save to storage
            self._save_monitoring_task_to_storage(task)
            
            # Add to tasks
            self.monitoring_tasks[task.uuid] = task
            
            logger.info(f"Started monitoring for competitor: {competitor_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}", exc_info=True)
            return None
    
    def _stop_monitoring(self, competitor_id: str) -> bool:
        """Stop monitoring a competitor"""
        try:
            # Find and deactivate monitoring tasks
            deactivated = False
            for task in self.monitoring_tasks.values():
                if task.competitor_id == competitor_id and task.is_active:
                    task.is_active = False
                    self._save_monitoring_task_to_storage(task)
                    deactivated = True
            
            if deactivated:
                logger.info(f"Stopped monitoring for competitor: {competitor_id}")
                return True
            else:
                logger.warning(f"No active monitoring found for competitor: {competitor_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}", exc_info=True)
            return False
    
    def _save_monitoring_task_to_storage(self, task: AdMonitoringTask):
        """Save monitoring task to storage"""
        try:
            task_data = {
                "uuid": task.uuid,
                "competitor_id": task.competitor_id,
                "platforms": task.platforms,
                "keywords": task.keywords,
                "frequency_hours": task.frequency_hours,
                "is_active": task.is_active,
                "last_run": task.last_run,
                "next_run": task.next_run,
                "created_at": task.created_at
            }
            
            self.storage.save_lead({
                "uuid": task.uuid,
                "source": "monitoring_task",
                "name": f"Monitoring Task {task.uuid}",
                "raw_content": json.dumps(task_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving monitoring task to storage: {str(e)}", exc_info=True)
    
    async def _monitoring_worker(self):
        """Worker for processing monitoring tasks"""
        while True:
            try:
                current_time = time.time()
                
                # Process due tasks
                for task in self.monitoring_tasks.values():
                    if task.is_active and current_time >= task.next_run:
                        await self._process_monitoring_task(task)
                
                # Sleep for next cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring worker: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _process_monitoring_task(self, task: AdMonitoringTask):
        """Process a monitoring task"""
        try:
            if task.competitor_id not in self.competitors:
                logger.warning(f"Competitor not found for monitoring task: {task.competitor_id}")
                return
            
            competitor = self.competitors[task.competitor_id]
            
            # Scan each platform
            for platform in task.platforms:
                if platform in self.config.platforms:
                    ads = await self._scan_competitor_ads(
                        task.competitor_id, 
                        platform, 
                        task.keywords
                    )
                    
                    if ads:
                        logger.info(f"Found {len(ads)} ads for {competitor.name} on {platform}")
            
            # Update task
            task.last_run = time.time()
            task.next_run = time.time() + (task.frequency_hours * 3600)
            self._save_monitoring_task_to_storage(task)
            
            # Update stats
            self.session_stats["monitoring_runs"] += 1
            
        except Exception as e:
            logger.error(f"Error processing monitoring task: {str(e)}", exc_info=True)
    
    async def _scan_competitor_ads(self, competitor_id: str, platform: str, 
                                 keywords: List[str]) -> List[CompetitorAd]:
        """Scan for competitor ads on a platform"""
        try:
            if competitor_id not in self.competitors:
                logger.warning(f"Competitor not found: {competitor_id}")
                return []
            
            competitor = self.competitors[competitor_id]
            platform_config = self.config.platforms.get(platform)
            
            if not platform_config:
                logger.warning(f"Platform not supported: {platform}")
                return []
            
            ads = []
            
            # Search for each keyword
            for keyword in keywords:
                search_url = f"{platform_config['search_url']}?q={keyword}"
                
                try:
                    # Make request with proxy rotation
                    proxy = self._get_next_proxy()
                    headers = {
                        "User-Agent": self._get_user_agent(),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            search_url,
                            headers=headers,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                        ) as response:
                            if response.status == 200:
                                html = await response.text()
                                platform_ads = self._extract_ads_from_html(
                                    html, 
                                    platform_config, 
                                    competitor,
                                    platform
                                )
                                ads.extend(platform_ads)
                            
                            # Rate limiting
                            await asyncio.sleep(self.config.default_delay)
                
                except Exception as e:
                    logger.error(f"Error scanning {platform} for keyword '{keyword}': {str(e)}", exc_info=True)
                    self.session_stats["scraping_errors"] += 1
            
            # Process and save new ads
            new_ads = []
            for ad in ads:
                if self._is_new_ad(competitor, ad):
                    competitor.ads.append(ad)
                    self._save_ad_to_storage(ad)
                    new_ads.append(ad)
                    
                    # Extract contact info
                    if ad.contact_info:
                        self.session_stats["contacts_extracted"] += 1
            
            # Update competitor
            competitor.updated_at = time.time()
            self._save_competitor_to_storage(competitor)
            
            # Update stats
            self.session_stats["ads_discovered"] += len(new_ads)
            
            return new_ads
            
        except Exception as e:
            logger.error(f"Error scanning competitor ads: {str(e)}", exc_info=True)
            return []
    
    def _extract_ads_from_html(self, html: str, platform_config: Dict[str, Any], 
                             competitor: CompetitorProfile, platform: str) -> List[CompetitorAd]:
        """Extract ads from HTML content"""
        ads = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find ad elements
            for selector in platform_config['ad_selectors']:
                ad_elements = soup.select(selector)
                
                for ad_element in ad_elements:
                    try:
                        # Extract ad information
                        headline = self._extract_text(ad_element, ['h1', 'h2', 'h3', '.title', '.headline'])
                        description = self._extract_text(ad_element, ['p', '.description', '.content'])
                        display_url = self._extract_url(ad_element, ['a'])
                        
                        # Check if ad belongs to competitor
                        if self._is_competitor_ad(display_url, competitor):
                            # Extract final URL
                            final_url = self._extract_final_url(ad_element)
                            
                            # Extract contact info
                            contact_info = self._extract_contact_info(ad_element, platform_config['contact_extractors'])
                            
                            # Create ad
                            ad = CompetitorAd(
                                uuid=str(uuid.uuid4()),
                                competitor_id=competitor.uuid,
                                ad_platform=platform,
                                headline=headline,
                                description=description,
                                display_url=display_url,
                                final_url=final_url,
                                landing_page=final_url,
                                contact_info=contact_info,
                                metadata={
                                    "scraped_at": time.time(),
                                    "html_snippet": str(ad_element)[:500]
                                }
                            )
                            
                            ads.append(ad)
                    
                    except Exception as e:
                        logger.debug(f"Error extracting ad: {str(e)}")
                        continue
            
            return ads
            
        except Exception as e:
            logger.error(f"Error extracting ads from HTML: {str(e)}", exc_info=True)
            return []
    
    def _extract_text(self, element, selectors):
        """Extract text from element using selectors"""
        for selector in selectors:
            found = element.select_one(selector)
            if found:
                return found.get_text(strip=True)
        return ""
    
    def _extract_url(self, element, selectors):
        """Extract URL from element using selectors"""
        for selector in selectors:
            found = element.select_one(selector)
            if found and found.get('href'):
                return found['href']
        return ""
    
    def _extract_final_url(self, element):
        """Extract final URL from ad element"""
        # Look for tracking parameters
        link = element.find('a')
        if link and link.get('href'):
            url = link['href']
            # Remove tracking parameters
            parsed = urlparse(url)
            clean_params = {k: v for k, v in parse_qs(parsed.query).items() 
                          if not k.startswith('utm_') and k not in ['gclid', 'fbclid']}
            if clean_params:
                return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{parse_qs(clean_params)}"
            else:
                return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return ""
    
    def _extract_contact_info(self, element, patterns):
        """Extract contact information from element"""
        contact_info = {}
        text = element.get_text()
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if 'tel:' in pattern:
                    contact_info['phone'] = matches[0]
                elif 'mailto:' in pattern:
                    contact_info['email'] = matches[0]
        
        return contact_info
    
    def _is_competitor_ad(self, url, competitor):
        """Check if ad belongs to competitor"""
        if not url:
            return False
        
        # Check domain
        domain = urlparse(url).netloc
        return competitor.domain in domain
    
    def _is_new_ad(self, competitor, ad):
        """Check if ad is new (not seen before)"""
        # Check by final URL
        for existing_ad in competitor.ads:
            if existing_ad.final_url == ad.final_url:
                # Update last seen
                existing_ad.last_seen = time.time()
                existing_ad.is_active = True
                return False
        
        return True
    
    def _save_ad_to_storage(self, ad: CompetitorAd):
        """Save ad to storage"""
        try:
            ad_data = {
                "uuid": ad.uuid,
                "competitor_id": ad.competitor_id,
                "ad_platform": ad.ad_platform,
                "ad_id": ad.ad_id,
                "headline": ad.headline,
                "description": ad.description,
                "display_url": ad.display_url,
                "final_url": ad.final_url,
                "landing_page": ad.landing_page,
                "contact_info": ad.contact_info,
                "pricing_info": ad.pricing_info,
                "first_seen": ad.first_seen,
                "last_seen": ad.last_seen,
                "is_active": ad.is_active,
                "metadata": ad.metadata
            }
            
            self.storage.save_lead({
                "uuid": ad.uuid,
                "source": "competitor_ad",
                "name": f"Ad {ad.uuid}",
                "raw_content": json.dumps(ad_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving ad to storage: {str(e)}", exc_info=True)
    
    async def _cleanup_worker(self):
        """Worker for cleanup tasks"""
        while True:
            try:
                # Perform cleanup
                self._perform_maintenance()
                
                # Sleep for next cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    def _cleanup_old_data(self):
        """Clean up old ads and history"""
        try:
            cutoff_time = time.time() - (self.config.ad_retention_days * 86400)
            history_cutoff = time.time() - (self.config.history_retention_days * 86400)
            
            # Clean up old ads
            for competitor in self.competitors.values():
                competitor.ads = [ad for ad in competitor.ads if ad.last_seen > cutoff_time]
                
                # Clean up old pricing history
                competitor.pricing_history = [
                    entry for entry in competitor.pricing_history 
                    if entry.get('timestamp', 0) > history_cutoff
                ]
                
                # Clean up old job postings
                competitor.job_postings = [
                    entry for entry in competitor.job_postings 
                    if entry.get('posted_date', 0) > history_cutoff
                ]
                
                # Save updated competitor
                self._save_competitor_to_storage(competitor)
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}", exc_info=True)
    
    def _get_user_agent(self) -> str:
        """Get a random user agent from the list"""
        if self.config.user_agents:
            import random
            return random.choice(self.config.user_agents)
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DRN.today/1.0"
    
    async def add_competitor(self, name: str, domain: str, **kwargs) -> Dict[str, Any]:
        """Public method to add a competitor"""
        competitor_data = {
            "name": name,
            "domain": domain,
            **kwargs
        }
        
        competitor = self._add_competitor(competitor_data)
        if competitor:
            return {
                "competitor_id": competitor.uuid,
                "name": competitor.name,
                "status": "added"
            }
        else:
            return {
                "status": "failed",
                "error": "Invalid competitor configuration or limit reached"
            }
    
    async def remove_competitor(self, competitor_id: str) -> Dict[str, Any]:
        """Public method to remove a competitor"""
        if self._remove_competitor(competitor_id):
            return {
                "competitor_id": competitor_id,
                "status": "removed"
            }
        else:
            return {
                "competitor_id": competitor_id,
                "status": "failed",
                "error": "Competitor not found"
            }
    
    async def start_monitoring(self, competitor_id: str, platforms: List[str] = None, 
                             keywords: List[str] = None, frequency_hours: int = 24) -> Dict[str, Any]:
        """Public method to start monitoring a competitor"""
        task = self._start_monitoring(competitor_id, platforms, keywords, frequency_hours)
        if task:
            return {
                "competitor_id": competitor_id,
                "task_id": task.uuid,
                "status": "monitoring_started"
            }
        else:
            return {
                "competitor_id": competitor_id,
                "status": "failed",
                "error": "Competitor not found or monitoring already active"
            }
    
    async def stop_monitoring(self, competitor_id: str) -> Dict[str, Any]:
        """Public method to stop monitoring a competitor"""
        if self._stop_monitoring(competitor_id):
            return {
                "competitor_id": competitor_id,
                "status": "monitoring_stopped"
            }
        else:
            return {
                "competitor_id": competitor_id,
                "status": "failed",
                "error": "No active monitoring found"
            }
    
    async def scan_ads(self, competitor_id: str, platform: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Public method to scan for competitor ads"""
        keywords = keywords or [competitor_id]
        ads = await self._scan_competitor_ads(competitor_id, platform, keywords)
        
        return {
            "competitor_id": competitor_id,
            "platform": platform,
            "ads_found": len(ads),
            "ads": [
                {
                    "uuid": ad.uuid,
                    "headline": ad.headline,
                    "description": ad.description,
                    "display_url": ad.display_url,
                    "contact_info": ad.contact_info
                }
                for ad in ads
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ad tracker statistics"""
        return {
            "session_stats": self.session_stats,
            "competitors": len(self.competitors),
            "active_competitors": len([c for c in self.competitors.values() if c.monitoring_enabled]),
            "monitoring_tasks": len(self.monitoring_tasks),
            "active_tasks": len([t for t in self.monitoring_tasks.values() if t.is_active]),
            "proxy_pool_size": len(self.proxy_pool)
        }
