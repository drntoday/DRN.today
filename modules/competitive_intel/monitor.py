import os
import json
import hashlib
import time
import schedule
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Local imports matching project structure
from engine.orchestrator import SystemOrchestrator
from engine.event_system import EventBus, Event, EventPriority
from engine.storage import SecureStorage
from ai.models.tinybert import TinyBERTModel
from ai.nlp import NLPProcessor
from modules.competitive_intel.scraper import CompetitiveScraper
from modules.competitive_intel.ad_tracker import AdTracker


class ChangeType(Enum):
    """Types of changes that can be detected"""
    NEW = "new"
    UPDATED = "updated"
    REMOVED = "removed"
    NO_CHANGE = "no_change"


class Platform(Enum):
    """Supported social platforms"""
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"


@dataclass
class CompetitorConfig:
    """Configuration for a competitor to monitor"""
    id: str
    name: str
    website: str
    social_profiles: Dict[Platform, str]
    pricing_url: Optional[str] = None
    landing_pages: List[str] = None
    job_board_url: Optional[str] = None
    ad_keywords: List[str] = None
    monitoring_frequency: int = 3600  # seconds

    def __post_init__(self):
        if self.landing_pages is None:
            self.landing_pages = []
        if self.ad_keywords is None:
            self.ad_keywords = []


@dataclass
class SocialMetrics:
    """Social media metrics for a platform"""
    platform: Platform
    followers: int
    following: int
    posts: int
    last_updated: datetime


@dataclass
class PricingData:
    """Pricing information"""
    plans: List[Dict[str, Any]]
    last_updated: datetime
    content_hash: str


@dataclass
class LandingPageData:
    """Landing page information"""
    url: str
    title: str
    content_hash: str
    last_updated: datetime
    change_detected: ChangeType


@dataclass
class JobPosting:
    """Job posting information"""
    title: str
    department: str
    location: str
    url: str
    first_seen: datetime
    last_seen: datetime
    status: str  # active, closed, etc.


@dataclass
class AdData:
    """Advertisement data"""
    keyword: str
    ad_copy: str
    landing_url: str
    contact_points: List[str]
    first_seen: datetime
    last_seen: datetime
    platform: str


@dataclass
class CompetitorState:
    """Complete state of a competitor's monitored data"""
    config: CompetitorConfig
    last_checked: datetime
    social_metrics: Dict[Platform, SocialMetrics]
    pricing: Optional[PricingData] = None
    landing_pages: List[LandingPageData] = None
    job_postings: List[JobPosting] = None
    ads: List[AdData] = None

    def __post_init__(self):
        if self.landing_pages is None:
            self.landing_pages = []
        if self.job_postings is None:
            self.job_postings = []
        if self.ads is None:
            self.ads = []


class CompetitiveMonitor:
    """
    Competitive Intelligence Monitor
    
    Monitors competitors across multiple platforms and tracks changes
    in their social presence, pricing, landing pages, job postings, and ads.
    """
    
    def __init__(self, orchestrator: SystemOrchestrator, storage: SecureStorage, event_system: EventBus):
        self.orchestrator = SystemOrchestrator
        self.storage = SecureStorage
        self.event_system = event_system
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI components
        self.nlp = NLPProcessor()
        self.tinybert = TinyBERTModel()
        
        # Initialize scrapers
        self.scraper = CompetitiveScraper()
        self.ad_tracker = AdTracker()
        
        # Competitor states
        self.competitors: Dict[str, CompetitorState] = {}
        
        # Monitoring status
        self.running = False
        self.next_run_time = None
        
        # Load competitors from storage
        self._load_competitors()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Schedule monitoring tasks
        self._schedule_tasks()
    
    def _register_event_handlers(self):
        """Register event handlers for the event system"""
        self.event_system.subscribe(
            event_type="competitor_added",
            handler=self._on_competitor_added,
            priority=EventPriority.HIGH
        )
        
        self.event_system.subscribe(
            event_type="competitor_removed",
            handler=self._on_competitor_removed,
            priority=EventPriority.HIGH
        )
        
        self.event_system.subscribe(
            event_type="system_shutdown",
            handler=self._on_system_shutdown,
            priority=EventPriority.CRITICAL
        )
    
    def _schedule_tasks(self):
        """Schedule periodic monitoring tasks"""
        schedule.every(1).hour.do(self._check_all_competitors)
        schedule.every().day.at("02:00").do(self._cleanup_old_data)
    
    def _load_competitors(self):
        """Load competitor configurations from storage"""
        try:
            competitors_data = self.storage.load("competitive_intel/competitors.json")
            if competitors_data:
                for comp_data in competitors_data:
                    config = CompetitorConfig(**comp_data["config"])
                    state_data = comp_data.get("state")
                    
                    if state_data:
                        # Reconstruct datetime objects
                        state_data["last_checked"] = datetime.fromisoformat(state_data["last_checked"])
                        
                        if state_data.get("pricing"):
                            state_data["pricing"]["last_updated"] = datetime.fromisoformat(
                                state_data["pricing"]["last_updated"]
                            )
                        
                        for lp in state_data.get("landing_pages", []):
                            lp["last_updated"] = datetime.fromisoformat(lp["last_updated"])
                        
                        for job in state_data.get("job_postings", []):
                            job["first_seen"] = datetime.fromisoformat(job["first_seen"])
                            job["last_seen"] = datetime.fromisoformat(job["last_seen"])
                        
                        for ad in state_data.get("ads", []):
                            ad["first_seen"] = datetime.fromisoformat(ad["first_seen"])
                            ad["last_seen"] = datetime.fromisoformat(ad["last_seen"])
                        
                        # Reconstruct enums
                        for platform_str, metrics in state_data.get("social_metrics", {}).items():
                            metrics["platform"] = Platform(platform_str)
                        
                        for lp in state_data.get("landing_pages", []):
                            lp["change_detected"] = ChangeType(lp["change_detected"])
                        
                        state = CompetitorState(config, **state_data)
                    else:
                        state = CompetitorState(config, last_checked=datetime.min)
                    
                    self.competitors[config.id] = state
                
                self.logger.info(f"Loaded {len(self.competitors)} competitors from storage")
        except Exception as e:
            self.logger.error(f"Error loading competitors: {str(e)}")
    
    def _save_competitors(self):
        """Save competitor configurations and states to storage"""
        try:
            competitors_data = []
            for comp_id, state in self.competitors.items():
                # Convert to dict and handle datetime serialization
                state_dict = asdict(state)
                state_dict["last_checked"] = state.last_checked.isoformat()
                
                if state.pricing:
                    state_dict["pricing"]["last_updated"] = state.pricing.last_updated.isoformat()
                
                for lp in state_dict.get("landing_pages", []):
                    lp["last_updated"] = lp["last_updated"].isoformat()
                    lp["change_detected"] = lp["change_detected"].value
                
                for job in state_dict.get("job_postings", []):
                    job["first_seen"] = job["first_seen"].isoformat()
                    job["last_seen"] = job["last_seen"].isoformat()
                
                for ad in state_dict.get("ads", []):
                    ad["first_seen"] = ad["first_seen"].isoformat()
                    ad["last_seen"] = ad["last_seen"].isoformat()
                
                for platform_str, metrics in state_dict.get("social_metrics", {}).items():
                    metrics["platform"] = metrics["platform"].value
                
                competitors_data.append({
                    "config": asdict(state.config),
                    "state": state_dict
                })
            
            self.storage.save("competitive_intel/competitors.json", competitors_data)
            self.logger.debug("Saved competitors to storage")
        except Exception as e:
            self.logger.error(f"Error saving competitors: {str(e)}")
    
    def add_competitor(self, config: CompetitorConfig) -> bool:
        """
        Add a new competitor to monitor
        
        Args:
            config: Competitor configuration
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            if config.id in self.competitors:
                self.logger.warning(f"Competitor {config.id} already exists")
                return False
            
            # Create initial state
            state = CompetitorState(
                config=config,
                last_checked=datetime.min,
                social_metrics={},
                landing_pages=[LandingPageData(
                    url=url,
                    title="",
                    content_hash="",
                    last_updated=datetime.min,
                    change_detected=ChangeType.NEW
                ) for url in config.landing_pages]
            )
            
            self.competitors[config.id] = state
            self._save_competitors()
            
            # Emit event
            self.event_system.publish(Event(
                type="competitor_added",
                data={"competitor_id": config.id},
                priority=EventPriority.HIGH
            ))
            
            self.logger.info(f"Added competitor: {config.name} ({config.id})")
            return True
        except Exception as e:
            self.logger.error(f"Error adding competitor {config.id}: {str(e)}")
            return False
    
    def remove_competitor(self, competitor_id: str) -> bool:
        """
        Remove a competitor from monitoring
        
        Args:
            competitor_id: ID of the competitor to remove
            
        Returns:
            bool: True if successfully removed, False otherwise
        """
        try:
            if competitor_id not in self.competitors:
                self.logger.warning(f"Competitor {competitor_id} not found")
                return False
            
            del self.competitors[competitor_id]
            self._save_competitors()
            
            # Emit event
            self.event_system.publish(Event(
                type="competitor_removed",
                data={"competitor_id": competitor_id},
                priority=EventPriority.HIGH
            ))
            
            self.logger.info(f"Removed competitor: {competitor_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error removing competitor {competitor_id}: {str(e)}")
            return False
    
    def update_competitor(self, competitor_id: str, config: CompetitorConfig) -> bool:
        """
        Update an existing competitor's configuration
        
        Args:
            competitor_id: ID of the competitor to update
            config: New configuration
            
        Returns:
            bool: True if successfully updated, False otherwise
        """
        try:
            if competitor_id not in self.competitors:
                self.logger.warning(f"Competitor {competitor_id} not found")
                return False
            
            # Preserve existing state but update config
            state = self.competitors[competitor_id]
            old_config = state.config
            state.config = config
            
            # Check for new landing pages
            existing_urls = {lp.url for lp in state.landing_pages}
            new_urls = set(config.landing_pages) - existing_urls
            
            for url in new_urls:
                state.landing_pages.append(LandingPageData(
                    url=url,
                    title="",
                    content_hash="",
                    last_updated=datetime.min,
                    change_detected=ChangeType.NEW
                ))
            
            self._save_competitors()
            
            # Emit event
            self.event_system.publish(Event(
                type="competitor_updated",
                data={
                    "competitor_id": competitor_id,
                    "old_config": asdict(old_config),
                    "new_config": asdict(config)
                },
                priority=EventPriority.HIGH
            ))
            
            self.logger.info(f"Updated competitor: {config.name} ({config.id})")
            return True
        except Exception as e:
            self.logger.error(f"Error updating competitor {competitor_id}: {str(e)}")
            return False
    
    def get_competitor(self, competitor_id: str) -> Optional[CompetitorState]:
        """
        Get a competitor's current state
        
        Args:
            competitor_id: ID of the competitor
            
        Returns:
            CompetitorState: Current state of the competitor or None if not found
        """
        return self.competitors.get(competitor_id)
    
    def get_all_competitors(self) -> Dict[str, CompetitorState]:
        """
        Get all competitors and their states
        
        Returns:
            Dict[str, CompetitorState]: Dictionary of competitor IDs to their states
        """
        return self.competitors.copy()
    
    def start_monitoring(self):
        """Start the monitoring process"""
        if self.running:
            self.logger.warning("Monitoring is already running")
            return
        
        self.running = True
        self.logger.info("Started competitive intelligence monitoring")
        
        # Schedule initial run
        self._check_all_competitors()
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        if not self.running:
            self.logger.warning("Monitoring is not running")
            return
        
        self.running = False
        self.logger.info("Stopped competitive intelligence monitoring")
    
    def _check_all_competitors(self):
        """Check all competitors for updates"""
        if not self.running:
            return
        
        self.logger.info("Checking all competitors for updates")
        
        for comp_id, state in self.competitors.items():
            try:
                self._check_competitor(state)
            except Exception as e:
                self.logger.error(f"Error checking competitor {comp_id}: {str(e)}")
        
        # Save updated states
        self._save_competitors()
        
        # Schedule next run
        self.next_run_time = datetime.now() + timedelta(seconds=3600)
    
    def _check_competitor(self, state: CompetitorState):
        """
        Check a single competitor for updates
        
        Args:
            state: Competitor state to check
        """
        now = datetime.now()
        
        # Check if it's time to monitor this competitor
        if (now - state.last_checked).total_seconds() < state.config.monitoring_frequency:
            return
        
        self.logger.debug(f"Checking competitor: {state.config.name}")
        
        # Check social metrics
        self._check_social_metrics(state)
        
        # Check pricing
        if state.config.pricing_url:
            self._check_pricing(state)
        
        # Check landing pages
        for lp in state.landing_pages:
            self._check_landing_page(state, lp)
        
        # Check job board
        if state.config.job_board_url:
            self._check_job_board(state)
        
        # Check ads
        if state.config.ad_keywords:
            self._check_ads(state)
        
        # Update last checked time
        state.last_checked = now
    
    def _check_social_metrics(self, state: CompetitorState):
        """
        Check social media metrics for a competitor
        
        Args:
            state: Competitor state to update
        """
        for platform, url in state.config.social_profiles.items():
            try:
                metrics = self.scraper.get_social_metrics(platform, url)
                
                if metrics:
                    # Check if metrics changed
                    old_metrics = state.social_metrics.get(platform)
                    changed = False
                    
                    if old_metrics:
                        if (old_metrics.followers != metrics.followers or
                            old_metrics.following != metrics.following or
                            old_metrics.posts != metrics.posts):
                            changed = True
                    else:
                        changed = True
                    
                    # Update state
                    state.social_metrics[platform] = metrics
                    
                    # Emit event if changed
                    if changed:
                        self.event_system.publish(Event(
                            type="competitor_social_change",
                            data={
                                "competitor_id": state.config.id,
                                "platform": platform.value,
                                "old_metrics": asdict(old_metrics) if old_metrics else None,
                                "new_metrics": asdict(metrics)
                            },
                            priority=EventPriority.MEDIUM
                        ))
            except Exception as e:
                self.logger.error(f"Error checking {platform.value} for {state.config.name}: {str(e)}")
    
    def _check_pricing(self, state: CompetitorState):
        """
        Check pricing page for changes
        
        Args:
            state: Competitor state to update
        """
        try:
            content = self.scraper.scrape_page(state.config.pricing_url)
            if not content:
                return
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Extract pricing plans
            plans = self.scraper.extract_pricing_plans(content)
            
            # Check if pricing changed
            changed = False
            if state.pricing:
                if state.pricing.content_hash != content_hash:
                    changed = True
            else:
                changed = True
            
            # Update state
            state.pricing = PricingData(
                plans=plans,
                last_updated=datetime.now(),
                content_hash=content_hash
            )
            
            # Emit event if changed
            if changed:
                self.event_system.publish(Event(
                    type="competitor_pricing_change",
                    data={
                        "competitor_id": state.config.id,
                        "old_pricing": asdict(state.pricing) if state.pricing else None,
                        "new_pricing": asdict(state.pricing)
                    },
                    priority=EventPriority.HIGH
                ))
        except Exception as e:
            self.logger.error(f"Error checking pricing for {state.config.name}: {str(e)}")
    
    def _check_landing_page(self, state: CompetitorState, lp: LandingPageData):
        """
        Check a landing page for changes
        
        Args:
            state: Competitor state to update
            lp: Landing page data to check
        """
        try:
            content = self.scraper.scrape_page(lp.url)
            if not content:
                return
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Extract title
            title = self.scraper.extract_title(content)
            
            # Check if content changed
            change_type = ChangeType.NO_CHANGE
            if lp.content_hash:
                if lp.content_hash != content_hash:
                    change_type = ChangeType.UPDATED
            else:
                change_type = ChangeType.NEW
            
            # Update landing page data
            lp.title = title
            lp.content_hash = content_hash
            lp.last_updated = datetime.now()
            lp.change_detected = change_type
            
            # Emit event if changed
            if change_type != ChangeType.NO_CHANGE:
                self.event_system.publish(Event(
                    type="competitor_landing_page_change",
                    data={
                        "competitor_id": state.config.id,
                        "url": lp.url,
                        "change_type": change_type.value,
                        "title": title
                    },
                    priority=EventPriority.MEDIUM
                ))
        except Exception as e:
            self.logger.error(f"Error checking landing page {lp.url} for {state.config.name}: {str(e)}")
    
    def _check_job_board(self, state: CompetitorState):
        """
        Check job board for new or updated postings
        
        Args:
            state: Competitor state to update
        """
        try:
            job_postings = self.scraper.scrape_job_postings(state.config.job_board_url)
            if not job_postings:
                return
            
            # Track existing job URLs
            existing_urls = {job.url for job in state.job_postings}
            new_urls = set()
            updated_urls = set()
            
            # Process new and updated job postings
            for job_data in job_postings:
                url = job_data.get("url")
                if not url:
                    continue
                
                if url in existing_urls:
                    # Update existing job posting
                    for job in state.job_postings:
                        if job.url == url:
                            job.last_seen = datetime.now()
                            # Check if other details changed
                            if (job.title != job_data.get("title") or
                                job.department != job_data.get("department") or
                                job.location != job_data.get("location")):
                                updated_urls.add(url)
                            break
                else:
                    # New job posting
                    new_job = JobPosting(
                        title=job_data.get("title", ""),
                        department=job_data.get("department", ""),
                        location=job_data.get("location", ""),
                        url=url,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        status=job_data.get("status", "active")
                    )
                    state.job_postings.append(new_job)
                    new_urls.add(url)
            
            # Check for removed job postings
            removed_urls = set()
            now = datetime.now()
            for job in state.job_postings[:]:  # Copy list to allow removal
                if job.url not in {j.get("url") for j in job_postings}:
                    # Mark as removed if not seen in 7 days
                    if (now - job.last_seen).days > 7:
                        state.job_postings.remove(job)
                        removed_urls.add(job.url)
            
            # Emit events for changes
            for url in new_urls:
                job = next(j for j in state.job_postings if j.url == url)
                self.event_system.publish(Event(
                    type="competitor_new_job_posting",
                    data={
                        "competitor_id": state.config.id,
                        "job": asdict(job)
                    },
                    priority=EventPriority.MEDIUM
                ))
            
            for url in updated_urls:
                job = next(j for j in state.job_postings if j.url == url)
                self.event_system.publish(Event(
                    type="competitor_updated_job_posting",
                    data={
                        "competitor_id": state.config.id,
                        "job": asdict(job)
                    },
                    priority=EventPriority.LOW
                ))
            
            for url in removed_urls:
                self.event_system.publish(Event(
                    type="competitor_removed_job_posting",
                    data={
                        "competitor_id": state.config.id,
                        "url": url
                    },
                    priority=EventPriority.LOW
                ))
        except Exception as e:
            self.logger.error(f"Error checking job board for {state.config.name}: {str(e)}")
    
    def _check_ads(self, state: CompetitorState):
        """
        Check for new or updated ads
        
        Args:
            state: Competitor state to update
        """
        try:
            for keyword in state.config.ad_keywords:
                ads = self.ad_tracker.track_ads(keyword)
                if not ads:
                    continue
                
                # Track existing ad identifiers
                existing_ids = {
                    (ad.keyword, ad.ad_copy, ad.landing_url) 
                    for ad in state.ads
                }
                new_ids = set()
                
                # Process new ads
                for ad_data in ads:
                    ad_id = (ad_data.get("keyword", ""), 
                            ad_data.get("ad_copy", ""), 
                            ad_data.get("landing_url", ""))
                    
                    if ad_id in existing_ids:
                        # Update existing ad
                        for ad in state.ads:
                            if (ad.keyword, ad.ad_copy, ad.landing_url) == ad_id:
                                ad.last_seen = datetime.now()
                                break
                    else:
                        # New ad
                        new_ad = AdData(
                            keyword=ad_data.get("keyword", ""),
                            ad_copy=ad_data.get("ad_copy", ""),
                            landing_url=ad_data.get("landing_url", ""),
                            contact_points=ad_data.get("contact_points", []),
                            first_seen=datetime.now(),
                            last_seen=datetime.now(),
                            platform=ad_data.get("platform", "")
                        )
                        state.ads.append(new_ad)
                        new_ids.add(ad_id)
                
                # Emit events for new ads
                for ad_id in new_ids:
                    ad = next(a for a in state.ads if (a.keyword, a.ad_copy, a.landing_url) == ad_id)
                    self.event_system.publish(Event(
                        type="competitor_new_ad",
                        data={
                            "competitor_id": state.config.id,
                            "ad": asdict(ad)
                        },
                        priority=EventPriority.MEDIUM
                    ))
        except Exception as e:
            self.logger.error(f"Error checking ads for {state.config.name}: {str(e)}")
    
    def _cleanup_old_data(self):
        """Clean up old data to manage storage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=90)  # Keep 90 days of data
            
            for comp_id, state in self.competitors.items():
                # Clean up old job postings
                state.job_postings = [
                    job for job in state.job_postings 
                    if job.last_seen > cutoff_date
                ]
                
                # Clean up old ads
                state.ads = [
                    ad for ad in state.ads 
                    if ad.last_seen > cutoff_date
                ]
            
            self._save_competitors()
            self.logger.info("Cleaned up old competitive intelligence data")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def _on_competitor_added(self, event: Event):
        """Handle competitor added event"""
        competitor_id = event.data.get("competitor_id")
        if competitor_id:
            self.logger.info(f"Received event: competitor_added for {competitor_id}")
    
    def _on_competitor_removed(self, event: Event):
        """Handle competitor removed event"""
        competitor_id = event.data.get("competitor_id")
        if competitor_id:
            self.logger.info(f"Received event: competitor_removed for {competitor_id}")
    
    def _on_system_shutdown(self, event: Event):
        """Handle system shutdown event"""
        self.logger.info("System shutdown requested, stopping monitoring")
        self.stop_monitoring()
        self._save_competitors()
    
    def run_scheduled_tasks(self):
        """Run scheduled tasks (should be called periodically)"""
        schedule.run_pending()
    
    def get_next_run_time(self) -> Optional[datetime]:
        """
        Get the next scheduled run time
        
        Returns:
            datetime: Next run time or None if not scheduled
        """
        return self.next_run_time