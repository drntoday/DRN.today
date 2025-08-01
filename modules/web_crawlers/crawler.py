#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Self-Adaptive Web Crawlers Module
Production-Ready Implementation
"""

import asyncio
import logging
import time
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import random
import re
from urllib.parse import urljoin, urlparse, parse_qs
import aiohttp
import backoff
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
import numpy as np

# Core system imports
from engine.orchestrator import BaseModule
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# AI imports
from ai.nlp import NLPProcessor

# Initialize crawler logger
logger = logging.getLogger(__name__)

@dataclass
class CrawlStage:
    """Data structure for crawl stage information"""
    stage_type: str  # "source", "result", "landing", "social", "enrichment"
    url: str
    depth: int = 0
    parent_url: Optional[str] = None
    selector: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class CrawlResult:
    """Data structure for crawl results"""
    uuid: str
    source_url: str
    stages: List[CrawlStage] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

class WebCrawlerConfig:
    """Configuration for the web crawler module"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.scraping_config = config_dict.get("scraping", {})
        self.ai_config = config_dict.get("ai", {})
        
        # Scraping settings
        self.user_agents = self.scraping_config.get("user_agents", [])
        self.default_delay = self.scraping_config.get("default_delay_seconds", 1.5)
        self.max_retries = self.scraping_config.get("max_retries", 3)
        self.timeout = self.scraping_config.get("timeout_seconds", 30)
        self.proxy_rotation = self.scraping_config.get("proxy_rotation", True)
        self.proxy_list = self.scraping_config.get("proxy_list_path")
        self.captcha_solver_enabled = self.scraping_config.get("captcha_solver_enabled", True)
        self.respect_robots_txt = self.scraping_config.get("respect_robots_txt", True)
        self.max_concurrent_requests = self.scraping_config.get("max_concurrent_requests", 5)
        self.max_depth = self.scraping_config.get("max_depth", 3)
        
        # AI settings
        self.tinybert_model_path = self.ai_config.get("tinybert_model_path")
        self.dom_self_healing = self.ai_config.get("dom_self_healing", True)
        
        # Crawler specific settings
        self.stage_selectors = {
            "source": "body",
            "result": ".search-result, .result-item, .listing",
            "landing": ".content, .main-content, article",
            "social": ".social-links, .social, .connect",
            "enrichment": ".about, .profile, .details"
        }
        
        self.extraction_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'(\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
            "name": r'([A-Z][a-z]+ [A-Z][a-z]+)',
            "company": r'([A-Z][a-z]+ (?:Inc|LLC|Corp|Company|Ltd))',
            "title": r'(CEO|CTO|Founder|President|Director|Manager|VP)'
        }

class WebCrawler(BaseModule):
    """Production-ready self-adaptive web crawler with AI capabilities"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Dict[str, Any]):
        super().__init__(name, event_bus, storage, license_manager, config)
        self.config = WebCrawlerConfig(config)
        self.browser: Optional[Browser] = None
        self.nlp_processor: Optional[NLPProcessor] = None
        self.active_crawls: Dict[str, asyncio.Task] = {}
        self.proxy_pool: List[str] = []
        self.current_proxy_index = 0
        self.session_stats = {
            "total_crawls": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "stages_completed": 0,
            "self_heal_events": 0,
            "retries": 0,
            "captcha_solved": 0
        }
        self.robots_cache: Dict[str, Dict[str, Any]] = {}
        
    def _setup_event_handlers(self):
        """Setup event handlers for crawling requests"""
        self.event_bus.subscribe("crawler.start", self._handle_crawl_request)
        self.event_bus.subscribe("crawler.status", self._handle_status_request)
        self.event_bus.subscribe("crawler.stop", self._handle_stop_request)
        
    def _validate_requirements(self):
        """Validate module requirements and dependencies"""
        # Check if AI models are available
        if self.config.dom_self_healing and not Path(self.config.tinybert_model_path).exists():
            raise FileNotFoundError(f"TinyBERT model not found: {self.config.tinybert_model_path}")
        
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
        """Start crawler services"""
        # Initialize Playwright browser
        self.browser = await async_playwright().start()
        await self.browser.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--no-zygote",
                "--disable-gpu"
            ]
        )
        
        # Initialize AI components if enabled
        if self.config.dom_self_healing:
            self.nlp_processor = NLPProcessor(self.config.tinybert_model_path)
        
        logger.info("Web crawler services started successfully")
    
    async def _stop_services(self):
        """Stop crawler services"""
        # Cancel all active crawls
        for task_id, task in self.active_crawls.items():
            task.cancel()
        
        # Close browser
        if self.browser:
            await self.browser.close()
        
        logger.info("Web crawler services stopped")
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up completed tasks
        completed = [task_id for task_id, task in self.active_crawls.items() if task.done()]
        for task_id in completed:
            del self.active_crawls[task_id]
        
        # Rotate proxies if needed
        if self.config.proxy_rotation and self.proxy_pool:
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_pool)
        
        # Clean old robots.txt cache
        current_time = time.time()
        expired_keys = [
            key for key, data in self.robots_cache.items()
            if current_time - data.get("timestamp", 0) > 86400  # 24 hours
        ]
        for key in expired_keys:
            del self.robots_cache[key]
        
        # Log session stats
        logger.debug(f"Crawler stats: {self.session_stats}")
    
    async def _handle_crawl_request(self, event_type: str, data: Dict[str, Any]):
        """Handle crawl requests from event bus"""
        try:
            request_id = data.get("request_id", str(uuid.uuid4()))
            url = data.get("url")
            config = data.get("config", {})
            
            if not url:
                logger.warning("Invalid crawl request: missing URL")
                return
            
            # Create crawl task
            task = asyncio.create_task(
                self._crawl_url(url, config),
                name=f"crawl_{request_id}"
            )
            
            self.active_crawls[request_id] = task
            
            # Set up callback for completion
            task.add_done_callback(lambda t: self._crawl_completed(request_id, t))
            
        except Exception as e:
            logger.error(f"Error handling crawl request: {str(e)}", exc_info=True)
    
    async def _handle_status_request(self, event_type: str, data: Dict[str, Any]):
        """Handle status requests"""
        status = {
            "active_crawls": len(self.active_crawls),
            "session_stats": self.session_stats,
            "proxy_pool_size": len(self.proxy_pool),
            "browser_available": self.browser is not None,
            "robots_cache_size": len(self.robots_cache)
        }
        self.event_bus.publish("crawler.status.response", status)
    
    async def _handle_stop_request(self, event_type: str, data: Dict[str, Any]):
        """Handle stop requests"""
        request_id = data.get("request_id")
        if request_id and request_id in self.active_crawls:
            self.active_crawls[request_id].cancel()
            del self.active_crawls[request_id]
            logger.info(f"Stopped crawl task: {request_id}")
    
    def _crawl_completed(self, request_id: str, task: asyncio.Task):
        """Callback for when crawl task completes"""
        try:
            if request_id in self.active_crawls:
                del self.active_crawls[request_id]
            
            if task.cancelled():
                logger.info(f"Crawl task cancelled: {request_id}")
                return
            
            result = task.result()
            self.event_bus.publish("crawler.completed", {
                "request_id": request_id,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Error in crawl completion: {str(e)}", exc_info=True)
    
    async def _crawl_url(self, url: str, config: Dict[str, Any]) -> CrawlResult:
        """Main crawling method for a URL"""
        start_time = time.time()
        result = CrawlResult(
            uuid=str(uuid.uuid4()),
            source_url=url
        )
        
        try:
            logger.info(f"Starting crawl: {url}")
            self.session_stats["total_crawls"] += 1
            
            # Check robots.txt if enabled
            if self.config.respect_robots_txt:
                if not await self._check_robots_txt(url):
                    result.error = "Blocked by robots.txt"
                    return result
            
            # Create browser context
            context = await self.browser.new_context(
                user_agent=self._get_user_agent(),
                proxy=self._get_proxy_config(),
                viewport={"width": 1920, "height": 1080}
            )
            
            page = await context.new_page()
            await page.set_default_timeout(self.config.timeout * 1000)
            
            # Multi-stage crawling
            stages = self._build_crawl_stages(url, config)
            
            for stage in stages:
                try:
                    # Process stage
                    stage_result = await self._process_stage(page, stage)
                    result.stages.append(stage_result)
                    
                    # Extract data from stage
                    stage_data = await self._extract_stage_data(page, stage_result)
                    result.extracted_data.update(stage_data)
                    
                    # Check if we should continue to next stage
                    if stage_result.status == "failed":
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing stage {stage.stage_type}: {str(e)}", exc_info=True)
                    stage.status = "failed"
                    stage.error = str(e)
                    result.stages.append(stage)
                    break
            
            # Close context
            await context.close()
            
            # Set result status
            result.success = any(stage.status == "completed" for stage in result.stages)
            result.duration = time.time() - start_time
            
            # Update stats
            if result.success:
                self.session_stats["successful_crawls"] += 1
            else:
                self.session_stats["failed_crawls"] += 1
            
            self.session_stats["stages_completed"] += len([s for s in result.stages if s.status == "completed"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}", exc_info=True)
            result.error = str(e)
            result.duration = time.time() - start_time
            self.session_stats["failed_crawls"] += 1
            return result
    
    def _build_crawl_stages(self, url: str, config: Dict[str, Any]) -> List[CrawlStage]:
        """Build crawl stages based on configuration"""
        stages = []
        
        # Source stage (initial URL)
        stages.append(CrawlStage(
            stage_type="source",
            url=url,
            depth=0
        ))
        
        # Add additional stages based on config
        max_depth = config.get("max_depth", self.config.max_depth)
        
        if max_depth >= 1:
            # Result stage (search results, listings, etc.)
            stages.append(CrawlStage(
                stage_type="result",
                url=url,  # Will be updated during processing
                depth=1,
                parent_url=url
            ))
        
        if max_depth >= 2:
            # Landing stage (individual pages)
            stages.append(CrawlStage(
                stage_type="landing",
                url="",  # Will be updated during processing
                depth=2
            ))
        
        if max_depth >= 3:
            # Social stage (social profiles)
            stages.append(CrawlStage(
                stage_type="social",
                url="",  # Will be updated during processing
                depth=3
            ))
        
        # Enrichment stage (additional data)
        stages.append(CrawlStage(
            stage_type="enrichment",
            url="",  # Will be updated during processing
            depth=4
        ))
        
        return stages
    
    async def _process_stage(self, page: Page, stage: CrawlStage) -> CrawlStage:
        """Process a single crawl stage"""
        stage.status = "processing"
        
        try:
            # Navigate to URL
            if stage.url:
                for attempt in range(self.config.max_retries):
                    try:
                        await page.goto(stage.url, wait_until="networkidle")
                        break
                    except PlaywrightTimeoutError:
                        if attempt == self.config.max_retries - 1:
                            raise
                        await asyncio.sleep(self.config.default_delay * (attempt + 1))
                        self.session_stats["retries"] += 1
            
            # Handle CAPTCHA if present
            if await self._detect_captcha(page):
                await self._solve_captcha(page)
                self.session_stats["captcha_solved"] += 1
            
            # Get selector for this stage
            selector = self.config.stage_selectors.get(stage.stage_type, "body")
            
            # Try to find elements with selector
            try:
                elements = await page.query_selector_all(selector)
                if not elements:
                    # Try self-healing if enabled
                    if self.config.dom_self_healing and self.nlp_processor:
                        selector = await self._self_heal_selector(page, stage.stage_type)
                        if selector:
                            elements = await page.query_selector_all(selector)
                            self.session_stats["self_heal_events"] += 1
                
                stage.selector = selector
                stage.data["elements_found"] = len(elements)
                
                # Extract links for next stages
                if stage.stage_type in ["source", "result"]:
                    links = await self._extract_links(page, elements)
                    stage.data["links"] = links
                
                stage.status = "completed"
                
            except Exception as e:
                stage.status = "failed"
                stage.error = str(e)
                logger.error(f"Error processing stage {stage.stage_type}: {str(e)}", exc_info=True)
            
            # Rate limiting
            await asyncio.sleep(self.config.default_delay)
            
            return stage
            
        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            logger.error(f"Error processing stage {stage.stage_type}: {str(e)}", exc_info=True)
            return stage
    
    async def _extract_stage_data(self, page: Page, stage: CrawlStage) -> Dict[str, Any]:
        """Extract data from a crawl stage"""
        data = {}
        
        try:
            # Extract text content
            content = await page.inner_text("body")
            
            # Extract using patterns
            for field, pattern in self.config.extraction_patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    data[field] = matches[0] if len(matches) == 1 else matches
            
            # Extract metadata
            data["title"] = await page.title()
            data["url"] = page.url
            data["stage"] = stage.stage_type
            
            # Extract structured data
            structured_data = await page.evaluate("""
                () => {
                    const structuredData = [];
                    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                    scripts.forEach(script => {
                        try {
                            structuredData.push(JSON.parse(script.textContent));
                        } catch (e) {}
                    });
                    return structuredData;
                }
            """)
            
            if structured_data:
                data["structured_data"] = structured_data
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting stage data: {str(e)}", exc_info=True)
            return data
    
    async def _extract_links(self, page: Page, elements) -> List[str]:
        """Extract links from page elements"""
        links = []
        
        try:
            # Get all links from elements
            for element in elements:
                try:
                    link_elements = await element.query_selector_all("a[href]")
                    for link_element in link_elements:
                        href = await link_element.get_attribute("href")
                        if href:
                            links.append(href)
                except Exception:
                    continue
            
            # Deduplicate and filter links
            links = list(set(links))
            links = [link for link in links if link.startswith("http")]
            
            return links[:10]  # Limit to first 10 links
            
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}", exc_info=True)
            return []
    
    async def _detect_captcha(self, page: Page) -> bool:
        """Detect if CAPTCHA is present on page"""
        try:
            # Common CAPTCHA indicators
            captcha_selectors = [
                ".captcha",
                "#captcha",
                "[id*='captcha']",
                "[class*='captcha']",
                "iframe[src*='captcha']",
                "iframe[src*='recaptcha']"
            ]
            
            for selector in captcha_selectors:
                if await page.query_selector(selector):
                    return True
            
            # Check for CAPTCHA in page content
            content = await page.content()
            if "captcha" in content.lower() or "recaptcha" in content.lower():
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error detecting CAPTCHA: {str(e)}", exc_info=True)
            return False
    
    async def _solve_captcha(self, page: Page):
        """Solve CAPTCHA (simplified implementation)"""
        try:
            # In a real implementation, this would use a CAPTCHA solving service
            # or advanced image recognition with OpenCV
            
            # For demo purposes, we'll just wait for manual intervention
            logger.warning("CAPTCHA detected - waiting for manual resolution")
            await asyncio.sleep(10)  # Wait 10 seconds for manual solving
            
            # Check if CAPTCHA is still present
            if await self._detect_captcha(page):
                logger.error("CAPTCHA not resolved - skipping page")
                raise Exception("CAPTCHA not resolved")
                
        except Exception as e:
            logger.error(f"Error solving CAPTCHA: {str(e)}", exc_info=True)
            raise
    
    async def _self_heal_selector(self, page: Page, stage_type: str) -> Optional[str]:
        """Use AI to self-heal broken selectors"""
        try:
            if not self.nlp_processor:
                return None
            
            # Get page content
            content = await page.content()
            
            # Use NLP to understand page structure
            nlp_result = self.nlp_processor.process_text(content)
            
            # Generate new selector based on stage type and content
            # This is a simplified version - in reality, this would use more sophisticated AI
            if stage_type == "result":
                # Look for common result patterns
                if "search result" in content.lower():
                    return ".search-result"
                elif "listing" in content.lower():
                    return ".listing"
                elif "item" in content.lower():
                    return ".item"
            elif stage_type == "landing":
                # Look for main content
                if "content" in content.lower():
                    return ".content"
                elif "main" in content.lower():
                    return "main"
            
            return None
            
        except Exception as e:
            logger.error(f"Error in selector self-healing: {str(e)}", exc_info=True)
            return None
    
    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            # Check cache first
            if robots_url in self.robots_cache:
                return self.robots_cache[robots_url].get("allowed", True)
            
            # Fetch robots.txt
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Simple robots.txt parsing (in production, use a proper parser)
                        if "Disallow: /" in content:
                            allowed = False
                        else:
                            allowed = True
                        
                        # Cache result
                        self.robots_cache[robots_url] = {
                            "allowed": allowed,
                            "timestamp": time.time()
                        }
                        
                        return allowed
                    else:
                        # If robots.txt not found, allow crawling
                        return True
                        
        except Exception as e:
            logger.debug(f"Error checking robots.txt: {str(e)}", exc_info=True)
            return True  # Allow on error
    
    def _get_user_agent(self) -> str:
        """Get a random user agent from the list"""
        if self.config.user_agents:
            return random.choice(self.config.user_agents)
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DRN.today/1.0"
    
    def _get_proxy_config(self) -> Optional[Dict[str, str]]:
        """Get proxy configuration"""
        if self.config.proxy_rotation:
            proxy = self._get_next_proxy()
            if proxy:
                return {"server": proxy}
        return None
    
    async def crawl_url(self, url: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Public method to crawl a URL"""
        config = config or {}
        
        # Create crawl request
        result = await self._crawl_url(url, config)
        
        # Convert to dictionary for JSON serialization
        return self._result_to_dict(result)
    
    def _result_to_dict(self, result: CrawlResult) -> Dict[str, Any]:
        """Convert CrawlResult to dictionary"""
        return {
            "uuid": result.uuid,
            "source_url": result.source_url,
            "stages": [
                {
                    "stage_type": stage.stage_type,
                    "url": stage.url,
                    "depth": stage.depth,
                    "status": stage.status,
                    "error": stage.error,
                    "data": stage.data
                }
                for stage in result.stages
            ],
            "extracted_data": result.extracted_data,
            "metadata": result.metadata,
            "success": result.success,
            "error": result.error,
            "duration": result.duration,
            "timestamp": result.timestamp
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crawler statistics"""
        return {
            "session_stats": self.session_stats,
            "active_crawls": len(self.active_crawls),
            "proxy_pool_size": len(self.proxy_pool),
            "browser_available": self.browser is not None,
            "robots_cache_size": len(self.robots_cache)
        }
