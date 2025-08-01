import asyncio
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import uuid
import base64
import socket
import whois
import dns.resolver
import requests
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
import cv2
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
import tldextract

# AI/ML imports
from ai.nlp import NLPProcessor
from ai.scoring import LeadScorer

# Core system imports
from engine.orchestrator import BaseModule
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# Initialize scraper logger
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Data structure for scraping results"""
    uuid: str
    source: str
    url: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    social_links: Dict[str, str] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    category: Optional[str] = None
    score: float = 0.0
    raw_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class ScraperConfig:
    """Configuration for the scraper module"""
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
        
        # AI settings
        self.tinybert_model_path = self.ai_config.get("tinybert_model_path")
        self.scoring_threshold = self.ai_config.get("scoring_threshold", 0.75)
        self.batch_size = self.ai_config.get("batch_size", 32)

class LeadScraper(BaseModule):
    """Production-ready multi-platform lead scraper with AI capabilities"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Dict[str, Any]):
        super().__init__(name, event_bus, storage, license_manager, config)
        self.config = ScraperConfig(config)
        self.browser: Optional[Browser] = None
        self.nlp_processor: Optional[NLPProcessor] = None
        self.lead_scorer: Optional[LeadScorer] = None
        self.active_scrapers: Dict[str, asyncio.Task] = {}
        self.proxy_pool: List[str] = []
        self.current_proxy_index = 0
        self.session_stats = {
            "total_scraped": 0,
            "successful": 0,
            "failed": 0,
            "captcha_solved": 0,
            "retries": 0
        }
        
    def _setup_event_handlers(self):
        """Setup event handlers for scraping requests"""
        self.event_bus.subscribe("scraping.request", self._handle_scraping_request)
        self.event_bus.subscribe("scraping.status", self._handle_status_request)
        self.event_bus.subscribe("scraping.stop", self._handle_stop_request)
        
    def _validate_requirements(self):
        """Validate module requirements and dependencies"""
        # Check if AI models are available
        if not Path(self.config.tinybert_model_path).exists():
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
            logger.error(f"Failed to load proxy pool: {str(e)}")
            self.config.proxy_rotation = False
    
    def _get_next_proxy(self) -> Optional[str]:
        """Get next proxy from rotation pool"""
        if not self.proxy_pool:
            return None
            
        proxy = self.proxy_pool[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_pool)
        return proxy
    
    async def _start_services(self):
        """Start scraper services"""
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
        
        # Initialize AI components
        self.nlp_processor = NLPProcessor(self.config.tinybert_model_path)
        self.lead_scorer = LeadScorer()
        
        logger.info("Scraper services started successfully")
    
    async def _stop_services(self):
        """Stop scraper services"""
        # Cancel all active scrapers
        for task_id, task in self.active_scrapers.items():
            task.cancel()
        
        # Close browser
        if self.browser:
            await self.browser.close()
        
        logger.info("Scraper services stopped")
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up completed tasks
        completed = [task_id for task_id, task in self.active_scrapers.items() if task.done()]
        for task_id in completed:
            del self.active_scrapers[task_id]
        
        # Rotate proxies if needed
        if self.config.proxy_rotation and self.proxy_pool:
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_pool)
        
        # Log session stats
        logger.debug(f"Scraper stats: {self.session_stats}")
    
    async def _handle_scraping_request(self, event_type: str, data: Dict[str, Any]):
        """Handle scraping requests from event bus"""
        try:
            request_id = data.get("request_id", str(uuid.uuid4()))
            source = data.get("source")
            query = data.get("query")
            params = data.get("params", {})
            
            if not source or not query:
                logger.warning("Invalid scraping request: missing source or query")
                return
            
            # Create scraping task
            task = asyncio.create_task(
                self._scrape_source(source, query, params),
                name=f"scrape_{source}_{request_id}"
            )
            
            self.active_scrapers[request_id] = task
            
            # Set up callback for completion
            task.add_done_callback(lambda t: self._scraping_completed(request_id, t))
            
        except Exception as e:
            logger.error(f"Error handling scraping request: {str(e)}", exc_info=True)
    
    async def _handle_status_request(self, event_type: str, data: Dict[str, Any]):
        """Handle status requests"""
        status = {
            "active_scrapers": len(self.active_scrapers),
            "session_stats": self.session_stats,
            "proxy_pool_size": len(self.proxy_pool),
            "browser_available": self.browser is not None
        }
        self.event_bus.publish("scraping.status.response", status)
    
    async def _handle_stop_request(self, event_type: str, data: Dict[str, Any]):
        """Handle stop requests"""
        request_id = data.get("request_id")
        if request_id and request_id in self.active_scrapers:
            self.active_scrapers[request_id].cancel()
            del self.active_scrapers[request_id]
            logger.info(f"Stopped scraping task: {request_id}")
    
    def _scraping_completed(self, request_id: str, task: asyncio.Task):
        """Callback for when scraping task completes"""
        try:
            if request_id in self.active_scrapers:
                del self.active_scrapers[request_id]
            
            if task.cancelled():
                logger.info(f"Scraping task cancelled: {request_id}")
                return
            
            result = task.result()
            self.event_bus.publish("scraping.completed", {
                "request_id": request_id,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Error in scraping completion: {str(e)}", exc_info=True)
    
    async def _scrape_source(self, source: str, query: str, params: Dict[str, Any]) -> List[ScrapingResult]:
        """Main scraping method for different sources"""
        try:
            logger.info(f"Starting scrape for {source}: {query}")
            
            # Route to appropriate scraper
            if source.lower() in ["google", "yahoo", "duckduckgo", "yandex"]:
                return await self._scrape_search_engine(source, query, params)
            elif source.lower() == "linkedin":
                return await self._scrape_linkedin(query, params)
            elif source.lower() in ["discord", "telegram", "reddit"]:
                return await self._scrape_social_platform(source, query, params)
            elif source.lower() == "github":
                return await self._scrape_github(query, params)
            elif source.lower() in ["crunchbase", "angellist", "producthunt"]:
                return await self._scrape_startup_platform(source, query, params)
            elif source.lower() == "whois":
                return await self._scrape_whois(query, params)
            elif source.lower() == "dns":
                return await self._scrape_dns(query, params)
            else:
                logger.warning(f"Unsupported source: {source}")
                return []
                
        except Exception as e:
            logger.error(f"Error scraping {source}: {str(e)}", exc_info=True)
            self.session_stats["failed"] += 1
            return []
    
    async def _scrape_search_engine(self, engine: str, query: str, params: Dict[str, Any]) -> List[ScrapingResult]:
        """Scrape search engine results"""
        results = []
        
        try:
            # Build search URL
            search_urls = {
                "google": f"https://www.google.com/search?q={query}",
                "yahoo": f"https://search.yahoo.com/search?p={query}",
                "duckduckgo": f"https://duckduckgo.com/?q={query}",
                "yandex": f"https://yandex.com/search/?text={query}"
            }
            
            url = search_urls.get(engine.lower())
            if not url:
                return results
            
            # Create browser context
            context = await self.browser.new_context(
                user_agent=self._get_user_agent(),
                proxy=self._get_proxy_config()
            )
            
            page = await context.new_page()
            
            # Set timeout and retry logic
            await page.set_default_timeout(self.config.timeout * 1000)
            
            # Navigate to search results
            for attempt in range(self.config.max_retries):
                try:
                    await page.goto(url, wait_until="networkidle")
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
            
            # Extract search results
            if engine.lower() == "google":
                results = await self._extract_google_results(page, query)
            elif engine.lower() == "yahoo":
                results = await self._extract_yahoo_results(page, query)
            elif engine.lower() == "duckduckgo":
                results = await self._extract_duckduckgo_results(page, query)
            elif engine.lower() == "yandex":
                results = await self._extract_yandex_results(page, query)
            
            # Process and score results
            processed_results = []
            for result in results:
                processed = await self._process_scraped_result(result, engine)
                if processed:
                    processed_results.append(processed)
            
            # Close context
            await context.close()
            
            self.session_stats["successful"] += len(processed_results)
            self.session_stats["total_scraped"] += len(results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error scraping {engine}: {str(e)}", exc_info=True)
            return []
    
    async def _extract_google_results(self, page: Page, query: str) -> List[ScrapingResult]:
        """Extract results from Google search page"""
        results = []
        
        try:
            # Wait for results to load
            await page.wait_for_selector("div.g", timeout=5000)
            
            # Extract result elements
            result_elements = await page.query_selector_all("div.g")
            
            for element in result_elements:
                try:
                    # Extract title
                    title_element = await element.query_selector("h3")
                    title = await title_element.inner_text() if title_element else ""
                    
                    # Extract URL
                    link_element = await element.query_selector("a")
                    url = await link_element.get_attribute("href") if link_element else ""
                    
                    # Extract description
                    desc_element = await element.query_selector("div[style='-webkit-line-clamp:2']")
                    description = await desc_element.inner_text() if desc_element else ""
                    
                    if url and title:
                        result = ScrapingResult(
                            uuid=str(uuid.uuid4()),
                            source="google",
                            url=url,
                            name=title,
                            raw_content=description,
                            metadata={"query": query, "position": len(results) + 1}
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error extracting Google result: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting Google results: {str(e)}", exc_info=True)
        
        return results
    
    async def _extract_yahoo_results(self, page: Page, query: str) -> List[ScrapingResult]:
        """Extract results from Yahoo search page"""
        results = []
        
        try:
            # Wait for results to load
            await page.wait_for_selector(".algo-sr", timeout=5000)
            
            # Extract result elements
            result_elements = await page.query_selector_all(".algo-sr")
            
            for element in result_elements:
                try:
                    # Extract title
                    title_element = await element.query_selector("h3 a")
                    title = await title_element.inner_text() if title_element else ""
                    url = await title_element.get_attribute("href") if title_element else ""
                    
                    # Extract description
                    desc_element = await element.query_selector(".compText")
                    description = await desc_element.inner_text() if desc_element else ""
                    
                    if url and title:
                        result = ScrapingResult(
                            uuid=str(uuid.uuid4()),
                            source="yahoo",
                            url=url,
                            name=title,
                            raw_content=description,
                            metadata={"query": query, "position": len(results) + 1}
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error extracting Yahoo result: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting Yahoo results: {str(e)}", exc_info=True)
        
        return results
    
    async def _extract_duckduckgo_results(self, page: Page, query: str) -> List[ScrapingResult]:
        """Extract results from DuckDuckGo search page"""
        results = []
        
        try:
            # Wait for results to load
            await page.wait_for_selector(".result", timeout=5000)
            
            # Extract result elements
            result_elements = await page.query_selector_all(".result")
            
            for element in result_elements:
                try:
                    # Extract title
                    title_element = await element.query_selector("h2 a")
                    title = await title_element.inner_text() if title_element else ""
                    url = await title_element.get_attribute("href") if title_element else ""
                    
                    # Extract description
                    desc_element = await element.query_selector(".result__snippet")
                    description = await desc_element.inner_text() if desc_element else ""
                    
                    if url and title:
                        result = ScrapingResult(
                            uuid=str(uuid.uuid4()),
                            source="duckduckgo",
                            url=url,
                            name=title,
                            raw_content=description,
                            metadata={"query": query, "position": len(results) + 1}
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error extracting DuckDuckGo result: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting DuckDuckGo results: {str(e)}", exc_info=True)
        
        return results
    
    async def _extract_yandex_results(self, page: Page, query: str) -> List[ScrapingResult]:
        """Extract results from Yandex search page"""
        results = []
        
        try:
            # Wait for results to load
            await page.wait_for_selector(".serp-item", timeout=5000)
            
            # Extract result elements
            result_elements = await page.query_selector_all(".serp-item")
            
            for element in result_elements:
                try:
                    # Extract title
                    title_element = await element.query_selector("h2 a")
                    title = await title_element.inner_text() if title_element else ""
                    url = await title_element.get_attribute("href") if title_element else ""
                    
                    # Extract description
                    desc_element = await element.query_selector(".organic__content-wrapper")
                    description = await desc_element.inner_text() if desc_element else ""
                    
                    if url and title:
                        result = ScrapingResult(
                            uuid=str(uuid.uuid4()),
                            source="yandex",
                            url=url,
                            name=title,
                            raw_content=description,
                            metadata={"query": query, "position": len(results) + 1}
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error extracting Yandex result: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting Yandex results: {str(e)}", exc_info=True)
        
        return results
    
    async def _scrape_linkedin(self, query: str, params: Dict[str, Any]) -> List[ScrapingResult]:
        """Scrape LinkedIn for professional leads"""
        results = []
        
        try:
            # Build search URL
            search_url = f"https://www.linkedin.com/search/results/people/?keywords={query}"
            
            # Create browser context
            context = await self.browser.new_context(
                user_agent=self._get_user_agent(),
                proxy=self._get_proxy_config()
            )
            
            page = await context.new_page()
            await page.set_default_timeout(self.config.timeout * 1000)
            
            # Navigate to LinkedIn
            for attempt in range(self.config.max_retries):
                try:
                    await page.goto(search_url, wait_until="networkidle")
                    break
                except PlaywrightTimeoutError:
                    if attempt == self.config.max_retries - 1:
                        raise
                    await asyncio.sleep(self.config.default_delay * (attempt + 1))
                    self.session_stats["retries"] += 1
            
            # Handle login if needed (simplified for demo)
            if await page.query_selector("#login-email"):
                logger.warning("LinkedIn login required - skipping")
                return results
            
            # Extract profile results
            await page.wait_for_selector(".search-result", timeout=5000)
            profile_elements = await page.query_selector_all(".search-result")
            
            for element in profile_elements:
                try:
                    # Extract name
                    name_element = await element.query_selector(".actor-name")
                    name = await name_element.inner_text() if name_element else ""
                    
                    # Extract title
                    title_element = await element.query_selector(".subline-level-1")
                    title = await title_element.inner_text() if title_element else ""
                    
                    # Extract company
                    company_element = await element.query_selector(".subline-level-2")
                    company = await company_element.inner_text() if company_element else ""
                    
                    # Extract profile URL
                    link_element = await element.query_selector("a.search-result__result-link")
                    url = await link_element.get_attribute("href") if link_element else ""
                    
                    if name and url:
                        result = ScrapingResult(
                            uuid=str(uuid.uuid4()),
                            source="linkedin",
                            url=url,
                            name=name,
                            company=company,
                            raw_content=f"{title} at {company}",
                            metadata={"query": query, "position": len(results) + 1}
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error extracting LinkedIn profile: {str(e)}")
                    continue
            
            # Close context
            await context.close()
            
            # Process results
            processed_results = []
            for result in results:
                processed = await self._process_scraped_result(result, "linkedin")
                if processed:
                    processed_results.append(processed)
            
            self.session_stats["successful"] += len(processed_results)
            self.session_stats["total_scraped"] += len(results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error scraping LinkedIn: {str(e)}", exc_info=True)
            return []
    
    async def _scrape_social_platform(self, platform: str, query: str, params: Dict[str, Any]) -> List[ScrapingResult]:
        """Scrape social platforms (Discord, Telegram, Reddit)"""
        # Simplified implementation for demo
        logger.info(f"Scraping {platform} for: {query}")
        
        # In a real implementation, this would use platform-specific APIs or web scraping
        # For now, return empty results
        return []
    
    async def _scrape_github(self, query: str, params: Dict[str, Any]) -> List[ScrapingResult]:
        """Scrape GitHub for developer leads"""
        results = []
        
        try:
            # Build search URL
            search_url = f"https://github.com/search?q={query}&type=users"
            
            # Create browser context
            context = await self.browser.new_context(
                user_agent=self._get_user_agent(),
                proxy=self._get_proxy_config()
            )
            
            page = await context.new_page()
            await page.set_default_timeout(self.config.timeout * 1000)
            
            # Navigate to GitHub
            await page.goto(search_url, wait_until="networkidle")
            
            # Extract user results
            await page.wait_for_selector(".user-list-item", timeout=5000)
            user_elements = await page.query_selector_all(".user-list-item")
            
            for element in user_elements:
                try:
                    # Extract username
                    username_element = await element.query_selector("a[data-hovercard-type='user']")
                    username = await username_element.inner_text() if username_element else ""
                    url = await username_element.get_attribute("href") if username_element else ""
                    
                    # Extract full name
                    name_element = await element.query_selector(".user-list-info")
                    name = await name_element.inner_text() if name_element else ""
                    
                    if username and url:
                        result = ScrapingResult(
                            uuid=str(uuid.uuid4()),
                            source="github",
                            url=url,
                            name=name or username,
                            metadata={"query": query, "username": username}
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error extracting GitHub user: {str(e)}")
                    continue
            
            # Close context
            await context.close()
            
            # Process results
            processed_results = []
            for result in results:
                processed = await self._process_scraped_result(result, "github")
                if processed:
                    processed_results.append(processed)
            
            self.session_stats["successful"] += len(processed_results)
            self.session_stats["total_scraped"] += len(results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error scraping GitHub: {str(e)}", exc_info=True)
            return []
    
    async def _scrape_startup_platform(self, platform: str, query: str, params: Dict[str, Any]) -> List[ScrapingResult]:
        """Scrape startup platforms (Crunchbase, AngelList, Product Hunt)"""
        # Simplified implementation for demo
        logger.info(f"Scraping {platform} for: {query}")
        return []
    
    async def _scrape_whois(self, domain: str, params: Dict[str, Any]) -> List[ScrapingResult]:
        """Scrape WHOIS information for email mining"""
        results = []
        
        try:
            # Get WHOIS data
            w = whois.whois(domain)
            
            # Extract contact information
            emails = w.emails or []
            name = w.org_name or w.name
            registrar = w.registrar
            
            for email in emails:
                result = ScrapingResult(
                    uuid=str(uuid.uuid4()),
                    source="whois",
                    url=f"https://{domain}",
                    name=name,
                    email=email,
                    company=registrar,
                    metadata={"domain": domain, "registrar": registrar}
                )
                results.append(result)
            
            # Process results
            processed_results = []
            for result in results:
                processed = await self._process_scraped_result(result, "whois")
                if processed:
                    processed_results.append(processed)
            
            self.session_stats["successful"] += len(processed_results)
            self.session_stats["total_scraped"] += len(results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error scraping WHOIS for {domain}: {str(e)}", exc_info=True)
            return []
    
    async def _scrape_dns(self, domain: str, params: Dict[str, Any]) -> List[ScrapingResult]:
        """Scrape DNS records for email mining"""
        results = []
        
        try:
            # Get MX records
            resolver = dns.resolver.Resolver()
            mx_records = resolver.resolve(domain, 'MX')
            
            # Extract mail server domains
            mail_domains = []
            for mx in mx_records:
                mail_domains.append(str(mx.exchange).rstrip('.'))
            
            # Common email patterns
            email_patterns = [
                "info@{domain}",
                "contact@{domain}",
                "admin@{domain}",
                "support@{domain}",
                "hello@{domain}"
            ]
            
            # Generate potential emails
            for mail_domain in mail_domains:
                for pattern in email_patterns:
                    email = pattern.format(domain=mail_domain)
                    
                    result = ScrapingResult(
                        uuid=str(uuid.uuid4()),
                        source="dns",
                        url=f"https://{domain}",
                        email=email,
                        metadata={"domain": domain, "mail_server": mail_domain}
                    )
                    results.append(result)
            
            # Process results
            processed_results = []
            for result in results:
                processed = await self._process_scraped_result(result, "dns")
                if processed:
                    processed_results.append(processed)
            
            self.session_stats["successful"] += len(processed_results)
            self.session_stats["total_scraped"] += len(results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error scraping DNS for {domain}: {str(e)}", exc_info=True)
            return []
    
    async def _process_scraped_result(self, result: ScrapingResult, source: str) -> Optional[ScrapingResult]:
        """Process and enrich scraped result with AI"""
        try:
            # Extract additional information from URL
            if result.url:
                parsed_url = urlparse(result.url)
                if not result.website:
                    result.website = f"{parsed_url.scheme}://{parsed_url.netloc}"
                
                # Extract domain
                extracted = tldextract.extract(result.url)
                domain = f"{extracted.domain}.{extracted.suffix}"
                
                # Try to get company name from domain
                if not result.company:
                    result.company = extracted.domain.title()
            
            # Extract contact information from raw content
            if result.raw_content:
                # Extract emails
                emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', result.raw_content)
                if emails and not result.email:
                    result.email = emails[0]
                
                # Extract phone numbers
                phones = re.findall(r'(\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', result.raw_content)
                if phones and not result.phone:
                    result.phone = phones[0]
                
                # Extract keywords using NLP
                if self.nlp_processor:
                    keywords = await self.nlp_processor.extract_keywords(result.raw_content)
                    result.keywords = keywords[:10]  # Top 10 keywords
            
            # Score the lead
            if self.lead_scorer:
                score_data = {
                    "name": result.name,
                    "email": result.email,
                    "phone": result.phone,
                    "company": result.company,
                    "source": source,
                    "keywords": result.keywords,
                    "raw_content": result.raw_content
                }
                result.score = await self.lead_scorer.score_lead(score_data)
            
            # Only return results above threshold
            if result.score >= self.config.scoring_threshold:
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error processing scraped result: {str(e)}", exc_info=True)
            return None
    
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
            logger.debug(f"Error detecting CAPTCHA: {str(e)}")
            return False
    
    async def _solve_captcha(self, page: Page):
        """Solve CAPTCHA using OpenCV (simplified implementation)"""
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
    
    def _get_user_agent(self) -> str:
        """Get a random user agent from the list"""
        if self.config.user_agents:
            import random
            return random.choice(self.config.user_agents)
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DRN.today/1.0"
    
    def _get_proxy_config(self) -> Optional[Dict[str, str]]:
        """Get proxy configuration"""
        if self.config.proxy_rotation:
            proxy = self._get_next_proxy()
            if proxy:
                return {"server": proxy}
        return None
    
    async def scrape_leads(self, source: str, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Public method to scrape leads from a source"""
        params = params or {}
        
        # Create scraping request
        request_data = {
            "source": source,
            "query": query,
            "params": params
        }
        
        # Handle the request synchronously for this method
        results = await self._scrape_source(source, query, params)
        
        # Convert to dictionaries for JSON serialization
        return [self._result_to_dict(result) for result in results]
    
    def _result_to_dict(self, result: ScrapingResult) -> Dict[str, Any]:
        """Convert ScrapingResult to dictionary"""
        return {
            "uuid": result.uuid,
            "source": result.source,
            "url": result.url,
            "name": result.name,
            "email": result.email,
            "phone": result.phone,
            "company": result.company,
            "website": result.website,
            "location": result.location,
            "social_links": result.social_links,
            "keywords": result.keywords,
            "category": result.category,
            "score": result.score,
            "raw_content": result.raw_content,
            "metadata": result.metadata,
            "timestamp": result.timestamp
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics"""
        return {
            "session_stats": self.session_stats,
            "active_scrapers": len(self.active_scrapers),
            "proxy_pool_size": len(self.proxy_pool),
            "browser_available": self.browser is not None
        }
