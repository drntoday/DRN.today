import os
import re
import json
import time
import random
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urlparse, urljoin

import playwright.async_api as pwt
from bs4 import BeautifulSoup
import cv2
import numpy as np
from PIL import Image

# Local imports matching project structure
from engine.orchestrator import SystemOrchestrator
from engine.event_system import EventBus, Event, EventPriority
from ai.models.tinybert import TinyBERTModel
from ai.nlp import NLPProcessor
from modules.web_crawlers.retry_logic import RetryStrategy
from modules.web_crawlers.self_healing import DOMSelfHealingEngine
from modules.compliance.restrictions import GeoRestrictions


class Platform(Enum):
    """Supported social platforms"""
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"


@dataclass
class SocialMetrics:
    """Social media metrics for a platform"""
    platform: Platform
    followers: int
    following: int
    posts: int
    last_updated: datetime


@dataclass
class PricingPlan:
    """Pricing plan information"""
    name: str
    price: str
    features: List[str]
    currency: str = "USD"
    period: str = "month"


@dataclass
class JobPosting:
    """Job posting information"""
    title: str
    department: str
    location: str
    url: str
    description: str = ""
    requirements: List[str] = None
    posted_date: Optional[datetime] = None

    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []


class CompetitiveScraper:
    """
    Competitive Intelligence Scraper
    
    Scrapes competitor data from various sources including social media,
    pricing pages, landing pages, and job boards with advanced features
    like DOM self-healing, smart retries, and CAPTCHA solving.
    """
    
    def __init__(self, orchestrator: SystemOrchestrator, event_system: EventBus):
        self.orchestrator = orchestrator
        self.event_system = event_system
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI components
        self.tinybert = TinyBERTModel()
        self.nlp = NLPProcessor()
        
        # Initialize helper components
        self.retry_logic = RetryStrategy()
        self.dom_healing = DOMSelfHealingEngine(self.tinybert)
        self.geo_restrictions = GeoRestrictions()
        
        # Browser configuration
        self.browser_config = {
            "headless": True,
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Playwright browser instance
        self.browser: Optional[pwt.Browser] = None
        self.context: Optional[pwt.BrowserContext] = None
        
        # Proxy configuration
        self.proxy_config = None
        self.current_proxy_index = 0
        self.proxies = []
        
        # CAPTCHA solving
        self.captcha_solver = CAPTCHASolver()
        
        # Platform-specific selectors
        self.platform_selectors = {
            Platform.LINKEDIN: {
                "followers": "div.org-top-card-summary__info h3",
                "following": "div.org-top-card-summary__info h3",
                "posts": "div.feed-shared-update-v2"
            },
            Platform.TWITTER: {
                "followers": "a[href*='/followers'] span",
                "following": "a[href*='/following'] span",
                "posts": "div[data-testid='tweet']"
            },
            Platform.FACEBOOK: {
                "followers": "a[href*='/followers'] span",
                "following": "a[href*='/following'] span",
                "posts": "div.userContent"
            },
            Platform.INSTAGRAM: {
                "followers": "a[href*='/followers'] span",
                "following": "a[href*='/following'] span",
                "posts": "div.v1Nh3"
            },
            Platform.YOUTUBE: {
                "followers": "yt-formatted-string#subscriber-count",
                "following": None,
                "posts": "div.ytd-rich-item-renderer"
            }
        }
        
        # Initialize browser
        self._initialize_browser()
    
    async def _initialize_browser(self):
        """Initialize the Playwright browser"""
        try:
            playwright = await pwt.async_playwright().start()
            
            # Configure browser launch options
            launch_options = {
                "headless": self.browser_config["headless"]
            }
            
            # Add proxy if configured
            if self.proxy_config:
                launch_options["proxy"] = self.proxy_config
            
            # Launch browser
            self.browser = await playwright.chromium.launch(**launch_options)
            
            # Create context with viewport and user agent
            self.context = await self.browser.new_context(
                viewport=self.browser_config["viewport"],
                user_agent=self.browser_config["user_agent"]
            )
            
            # Set default timeout
            self.context.set_default_timeout(30000)
            
            self.logger.info("Browser initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing browser: {str(e)}")
            raise
    
    async def close_browser(self):
        """Close the browser and clean up resources"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            self.logger.info("Browser closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing browser: {str(e)}")
    
    def set_proxies(self, proxies: List[Dict[str, str]]):
        """
        Set proxy servers to use for scraping
        
        Args:
            proxies: List of proxy configurations
        """
        self.proxies = proxies
        if proxies:
            self.current_proxy_index = 0
            self._rotate_proxy()
    
    def _rotate_proxy(self):
        """Rotate to the next proxy in the list"""
        if not self.proxies:
            return
        
        proxy = self.proxies[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        
        self.proxy_config = {
            "server": proxy.get("server"),
            "username": proxy.get("username"),
            "password": proxy.get("password")
        }
        
        # Restart browser with new proxy
        asyncio.create_task(self._restart_browser())
    
    async def _restart_browser(self):
        """Restart the browser with current configuration"""
        await self.close_browser()
        await self._initialize_browser()
    
    async def get_social_metrics(self, platform: Platform, url: str) -> Optional[SocialMetrics]:
        """
        Get social media metrics for a platform
        
        Args:
            platform: Social media platform
            url: Profile URL
            
        Returns:
            SocialMetrics: Platform metrics or None if failed
        """
        try:
            # Check if scraping is allowed for this domain
            domain = urlparse(url).netloc
            if not self.geo_restrictions.is_scraping_allowed(domain):
                self.logger.warning(f"Scraping not allowed for domain: {domain}")
                return None
            
            # Get platform-specific selectors
            selectors = self.platform_selectors.get(platform)
            if not selectors:
                self.logger.error(f"Unsupported platform: {platform.value}")
                return None
            
            # Create new page
            page = await self.context.new_page()
            
            try:
                # Navigate to URL with retry logic
                await self.retry_logic.execute_async(
                    lambda: page.goto(url, wait_until="networkidle"),
                    max_retries=3,
                    retry_delay=2
                )
                
                # Wait for page to load
                await page.wait_for_load_state("networkidle")
                
                # Check for CAPTCHA
                if await self._is_captcha_present(page):
                    await self._handle_captcha(page)
                
                # Extract metrics
                metrics_data = {}
                
                # Extract followers count
                if selectors["followers"]:
                    followers_element = await self._extract_element_with_healing(
                        page, selectors["followers"], platform.value
                    )
                    if followers_element:
                        followers_text = await followers_element.inner_text()
                        metrics_data["followers"] = self._parse_count(followers_text)
                
                # Extract following count
                if selectors["following"]:
                    following_element = await self._extract_element_with_healing(
                        page, selectors["following"], platform.value
                    )
                    if following_element:
                        following_text = await following_element.inner_text()
                        metrics_data["following"] = self._parse_count(following_text)
                
                # Extract posts count
                if selectors["posts"]:
                    posts_elements = await page.query_selector_all(selectors["posts"])
                    metrics_data["posts"] = len(posts_elements)
                
                # Create metrics object
                metrics = SocialMetrics(
                    platform=platform,
                    followers=metrics_data.get("followers", 0),
                    following=metrics_data.get("following", 0),
                    posts=metrics_data.get("posts", 0),
                    last_updated=datetime.now()
                )
                
                return metrics
            finally:
                await page.close()
        except Exception as e:
            self.logger.error(f"Error getting social metrics for {platform.value}: {str(e)}")
            return None
    
    async def scrape_page(self, url: str) -> Optional[str]:
        """
        Scrape content from a web page
        
        Args:
            url: URL to scrape
            
        Returns:
            str: Page content or None if failed
        """
        try:
            # Check if scraping is allowed for this domain
            domain = urlparse(url).netloc
            if not self.geo_restrictions.is_scraping_allowed(domain):
                self.logger.warning(f"Scraping not allowed for domain: {domain}")
                return None
            
            # Create new page
            page = await self.context.new_page()
            
            try:
                # Navigate to URL with retry logic
                await self.retry_logic.execute_async(
                    lambda: page.goto(url, wait_until="networkidle"),
                    max_retries=3,
                    retry_delay=2
                )
                
                # Wait for page to load
                await page.wait_for_load_state("networkidle")
                
                # Check for CAPTCHA
                if await self._is_captcha_present(page):
                    await self._handle_captcha(page)
                
                # Get page content
                content = await page.content()
                return content
            finally:
                await page.close()
        except Exception as e:
            self.logger.error(f"Error scraping page {url}: {str(e)}")
            return None
    
    def extract_pricing_plans(self, content: str) -> List[PricingPlan]:
        """
        Extract pricing plans from page content
        
        Args:
            content: HTML content
            
        Returns:
            List[PricingPlan]: Extracted pricing plans
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            plans = []
            
            # Common pricing selectors
            pricing_selectors = [
                '.pricing-plan',
                '.price-card',
                '.plan',
                '.tier',
                '[class*="pricing"]',
                '[class*="plan"]',
                '[class*="tier"]'
            ]
            
            # Find pricing elements
            pricing_elements = []
            for selector in pricing_selectors:
                elements = soup.select(selector)
                if elements:
                    pricing_elements.extend(elements)
            
            # If no specific pricing elements found, try to find by structure
            if not pricing_elements:
                # Look for elements with price patterns
                price_pattern = re.compile(r'\$[\d,]+\.?\d*|\€[\d,]+\.?\d*|\£[\d,]+\.?\d*')
                for element in soup.find_all(['div', 'section', 'article']):
                    text = element.get_text()
                    if price_pattern.search(text):
                        pricing_elements.append(element)
            
            # Extract plan details
            for element in pricing_elements:
                plan = self._extract_plan_from_element(element)
                if plan:
                    plans.append(plan)
            
            return plans
        except Exception as e:
            self.logger.error(f"Error extracting pricing plans: {str(e)}")
            return []
    
    def _extract_plan_from_element(self, element) -> Optional[PricingPlan]:
        """Extract a single pricing plan from an HTML element"""
        try:
            # Extract plan name
            name_element = element.select_one('h1, h2, h3, h4, h5, .title, .name, [class*="title"], [class*="name"]')
            name = name_element.get_text().strip() if name_element else "Unknown"
            
            # Extract price
            price_text = ""
            price_element = element.select_one('.price, .cost, [class*="price"], [class*="cost"]')
            if price_element:
                price_text = price_element.get_text().strip()
            else:
                # Try to find price using regex
                text = element.get_text()
                price_match = re.search(r'\$[\d,]+\.?\d*|\€[\d,]+\.?\d*|\£[\d,]+\.?\d*', text)
                if price_match:
                    price_text = price_match.group(0)
            
            # Parse price and currency
            price = price_text
            currency = "USD"
            
            if price_text.startswith('$'):
                currency = "USD"
                price = price_text[1:]
            elif price_text.startswith('€'):
                currency = "EUR"
                price = price_text[1:]
            elif price_text.startswith('£'):
                currency = "GBP"
                price = price_text[1:]
            
            # Extract period
            period = "month"
            period_match = re.search(r'per\s+(month|year|week|day)', price_text.lower())
            if period_match:
                period = period_match.group(1)
            
            # Extract features
            features = []
            feature_elements = element.select('li, .feature, [class*="feature"]')
            for feature_element in feature_elements:
                feature_text = feature_element.get_text().strip()
                if feature_text and not feature_text.lower().startswith(('$', '€', '£')):
                    features.append(feature_text)
            
            # If no features found, try to extract from paragraphs
            if not features:
                for p in element.select('p'):
                    text = p.get_text().strip()
                    if text and len(text) > 10 and not text.lower().startswith(('$', '€', '£')):
                        features.append(text)
            
            return PricingPlan(
                name=name,
                price=price,
                features=features,
                currency=currency,
                period=period
            )
        except Exception as e:
            self.logger.error(f"Error extracting plan from element: {str(e)}")
            return None
    
    def extract_title(self, content: str) -> str:
        """
        Extract title from page content
        
        Args:
            content: HTML content
            
        Returns:
            str: Extracted title
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Try to find title tag
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.get_text().strip()
            
            # Try to find h1 tag
            h1_tag = soup.find('h1')
            if h1_tag:
                return h1_tag.get_text().strip()
            
            # Try to find other heading tags
            for tag in ['h2', 'h3', 'h4', 'h5', 'h6']:
                heading = soup.find(tag)
                if heading:
                    return heading.get_text().strip()
            
            # If no title found, return domain
            return "Untitled"
        except Exception as e:
            self.logger.error(f"Error extracting title: {str(e)}")
            return "Untitled"
    
    async def scrape_job_postings(self, url: str) -> List[Dict[str, Any]]:
        """
        Scrape job postings from a job board URL
        
        Args:
            url: Job board URL
            
        Returns:
            List[Dict]: Job posting data
        """
        try:
            # Check if scraping is allowed for this domain
            domain = urlparse(url).netloc
            if not self.geo_restrictions.is_scraping_allowed(domain):
                self.logger.warning(f"Scraping not allowed for domain: {domain}")
                return []
            
            # Create new page
            page = await self.context.new_page()
            
            try:
                # Navigate to URL with retry logic
                await self.retry_logic.execute_async(
                    lambda: page.goto(url, wait_until="networkidle"),
                    max_retries=3,
                    retry_delay=2
                )
                
                # Wait for page to load
                await page.wait_for_load_state("networkidle")
                
                # Check for CAPTCHA
                if await self._is_captcha_present(page):
                    await self._handle_captcha(page)
                
                # Get page content
                content = await page.content()
                
                # Extract job postings
                job_postings = self._extract_job_postings(content, url)
                
                # If no jobs found, try to click on job links and extract details
                if not job_postings:
                    job_postings = await self._extract_job_postings_with_navigation(page, url)
                
                return job_postings
            finally:
                await page.close()
        except Exception as e:
            self.logger.error(f"Error scraping job postings from {url}: {str(e)}")
            return []
    
    def _extract_job_postings(self, content: str, base_url: str) -> List[Dict[str, Any]]:
        """Extract job postings from HTML content"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            job_postings = []
            
            # Common job selectors
            job_selectors = [
                '.job-posting',
                '.job-listing',
                '.job-card',
                '.position',
                '[class*="job"]',
                '[class*="position"]'
            ]
            
            # Find job elements
            job_elements = []
            for selector in job_selectors:
                elements = soup.select(selector)
                if elements:
                    job_elements.extend(elements)
            
            # If no specific job elements found, try to find by structure
            if not job_elements:
                # Look for links with job-related keywords
                job_keywords = ['job', 'career', 'position', 'opening', 'role']
                for link in soup.find_all('a', href=True):
                    text = link.get_text().lower()
                    if any(keyword in text for keyword in job_keywords):
                        job_elements.append(link.parent)
            
            # Extract job details
            for element in job_elements:
                job = self._extract_job_from_element(element, base_url)
                if job:
                    job_postings.append(job)
            
            return job_postings
        except Exception as e:
            self.logger.error(f"Error extracting job postings: {str(e)}")
            return []
    
    def _extract_job_from_element(self, element, base_url: str) -> Optional[Dict[str, Any]]:
        """Extract a single job posting from an HTML element"""
        try:
            # Extract job title
            title_element = element.select_one('h1, h2, h3, h4, h5, .title, .job-title, [class*="title"], [class*="job-title"]')
            title = title_element.get_text().strip() if title_element else ""
            
            if not title:
                return None
            
            # Extract department
            department = ""
            department_element = element.select_one('.department, .team, [class*="department"], [class*="team"]')
            if department_element:
                department = department_element.get_text().strip()
            
            # Extract location
            location = ""
            location_element = element.select_one('.location, [class*="location"]')
            if location_element:
                location = location_element.get_text().strip()
            
            # Extract URL
            url = ""
            link_element = element.select_one('a[href]')
            if link_element:
                url = urljoin(base_url, link_element['href'])
            
            # Extract description
            description = ""
            desc_element = element.select_one('.description, .job-description, [class*="description"]')
            if desc_element:
                description = desc_element.get_text().strip()
            
            # Extract requirements
            requirements = []
            req_elements = element.select('li, .requirement, [class*="requirement"]')
            for req_element in req_elements:
                req_text = req_element.get_text().strip()
                if req_text:
                    requirements.append(req_text)
            
            # Extract posted date
            posted_date = None
            date_element = element.select_one('.date, .posted-date, [class*="date"]')
            if date_element:
                date_text = date_element.get_text().strip()
                posted_date = self._parse_date(date_text)
            
            return {
                "title": title,
                "department": department,
                "location": location,
                "url": url,
                "description": description,
                "requirements": requirements,
                "posted_date": posted_date.isoformat() if posted_date else None
            }
        except Exception as e:
            self.logger.error(f"Error extracting job from element: {str(e)}")
            return None
    
    async def _extract_job_postings_with_navigation(self, page: pwt.Page, base_url: str) -> List[Dict[str, Any]]:
        """Extract job postings by navigating to job detail pages"""
        try:
            job_postings = []
            
            # Find job links
            job_links = await page.query_selector_all('a[href]')
            
            # Filter job links
            job_urls = []
            for link in job_links:
                href = await link.get_attribute('href')
                if href and any(keyword in href.lower() for keyword in ['job', 'career', 'position']):
                    job_url = urljoin(base_url, href)
                    job_urls.append(job_url)
            
            # Limit to first 10 jobs to avoid too many requests
            job_urls = job_urls[:10]
            
            # Navigate to each job URL and extract details
            for job_url in job_urls:
                try:
                    # Create new page for each job
                    job_page = await self.context.new_page()
                    
                    try:
                        # Navigate to job URL
                        await job_page.goto(job_url, wait_until="networkidle")
                        await job_page.wait_for_load_state("networkidle")
                        
                        # Get page content
                        content = await job_page.content()
                        
                        # Extract job details
                        job = self._extract_job_from_content(content, job_url)
                        if job:
                            job_postings.append(job)
                    finally:
                        await job_page.close()
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
                except Exception as e:
                    self.logger.error(f"Error extracting job from {job_url}: {str(e)}")
            
            return job_postings
        except Exception as e:
            self.logger.error(f"Error extracting job postings with navigation: {str(e)}")
            return []
    
    def _extract_job_from_content(self, content: str, url: str) -> Optional[Dict[str, Any]]:
        """Extract job posting from HTML content"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract job title
            title_element = soup.select_one('h1, h2, h3, .job-title, [class*="job-title"]')
            title = title_element.get_text().strip() if title_element else ""
            
            if not title:
                return None
            
            # Extract department
            department = ""
            department_element = soup.select_one('.department, .team, [class*="department"], [class*="team"]')
            if department_element:
                department = department_element.get_text().strip()
            
            # Extract location
            location = ""
            location_element = soup.select_one('.location, [class*="location"]')
            if location_element:
                location = location_element.get_text().strip()
            
            # Extract description
            description = ""
            desc_element = soup.select_one('.description, .job-description, [class*="description"]')
            if desc_element:
                description = desc_element.get_text().strip()
            
            # Extract requirements
            requirements = []
            req_elements = soup.select('li, .requirement, [class*="requirement"]')
            for req_element in req_elements:
                req_text = req_element.get_text().strip()
                if req_text:
                    requirements.append(req_text)
            
            # Extract posted date
            posted_date = None
            date_element = soup.select_one('.date, .posted-date, [class*="date"]')
            if date_element:
                date_text = date_element.get_text().strip()
                posted_date = self._parse_date(date_text)
            
            return {
                "title": title,
                "department": department,
                "location": location,
                "url": url,
                "description": description,
                "requirements": requirements,
                "posted_date": posted_date.isoformat() if posted_date else None
            }
        except Exception as e:
            self.logger.error(f"Error extracting job from content: {str(e)}")
            return None
    
    async def _extract_element_with_healing(self, page: pwt.Page, selector: str, context: str) -> Optional[pwt.ElementHandle]:
        """
        Extract element with DOM self-healing
        
        Args:
            page: Playwright page
            selector: CSS selector
            context: Context for healing
            
        Returns:
            ElementHandle: Found element or None
        """
        try:
            # Try to find element with original selector
            element = await page.query_selector(selector)
            if element:
                return element
            
            # If not found, try DOM self-healing
            self.logger.info(f"Element not found with selector '{selector}', attempting DOM self-healing")
            
            # Get page content for healing
            content = await page.content()
            
            # Use DOM self-healing to find alternative selector
            new_selector = await self.dom_healing.heal_selector(content, selector, context)
            if new_selector:
                self.logger.info(f"DOM self-healing found alternative selector: '{new_selector}'")
                element = await page.query_selector(new_selector)
                if element:
                    return element
            
            # If still not found, try to find by text content
            text_content = selector.split(' ')[-1].strip()
            if text_content:
                xpath = f"//*[contains(text(), '{text_content}')]"
                element = await page.query_selector(xpath)
                if element:
                    return element
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting element with healing: {str(e)}")
            return None
    
    async def _is_captcha_present(self, page: pwt.Page) -> bool:
        """
        Check if CAPTCHA is present on the page
        
        Args:
            page: Playwright page
            
        Returns:
            bool: True if CAPTCHA is detected
        """
        try:
            # Common CAPTCHA selectors
            captcha_selectors = [
                '.captcha',
                '#captcha',
                '[class*="captcha"]',
                '[id*="captcha"]',
                '.g-recaptcha',
                '#recaptcha',
                '.h-captcha',
                '#hcaptcha'
            ]
            
            for selector in captcha_selectors:
                if await page.query_selector(selector):
                    return True
            
            # Check for CAPTCHA keywords in page text
            content = await page.text_content('body')
            if content and any(keyword in content.lower() for keyword in ['captcha', 'verify', 'robot', 'human']):
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error checking for CAPTCHA: {str(e)}")
            return False
    
    async def _handle_captcha(self, page: pwt.Page):
        """
        Handle CAPTCHA if present
        
        Args:
            page: Playwright page
        """
        try:
            self.logger.info("CAPTCHA detected, attempting to solve")
            
            # Take screenshot for CAPTCHA solving
            screenshot = await page.screenshot()
            
            # Solve CAPTCHA
            solution = await self.captcha_solver.solve(screenshot)
            
            if solution:
                # Try to input solution
                captcha_input = await page.query_selector('input[name="captcha"], input[id="captcha"]')
                if captcha_input:
                    await captcha_input.fill(solution)
                    
                    # Click submit button if present
                    submit_button = await page.query_selector('button[type="submit"], input[type="submit"]')
                    if submit_button:
                        await submit_button.click()
                        await page.wait_for_load_state("networkidle")
                
                self.logger.info("CAPTCHA solution submitted")
            else:
                self.logger.warning("Failed to solve CAPTCHA")
                
                # Try to wait for manual intervention
                await asyncio.sleep(10)
                
                # Check if CAPTCHA is still present
                if await self._is_captcha_present(page):
                    self.logger.error("CAPTCHA still present after waiting")
                    raise Exception("CAPTCHA could not be solved")
        except Exception as e:
            self.logger.error(f"Error handling CAPTCHA: {str(e)}")
            raise
    
    def _parse_count(self, text: str) -> int:
        """
        Parse count from text (e.g., "1.2K followers" -> 1200)
        
        Args:
            text: Text containing count
            
        Returns:
            int: Parsed count
        """
        try:
            # Extract number from text
            match = re.search(r'([\d,]+\.?\d*)\s*[KkMmBb]?', text)
            if not match:
                return 0
            
            num_str = match.group(1).replace(',', '')
            num = float(num_str)
            
            # Handle suffixes
            if 'K' in text or 'k' in text:
                num *= 1000
            elif 'M' in text or 'm' in text:
                num *= 1000000
            elif 'B' in text or 'b' in text:
                num *= 1000000000
            
            return int(num)
        except Exception as e:
            self.logger.error(f"Error parsing count from '{text}': {str(e)}")
            return 0
    
    def _parse_date(self, text: str) -> Optional[datetime]:
        """
        Parse date from text
        
        Args:
            text: Text containing date
            
        Returns:
            datetime: Parsed date or None
        """
        try:
            # Common date patterns
            patterns = [
                r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
                r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
                r'(\d{1,2})\s+(\w+)\s+(\d{4})',  # DD Month YYYY
                r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # Month DD, YYYY
                r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago'  # Relative date
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if 'ago' in pattern:
                        # Handle relative dates
                        num = int(match.group(1))
                        unit = match.group(2).lower()
                        
                        if 'day' in unit:
                            delta = timedelta(days=num)
                        elif 'week' in unit:
                            delta = timedelta(weeks=num)
                        elif 'month' in unit:
                            delta = timedelta(days=num * 30)
                        elif 'year' in unit:
                            delta = timedelta(days=num * 365)
                        else:
                            return None
                        
                        return datetime.now() - delta
                    else:
                        # Handle absolute dates
                        groups = match.groups()
                        if len(groups) == 3:
                            if pattern == patterns[0]:  # MM/DD/YYYY
                                month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                                return datetime(year, month, day)
                            elif pattern == patterns[1]:  # YYYY-MM-DD
                                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                                return datetime(year, month, day)
                            elif pattern == patterns[2]:  # DD Month YYYY
                                day, month_str, year = int(groups[0]), groups[1], int(groups[2])
                                month = self._month_name_to_number(month_str)
                                if month:
                                    return datetime(year, month, day)
                            elif pattern == patterns[3]:  # Month DD, YYYY
                                month_str, day, year = groups[0], int(groups[1]), int(groups[2])
                                month = self._month_name_to_number(month_str)
                                if month:
                                    return datetime(year, month, day)
            
            return None
        except Exception as e:
            self.logger.error(f"Error parsing date from '{text}': {str(e)}")
            return None
    
    def _month_name_to_number(self, month_str: str) -> Optional[int]:
        """Convert month name to number"""
        month_str = month_str.lower()
        months = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        return months.get(month_str)


class CAPTCHASolver:
    """
    CAPTCHA solver using OpenCV and OCR
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def solve(self, image_data: bytes) -> Optional[str]:
        """
        Solve CAPTCHA from image data
        
        Args:
            image_data: Image bytes
            
        Returns:
            str: CAPTCHA solution or None
        """
        try:
            # Convert image data to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Preprocess image
            processed = self._preprocess_image(img)
            
            # Try OCR with different methods
            solution = self._ocr_image(processed)
            
            if solution:
                self.logger.info(f"CAPTCHA solved: {solution}")
                return solution
            
            return None
        except Exception as e:
            self.logger.error(f"Error solving CAPTCHA: {str(e)}")
            return None
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Remove noise
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Dilate to connect text
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(opening, kernel, iterations=1)
            
            return dilated
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return img
    
    def _ocr_image(self, img: np.ndarray) -> Optional[str]:
        """Extract text from image using OCR"""
        try:
            # Try to use Tesseract OCR if available
            try:
                import pytesseract
                text = pytesseract.image_to_string(img)
                # Extract only alphanumeric characters
                solution = re.sub(r'[^a-zA-Z0-9]', '', text)
                if solution and len(solution) >= 4:
                    return solution
            except ImportError:
                self.logger.warning("Tesseract OCR not available")
            
            # Simple pattern matching for common CAPTCHA patterns
            # This is a fallback method with limited accuracy
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by x-coordinate
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            
            # Extract bounding boxes
            solution = ""
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 5 and h > 5:  # Filter small contours
                    # Simple character recognition based on contour properties
                    char = self._recognize_char(img[y:y+h, x:x+w])
                    if char:
                        solution += char
            
            if solution and len(solution) >= 4:
                return solution
            
            return None
        except Exception as e:
            self.logger.error(f"Error in OCR: {str(e)}")
            return None
    
    def _recognize_char(self, char_img: np.ndarray) -> Optional[str]:
        """Recognize a single character from image"""
        try:
            # This is a very simple character recognition
            # In a real implementation, you would use a trained model
            
            # Calculate aspect ratio
            h, w = char_img.shape
            aspect_ratio = w / max(h, 1)
            
            # Calculate density of white pixels
            density = np.sum(char_img > 0) / (w * h) if w * h > 0 else 0
            
            # Simple heuristic-based recognition
            if aspect_ratio < 0.5:
                return "1"
            elif aspect_ratio > 1.5:
                return "0"
            elif density > 0.7:
                return "8"
            elif density > 0.5:
                return "6"
            elif density > 0.3:
                return "4"
            else:
                return "2"
        except Exception as e:
            self.logger.error(f"Error recognizing character: {str(e)}")
            return None