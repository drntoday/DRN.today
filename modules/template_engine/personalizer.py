# modules/template_engine/personalizer.py

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse

import numpy as np
from bs4 import BeautifulSoup
from playwright.async_api import Page, Error as PlaywrightError

from engine.SecureStorage import SecureStorage
from ai.nlp import NLPProcessor
from modules.web_crawlers.self_healing import DOMSelfHealingEngine


class PersonalizationLevel(Enum):
    BASIC = "basic"  # Simple name/company insertion
    INTERMEDIATE = "intermediate"  # Industry/role-based content
    ADVANCED = "advanced"  # Website-aware deep personalization
    HYPER_PERSONALIZED = "hyper_personalized"  # AI-generated unique content


@dataclass
class WebsiteData:
    url: str
    title: str
    description: str
    industry: str
    company_size: str
    key_products: List[str] = field(default_factory=list)
    key_services: List[str] = field(default_factory=list)
    recent_news: List[str] = field(default_factory=list)
    company_values: List[str] = field(default_factory=list)
    target_audience: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PersonalizationToken:
    name: str
    description: str
    source: str  # 'lead', 'website', 'enrichment', 'ai_generated'
    value: str
    confidence: float = 1.0
    fallback: str = ""


class TemplatePersonalizer:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor, 
                 self_healing_engine: DOMSelfHealingEngine = None):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.self_healing_engine = self_healing_engine
        self.logger = logging.getLogger("template_personalizer")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Personalization patterns
        self.personalization_patterns = self._initialize_personalization_patterns()
        
        # Industry-specific templates
        self.industry_templates = self._initialize_industry_templates()
        
        # Cache for website data
        self.website_cache = {}

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS website_data (
            url TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            industry TEXT,
            company_size TEXT,
            key_products TEXT,
            key_services TEXT,
            recent_news TEXT,
            company_values TEXT,
            target_audience TEXT,
            technologies TEXT,
            pain_points TEXT,
            last_updated TEXT
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS personalization_tokens (
            id TEXT PRIMARY KEY,
            lead_id TEXT,
            token_name TEXT,
            token_value TEXT,
            source TEXT,
            confidence REAL,
            fallback TEXT,
            created_at TEXT,
            FOREIGN KEY (lead_id) REFERENCES leads (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS personalization_history (
            id TEXT PRIMARY KEY,
            lead_id TEXT,
            template_id TEXT,
            personalization_level TEXT,
            tokens_used TEXT,
            created_at TEXT,
            FOREIGN KEY (lead_id) REFERENCES leads (id)
        )
        """)

    def _initialize_personalization_patterns(self) -> Dict:
        """Initialize patterns for extracting personalization data"""
        return {
            "industry": {
                "selectors": [
                    "meta[name='industry']",
                    "meta[name='sector']",
                    ".industry",
                    ".sector",
                    "footer .industry"
                ],
                "keywords": [
                    "industry", "sector", "vertical", "market", "domain"
                ]
            },
            "company_size": {
                "selectors": [
                    "meta[name='company-size']",
                    ".company-size",
                    ".employees",
                    ".team-size"
                ],
                "patterns": [
                    r"(\d+)\s*employees?",
                    r"team\s*of\s*(\d+)",
                    r"(\d+)\s*people"
                ]
            },
            "products": {
                "selectors": [
                    ".products .product",
                    ".services .service",
                    ".offerings .offering",
                    ".solutions .solution"
                ],
                "keywords": [
                    "product", "service", "solution", "offering", "feature"
                ]
            },
            "values": {
                "selectors": [
                    ".values .value",
                    ".mission",
                    ".vision",
                    ".about-us .core-values"
                ],
                "keywords": [
                    "mission", "vision", "values", "culture", "principles"
                ]
            },
            "technologies": {
                "selectors": [
                    ".tech-stack",
                    ".technologies",
                    ".tools",
                    ".platforms"
                ],
                "keywords": [
                    "technology", "stack", "platform", "tool", "software"
                ]
            }
        }

    def _initialize_industry_templates(self) -> Dict:
        """Initialize industry-specific personalization templates"""
        return {
            "Technology": {
                "pain_points": [
                    "scalability challenges",
                    "technical debt",
                    "integration issues",
                    "security concerns",
                    "talent acquisition"
                ],
                "value_propositions": [
                    "improve development efficiency",
                    "enhance system reliability",
                    "accelerate time-to-market",
                    "reduce operational costs",
                    "improve user experience"
                ]
            },
            "Healthcare": {
                "pain_points": [
                    "regulatory compliance",
                    "patient data security",
                    "operational inefficiencies",
                    "staff shortages",
                    "cost management"
                ],
                "value_propositions": [
                    "improve patient outcomes",
                    "streamline clinical workflows",
                    "ensure data compliance",
                    "reduce administrative burden",
                    "enhance care coordination"
                ]
            },
            "Finance": {
                "pain_points": [
                    "regulatory compliance",
                    "fraud detection",
                    "customer experience",
                    "operational costs",
                    "digital transformation"
                ],
                "value_propositions": [
                    "improve security posture",
                    "enhance customer experience",
                    "streamline compliance",
                    "reduce operational risk",
                    "accelerate digital initiatives"
                ]
            },
            "Retail": {
                "pain_points": [
                    "inventory management",
                    "customer retention",
                    "supply chain efficiency",
                    "omnichannel experience",
                    "price competition"
                ],
                "value_propositions": [
                    "improve customer loyalty",
                    "optimize inventory turnover",
                    "enhance shopping experience",
                    "streamline operations",
                    "increase sales conversion"
                ]
            },
            "Manufacturing": {
                "pain_points": [
                    "supply chain disruptions",
                    "production efficiency",
                    "quality control",
                    "equipment maintenance",
                    "workforce safety"
                ],
                "value_propositions": [
                    "optimize production processes",
                    "reduce downtime",
                    "improve product quality",
                    "enhance worker safety",
                    "streamline supply chain"
                ]
            }
        }

    async def scrape_website_data(self, url: str, page: Page = None) -> Optional[WebsiteData]:
        """Scrape website data for personalization"""
        # Check cache first
        if url in self.website_cache:
            cached_data = self.website_cache[url]
            # Check if cache is still valid (less than 7 days old)
            if (datetime.now() - cached_data.last_updated).days < 7:
                return cached_data
        
        try:
            # If no page provided, create one
            if page is None:
                from playwright.async_api import async_playwright
                async with async_playwright() as p:
                    browser = await p.chromium.launch()
                    context = await browser.new_context()
                    page = await context.new_page()
                    
                    try:
                        await page.goto(url, timeout=30000)
                        website_data = await self._extract_website_data(page, url)
                        return website_data
                    finally:
                        await browser.close()
            else:
                await page.goto(url, timeout=30000)
                website_data = await self._extract_website_data(page, url)
                return website_data
                
        except Exception as e:
            self.logger.error(f"Error scraping website {url}: {str(e)}")
            return None

    async def _extract_website_data(self, page: Page, url: str) -> WebsiteData:
        """Extract structured data from a website"""
        # Get page content
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract basic information
        title = await page.title()
        description = self._extract_meta_content(soup, 'description') or ""
        
        # Extract industry
        industry = await self._extract_industry(page, soup)
        
        # Extract company size
        company_size = await self._extract_company_size(page, soup)
        
        # Extract products/services
        products = await self._extract_products(page, soup)
        services = await self._extract_services(page, soup)
        
        # Extract company values
        values = await self._extract_values(page, soup)
        
        # Extract technologies
        technologies = await self._extract_technologies(page, soup)
        
        # Extract target audience
        audience = await self._extract_target_audience(page, soup)
        
        # Extract pain points (inferred)
        pain_points = self._infer_pain_points(industry, products, services)
        
        # Create website data object
        website_data = WebsiteData(
            url=url,
            title=title,
            description=description,
            industry=industry,
            company_size=company_size,
            key_products=products,
            key_services=services,
            company_values=values,
            technologies=technologies,
            target_audience=audience,
            pain_points=pain_points
        )
        
        # Cache the data
        self.website_cache[url] = website_data
        
        # Save to database
        self._save_website_data(website_data)
        
        return website_data

    def _extract_meta_content(self, soup: BeautifulSoup, name: str) -> Optional[str]:
        """Extract meta tag content by name"""
        meta_tag = soup.find('meta', attrs={'name': name})
        if meta_tag and 'content' in meta_tag.attrs:
            return meta_tag['content']
        return None

    async def _extract_industry(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract industry information from website"""
        # Try meta tags first
        industry = self._extract_meta_content(soup, 'industry')
        if industry:
            return industry
        
        # Try structured data
        industry = self._extract_from_structured_data(soup, 'industry')
        if industry:
            return industry
        
        # Try common selectors
        for selector in self.personalization_patterns["industry"]["selectors"]:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    if text and text.strip():
                        return text.strip()
            except PlaywrightError:
                continue
        
        # Use NLP to infer from content
        try:
            content = await page.text_content("body")
            if content:
                industry = self.nlp.classify_industry(content)
                if industry and industry != "Unknown":
                    return industry
        except:
            pass
        
        return "Unknown"

    async def _extract_company_size(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract company size information"""
        # Try meta tags first
        size = self._extract_meta_content(soup, 'company-size')
        if size:
            return size
        
        # Try structured data
        size = self._extract_from_structured_data(soup, 'numberOfEmployees')
        if size:
            return size
        
        # Try common selectors
        for selector in self.personalization_patterns["company_size"]["selectors"]:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    if text and text.strip():
                        return text.strip()
            except PlaywrightError:
                continue
        
        # Try to find size patterns in text
        try:
            content = await page.text_content("body")
            for pattern in self.personalization_patterns["company_size"]["patterns"]:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return f"{match.group(1)} employees"
        except:
            pass
        
        return "Unknown"

    async def _extract_products(self, page: Page, soup: BeautifulSoup) -> List[str]:
        """Extract product information"""
        products = []
        
        # Try structured data
        structured_products = self._extract_list_from_structured_data(soup, 'product')
        if structured_products:
            products.extend(structured_products)
        
        # Try common selectors
        for selector in self.personalization_patterns["products"]["selectors"]:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text and text.strip():
                        products.append(text.strip())
            except PlaywrightError:
                continue
        
        # Remove duplicates and limit to top 5
        products = list(set(products))[:5]
        return products

    async def _extract_services(self, page: Page, soup: BeautifulSoup) -> List[str]:
        """Extract service information"""
        services = []
        
        # Try structured data
        structured_services = self._extract_list_from_structured_data(soup, 'service')
        if structured_services:
            services.extend(structured_services)
        
        # Try common selectors
        for selector in self.personalization_patterns["products"]["selectors"]:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text and text.strip():
                        services.append(text.strip())
            except PlaywrightError:
                continue
        
        # Remove duplicates and limit to top 5
        services = list(set(services))[:5]
        return services

    async def _extract_values(self, page: Page, soup: BeautifulSoup) -> List[str]:
        """Extract company values"""
        values = []
        
        # Try common selectors
        for selector in self.personalization_patterns["values"]["selectors"]:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text and text.strip():
                        values.append(text.strip())
            except PlaywrightError:
                continue
        
        # Remove duplicates and limit to top 5
        values = list(set(values))[:5]
        return values

    async def _extract_technologies(self, page: Page, soup: BeautifulSoup) -> List[str]:
        """Extract technology information"""
        technologies = []
        
        # Try common selectors
        for selector in self.personalization_patterns["technologies"]["selectors"]:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text and text.strip():
                        technologies.append(text.strip())
            except PlaywrightError:
                continue
        
        # Remove duplicates and limit to top 5
        technologies = list(set(technologies))[:5]
        return technologies

    async def _extract_target_audience(self, page: Page, soup: BeautifulSoup) -> List[str]:
        """Extract target audience information"""
        audience = []
        
        # Try to find audience information in content
        try:
            content = await page.text_content("body")
            # Use NLP to extract audience information
            audience = self.nlp.extract_target_audience(content)
        except:
            pass
        
        # Remove duplicates and limit to top 5
        audience = list(set(audience))[:5]
        return audience

    def _extract_from_structured_data(self, soup: BeautifulSoup, field: str) -> Optional[str]:
        """Extract information from structured data (JSON-LD)"""
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Direct field
                    if field in data:
                        return str(data[field])
                    
                    # Nested in mainEntity
                    if 'mainEntity' in data and isinstance(data['mainEntity'], dict):
                        if field in data['mainEntity']:
                            return str(data['mainEntity'][field])
                    
                    # Nested in about
                    if 'about' in data and isinstance(data['about'], dict):
                        if field in data['about']:
                            return str(data['about'][field])
            except:
                continue
        
        return None

    def _extract_list_from_structured_data(self, soup: BeautifulSoup, field: str) -> List[str]:
        """Extract list information from structured data"""
        results = []
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Direct list field
                    if field in data and isinstance(data[field], list):
                        results.extend([str(item) for item in data[field]])
                    
                    # Nested in mainEntity
                    if 'mainEntity' in data and isinstance(data['mainEntity'], dict):
                        if field in data['mainEntity'] and isinstance(data['mainEntity'][field], list):
                            results.extend([str(item) for item in data['mainEntity'][field]])
            except:
                continue
        
        return results

    def _infer_pain_points(self, industry: str, products: List[str], services: List[str]) -> List[str]:
        """Infer pain points based on industry and offerings"""
        pain_points = []
        
        # Get industry-specific pain points
        if industry in self.industry_templates:
            pain_points.extend(self.industry_templates[industry]["pain_points"])
        
        # Infer from products/services
        all_offerings = products + services
        if any("security" in offering.lower() for offering in all_offerings):
            pain_points.append("security concerns")
        if any("integration" in offering.lower() for offering in all_offerings):
            pain_points.append("integration challenges")
        if any("analytics" in offering.lower() for offering in all_offerings):
            pain_points.append("data visibility")
        if any("automation" in offering.lower() for offering in all_offerings):
            pain_points.append("manual processes")
        
        # Remove duplicates and limit to top 5
        pain_points = list(set(pain_points))[:5]
        return pain_points

    def _save_website_data(self, website_data: WebsiteData):
        """Save website data to database"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO website_data 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                website_data.url,
                website_data.title,
                website_data.description,
                website_data.industry,
                website_data.company_size,
                json.dumps(website_data.key_products),
                json.dumps(website_data.key_services),
                json.dumps(website_data.recent_news),
                json.dumps(website_data.company_values),
                json.dumps(website_data.target_audience),
                json.dumps(website_data.technologies),
                json.dumps(website_data.pain_points),
                website_data.last_updated.isoformat()
            )
        )

    def get_website_data(self, url: str) -> Optional[WebsiteData]:
        """Get website data from database or cache"""
        # Check cache first
        if url in self.website_cache:
            return self.website_cache[url]
        
        # Check database
        row = self.SecureStorage.query(
            "SELECT * FROM website_data WHERE url = ?",
            (url,)
        ).fetchone()
        
        if not row:
            return None
        
        website_data = WebsiteData(
            url=row['url'],
            title=row['title'],
            description=row['description'],
            industry=row['industry'],
            company_size=row['company_size'],
            key_products=json.loads(row['key_products']) if row['key_products'] else [],
            key_services=json.loads(row['key_services']) if row['key_services'] else [],
            recent_news=json.loads(row['recent_news']) if row['recent_news'] else [],
            company_values=json.loads(row['company_values']) if row['company_values'] else [],
            target_audience=json.loads(row['target_audience']) if row['target_audience'] else [],
            technologies=json.loads(row['technologies']) if row['technologies'] else [],
            pain_points=json.loads(row['pain_points']) if row['pain_points'] else [],
            last_updated=datetime.fromisoformat(row['last_updated'])
        )
        
        # Update cache
        self.website_cache[url] = website_data
        
        return website_data

    def create_personalization_tokens(self, lead_id: str, lead_data: Dict, 
                                    website_data: WebsiteData = None) -> List[PersonalizationToken]:
        """Create personalization tokens for a lead"""
        tokens = []
        
        # Basic tokens from lead data
        if 'name' in lead_data:
            tokens.append(PersonalizationToken(
                name="first_name",
                description="Lead's first name",
                source="lead",
                value=lead_data['name'].split()[0],
                fallback="there"
            ))
        
        if 'company' in lead_data:
            tokens.append(PersonalizationToken(
                name="company",
                description="Lead's company name",
                source="lead",
                value=lead_data['company'],
                fallback="your company"
            ))
        
        if 'job_title' in lead_data:
            tokens.append(PersonalizationToken(
                name="job_title",
                description="Lead's job title",
                source="lead",
                value=lead_data['job_title'],
                fallback="your role"
            ))
        
        # Website-based tokens
        if website_data:
            # Industry token
            tokens.append(PersonalizationToken(
                name="industry",
                description="Company's industry",
                source="website",
                value=website_data.industry,
                fallback="your industry"
            ))
            
            # Company size token
            tokens.append(PersonalizationToken(
                name="company_size",
                description="Company size",
                source="website",
                value=website_data.company_size,
                fallback="your team"
            ))
            
            # Pain points token
            if website_data.pain_points:
                pain_points_text = ", ".join(website_data.pain_points[:2])
                tokens.append(PersonalizationToken(
                    name="pain_points",
                    description="Company's likely pain points",
                    source="website",
                    value=pain_points_text,
                    fallback="your challenges"
                ))
            
            # Value propositions token
            if website_data.industry in self.industry_templates:
                value_props = self.industry_templates[website_data.industry]["value_propositions"]
                value_props_text = ", ".join(value_props[:2])
                tokens.append(PersonalizationToken(
                    name="value_propositions",
                    description="Relevant value propositions",
                    source="website",
                    value=value_props_text,
                    fallback="value to your business"
                ))
            
            # Products token
            if website_data.key_products:
                products_text = ", ".join(website_data.key_products[:2])
                tokens.append(PersonalizationToken(
                    name="products",
                    description="Company's key products",
                    source="website",
                    value=products_text,
                    fallback="your offerings"
                ))
        
        # Save tokens to database
        for token in tokens:
            self._save_personalization_token(lead_id, token)
        
        return tokens

    def _save_personalization_token(self, lead_id: str, token: PersonalizationToken):
        """Save a personalization token to database"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO personalization_tokens 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"token_{lead_id}_{token.name}",
                lead_id,
                token.name,
                token.value,
                token.source,
                token.confidence,
                token.fallback
            )
        )

    def get_personalization_tokens(self, lead_id: str) -> List[PersonalizationToken]:
        """Get personalization tokens for a lead"""
        tokens = []
        
        for row in self.SecureStorage.query(
            "SELECT * FROM personalization_tokens WHERE lead_id = ?",
            (lead_id,)
        ):
            token = PersonalizationToken(
                name=row['token_name'],
                description="",
                source=row['source'],
                value=row['token_value'],
                confidence=row['confidence'],
                fallback=row['fallback']
            )
            tokens.append(token)
        
        return tokens

    def personalize_template(self, template: str, lead_id: str, 
                           lead_data: Dict, website_data: WebsiteData = None,
                           level: PersonalizationLevel = PersonalizationLevel.ADVANCED) -> str:
        """Personalize an email template for a lead"""
        # Create personalization tokens
        tokens = self.create_personalization_tokens(lead_id, lead_data, website_data)
        
        # Replace tokens in template
        personalized_template = template
        
        for token in tokens:
            # Replace both {{token_name}} and {token_name} formats
            patterns = [
                f"{{{{{token.name}}}}}",
                f"{{{token.name}}}"
            ]
            
            for pattern in patterns:
                if pattern in personalized_template:
                    personalized_template = personalized_template.replace(pattern, token.value)
        
        # Apply level-specific personalization
        if level == PersonalizationLevel.INTERMEDIATE:
            personalized_template = self._apply_intermediate_personalization(
                personalized_template, lead_data, website_data
            )
        elif level == PersonalizationLevel.ADVANCED:
            personalized_template = self._apply_advanced_personalization(
                personalized_template, lead_data, website_data
            )
        elif level == PersonalizationLevel.HYPER_PERSONALIZED:
            personalized_template = self._apply_hyper_personalization(
                personalized_template, lead_data, website_data
            )
        
        # Record personalization history
        self._record_personalization(lead_id, template, level, tokens)
        
        return personalized_template

    def _apply_intermediate_personalization(self, template: str, lead_data: Dict, 
                                          website_data: WebsiteData = None) -> str:
        """Apply intermediate level personalization"""
        personalized = template
        
        # Add industry-specific content if available
        if website_data and website_data.industry in self.industry_templates:
            industry_template = self.industry_templates[website_data.industry]
            
            # Add pain points section
            if "pain_points" not in personalized.lower():
                pain_points = industry_template["pain_points"][:2]
                pain_section = f"\n\nI understand that companies in the {website_data.industry} industry often face challenges with {', '.join(pain_points)}."
                personalized += pain_section
            
            # Add value proposition section
            if "value_propositions" not in personalized.lower():
                value_props = industry_template["value_propositions"][:2]
                value_section = f"\n\nOur solution helps {website_data.industry} companies {value_props[0]} and {value_props[1]}."
                personalized += value_section
        
        return personalized

    def _apply_advanced_personalization(self, template: str, lead_data: Dict, 
                                      website_data: WebsiteData = None) -> str:
        """Apply advanced level personalization"""
        # First apply intermediate personalization
        personalized = self._apply_intermediate_personalization(template, lead_data, website_data)
        
        if website_data:
            # Add product-specific content
            if website_data.key_products and "products" not in personalized.lower():
                products_text = ", ".join(website_data.key_products[:2])
                product_section = f"\n\nI noticed you offer {products_text}. Our solution integrates seamlessly with such offerings to enhance their capabilities."
                personalized += product_section
            
            # Add company values alignment
            if website_data.company_values and "values" not in personalized.lower():
                values_text = ", ".join(website_data.company_values[:2])
                values_section = f"\n\nOur company shares your commitment to {values_text}, which is why we've designed our solution to support these principles."
                personalized += values_section
            
            # Add technology alignment
            if website_data.technologies and "technology" not in personalized.lower():
                tech_text = ", ".join(website_data.technologies[:2])
                tech_section = f"\n\nOur solution is built to work with {tech_text}, ensuring smooth integration with your existing tech stack."
                personalized += tech_section
        
        return personalized

    def _apply_hyper_personalization(self, template: str, lead_data: Dict, 
                                    website_data: WebsiteData = None) -> str:
        """Apply hyper-personalization using AI"""
        # First apply advanced personalization
        personalized = self._apply_advanced_personalization(template, lead_data, website_data)
        
        # Use AI to generate unique content based on lead and website data
        if website_data:
            # Create context for AI generation
            context = {
                "company": lead_data.get('company', ''),
                "industry": website_data.industry,
                "products": website_data.key_products,
                "values": website_data.company_values,
                "pain_points": website_data.pain_points
            }
            
            # Generate personalized opening
            opening = self.nlp.generate_personalized_opening(context)
            if opening:
                # Replace generic opening with AI-generated one
                lines = personalized.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('Hi') and not line.startswith('Hello'):
                        lines.insert(i, opening)
                        break
                personalized = '\n'.join(lines)
            
            # Generate personalized value proposition
            value_prop = self.nlp.generate_personalized_value_prop(context)
            if value_prop:
                # Add to template if not already present
                if "value_prop" not in personalized.lower():
                    personalized += f"\n\n{value_prop}"
        
        return personalized

    def _record_personalization(self, lead_id: str, template: str, 
                               level: PersonalizationLevel, tokens: List[PersonalizationToken]):
        """Record personalization history"""
        self.SecureStorage.execute(
            """
            INSERT INTO personalization_history 
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"hist_{lead_id}_{int(datetime.now().timestamp())}",
                lead_id,
                "template_id",  # In a real implementation, we would pass the actual template ID
                level.value,
                json.dumps([t.name for t in tokens]),
                datetime.now().isoformat()
            )
        )

    def get_personalization_stats(self, days: int = 30) -> Dict:
        """Get personalization statistics"""
        since = datetime.now() - timedelta(days=days)
        
        # Get personalization counts by level
        level_counts = {level.value: 0 for level in PersonalizationLevel}
        total_personalizations = 0
        
        for row in self.SecureStorage.query(
            "SELECT personalization_level, COUNT(*) as count FROM personalization_history WHERE created_at >= ? GROUP BY personalization_level",
            (since.isoformat(),)
        ):
            level_counts[row['personalization_level']] = row['count']
            total_personalizations += row['count']
        
        # Get token usage statistics
        token_usage = {}
        for row in self.SecureStorage.query(
            "SELECT token_name, COUNT(*) as count FROM personalization_tokens WHERE created_at >= ? GROUP BY token_name ORDER BY count DESC LIMIT 10",
            (since.isoformat(),)
        ):
            token_usage[row['token_name']] = row['count']
        
        # Get website data freshness
        website_count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM website_data WHERE last_updated >= ?",
            (since.isoformat(),)
        ).fetchone()[0]
        
        return {
            "total_personalizations": total_personalizations,
            "level_distribution": level_counts,
            "top_tokens": token_usage,
            "websites_scraped": website_count
        }

    def refresh_website_data(self, url: str) -> Optional[WebsiteData]:
        """Force refresh website data"""
        # Remove from cache
        if url in self.website_cache:
            del self.website_cache[url]
        
        # Scrape fresh data
        # In a real implementation, we would call scrape_website_data
        # For now, we'll just return None
        return None
