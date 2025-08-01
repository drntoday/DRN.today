# modules/web_crawlers/self_healing.py

import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from playwright.async_api import Page, ElementHandle, TimeoutError as PlaywrightTimeoutError

from engine.SecureStorage import SecureStorage
from ai.nlp import NLPProcessor


class SelectorType(Enum):
    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"
    ATTRIBUTE = "attribute"
    CUSTOM = "custom"


@dataclass
class SelectorPattern:
    id: str
    url_pattern: str
    selector_type: SelectorType
    selector_value: str
    context_description: str
    success_count: int = 0
    failure_count: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    healing_attempts: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HealingResult:
    original_selector: str
    healed_selector: str
    confidence: float
    healing_method: str
    timestamp: datetime = field(default_factory=datetime.now)


class DOMSelfHealingEngine:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.logger = logging.getLogger("dom_self_healing")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load existing selector patterns
        self.selector_patterns: Dict[str, SelectorPattern] = {}
        self._load_selector_patterns()
        
        # Healing methods configuration
        self.healing_methods = {
            "semantic_similarity": 0.9,  # Weight for semantic similarity
            "structural_similarity": 0.7,  # Weight for structural similarity
            "attribute_matching": 0.8,  # Weight for attribute matching
            "text_content": 0.6,  # Weight for text content matching
            "position_similarity": 0.5  # Weight for position similarity
        }

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS selector_patterns (
            id TEXT PRIMARY KEY,
            url_pattern TEXT NOT NULL,
            selector_type TEXT NOT NULL,
            selector_value TEXT NOT NULL,
            context_description TEXT,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            last_success TEXT,
            last_failure TEXT,
            healing_attempts TEXT,
            created_at TEXT
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS healing_history (
            id TEXT PRIMARY KEY,
            selector_pattern_id TEXT,
            original_selector TEXT,
            healed_selector TEXT,
            confidence REAL,
            healing_method TEXT,
            timestamp TEXT,
            FOREIGN KEY (selector_pattern_id) REFERENCES selector_patterns (id)
        )
        """)

    def _load_selector_patterns(self):
        """Load selector patterns from SecureStorage"""
        for row in self.SecureStorage.query("SELECT * FROM selector_patterns"):
            pattern = SelectorPattern(
                id=row['id'],
                url_pattern=row['url_pattern'],
                selector_type=SelectorType(row['selector_type']),
                selector_value=row['selector_value'],
                context_description=row['context_description'],
                success_count=row['success_count'],
                failure_count=row['failure_count'],
                last_success=datetime.fromisoformat(row['last_success']) if row['last_success'] else None,
                last_failure=datetime.fromisoformat(row['last_failure']) if row['last_failure'] else None,
                healing_attempts=json.loads(row['healing_attempts']) if row['healing_attempts'] else [],
                created_at=datetime.fromisoformat(row['created_at'])
            )
            self.selector_patterns[pattern.id] = pattern

    def save_selector_pattern(self, pattern: SelectorPattern):
        """Save a selector pattern to SecureStorage"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO selector_patterns 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern.id,
                pattern.url_pattern,
                pattern.selector_type.value,
                pattern.selector_value,
                pattern.context_description,
                pattern.success_count,
                pattern.failure_count,
                pattern.last_success.isoformat() if pattern.last_success else None,
                pattern.last_failure.isoformat() if pattern.last_failure else None,
                json.dumps(pattern.healing_attempts),
                pattern.created_at.isoformat()
            )
        )

    def record_selector_success(self, pattern_id: str):
        """Record a successful selector use"""
        if pattern_id in self.selector_patterns:
            pattern = self.selector_patterns[pattern_id]
            pattern.success_count += 1
            pattern.last_success = datetime.now()
            self.save_selector_pattern(pattern)

    def record_selector_failure(self, pattern_id: str):
        """Record a failed selector use"""
        if pattern_id in self.selector_patterns:
            pattern = self.selector_patterns[pattern_id]
            pattern.failure_count += 1
            pattern.last_failure = datetime.now()
            self.save_selector_pattern(pattern)

    def get_selector_pattern(self, url: str, selector: str) -> Optional[SelectorPattern]:
        """Get a selector pattern for a URL and selector"""
        # Find matching pattern by URL pattern and selector value
        for pattern in self.selector_patterns.values():
            if self._url_matches_pattern(url, pattern.url_pattern) and pattern.selector_value == selector:
                return pattern
        return None

    def _url_matches_pattern(self, url: str, pattern: str) -> bool:
        """Check if a URL matches a pattern"""
        # Convert pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".?")
        return re.match(regex_pattern, url) is not None

    async def heal_selector(self, page: Page, original_selector: str, 
                          selector_type: SelectorType, context: str) -> HealingResult:
        """Attempt to heal a failed selector"""
        self.logger.info(f"Attempting to heal selector: {original_selector}")
        
        # Get current URL
        url = page.url
        
        # Get selector pattern if exists
        pattern = self.get_selector_pattern(url, original_selector)
        
        # If no pattern exists, create one
        if not pattern:
            pattern_id = f"pattern_{int(time.time())}"
            pattern = SelectorPattern(
                id=pattern_id,
                url_pattern=self._extract_url_pattern(url),
                selector_type=selector_type,
                selector_value=original_selector,
                context_description=context
            )
            self.selector_patterns[pattern_id] = pattern
            self.save_selector_pattern(pattern)
        
        # Record the failure
        self.record_selector_failure(pattern.id)
        
        # Get all candidate elements from the page
        candidate_elements = await self._get_candidate_elements(page, selector_type)
        
        # If no candidates found, return failure
        if not candidate_elements:
            self.logger.warning("No candidate elements found for healing")
            return HealingResult(
                original_selector=original_selector,
                healed_selector=original_selector,
                confidence=0.0,
                healing_method="no_candidates"
            )
        
        # Score each candidate element
        scored_candidates = []
        for element, selector in candidate_elements:
            score = await self._score_element(page, element, context, selector_type)
            scored_candidates.append((element, selector, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Get the best candidate
        best_element, best_selector, best_score = scored_candidates[0]
        
        # If confidence is too low, return original selector
        if best_score < 0.5:
            self.logger.warning(f"Healing confidence too low: {best_score}")
            return HealingResult(
                original_selector=original_selector,
                healed_selector=original_selector,
                confidence=best_score,
                healing_method="low_confidence"
            )
        
        # Record the healing attempt
        healing_result = HealingResult(
            original_selector=original_selector,
            healed_selector=best_selector,
            confidence=best_score,
            healing_method="semantic_similarity"
        )
        
        pattern.healing_attempts.append({
            "timestamp": datetime.now().isoformat(),
            "original_selector": original_selector,
            "healed_selector": best_selector,
            "confidence": best_score,
            "method": "semantic_similarity"
        })
        
        self.save_selector_pattern(pattern)
        
        # Save to healing history
        self.SecureStorage.execute(
            """
            INSERT INTO healing_history 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"healing_{int(time.time())}",
                pattern.id,
                original_selector,
                best_selector,
                best_score,
                datetime.now().isoformat()
            )
        )
        
        self.logger.info(f"Successfully healed selector: {original_selector} -> {best_selector} (confidence: {best_score:.2f})")
        return healing_result

    async def _get_candidate_elements(self, page: Page, selector_type: SelectorType) -> List[Tuple[ElementHandle, str]]:
        """Get candidate elements from the page based on selector type"""
        candidates = []
        
        if selector_type == SelectorType.CSS:
            # Get all interactive elements
            elements = await page.query_selector_all("button, a, input, select, textarea, [onclick], [role='button']")
            for element in elements:
                try:
                    selector = await self._generate_css_selector(element)
                    candidates.append((element, selector))
                except Exception as e:
                    self.logger.error(f"Error generating CSS selector: {str(e)}")
        
        elif selector_type == SelectorType.XPATH:
            # Get all elements with text content
            elements = await page.query_selector_all("//*[normalize-space(text())]")
            for element in elements:
                try:
                    selector = await self._generate_xpath_selector(element)
                    candidates.append((element, selector))
                except Exception as e:
                    self.logger.error(f"Error generating XPath selector: {str(e)}")
        
        elif selector_type == SelectorType.TEXT:
            # Get all elements with text content
            elements = await page.query_selector_all("//*[normalize-space(text())]")
            for element in elements:
                try:
                    text = await element.text_content()
                    if text and text.strip():
                        candidates.append((element, f"text={text.strip()}"))
                except Exception as e:
                    self.logger.error(f"Error getting text content: {str(e)}")
        
        return candidates

    async def _generate_css_selector(self, element: ElementHandle) -> str:
        """Generate a CSS selector for an element"""
        # Try to get ID first
        element_id = await element.get_attribute("id")
        if element_id:
            return f"#{element_id}"
        
        # Try to get unique class
        classes = await element.get_attribute("class")
        if classes:
            class_list = classes.split()
            if class_list:
                # Check if class is unique
                unique_class = None
                for cls in class_list:
                    count = await element.count(f".{cls}")
                    if count == 1:
                        unique_class = cls
                        break
                
                if unique_class:
                    return f".{unique_class}"
        
        # Generate path-based selector
        path = await self._get_element_path(element, "css")
        return path

    async def _generate_xpath_selector(self, element: ElementHandle) -> str:
        """Generate an XPath selector for an element"""
        # Try to get ID first
        element_id = await element.get_attribute("id")
        if element_id:
            return f"//*[@id='{element_id}']"
        
        # Generate path-based selector
        path = await self._get_element_path(element, "xpath")
        return path

    async def _get_element_path(self, element: ElementHandle, selector_type: SelectorType) -> str:
        """Get the path to an element"""
        path_parts = []
        current = element
        
        while current:
            try:
                tag_name = await current.evaluate("el => el.tagName.toLowerCase()")
                
                if selector_type == SelectorType.CSS:
                    # Get index among siblings
                    siblings = await current.query_selector_all(f"~ {tag_name}, + {tag_name}")
                    index = len(siblings) + 1
                    
                    if index == 1:
                        path_parts.append(tag_name)
                    else:
                        path_parts.append(f"{tag_name}:nth-of-type({index})")
                
                elif selector_type == SelectorType.XPATH:
                    # Get index among siblings
                    siblings = await current.query_selector_all(f"preceding-sibling::{tag_name}")
                    index = len(siblings) + 1
                    
                    path_parts.append(f"{tag_name}[{index}]")
                
                # Move to parent
                current = await current.query_selector_xpath("..")
            except Exception:
                break
        
        # Reverse path and join
        path_parts.reverse()
        
        if selector_type == SelectorType.CSS:
            return " > ".join(path_parts)
        elif selector_type == SelectorType.XPATH:
            return "/" + "/".join(path_parts)
        
        return ""

    async def _score_element(self, page: Page, element: ElementHandle, 
                           context: str, selector_type: SelectorType) -> float:
        """Score an element based on similarity to the context"""
        scores = {}
        
        # 1. Semantic similarity using TinyBERT
        semantic_score = await self._calculate_semantic_similarity(element, context)
        scores["semantic_similarity"] = semantic_score * self.healing_methods["semantic_similarity"]
        
        # 2. Structural similarity
        structural_score = await self._calculate_structural_similarity(element, selector_type)
        scores["structural_similarity"] = structural_score * self.healing_methods["structural_similarity"]
        
        # 3. Attribute matching
        attribute_score = await self._calculate_attribute_similarity(element, context)
        scores["attribute_matching"] = attribute_score * self.healing_methods["attribute_matching"]
        
        # 4. Text content matching
        text_score = await self._calculate_text_similarity(element, context)
        scores["text_content"] = text_score * self.healing_methods["text_content"]
        
        # 5. Position similarity
        position_score = await self._calculate_position_similarity(element)
        scores["position_similarity"] = position_score * self.healing_methods["position_similarity"]
        
        # Calculate weighted average
        total_weight = sum(self.healing_methods.values())
        weighted_score = sum(scores.values()) / total_weight
        
        return min(weighted_score, 1.0)

    async def _calculate_semantic_similarity(self, element: ElementHandle, context: str) -> float:
        """Calculate semantic similarity using TinyBERT"""
        try:
            # Get element text content
            element_text = await element.text_content() or ""
            
            # Get element attributes as text
            attributes = await element.evaluate("""
                el => {
                    const attrs = {};
                    for (const attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }
            """)
            
            # Combine text and attributes
            element_context = f"{element_text} {json.dumps(attributes)}"
            
            # Calculate similarity using NLP processor
            similarity = self.nlp.calculate_similarity(context, element_context)
            return similarity
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    async def _calculate_structural_similarity(self, element: ElementHandle, selector_type: SelectorType) -> float:
        """Calculate structural similarity based on element type and position"""
        try:
            # Get element tag name
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            
            # Get parent structure
            parent = await element.query_selector_xpath("..")
            parent_tag = ""
            if parent:
                parent_tag = await parent.evaluate("el => el.tagName.toLowerCase()")
            
            # Simple scoring based on common interactive elements
            interactive_elements = ["button", "a", "input", "select", "textarea"]
            if tag_name in interactive_elements:
                return 0.8
            
            # Check if it's in a common container
            common_containers = ["nav", "header", "footer", "main", "section", "div"]
            if parent_tag in common_containers:
                return 0.6
            
            return 0.3
        except Exception as e:
            self.logger.error(f"Error calculating structural similarity: {str(e)}")
            return 0.0

    async def _calculate_attribute_similarity(self, element: ElementHandle, context: str) -> float:
        """Calculate similarity based on element attributes"""
        try:
            # Get element attributes
            attributes = await element.evaluate("""
                el => {
                    const attrs = {};
                    for (const attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }
            """)
            
            # Check for important attributes
            important_attrs = ["id", "class", "name", "title", "role", "aria-label"]
            score = 0.0
            
            for attr in important_attrs:
                if attr in attributes:
                    attr_value = attributes[attr].lower()
                    context_lower = context.lower()
                    
                    # Check if attribute value contains context keywords
                    if any(keyword in attr_value for keyword in context_lower.split()):
                        score += 0.3
            
            return min(score, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating attribute similarity: {str(e)}")
            return 0.0

    async def _calculate_text_similarity(self, element: ElementHandle, context: str) -> float:
        """Calculate similarity based on text content"""
        try:
            # Get element text content
            element_text = await element.text_content() or ""
            element_text = element_text.strip().lower()
            context_lower = context.lower()
            
            # Calculate keyword overlap
            element_words = set(element_text.split())
            context_words = set(context_lower.split())
            
            if not element_words or not context_words:
                return 0.0
            
            intersection = element_words.intersection(context_words)
            union = element_words.union(context_words)
            
            jaccard = len(intersection) / len(union)
            return jaccard
        except Exception as e:
            self.logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0

    async def _calculate_position_similarity(self, element: ElementHandle) -> float:
        """Calculate similarity based on element position"""
        try:
            # Get element position
            bounding_box = await element.bounding_box()
            if not bounding_box:
                return 0.0
            
            # Get viewport size
            viewport_size = await element.evaluate("""
                () => ({
                    width: window.innerWidth,
                    height: window.innerHeight
                })
            """)
            
            # Calculate normalized position
            x_norm = bounding_box['x'] / viewport_size['width']
            y_norm = bounding_box['y'] / viewport_size['height']
            
            # Prefer elements in the top half and left side of the page
            # (common for important interactive elements)
            if y_norm < 0.5 and x_norm < 0.7:
                return 0.8
            elif y_norm < 0.7:
                return 0.6
            else:
                return 0.3
        except Exception as e:
            self.logger.error(f"Error calculating position similarity: {str(e)}")
            return 0.0

    def _extract_url_pattern(self, url: str) -> str:
        """Extract a URL pattern from a full URL"""
        # Simple pattern extraction - replace specific parts with wildcards
        pattern = re.sub(r'\d+', '*', url)  # Replace numbers with wildcards
        pattern = re.sub(r'/[a-f0-9]{8,}', '*', pattern)  # Replace IDs with wildcards
        pattern = re.sub(r'\?.*', '', pattern)  # Remove query parameters
        return pattern

    async def find_element_with_healing(self, page: Page, selector: str, 
                                      selector_type: SelectorType, 
                                      context: str, timeout: int = 5000) -> Optional[ElementHandle]:
        """Find an element with self-healing capability"""
        try:
            # First try the original selector
            if selector_type == SelectorType.CSS:
                element = await page.wait_for_selector(selector, timeout=timeout)
            elif selector_type == SelectorType.XPATH:
                element = await page.wait_for_selector(selector, timeout=timeout)
            elif selector_type == SelectorType.TEXT:
                # For text selectors, we need a different approach
                element = await page.wait_for_selector(f"text={selector}", timeout=timeout)
            else:
                element = await page.wait_for_selector(selector, timeout=timeout)
            
            # If successful, record it
            pattern = self.get_selector_pattern(page.url, selector)
            if pattern:
                self.record_selector_success(pattern.id)
            
            return element
        except PlaywrightTimeoutError:
            # Selector failed, attempt healing
            self.logger.info(f"Selector failed: {selector}, attempting healing")
            
            healing_result = await self.heal_selector(page, selector, selector_type, context)
            
            if healing_result.confidence > 0.5:
                try:
                    # Try the healed selector
                    if selector_type == SelectorType.CSS:
                        element = await page.wait_for_selector(healing_result.healed_selector, timeout=timeout)
                    elif selector_type == SelectorType.XPATH:
                        element = await page.wait_for_selector(healing_result.healed_selector, timeout=timeout)
                    elif selector_type == SelectorType.TEXT:
                        element = await page.wait_for_selector(f"text={healing_result.healed_selector}", timeout=timeout)
                    else:
                        element = await page.wait_for_selector(healing_result.healed_selector, timeout=timeout)
                    
                    # Record success
                    pattern = self.get_selector_pattern(page.url, selector)
                    if pattern:
                        self.record_selector_success(pattern.id)
                    
                    return element
                except PlaywrightTimeoutError:
                    self.logger.error(f"Healed selector also failed: {healing_result.healed_selector}")
                    return None
            else:
                self.logger.error(f"Failed to heal selector: {selector}")
                return None

    def get_healing_stats(self) -> Dict:
        """Get statistics about selector healing"""
        total_patterns = len(self.selector_patterns)
        total_healings = sum(len(p.healing_attempts) for p in self.selector_patterns.values())
        
        success_rates = {}
        for pattern_id, pattern in self.selector_patterns.items():
            total_attempts = pattern.success_count + pattern.failure_count
            if total_attempts > 0:
                success_rates[pattern_id] = pattern.success_count / total_attempts
        
        avg_success_rate = sum(success_rates.values()) / len(success_rates) if success_rates else 0
        
        return {
            "total_selector_patterns": total_patterns,
            "total_healing_attempts": total_healings,
            "average_success_rate": avg_success_rate,
            "healing_methods": list(self.healing_methods.keys())
        }

    def optimize_healing_methods(self, feedback_data: List[Dict]):
        """Optimize healing method weights based on feedback"""
        # This would be implemented to adjust weights based on success rates
        # For now, it's a placeholder
        self.logger.info("Optimizing healing methods based on feedback")
        pass
