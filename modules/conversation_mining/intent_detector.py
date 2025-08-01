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
import numpy as np
from collections import defaultdict, Counter

# Core system imports
from engine.orchestrator import BaseModule
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# AI imports
from ai.nlp import NLPProcessor

# Initialize intent detector logger
logger = logging.getLogger(__name__)

@dataclass
class IntentPattern:
    """Intent pattern configuration"""
    uuid: str
    name: str
    pattern: str
    intent_type: str
    confidence_threshold: float = 0.7
    weight: float = 1.0
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectedIntent:
    """Detected intent data structure"""
    uuid: str
    source_id: str
    source_type: str  # "reddit", "discord", "telegram", "forum"
    intent_type: str
    confidence: float
    text: str
    pattern_uuid: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    lead_score: float = 0.0
    urgency_level: str = "medium"  # "low", "medium", "high", "critical"
    timestamp: float = field(default_factory=time.time)

@dataclass
class ConversationThread:
    """Conversation thread data structure"""
    uuid: str
    source_id: str
    source_type: str
    title: str
    url: Optional[str] = None
    author: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    intents: List[DetectedIntent] = field(default_factory=list)
    cluster_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class IntentDetectorConfig:
    """Configuration for the intent detector module"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.ai_config = config_dict.get("ai", {})
        self.scraping_config = config_dict.get("scraping", {})
        
        # AI settings
        self.tinybert_model_path = self.ai_config.get("tinybert_model_path")
        self.confidence_threshold = self.ai_config.get("confidence_threshold", 0.75)
        self.batch_size = self.ai_config.get("batch_size", 32)
        
        # Intent detection settings
        self.intent_types = [
            "buying_signal",
            "information_request",
            "recommendation_request",
            "complaint",
            "praise",
            "comparison",
            "pricing_inquiry",
            "demo_request",
            "trial_request",
            "partnership_inquiry"
        ]
        
        # Pattern settings
        self.patterns_file = self.scraping_config.get("patterns_file", "resources/intent_patterns.json")
        self.custom_patterns_enabled = self.scraping_config.get("custom_patterns_enabled", True)
        
        # Processing settings
        self.max_thread_length = self.scraping_config.get("max_thread_length", 10000)
        self.min_message_length = self.scraping_config.get("min_message_length", 10)
        self.context_window_size = self.scraping_config.get("context_window_size", 5)
        
        # Scoring settings
        self.urgency_keywords = {
            "critical": ["urgent", "asap", "immediately", "emergency", "now"],
            "high": ["soon", "quickly", "fast", "need", "required"],
            "medium": ["looking", "searching", "considering", "interested"],
            "low": ["maybe", "someday", "future", "possibly"]
        }

class IntentDetector(BaseModule):
    """Production-ready intent detector for conversation mining"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Dict[str, Any]):
        super().__init__(name, event_bus, storage, license_manager, config)
        self.config = IntentDetectorConfig(config)
        self.nlp_processor: Optional[NLPProcessor] = None
        self.patterns: Dict[str, IntentPattern] = {}
        self.thread_cache: Dict[str, ConversationThread] = {}
        self.session_stats = {
            "threads_processed": 0,
            "intents_detected": 0,
            "high_value_intents": 0,
            "patterns_used": 0,
            "ai_inferences": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    def _setup_event_handlers(self):
        """Setup event handlers for intent detection requests"""
        self.event_bus.subscribe("intent.detect", self._handle_detection_request)
        self.event_bus.subscribe("intent.pattern.add", self._handle_pattern_add_request)
        self.event_bus.subscribe("intent.pattern.remove", self._handle_pattern_remove_request)
        self.event_bus.subscribe("intent.status", self._handle_status_request)
        
    def _validate_requirements(self):
        """Validate module requirements and dependencies"""
        # Check if AI models are available
        if not Path(self.config.tinybert_model_path).exists():
            raise FileNotFoundError(f"TinyBERT model not found: {self.config.tinybert_model_path}")
        
        # Load intent patterns
        self._load_patterns()
        
    async def _start_services(self):
        """Start intent detector services"""
        # Initialize AI components
        self.nlp_processor = NLPProcessor(self.config.tinybert_model_path)
        
        logger.info("Intent detector services started successfully")
    
    async def _stop_services(self):
        """Stop intent detector services"""
        logger.info("Intent detector services stopped")
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, thread in self.thread_cache.items()
            if current_time - thread.updated_at > 86400  # 24 hours
        ]
        for key in expired_keys:
            del self.thread_cache[key]
        
        # Log session stats
        logger.debug(f"Intent detector stats: {self.session_stats}")
    
    async def _handle_detection_request(self, event_type: str, data: Dict[str, Any]):
        """Handle intent detection requests"""
        try:
            thread_data = data.get("thread")
            if not thread_data:
                logger.warning("Invalid detection request: missing thread data")
                return
            
            # Detect intents
            intents = await self.detect_intents(thread_data)
            
            self.event_bus.publish("intent.detected", {
                "thread_id": thread_data.get("uuid"),
                "intents": intents
            })
            
        except Exception as e:
            logger.error(f"Error handling detection request: {str(e)}", exc_info=True)
    
    async def _handle_pattern_add_request(self, event_type: str, data: Dict[str, Any]):
        """Handle pattern addition requests"""
        try:
            pattern_data = data.get("pattern")
            if not pattern_data:
                logger.warning("Invalid pattern add request: missing pattern data")
                return
            
            # Add pattern
            pattern = self._add_pattern(pattern_data)
            if pattern:
                self.event_bus.publish("intent.pattern.added", {
                    "pattern_uuid": pattern.uuid,
                    "name": pattern.name
                })
            
        except Exception as e:
            logger.error(f"Error handling pattern add request: {str(e)}", exc_info=True)
    
    async def _handle_pattern_remove_request(self, event_type: str, data: Dict[str, Any]):
        """Handle pattern removal requests"""
        try:
            pattern_uuid = data.get("pattern_uuid")
            if not pattern_uuid:
                logger.warning("Invalid pattern remove request: missing pattern UUID")
                return
            
            # Remove pattern
            if self._remove_pattern(pattern_uuid):
                self.event_bus.publish("intent.pattern.removed", {
                    "pattern_uuid": pattern_uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling pattern remove request: {str(e)}", exc_info=True)
    
    async def _handle_status_request(self, event_type: str, data: Dict[str, Any]):
        """Handle status requests"""
        status = {
            "patterns": len(self.patterns),
            "active_patterns": len([p for p in self.patterns.values() if p.is_active]),
            "thread_cache_size": len(self.thread_cache),
            "session_stats": self.session_stats,
            "nlp_available": self.nlp_processor is not None
        }
        self.event_bus.publish("intent.status.response", status)
    
    def _load_patterns(self):
        """Load intent patterns from file and defaults"""
        try:
            # Load default patterns
            self._load_default_patterns()
            
            # Load custom patterns from file if enabled
            if self.config.custom_patterns_enabled and Path(self.config.patterns_file).exists():
                self._load_custom_patterns()
            
            logger.info(f"Loaded {len(self.patterns)} intent patterns")
            
        except Exception as e:
            logger.error(f"Error loading patterns: {str(e)}", exc_info=True)
    
    def _load_default_patterns(self):
        """Load default intent patterns"""
        default_patterns = [
            {
                "name": "Looking for service",
                "pattern": r"(looking for|need|searching for|seeking)\s+(a\s+)?(service|tool|solution|software|platform)",
                "intent_type": "buying_signal",
                "confidence_threshold": 0.8,
                "weight": 1.0
            },
            {
                "name": "Recommendation request",
                "pattern": r"(recommend|suggest)\s+(a\s+)?(good|best|top)\s+(tool|service|software|platform)",
                "intent_type": "recommendation_request",
                "confidence_threshold": 0.7,
                "weight": 0.8
            },
            {
                "name": "Pricing inquiry",
                "pattern": r"(how much|what's the price|cost|pricing|fee|rate)",
                "intent_type": "pricing_inquiry",
                "confidence_threshold": 0.7,
                "weight": 0.9
            },
            {
                "name": "Demo request",
                "pattern": r"(demo|demonstration|show me|how does it work)",
                "intent_type": "demo_request",
                "confidence_threshold": 0.8,
                "weight": 1.0
            },
            {
                "name": "Trial request",
                "pattern": r"(trial|free trial|test drive|try out|test)",
                "intent_type": "trial_request",
                "confidence_threshold": 0.8,
                "weight": 0.9
            },
            {
                "name": "Partnership inquiry",
                "pattern": r"(partner|partnership|collaborate|work together)",
                "intent_type": "partnership_inquiry",
                "confidence_threshold": 0.7,
                "weight": 0.7
            }
        ]
        
        for pattern_data in default_patterns:
            pattern = IntentPattern(
                uuid=str(uuid.uuid4()),
                name=pattern_data["name"],
                pattern=pattern_data["pattern"],
                intent_type=pattern_data["intent_type"],
                confidence_threshold=pattern_data.get("confidence_threshold", 0.7),
                weight=pattern_data.get("weight", 1.0)
            )
            self.patterns[pattern.uuid] = pattern
    
    def _load_custom_patterns(self):
        """Load custom patterns from file"""
        try:
            with open(self.config.patterns_file, 'r') as f:
                patterns_data = json.load(f)
            
            for pattern_data in patterns_data:
                pattern = IntentPattern(
                    uuid=str(uuid.uuid4()),
                    name=pattern_data.get("name"),
                    pattern=pattern_data.get("pattern"),
                    intent_type=pattern_data.get("intent_type"),
                    confidence_threshold=pattern_data.get("confidence_threshold", 0.7),
                    weight=pattern_data.get("weight", 1.0),
                    is_active=pattern_data.get("is_active", True)
                )
                self.patterns[pattern.uuid] = pattern
                
        except Exception as e:
            logger.error(f"Error loading custom patterns: {str(e)}", exc_info=True)
    
    def _add_pattern(self, pattern_data: Dict[str, Any]) -> Optional[IntentPattern]:
        """Add a new intent pattern"""
        try:
            pattern = IntentPattern(
                uuid=str(uuid.uuid4()),
                name=pattern_data.get("name"),
                pattern=pattern_data.get("pattern"),
                intent_type=pattern_data.get("intent_type"),
                confidence_threshold=pattern_data.get("confidence_threshold", 0.7),
                weight=pattern_data.get("weight", 1.0),
                is_active=pattern_data.get("is_active", True)
            )
            
            # Validate pattern
            if not self._validate_pattern(pattern):
                logger.error(f"Invalid pattern: {pattern.name}")
                return None
            
            # Add to patterns
            self.patterns[pattern.uuid] = pattern
            
            logger.info(f"Added intent pattern: {pattern.name}")
            return pattern
            
        except Exception as e:
            logger.error(f"Error adding pattern: {str(e)}", exc_info=True)
            return None
    
    def _remove_pattern(self, pattern_uuid: str) -> bool:
        """Remove an intent pattern"""
        try:
            if pattern_uuid not in self.patterns:
                logger.warning(f"Pattern not found: {pattern_uuid}")
                return False
            
            pattern = self.patterns[pattern_uuid]
            del self.patterns[pattern_uuid]
            
            logger.info(f"Removed intent pattern: {pattern.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing pattern: {str(e)}", exc_info=True)
            return False
    
    def _validate_pattern(self, pattern: IntentPattern) -> bool:
        """Validate intent pattern"""
        try:
            # Check required fields
            if not all([pattern.name, pattern.pattern, pattern.intent_type]):
                return False
            
            # Validate intent type
            if pattern.intent_type not in self.config.intent_types:
                return False
            
            # Validate regex pattern
            re.compile(pattern.pattern)
            
            return True
            
        except Exception as e:
            logger.error(f"Pattern validation failed: {str(e)}", exc_info=True)
            return False
    
    async def detect_intents(self, thread_data: Dict[str, Any]) -> List[DetectedIntent]:
        """Detect intents in a conversation thread"""
        try:
            # Create or get thread from cache
            thread_id = thread_data.get("uuid")
            if thread_id in self.thread_cache:
                thread = self.thread_cache[thread_id]
                self.session_stats["cache_hits"] += 1
            else:
                thread = ConversationThread(
                    uuid=thread_id,
                    source_id=thread_data.get("source_id"),
                    source_type=thread_data.get("source_type"),
                    title=thread_data.get("title"),
                    url=thread_data.get("url"),
                    author=thread_data.get("author"),
                    created_at=thread_data.get("created_at", time.time()),
                    updated_at=thread_data.get("updated_at", time.time()),
                    messages=thread_data.get("messages", []),
                    metadata=thread_data.get("metadata", {})
                )
                self.thread_cache[thread_id] = thread
                self.session_stats["cache_misses"] += 1
            
            # Detect intents in each message
            intents = []
            for message in thread.messages:
                message_intents = await self._detect_message_intents(
                    message.get("text", ""),
                    message.get("author"),
                    thread
                )
                intents.extend(message_intents)
            
            # Update thread intents
            thread.intents = intents
            thread.updated_at = time.time()
            
            # Calculate cluster score
            thread.cluster_score = self._calculate_cluster_score(intents)
            
            # Update stats
            self.session_stats["threads_processed"] += 1
            self.session_stats["intents_detected"] += len(intents)
            self.session_stats["high_value_intents"] += len([i for i in intents if i.lead_score >= 0.8])
            
            return intents
            
        except Exception as e:
            logger.error(f"Error detecting intents: {str(e)}", exc_info=True)
            return []
    
    async def _detect_message_intents(self, text: str, author: str, thread: ConversationThread) -> List[DetectedIntent]:
        """Detect intents in a single message"""
        intents = []
        
        try:
            # Skip short messages
            if len(text) < self.config.min_message_length:
                return intents
            
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Get context messages
            context_messages = self._get_context_messages(thread, text)
            
            # Detect intents using patterns
            pattern_intents = self._detect_pattern_intents(text, author, thread)
            intents.extend(pattern_intents)
            
            # Detect intents using AI
            if self.nlp_processor:
                ai_intents = await self._detect_ai_intents(text, context_messages, author, thread)
                intents.extend(ai_intents)
                self.session_stats["ai_inferences"] += 1
            
            # Remove duplicates and sort by confidence
            intents = self._deduplicate_intents(intents)
            intents.sort(key=lambda x: x.confidence, reverse=True)
            
            # Calculate lead scores and urgency
            for intent in intents:
                intent.lead_score = self._calculate_lead_score(intent, thread)
                intent.urgency_level = self._determine_urgency_level(intent, thread)
            
            return intents
            
        except Exception as e:
            logger.error(f"Error detecting message intents: {str(e)}", exc_info=True)
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for intent detection"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}", exc_info=True)
            return text
    
    def _get_context_messages(self, thread: ConversationThread, current_text: str) -> List[str]:
        """Get context messages for AI processing"""
        try:
            # Find current message index
            current_index = -1
            for i, message in enumerate(thread.messages):
                if message.get("text") == current_text:
                    current_index = i
                    break
            
            if current_index == -1:
                return []
            
            # Get context window
            start_index = max(0, current_index - self.config.context_window_size)
            end_index = min(len(thread.messages), current_index + self.config.context_window_size)
            
            context_messages = []
            for i in range(start_index, end_index):
                if i != current_index:
                    context_messages.append(thread.messages[i].get("text", ""))
            
            return context_messages
            
        except Exception as e:
            logger.error(f"Error getting context messages: {str(e)}", exc_info=True)
            return []
    
    def _detect_pattern_intents(self, text: str, author: str, thread: ConversationThread) -> List[DetectedIntent]:
        """Detect intents using regex patterns"""
        intents = []
        
        try:
            # Check each pattern
            for pattern in self.patterns.values():
                if not pattern.is_active:
                    continue
                
                # Search for pattern
                matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Calculate confidence based on match quality and pattern weight
                    confidence = pattern.weight * (len(match.group()) / len(text))
                    
                    # Apply threshold
                    if confidence >= pattern.confidence_threshold:
                        intent = DetectedIntent(
                            uuid=str(uuid.uuid4()),
                            source_id=thread.uuid,
                            source_type=thread.source_type,
                            intent_type=pattern.intent_type,
                            confidence=confidence,
                            pattern_uuid=pattern.uuid,
                            text=match.group(),
                            context={
                                "author": author,
                                "match_start": match.start(),
                                "match_end": match.end()
                            }
                        )
                        intents.append(intent)
                        self.session_stats["patterns_used"] += 1
            
            return intents
            
        except Exception as e:
            logger.error(f"Error detecting pattern intents: {str(e)}", exc_info=True)
            return []
    
    async def _detect_ai_intents(self, text: str, context_messages: List[str], author: str, thread: ConversationThread) -> List[DetectedIntent]:
        """Detect intents using AI (TinyBERT)"""
        intents = []
        
        try:
            # Combine text with context
            full_text = text
            if context_messages:
                context_text = " ".join(context_messages[-3:])  # Use last 3 context messages
                full_text = f"{context_text} [SEP] {text}"
            
            # Process with NLP
            nlp_result = self.nlp_processor.process_text(
                full_text,
                categories=self.config.intent_types
            )
            
            # Extract intents from NLP result
            for intent_type in self.config.intent_types:
                if intent_type in nlp_result.categories:
                    confidence = nlp_result.categories[intent_type]
                    
                    # Apply threshold
                    if confidence >= self.config.confidence_threshold:
                        intent = DetectedIntent(
                            uuid=str(uuid.uuid4()),
                            source_id=thread.uuid,
                            source_type=thread.source_type,
                            intent_type=intent_type,
                            confidence=confidence,
                            text=text,
                            context={
                                "author": author,
                                "nlp_sentiment": nlp_result.sentiment,
                                "nlp_keywords": nlp_result.keywords
                            }
                        )
                        intents.append(intent)
            
            return intents
            
        except Exception as e:
            logger.error(f"Error detecting AI intents: {str(e)}", exc_info=True)
            return []
    
    def _deduplicate_intents(self, intents: List[DetectedIntent]) -> List[DetectedIntent]:
        """Remove duplicate intents, keeping the highest confidence"""
        seen = set()
        unique_intents = []
        
        for intent in intents:
            # Create key based on intent type and text
            key = (intent.intent_type, intent.text[:50])
            
            if key not in seen:
                seen.add(key)
                unique_intents.append(intent)
            else:
                # Replace with higher confidence if exists
                for i, existing in enumerate(unique_intents):
                    if (existing.intent_type, existing.text[:50]) == key:
                        if intent.confidence > existing.confidence:
                            unique_intents[i] = intent
                        break
        
        return unique_intents
    
    def _calculate_lead_score(self, intent: DetectedIntent, thread: ConversationThread) -> float:
        """Calculate lead score for detected intent"""
        try:
            score = 0.0
            
            # Base score from confidence
            score += intent.confidence * 0.5
            
            # Intent type weight
            intent_weights = {
                "buying_signal": 0.3,
                "demo_request": 0.25,
                "trial_request": 0.25,
                "pricing_inquiry": 0.2,
                "partnership_inquiry": 0.15,
                "recommendation_request": 0.1,
                "information_request": 0.05,
                "complaint": 0.0,
                "praise": 0.05,
                "comparison": 0.1
            }
            
            score += intent_weights.get(intent.intent_type, 0.0)
            
            # Context bonus
            if intent.context.get("nlp_sentiment", {}).get("positive", 0) > 0.5:
                score += 0.1
            
            # Thread activity bonus
            if len(thread.messages) > 5:
                score += 0.05
            
            # Author contribution bonus
            author_messages = [m for m in thread.messages if m.get("author") == intent.context.get("author")]
            if len(author_messages) > 2:
                score += 0.05
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating lead score: {str(e)}", exc_info=True)
            return 0.0
    
    def _determine_urgency_level(self, intent: DetectedIntent, thread: ConversationThread) -> str:
        """Determine urgency level of intent"""
        try:
            text = intent.text.lower()
            
            # Check for urgency keywords
            for level, keywords in self.config.urgency_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        return level
            
            # Default based on intent type
            if intent.intent_type in ["buying_signal", "demo_request", "trial_request"]:
                return "high"
            elif intent.intent_type in ["pricing_inquiry", "partnership_inquiry"]:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Error determining urgency level: {str(e)}", exc_info=True)
            return "medium"
    
    def _calculate_cluster_score(self, intents: List[DetectedIntent]) -> float:
        """Calculate cluster score for conversation thread"""
        try:
            if not intents:
                return 0.0
            
            # Calculate weighted average of intent scores
            total_weight = 0.0
            weighted_sum = 0.0
            
            for intent in intents:
                weight = 1.0  # Could be adjusted based on recency or other factors
                weighted_sum += intent.lead_score * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating cluster score: {str(e)}", exc_info=True)
            return 0.0
    
    async def detect_intent(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Public method to detect intent in text"""
        try:
            # Create mock thread for detection
            thread_data = {
                "uuid": str(uuid.uuid4()),
                "source_id": "manual",
                "source_type": "manual",
                "title": "Manual Intent Detection",
                "messages": [{"text": text, "author": "user"}],
                "metadata": context or {}
            }
            
            # Detect intents
            intents = await self.detect_intents(thread_data)
            
            if intents:
                return {
                    "intent_type": intents[0].intent_type,
                    "confidence": intents[0].confidence,
                    "lead_score": intents[0].lead_score,
                    "urgency_level": intents[0].urgency_level,
                    "all_intents": [
                        {
                            "intent_type": i.intent_type,
                            "confidence": i.confidence,
                            "lead_score": i.lead_score,
                            "urgency_level": i.urgency_level
                        }
                        for i in intents
                    ]
                }
            else:
                return {
                    "intent_type": "unknown",
                    "confidence": 0.0,
                    "lead_score": 0.0,
                    "urgency_level": "low",
                    "all_intents": []
                }
                
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}", exc_info=True)
            return {
                "intent_type": "error",
                "confidence": 0.0,
                "lead_score": 0.0,
                "urgency_level": "low",
                "all_intents": []
            }
    
    def add_intent_pattern(self, name: str, pattern: str, intent_type: str, **kwargs) -> Dict[str, Any]:
        """Public method to add an intent pattern"""
        pattern_data = {
            "name": name,
            "pattern": pattern,
            "intent_type": intent_type,
            **kwargs
        }
        
        pattern = self._add_pattern(pattern_data)
        if pattern:
            return {
                "pattern_uuid": pattern.uuid,
                "name": pattern.name,
                "status": "added"
            }
        else:
            return {
                "status": "failed",
                "error": "Invalid pattern configuration"
            }
    
    def remove_intent_pattern(self, pattern_uuid: str) -> Dict[str, Any]:
        """Public method to remove an intent pattern"""
        if self._remove_pattern(pattern_uuid):
            return {
                "pattern_uuid": pattern_uuid,
                "status": "removed"
            }
        else:
            return {
                "pattern_uuid": pattern_uuid,
                "status": "failed",
                "error": "Pattern not found"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get intent detector statistics"""
        return {
            "session_stats": self.session_stats,
            "patterns": len(self.patterns),
            "active_patterns": len([p for p in self.patterns.values() if p.is_active]),
            "thread_cache_size": len(self.thread_cache),
            "nlp_available": self.nlp_processor is not None
        }
