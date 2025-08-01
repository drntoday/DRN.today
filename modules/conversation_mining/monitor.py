# modules/conversation_mining/monitor.py

import asyncio
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from engine.storage import Storage
from ai.nlp import NLPProcessor
from modules.conversation_mining.intent_detector import IntentDetector
from modules.conversation_mining.classifier import ConversationClassifier


class Platform(Enum):
    REDDIT = "reddit"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    FORUMS = "forums"
    COMMENTS = "comments"


@dataclass
class Conversation:
    id: str
    platform: Platform
    author: str
    content: str
    timestamp: datetime
    url: str
    metadata: Dict = field(default_factory=dict)
    processed: bool = False
    intent_detected: Optional[str] = None
    intent_confidence: float = 0.0
    classification: Optional[str] = None
    classification_confidence: float = 0.0
    keywords_found: List[str] = field(default_factory=list)


class ConversationMonitor:
    def __init__(self, storage: Storage, nlp_processor: NLPProcessor, 
                 intent_detector: IntentDetector, classifier: ConversationClassifier):
        self.storage = storage
        self.nlp = nlp_processor
        self.intent_detector = intent_detector
        self.classifier = classifier
        
        self.monitored_platforms: Set[Platform] = set()
        self.monitoring_active = False
        self.monitor_threads: Dict[Platform, threading.Thread] = {}
        self.conversation_queue = Queue()
        self.processing_thread = None
        
        self._initialize_tables()
        self._load_configuration()
        
        # Platform-specific configurations
        self.platform_configs = {
            Platform.REDDIT: {
                "subreddits": ["SaaS", "startups", "Entrepreneur", "technology"],
                "keywords": ["looking for", "recommend", "need", "help", "tool", "service"],
                "min_upvotes": 5,
                "max_age_days": 30
            },
            Platform.DISCORD: {
                "servers": ["Startup Discord", "SaaS Community", "Tech Entrepreneurs"],
                "channels": ["general", "help", "recommendations"],
                "keywords": ["looking for", "recommend", "need", "help", "tool", "service"],
                "min_reactions": 3
            },
            Platform.TELEGRAM: {
                "groups": ["Startup Chat", "SaaS Founders", "Tech Entrepreneurs"],
                "keywords": ["looking for", "recommend", "need", "help", "tool", "service"],
                "min_views": 100
            },
            Platform.FORUMS: {
                "sites": ["Indie Hackers", "Product Hunt", "Hacker News"],
                "keywords": ["looking for", "recommend", "need", "help", "tool", "service"],
                "min_replies": 5
            },
            Platform.COMMENTS: {
                "sites": ["TechCrunch", "VentureBeat", "Product Hunt"],
                "keywords": ["looking for", "recommend", "need", "help", "tool", "service"],
                "min_upvotes": 3
            }
        }
        
        # Rate limiting configuration
        self.rate_limits = {
            Platform.REDDIT: 60,  # requests per minute
            Platform.DISCORD: 120,
            Platform.TELEGRAM: 100,
            Platform.FORUMS: 90,
            Platform.COMMENTS: 150
        }
        
        self.last_request_time = {platform: 0 for platform in Platform}
        
        # Set up logging
        self.logger = logging.getLogger("conversation_monitor")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS monitored_conversations (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            author TEXT,
            content TEXT,
            timestamp TEXT,
            url TEXT,
            metadata TEXT,
            processed INTEGER DEFAULT 0,
            intent_detected TEXT,
            intent_confidence REAL,
            classification TEXT,
            classification_confidence REAL,
            keywords_found TEXT,
            created_at TEXT
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_config (
            platform TEXT PRIMARY KEY,
            config TEXT,
            active INTEGER DEFAULT 0,
            last_check TEXT
        )
        """)

    def _load_configuration(self):
        """Load monitoring configuration from storage"""
        for row in self.storage.query("SELECT * FROM monitoring_config"):
            platform = Platform(row['platform'])
            config = json.loads(row['config'])
            active = bool(row['active'])
            
            if active:
                self.monitored_platforms.add(platform)
                self.platform_configs[platform].update(config)

    def save_configuration(self):
        """Save current monitoring configuration to storage"""
        for platform in Platform:
            self.storage.execute(
                "INSERT OR REPLACE INTO monitoring_config VALUES (?, ?, ?, ?)",
                (
                    platform.value,
                    json.dumps(self.platform_configs[platform]),
                    1 if platform in self.monitored_platforms else 0,
                    datetime.now().isoformat()
                )
            )

    def start_monitoring(self, platforms: List[Platform] = None):
        """Start monitoring conversations on specified platforms"""
        if platforms is None:
            platforms = list(Platform)
            
        for platform in platforms:
            if platform not in self.monitored_platforms:
                self.monitored_platforms.add(platform)
                
                # Start monitoring thread for this platform
                thread = threading.Thread(
                    target=self._monitor_platform,
                    args=(platform,),
                    daemon=True
                )
                self.monitor_threads[platform] = thread
                thread.start()
                self.logger.info(f"Started monitoring {platform.value}")
        
        # Start processing thread if not already running
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(
                target=self._process_conversations,
                daemon=True
            )
            self.processing_thread.start()
            
        self.monitoring_active = True
        self.save_configuration()
        self.logger.info("Conversation monitoring started")

    def stop_monitoring(self, platforms: List[Platform] = None):
        """Stop monitoring conversations on specified platforms"""
        if platforms is None:
            platforms = list(self.monitored_platforms)
            
        for platform in platforms:
            if platform in self.monitored_platforms:
                self.monitored_platforms.remove(platform)
                self.logger.info(f"Stopped monitoring {platform.value}")
        
        # Stop processing thread if no platforms are being monitored
        if not self.monitored_platforms and self.processing_thread and self.processing_thread.is_alive():
            # The thread will exit naturally when the queue is empty and monitoring is stopped
            self.monitoring_active = False
            self.logger.info("Conversation monitoring stopped")
        
        self.save_configuration()

    def _monitor_platform(self, platform: Platform):
        """Monitor a specific platform for conversations"""
        config = self.platform_configs[platform]
        
        while platform in self.monitored_platforms:
            try:
                if platform == Platform.REDDIT:
                    self._monitor_reddit(config)
                elif platform == Platform.DISCORD:
                    self._monitor_discord(config)
                elif platform == Platform.TELEGRAM:
                    self._monitor_telegram(config)
                elif platform == Platform.FORUMS:
                    self._monitor_forums(config)
                elif platform == Platform.COMMENTS:
                    self._monitor_comments(config)
                    
                # Sleep to avoid rate limiting
                time.sleep(60 / self.rate_limits[platform])
            except Exception as e:
                self.logger.error(f"Error monitoring {platform.value}: {str(e)}")
                time.sleep(30)  # Wait before retrying

    def _monitor_reddit(self, config: Dict):
        """Monitor Reddit for conversations"""
        subreddits = config.get("subreddits", [])
        keywords = config.get("keywords", [])
        min_upvotes = config.get("min_upvotes", 5)
        max_age_days = config.get("max_age_days", 30)
        
        for subreddit in subreddits:
            try:
                # Apply rate limiting
                self._apply_rate_limit(Platform.REDDIT)
                
                # Fetch recent posts
                url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=100"
                response = requests.get(url, headers={"User-Agent": "DRN.today/1.0"})
                data = response.json()
                
                for post in data["data"]["children"]:
                    post_data = post["data"]
                    
                    # Check if post meets criteria
                    if (post_data["score"] < min_upvotes or 
                        (datetime.now() - datetime.fromtimestamp(post_data["created_utc"])).days > max_age_days):
                        continue
                    
                    # Check for keywords in title
                    title_lower = post_data["title"].lower()
                    if any(keyword.lower() in title_lower for keyword in keywords):
                        # Create conversation object
                        conversation = Conversation(
                            id=f"reddit_{post_data['id']}",
                            platform=Platform.REDDIT,
                            author=post_data["author"],
                            content=post_data["title"],
                            timestamp=datetime.fromtimestamp(post_data["created_utc"]),
                            url=f"https://www.reddit.com{post_data['permalink']}",
                            metadata={
                                "subreddit": subreddit,
                                "upvotes": post_data["score"],
                                "num_comments": post_data["num_comments"]
                            }
                        )
                        
                        # Add to queue for processing
                        self.conversation_queue.put(conversation)
                        
                        # Also fetch comments if post looks promising
                        if post_data["num_comments"] > 10:
                            self._fetch_reddit_comments(post_data["permalink"], keywords)
                            
            except Exception as e:
                self.logger.error(f"Error monitoring Reddit subreddit {subreddit}: {str(e)}")

    def _fetch_reddit_comments(self, post_url: str, keywords: List[str]):
        """Fetch comments from a Reddit post"""
        try:
            # Apply rate limiting
            self._apply_rate_limit(Platform.REDDIT)
            
            url = f"{post_url}.json"
            response = requests.get(url, headers={"User-Agent": "DRN.today/1.0"})
            data = response.json()
            
            # The second element in the response array contains the comments
            if len(data) > 1 and "data" in data[1] and "children" in data[1]["data"]:
                for comment in data[1]["data"]["children"]:
                    comment_data = comment["data"]
                    
                    # Skip deleted comments
                    if comment_data["author"] == "[deleted]":
                        continue
                    
                    # Check for keywords in comment
                    comment_lower = comment_data["body"].lower()
                    if any(keyword.lower() in comment_lower for keyword in keywords):
                        # Create conversation object
                        conversation = Conversation(
                            id=f"reddit_comment_{comment_data['id']}",
                            platform=Platform.REDDIT,
                            author=comment_data["author"],
                            content=comment_data["body"],
                            timestamp=datetime.fromtimestamp(comment_data["created_utc"]),
                            url=f"https://www.reddit.com{comment_data['permalink']}",
                            metadata={
                                "parent_post": post_url,
                                "upvotes": comment_data["score"],
                                "is_comment": True
                            }
                        )
                        
                        # Add to queue for processing
                        self.conversation_queue.put(conversation)
                        
        except Exception as e:
            self.logger.error(f"Error fetching Reddit comments from {post_url}: {str(e)}")

    def _monitor_discord(self, config: Dict):
        """Monitor Discord for conversations"""
        # Note: Discord monitoring would typically require a bot token
        # This is a simplified implementation that would need to be expanded
        servers = config.get("servers", [])
        channels = config.get("channels", [])
        keywords = config.get("keywords", [])
        
        # In a real implementation, we would use Discord.py or similar
        # For now, we'll simulate with a placeholder
        self.logger.info("Discord monitoring requires bot token - placeholder implementation")
        
        # Simulate finding a conversation
        if servers and channels and keywords:
            conversation = Conversation(
                id="discord_simulated",
                platform=Platform.DISCORD,
                author="simulated_user",
                content="Looking for a good CRM tool for my startup",
                timestamp=datetime.now(),
                url="https://discord.com/channels/simulated/123456",
                metadata={
                    "server": servers[0],
                    "channel": channels[0]
                }
            )
            self.conversation_queue.put(conversation)

    def _monitor_telegram(self, config: Dict):
        """Monitor Telegram for conversations"""
        # Note: Telegram monitoring would typically require API access
        # This is a simplified implementation that would need to be expanded
        groups = config.get("groups", [])
        keywords = config.get("keywords", [])
        
        # In a real implementation, we would use Telethon or similar
        # For now, we'll simulate with a placeholder
        self.logger.info("Telegram monitoring requires API access - placeholder implementation")
        
        # Simulate finding a conversation
        if groups and keywords:
            conversation = Conversation(
                id="telegram_simulated",
                platform=Platform.TELEGRAM,
                author="simulated_user",
                content="Can anyone recommend a good email marketing tool?",
                timestamp=datetime.now(),
                url="https://t.me/simulated_group/123456",
                metadata={
                    "group": groups[0]
                }
            )
            self.conversation_queue.put(conversation)

    def _monitor_forums(self, config: Dict):
        """Monitor forums for conversations"""
        sites = config.get("sites", [])
        keywords = config.get("keywords", [])
        min_replies = config.get("min_replies", 5)
        
        for site in sites:
            try:
                # Apply rate limiting
                self._apply_rate_limit(Platform.FORUMS)
                
                # This would be site-specific
                if site == "Indie Hackers":
                    self._scrape_indiehackers(keywords, min_replies)
                elif site == "Product Hunt":
                    self._scrape_producthunt(keywords, min_replies)
                elif site == "Hacker News":
                    self._scrape_hackernews(keywords, min_replies)
                    
            except Exception as e:
                self.logger.error(f"Error monitoring forum {site}: {str(e)}")

    def _scrape_indiehackers(self, keywords: List[str], min_replies: int):
        """Scrape Indie Hackers for conversations"""
        url = "https://www.indiehackers.com/posts"
        response = requests.get(url, headers={"User-Agent": "DRN.today/1.0"})
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for post in soup.select('.post-item'):
            try:
                title_element = post.select_one('.post-item__title a')
                if not title_element:
                    continue
                    
                title = title_element.get_text(strip=True)
                post_url = "https://www.indiehackers.com" + title_element['href']
                
                # Check for keywords
                title_lower = title.lower()
                if not any(keyword.lower() in title_lower for keyword in keywords):
                    continue
                
                # Get post details
                author = post.select_one('.post-item__author-name').get_text(strip=True)
                replies_text = post.select_one('.post-item__replies').get_text(strip=True)
                replies = int(replies_text.split()[0]) if replies_text else 0
                
                if replies < min_replies:
                    continue
                
                # Create conversation object
                conversation = Conversation(
                    id=f"ih_{post_url.split('/')[-1]}",
                    platform=Platform.FORUMS,
                    author=author,
                    content=title,
                    timestamp=datetime.now(),  # Would parse actual timestamp
                    url=post_url,
                    metadata={
                        "site": "Indie Hackers",
                        "replies": replies
                    }
                )
                
                # Add to queue for processing
                self.conversation_queue.put(conversation)
                
            except Exception as e:
                self.logger.error(f"Error processing Indie Hackers post: {str(e)}")

    def _scrape_producthunt(self, keywords: List[str], min_replies: int):
        """Scrape Product Hunt for conversations"""
        # Similar implementation to Indie Hackers
        # Placeholder for brevity
        pass

    def _scrape_hackernews(self, keywords: List[str], min_replies: int):
        """Scrape Hacker News for conversations"""
        # Similar implementation to Indie Hackers
        # Placeholder for brevity
        pass

    def _monitor_comments(self, config: Dict):
        """Monitor comment sections for conversations"""
        sites = config.get("sites", [])
        keywords = config.get("keywords", [])
        min_upvotes = config.get("min_upvotes", 3)
        
        for site in sites:
            try:
                # Apply rate limiting
                self._apply_rate_limit(Platform.COMMENTS)
                
                # This would be site-specific
                if site == "TechCrunch":
                    self._scrape_techcrunch_comments(keywords, min_upvotes)
                elif site == "VentureBeat":
                    self._scrape_venturebeat_comments(keywords, min_upvotes)
                elif site == "Product Hunt":
                    self._scrape_producthunt_comments(keywords, min_upvotes)
                    
            except Exception as e:
                self.logger.error(f"Error monitoring comments on {site}: {str(e)}")

    def _scrape_techcrunch_comments(self, keywords: List[str], min_upvotes: int):
        """Scrape TechCrunch comments for conversations"""
        # Implementation would use Playwright to handle JavaScript-rendered comments
        # Placeholder for brevity
        pass

    def _scrape_venturebeat_comments(self, keywords: List[str], min_upvotes: int):
        """Scrape VentureBeat comments for conversations"""
        # Implementation would use Playwright to handle JavaScript-rendered comments
        # Placeholder for brevity
        pass

    def _scrape_producthunt_comments(self, keywords: List[str], min_upvotes: int):
        """Scrape Product Hunt comments for conversations"""
        # Implementation would use Playwright to handle JavaScript-rendered comments
        # Placeholder for brevity
        pass

    def _apply_rate_limit(self, platform: Platform):
        """Apply rate limiting for a platform"""
        now = time.time()
        elapsed = now - self.last_request_time[platform]
        min_interval = 60 / self.rate_limits[platform]
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            
        self.last_request_time[platform] = time.time()

    def _process_conversations(self):
        """Process conversations from the queue"""
        while self.monitoring_active or not self.conversation_queue.empty():
            try:
                # Get conversation from queue with timeout
                conversation = self.conversation_queue.get(timeout=5)
                
                # Process conversation
                self._process_single_conversation(conversation)
                
                # Mark task as done
                self.conversation_queue.task_done()
                
            except Empty:
                # Queue is empty, continue loop
                continue
            except Exception as e:
                self.logger.error(f"Error processing conversation: {str(e)}")

    def _process_single_conversation(self, conversation: Conversation):
        """Process a single conversation"""
        try:
            # Detect intent
            intent_result = self.intent_detector.detect_intent(conversation.content)
            conversation.intent_detected = intent_result["intent"]
            conversation.intent_confidence = intent_result["confidence"]
            conversation.keywords_found = intent_result["keywords"]
            
            # Classify conversation
            classification_result = self.classifier.classify(conversation)
            conversation.classification = classification_result["classification"]
            conversation.classification_confidence = classification_result["confidence"]
            
            # Mark as processed
            conversation.processed = True
            
            # Save to storage
            self._save_conversation(conversation)
            
            # Log high-intent conversations
            if (conversation.intent_detected == "buying_signal" and 
                conversation.intent_confidence > 0.7):
                self.logger.info(
                    f"High-intent conversation found: {conversation.platform.value} - {conversation.url}"
                )
                
        except Exception as e:
            self.logger.error(f"Error processing conversation {conversation.id}: {str(e)}")
            # Still save the conversation even if processing failed
            self._save_conversation(conversation)

    def _save_conversation(self, conversation: Conversation):
        """Save conversation to storage"""
        self.storage.execute(
            """
            INSERT OR REPLACE INTO monitored_conversations 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation.id,
                conversation.platform.value,
                conversation.author,
                conversation.content,
                conversation.timestamp.isoformat(),
                conversation.url,
                json.dumps(conversation.metadata),
                1 if conversation.processed else 0,
                conversation.intent_detected,
                conversation.intent_confidence,
                conversation.classification,
                conversation.classification_confidence,
                json.dumps(conversation.keywords_found),
                datetime.now().isoformat()
            )
        )

    def get_recent_conversations(self, platform: Platform = None, 
                               hours: int = 24, 
                               min_intent_confidence: float = 0.5) -> List[Conversation]:
        """Get recent conversations with filtering options"""
        since = datetime.now() - timedelta(hours=hours)
        
        query = """
        SELECT * FROM monitored_conversations 
        WHERE timestamp >= ? AND processed = 1
        """
        params = [since.isoformat()]
        
        if platform:
            query += " AND platform = ?"
            params.append(platform.value)
            
        if min_intent_confidence > 0:
            query += " AND intent_confidence >= ?"
            params.append(min_intent_confidence)
            
        query += " ORDER BY timestamp DESC LIMIT 100"
        
        conversations = []
        for row in self.storage.query(query, params):
            conversation = Conversation(
                id=row['id'],
                platform=Platform(row['platform']),
                author=row['author'],
                content=row['content'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                url=row['url'],
                metadata=json.loads(row['metadata']),
                processed=bool(row['processed']),
                intent_detected=row['intent_detected'],
                intent_confidence=row['intent_confidence'],
                classification=row['classification'],
                classification_confidence=row['classification_confidence'],
                keywords_found=json.loads(row['keywords_found'])
            )
            conversations.append(conversation)
            
        return conversations

    def get_high_intent_conversations(self, platform: Platform = None, 
                                   hours: int = 24) -> List[Conversation]:
        """Get high-intent conversations (buying signals with high confidence)"""
        return self.get_recent_conversations(
            platform=platform,
            hours=hours,
            min_intent_confidence=0.7
        )

    def get_conversation_stats(self, platform: Platform = None, 
                             days: int = 7) -> Dict:
        """Get statistics about monitored conversations"""
        since = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            platform,
            COUNT(*) as total,
            SUM(CASE WHEN intent_detected = 'buying_signal' THEN 1 ELSE 0 END) as buying_signals,
            AVG(intent_confidence) as avg_intent_confidence,
            AVG(classification_confidence) as avg_classification_confidence
        FROM monitored_conversations 
        WHERE timestamp >= ? AND processed = 1
        """
        params = [since.isoformat()]
        
        if platform:
            query += " AND platform = ?"
            params.append(platform.value)
            
        query += " GROUP BY platform"
        
        stats = {}
        for row in self.storage.query(query, params):
            stats[row['platform']] = {
                "total_conversations": row['total'],
                "buying_signals": row['buying_signals'],
                "avg_intent_confidence": row['avg_intent_confidence'],
                "avg_classification_confidence": row['avg_classification_confidence']
            }
            
        return stats

    def update_platform_config(self, platform: Platform, config: Dict):
        """Update configuration for a platform"""
        self.platform_configs[platform].update(config)
        self.save_configuration()

    def get_platform_config(self, platform: Platform) -> Dict:
        """Get configuration for a platform"""
        return self.platform_configs.get(platform, {})