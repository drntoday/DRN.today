# modules/web_crawlers/retry_logic.py

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

import aiohttp
from playwright.async_api import Page, Error as PlaywrightError

from engine.storage import SecureStorage
from modules.web_crawlers.self_healing import DOMSelfHealingEngine


class RetryStrategy(Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"


class ErrorType(Enum):
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    CAPTCHA = "captcha"
    RATE_LIMIT = "rate_limit"
    BLOCKED = "blocked"
    SELECTOR_ERROR = "selector_error"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_on_errors: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.TIMEOUT,
        ErrorType.CONNECTION_ERROR,
        ErrorType.HTTP_ERROR,
        ErrorType.CAPTCHA,
        ErrorType.RATE_LIMIT,
        ErrorType.BLOCKED,
        ErrorType.SELECTOR_ERROR
    ])
    use_proxy_rotation: bool = True
    use_user_agent_rotation: bool = True
    enable_self_healing: bool = True
    adaptive_threshold: float = 0.7  # For adaptive strategy


@dataclass
class RetryResult:
    success: bool
    attempts: int
    final_error: Optional[Exception] = None
    used_proxies: List[str] = field(default_factory=list)
    used_user_agents: List[str] = field(default_factory=list)
    healing_attempts: int = 0
    total_time: float = 0.0
    strategy_used: str = ""


class RetryManager:
    def __init__(self, SecureStorage: SecureStorage, self_healing_engine: DOMSelfHealingEngine = None):
        self.SecureStorage = SecureStorage
        self.self_healing_engine = self_healing_engine
        self.logger = logging.getLogger("retry_manager")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load proxy and user agent pools
        self.proxy_pool = self._load_proxy_pool()
        self.user_agent_pool = self._load_user_agent_pool()
        
        # Track retry statistics
        self.retry_stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "error_distribution": {error_type.value: 0 for error_type in ErrorType}
        }

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS retry_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            error_type TEXT,
            error_message TEXT,
            attempt_number INTEGER,
            timestamp TEXT,
            success INTEGER,
            proxy_used TEXT,
            user_agent_used TEXT,
            healing_attempted INTEGER,
            strategy_used TEXT
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS proxy_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proxy_url TEXT UNIQUE,
            last_used TEXT,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS user_agent_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_agent TEXT UNIQUE,
            last_used TEXT,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1
        )
        """)

    def _load_proxy_pool(self) -> List[str]:
        """Load proxy pool from database"""
        proxies = []
        for row in self.SecureStorage.query("SELECT proxy_url FROM proxy_pool WHERE active = 1"):
            proxies.append(row['proxy_url'])
        return proxies

    def _load_user_agent_pool(self) -> List[str]:
        """Load user agent pool from database"""
        user_agents = []
        for row in self.SecureStorage.query("SELECT user_agent FROM user_agent_pool WHERE active = 1"):
            user_agents.append(row['user_agent'])
        return user_agents

    def add_proxy(self, proxy_url: str):
        """Add a new proxy to the pool"""
        try:
            self.SecureStorage.execute(
                "INSERT OR IGNORE INTO proxy_pool (proxy_url) VALUES (?)",
                (proxy_url,)
            )
            if proxy_url not in self.proxy_pool:
                self.proxy_pool.append(proxy_url)
            self.logger.info(f"Added proxy: {proxy_url}")
        except Exception as e:
            self.logger.error(f"Error adding proxy {proxy_url}: {str(e)}")

    def add_user_agent(self, user_agent: str):
        """Add a new user agent to the pool"""
        try:
            self.SecureStorage.execute(
                "INSERT OR IGNORE INTO user_agent_pool (user_agent) VALUES (?)",
                (user_agent,)
            )
            if user_agent not in self.user_agent_pool:
                self.user_agent_pool.append(user_agent)
            self.logger.info(f"Added user agent: {user_agent[:50]}...")
        except Exception as e:
            self.logger.error(f"Error adding user agent: {str(e)}")

    def get_next_proxy(self) -> Optional[str]:
        """Get the next proxy from the pool using round-robin with success weighting"""
        if not self.proxy_pool:
            return None
            
        # Get proxy stats
        proxy_stats = {}
        for proxy in self.proxy_pool:
            row = self.SecureStorage.query(
                "SELECT success_count, failure_count FROM proxy_pool WHERE proxy_url = ?",
                (proxy,)
            ).fetchone()
            
            if row:
                success = row['success_count']
                failure = row['failure_count']
                total = success + failure
                if total > 0:
                    success_rate = success / total
                else:
                    success_rate = 0.5  # Default for new proxies
                proxy_stats[proxy] = success_rate
        
        # Sort by success rate (descending)
        sorted_proxies = sorted(proxy_stats.items(), key=lambda x: x[1], reverse=True)
        
        # Use weighted random selection (top 3 proxies)
        top_proxies = [p[0] for p in sorted_proxies[:3]]
        weights = [p[1] for p in sorted_proxies[:3]]
        
        if not weights:
            return random.choice(self.proxy_pool)
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(top_proxies)
            
        normalized_weights = [w / total_weight for w in weights]
        
        # Select proxy
        selected_proxy = random.choices(top_proxies, weights=normalized_weights)[0]
        
        # Update last used time
        self.SecureStorage.execute(
            "UPDATE proxy_pool SET last_used = ? WHERE proxy_url = ?",
            (datetime.now().isoformat(), selected_proxy)
        )
        
        return selected_proxy

    def get_next_user_agent(self) -> Optional[str]:
        """Get the next user agent from the pool using round-robin with success weighting"""
        if not self.user_agent_pool:
            return None
            
        # Get user agent stats
        ua_stats = {}
        for ua in self.user_agent_pool:
            row = self.SecureStorage.query(
                "SELECT success_count, failure_count FROM user_agent_pool WHERE user_agent = ?",
                (ua,)
            ).fetchone()
            
            if row:
                success = row['success_count']
                failure = row['failure_count']
                total = success + failure
                if total > 0:
                    success_rate = success / total
                else:
                    success_rate = 0.5  # Default for new user agents
                ua_stats[ua] = success_rate
        
        # Sort by success rate (descending)
        sorted_uas = sorted(ua_stats.items(), key=lambda x: x[1], reverse=True)
        
        # Use weighted random selection (top 3 user agents)
        top_uas = [p[0] for p in sorted_uas[:3]]
        weights = [p[1] for p in sorted_uas[:3]]
        
        if not weights:
            return random.choice(self.user_agent_pool)
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(top_uas)
            
        normalized_weights = [w / total_weight for w in weights]
        
        # Select user agent
        selected_ua = random.choices(top_uas, weights=normalized_weights)[0]
        
        # Update last used time
        self.SecureStorage.execute(
            "UPDATE user_agent_pool SET last_used = ? WHERE user_agent = ?",
            (datetime.now().isoformat(), selected_ua)
        )
        
        return selected_ua

    def record_proxy_result(self, proxy_url: str, success: bool):
        """Record the result of using a proxy"""
        if success:
            self.SecureStorage.execute(
                "UPDATE proxy_pool SET success_count = success_count + 1 WHERE proxy_url = ?",
                (proxy_url,)
            )
        else:
            self.SecureStorage.execute(
                "UPDATE proxy_pool SET failure_count = failure_count + 1 WHERE proxy_url = ?",
                (proxy_url,)
            )
            
            # Deactivate proxy if failure rate is too high
            row = self.SecureStorage.query(
                "SELECT success_count, failure_count FROM proxy_pool WHERE proxy_url = ?",
                (proxy_url,)
            ).fetchone()
            
            if row:
                success = row['success_count']
                failure = row['failure_count']
                total = success + failure
                if total >= 5 and failure / total > 0.7:
                    self.SecureStorage.execute(
                        "UPDATE proxy_pool SET active = 0 WHERE proxy_url = ?",
                        (proxy_url,)
                    )
                    self.logger.warning(f"Deactivated proxy due to high failure rate: {proxy_url}")
                    if proxy_url in self.proxy_pool:
                        self.proxy_pool.remove(proxy_url)

    def record_user_agent_result(self, user_agent: str, success: bool):
        """Record the result of using a user agent"""
        if success:
            self.SecureStorage.execute(
                "UPDATE user_agent_pool SET success_count = success_count + 1 WHERE user_agent = ?",
                (user_agent,)
            )
        else:
            self.SecureStorage.execute(
                "UPDATE user_agent_pool SET failure_count = failure_count + 1 WHERE user_agent = ?",
                (user_agent,)
            )
            
            # Deactivate user agent if failure rate is too high
            row = self.SecureStorage.query(
                "SELECT success_count, failure_count FROM user_agent_pool WHERE user_agent = ?",
                (user_agent,)
            ).fetchone()
            
            if row:
                success = row['success_count']
                failure = row['failure_count']
                total = success + failure
                if total >= 5 and failure / total > 0.7:
                    self.SecureStorage.execute(
                        "UPDATE user_agent_pool SET active = 0 WHERE user_agent = ?",
                        (user_agent,)
                    )
                    self.logger.warning(f"Deactivated user agent due to high failure rate: {user_agent[:50]}...")
                    if user_agent in self.user_agent_pool:
                        self.user_agent_pool.remove(user_agent)

    def calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay based on retry strategy"""
        if config.strategy == RetryStrategy.FIXED:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.base_delay * attempt
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.base_delay * (2 ** (attempt - 1))
        elif config.strategy == RetryStrategy.FIBONACCI:
            delay = config.base_delay * self._fibonacci(attempt)
        elif config.strategy == RetryStrategy.ADAPTIVE:
            # Adaptive strategy adjusts based on recent success rate
            recent_attempts = self.SecureStorage.query(
                "SELECT success FROM retry_attempts ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()
            
            if recent_attempts:
                success_rate = sum(1 for row in recent_attempts if row['success']) / len(recent_attempts)
                if success_rate < config.adaptive_threshold:
                    # Increase delay when success rate is low
                    delay = config.base_delay * (2 ** (attempt - 1)) * 1.5
                else:
                    # Use standard exponential delay
                    delay = config.base_delay * (2 ** (attempt - 1))
            else:
                delay = config.base_delay * (2 ** (attempt - 1))
        else:
            delay = config.base_delay
        
        # Apply jitter if enabled
        if config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)  # 50% to 100% of calculated delay
        
        # Ensure delay is within bounds
        delay = max(config.base_delay, min(delay, config.max_delay))
        
        return delay

    def _fibonacci(self, n: int) -> int:
        """Calculate the nth Fibonacci number"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    def classify_error(self, error: Exception, url: str = "") -> ErrorType:
        """Classify an error into one of the defined error types"""
        error_msg = str(error).lower()
        
        if isinstance(error, asyncio.TimeoutError) or "timeout" in error_msg:
            return ErrorType.TIMEOUT
        elif isinstance(error, aiohttp.ClientConnectorError) or "connection" in error_msg:
            return ErrorType.CONNECTION_ERROR
        elif isinstance(error, aiohttp.ClientResponseError):
            status = getattr(error, 'status', 0)
            if status == 429:
                return ErrorType.RATE_LIMIT
            elif status in [403, 407]:
                return ErrorType.BLOCKED
            else:
                return ErrorType.HTTP_ERROR
        elif "captcha" in error_msg or "robot" in error_msg:
            return ErrorType.CAPTCHA
        elif "selector" in error_msg or "element" in error_msg:
            return ErrorType.SELECTOR_ERROR
        else:
            return ErrorType.UNKNOWN

    def record_attempt(self, url: str, error_type: ErrorType, error_message: str, 
                      attempt_number: int, success: bool, proxy_used: str = None,
                      user_agent_used: str = None, healing_attempted: bool = False,
                      strategy_used: str = ""):
        """Record a retry attempt in the database"""
        self.SecureStorage.execute(
            """
            INSERT INTO retry_attempts 
            (url, error_type, error_message, attempt_number, timestamp, success, 
             proxy_used, user_agent_used, healing_attempted, strategy_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                url,
                error_type.value,
                error_message,
                attempt_number,
                datetime.now().isoformat(),
                1 if success else 0,
                proxy_used,
                user_agent_used,
                1 if healing_attempted else 0,
                strategy_used
            )
        )
        
        # Update statistics
        self.retry_stats["total_attempts"] += 1
        self.retry_stats["error_distribution"][error_type.value] += 1
        
        if success and attempt_number > 1:
            self.retry_stats["successful_retries"] += 1
        elif not success:
            self.retry_stats["failed_retries"] += 1

    async def retry_http_request(self, url: str, method: str = "GET", 
                                headers: Dict = None, data: Any = None,
                                config: RetryConfig = None) -> Tuple[bool, Optional[aiohttp.ClientResponse], Optional[Exception]]:
        """Retry an HTTP request with smart retry logic"""
        if config is None:
            config = RetryConfig()
            
        start_time = time.time()
        used_proxies = []
        used_user_agents = []
        last_error = None
        
        for attempt in range(1, config.max_attempts + 1):
            proxy = None
            user_agent = None
            
            try:
                # Prepare request
                if headers is None:
                    headers = {}
                
                # Use proxy rotation if enabled
                if config.use_proxy_rotation and self.proxy_pool:
                    proxy = self.get_next_proxy()
                    if proxy:
                        used_proxies.append(proxy)
                        headers["Proxy-Authorization"] = f"Bearer {proxy}"  # Example auth
                
                # Use user agent rotation if enabled
                if config.use_user_agent_rotation and self.user_agent_pool:
                    user_agent = self.get_next_user_agent()
                    if user_agent:
                        used_user_agents.append(user_agent)
                        headers["User-Agent"] = user_agent
                
                # Make request
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method, url, headers=headers, data=data, 
                        proxy=proxy, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        # Check if status code indicates a retry
                        if response.status in config.retry_on_status:
                            error_type = ErrorType.HTTP_ERROR
                            error_msg = f"HTTP status {response.status}"
                            raise aiohttp.ClientResponseError(
                                request_info=None, 
                                history=None, 
                                status=response.status,
                                message=error_msg
                            )
                        
                        # Success
                        self.record_attempt(
                            url, ErrorType.UNKNOWN, "Success", attempt, True,
                            proxy, user_agent, False, config.strategy.value
                        )
                        
                        if proxy:
                            self.record_proxy_result(proxy, True)
                        if user_agent:
                            self.record_user_agent_result(user_agent, True)
                        
                        return True, response, None
                
            except Exception as e:
                last_error = e
                error_type = self.classify_error(e, url)
                error_msg = str(e)
                
                # Check if we should retry on this error type
                if error_type not in config.retry_on_errors:
                    self.record_attempt(
                        url, error_type, error_msg, attempt, False,
                        proxy, user_agent, False, config.strategy.value
                    )
                    
                    if proxy:
                        self.record_proxy_result(proxy, False)
                    if user_agent:
                        self.record_user_agent_result(user_agent, False)
                    
                    return False, None, e
                
                # Record the attempt
                self.record_attempt(
                    url, error_type, error_msg, attempt, False,
                    proxy, user_agent, False, config.strategy.value
                )
                
                if proxy:
                    self.record_proxy_result(proxy, False)
                if user_agent:
                    self.record_user_agent_result(user_agent, False)
                
                # Calculate delay
                delay = self.calculate_delay(attempt, config)
                self.logger.info(f"Attempt {attempt} failed for {url}. Retrying in {delay:.2f}s. Error: {error_msg}")
                
                # Wait before retrying
                await asyncio.sleep(delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        self.logger.error(f"All {config.max_attempts} attempts failed for {url}. Final error: {last_error}")
        
        return False, None, last_error

    async def retry_page_operation(self, page: Page, operation: Callable, 
                                  operation_args: Tuple = None, 
                                  operation_kwargs: Dict = None,
                                  config: RetryConfig = None,
                                  context: str = "") -> Tuple[bool, Any, Optional[Exception]]:
        """Retry a page operation with smart retry logic"""
        if config is None:
            config = RetryConfig()
            
        if operation_args is None:
            operation_args = ()
        if operation_kwargs is None:
            operation_kwargs = {}
            
        start_time = time.time()
        used_proxies = []
        used_user_agents = []
        healing_attempts = 0
        last_error = None
        
        for attempt in range(1, config.max_attempts + 1):
            proxy = None
            user_agent = None
            
            try:
                # Use proxy rotation if enabled
                if config.use_proxy_rotation and self.proxy_pool:
                    proxy = self.get_next_proxy()
                    if proxy:
                        used_proxies.append(proxy)
                        await page.set_extra_http_headers({"Proxy-Authorization": f"Bearer {proxy}"})
                
                # Use user agent rotation if enabled
                if config.use_user_agent_rotation and self.user_agent_pool:
                    user_agent = self.get_next_user_agent()
                    if user_agent:
                        used_user_agents.append(user_agent)
                        await page.set_extra_http_headers({"User-Agent": user_agent})
                
                # Try the operation
                result = await operation(*operation_args, **operation_kwargs)
                
                # Success
                self.record_attempt(
                    page.url, ErrorType.UNKNOWN, "Success", attempt, True,
                    proxy, user_agent, healing_attempts > 0, config.strategy.value
                )
                
                if proxy:
                    self.record_proxy_result(proxy, True)
                if user_agent:
                    self.record_user_agent_result(user_agent, True)
                
                return True, result, None
                
            except Exception as e:
                last_error = e
                error_type = self.classify_error(e, page.url)
                error_msg = str(e)
                
                # Check if we should retry on this error type
                if error_type not in config.retry_on_errors:
                    self.record_attempt(
                        page.url, error_type, error_msg, attempt, False,
                        proxy, user_agent, healing_attempts > 0, config.strategy.value
                    )
                    
                    if proxy:
                        self.record_proxy_result(proxy, False)
                    if user_agent:
                        self.record_user_agent_result(user_agent, False)
                    
                    return False, None, e
                
                # Try self-healing if enabled and it's a selector error
                if (config.enable_self_healing and 
                    error_type == ErrorType.SELECTOR_ERROR and 
                    self.self_healing_engine is not None and
                    context):
                    
                    try:
                        self.logger.info(f"Attempting self-healing for selector error on {page.url}")
                        
                        # Extract selector from error message (simplified)
                        selector = ""
                        if "selector" in error_msg.lower():
                            # Try to extract selector from error message
                            match = re.search(r"selector\s*[:=]\s*['\"]([^'\"]+)['\"]", error_msg)
                            if match:
                                selector = match.group(1)
                        
                        if selector:
                            # Attempt healing
                            healing_result = await self.self_healing_engine.heal_selector(
                                page, selector, 
                                self.self_healing_engine.SelectorType.CSS,  # Default to CSS
                                context
                            )
                            
                            if healing_result.confidence > 0.5:
                                healing_attempts += 1
                                self.logger.info(f"Self-healing successful with confidence {healing_result.confidence}")
                                
                                # Update operation to use healed selector
                                if "selector" in operation_kwargs:
                                    operation_kwargs["selector"] = healing_result.healed_selector
                                
                                # Continue to next attempt
                                pass
                            else:
                                self.logger.warning(f"Self-healing failed with confidence {healing_result.confidence}")
                    except Exception as healing_error:
                        self.logger.error(f"Error during self-healing: {str(healing_error)}")
                
                # Record the attempt
                self.record_attempt(
                    page.url, error_type, error_msg, attempt, False,
                    proxy, user_agent, healing_attempts > 0, config.strategy.value
                )
                
                if proxy:
                    self.record_proxy_result(proxy, False)
                if user_agent:
                    self.record_user_agent_result(user_agent, False)
                
                # Calculate delay
                delay = self.calculate_delay(attempt, config)
                self.logger.info(f"Attempt {attempt} failed for {page.url}. Retrying in {delay:.2f}s. Error: {error_msg}")
                
                # Wait before retrying
                await asyncio.sleep(delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        self.logger.error(f"All {config.max_attempts} attempts failed for {page.url}. Final error: {last_error}")
        
        return False, None, last_error

    def get_retry_stats(self, days: int = 7) -> Dict:
        """Get retry statistics for the specified time period"""
        since = datetime.now() - timedelta(days=days)
        
        # Get retry attempts from database
        attempts = self.SecureStorage.query(
            "SELECT * FROM retry_attempts WHERE timestamp >= ?",
            (since.isoformat(),)
        ).fetchall()
        
        if not attempts:
            return {
                "total_attempts": 0,
                "successful_retries": 0,
                "failed_retries": 0,
                "success_rate": 0.0,
                "error_distribution": {error_type.value: 0 for error_type in ErrorType},
                "proxy_usage": 0,
                "user_agent_usage": 0,
                "healing_usage": 0
            }
        
        # Calculate statistics
        total_attempts = len(attempts)
        successful_retries = sum(1 for a in attempts if a['success'] and a['attempt_number'] > 1)
        failed_retries = sum(1 for a in attempts if not a['success'])
        
        success_rate = successful_retries / total_attempts if total_attempts > 0 else 0.0
        
        # Error distribution
        error_distribution = {error_type.value: 0 for error_type in ErrorType}
        for attempt in attempts:
            error_type = attempt['error_type']
            if error_type in error_distribution:
                error_distribution[error_type] += 1
        
        # Proxy and user agent usage
        proxy_usage = sum(1 for a in attempts if a['proxy_used'])
        user_agent_usage = sum(1 for a in attempts if a['user_agent_used'])
        healing_usage = sum(1 for a in attempts if a['healing_attempted'])
        
        # Strategy distribution
        strategy_distribution = {}
        for attempt in attempts:
            strategy = attempt['strategy_used']
            if strategy:
                strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
        
        return {
            "total_attempts": total_attempts,
            "successful_retries": successful_retries,
            "failed_retries": failed_retries,
            "success_rate": success_rate,
            "error_distribution": error_distribution,
            "proxy_usage": proxy_usage,
            "user_agent_usage": user_agent_usage,
            "healing_usage": healing_usage,
            "strategy_distribution": strategy_distribution
        }

    def retry_decorator(self, config: RetryConfig = None):
        """Decorator to add retry functionality to any function"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if config is None:
                    retry_config = RetryConfig()
                else:
                    retry_config = config
                
                # Extract context if available
                context = kwargs.get("context", "")
                
                # Check if it's a page operation
                if args and hasattr(args[0], 'url'):  # Likely a Page object
                    page = args[0]
                    return await self.retry_page_operation(
                        page, func, args, kwargs, retry_config, context
                    )
                else:
                    # Generic retry logic
                    last_error = None
                    for attempt in range(1, retry_config.max_attempts + 1):
                        try:
                            result = await func(*args, **kwargs)
                            return result
                        except Exception as e:
                            last_error = e
                            error_type = self.classify_error(e)
                            
                            if error_type not in retry_config.retry_on_errors:
                                raise
                            
                            delay = self.calculate_delay(attempt, retry_config)
                            self.logger.info(f"Attempt {attempt} failed for {func.__name__}. Retrying in {delay:.2f}s. Error: {str(e)}")
                            await asyncio.sleep(delay)
                    
                    # All attempts failed
                    raise last_error
            
            return wrapper
        return decorator
