import threading
import queue
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple
from collections import deque
from enum import Enum, auto

# Initialize event system logger
logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Event priority levels"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

@dataclass
class Event:
    """Event data structure with metadata"""
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class EventHandler:
    """Base class for event handlers"""
    
    def __init__(self, event_type: str, handler: Callable, 
                 priority_filter: Optional[EventPriority] = None,
                 source_filter: Optional[str] = None):
        self.event_type = event_type
        self.handler = handler
        self.priority_filter = priority_filter
        self.source_filter = source_filter
        self.execution_count = 0
        self.error_count = 0
        self.last_execution = 0
        self.average_execution_time = 0.0
        
    def matches(self, event: Event) -> bool:
        """Check if handler should process this event"""
        if self.event_type != event.event_type:
            return False
            
        if self.priority_filter and event.priority != self.priority_filter:
            return False
            
        if self.source_filter and event.source != self.source_filter:
            return False
            
        return True
    
    def execute(self, event: Event) -> bool:
        """Execute the event handler with error handling and metrics"""
        start_time = time.time()
        
        try:
            self.handler(event.event_type, event.data)
            execution_time = time.time() - start_time
            
            # Update metrics
            self.execution_count += 1
            self.last_execution = time.time()
            self.average_execution_time = (
                (self.average_execution_time * (self.execution_count - 1) + execution_time) 
                / self.execution_count
            )
            
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Event handler error for {event.event_type}: {str(e)}", 
                        exc_info=True)
            return False

class EventMonitor:
    """Production-grade event system monitoring"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.event_history = deque(maxlen=max_history)
        self.error_history = deque(maxlen=100)
        self.metrics = {
            "total_events": 0,
            "processed_events": 0,
            "failed_events": 0,
            "average_processing_time": 0.0,
            "queue_size": 0,
            "active_handlers": 0,
            "last_error": None
        }
        self.lock = threading.Lock()
        
    def record_event(self, event: Event, processing_time: float, success: bool):
        """Record event processing metrics"""
        with self.lock:
            self.metrics["total_events"] += 1
            
            if success:
                self.metrics["processed_events"] += 1
            else:
                self.metrics["failed_events"] += 1
                
            # Update average processing time
            current_avg = self.metrics["average_processing_time"]
            total_processed = self.metrics["processed_events"] + self.metrics["failed_events"]
            self.metrics["average_processing_time"] = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )
            
            # Add to history
            self.event_history.append({
                "event_type": event.event_type,
                "source": event.source,
                "priority": event.priority.name,
                "timestamp": event.timestamp,
                "processing_time": processing_time,
                "success": success
            })
    
    def record_error(self, event: Event, error: str):
        """Record event processing errors"""
        with self.lock:
            self.metrics["last_error"] = {
                "timestamp": time.time(),
                "event_type": event.event_type,
                "error": error
            }
            self.error_history.append(self.metrics["last_error"])
    
    def update_queue_size(self, size: int):
        """Update current queue size metric"""
        with self.lock:
            self.metrics["queue_size"] = size
    
    def update_handler_count(self, count: int):
        """Update active handler count"""
        with self.lock:
            self.metrics["active_handlers"] = count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        with self.lock:
            return self.metrics.copy()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        metrics = self.get_metrics()
        
        # Calculate health score (0-100)
        health_score = 100
        
        # Deduct for errors
        if metrics["total_events"] > 0:
            error_rate = metrics["failed_events"] / metrics["total_events"]
            health_score -= min(50, error_rate * 100)
        
        # Deduct for queue size
        if metrics["queue_size"] > 1000:
            health_score -= min(30, (metrics["queue_size"] - 1000) / 100)
        
        # Deduct for slow processing
        if metrics["average_processing_time"] > 1.0:
            health_score -= min(20, metrics["average_processing_time"] * 10)
        
        return {
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "unhealthy",
            "health_score": max(0, health_score),
            "metrics": metrics,
            "recent_errors": list(self.error_history)[-5:] if self.error_history else []
        }

class EventBus:
    """Production-ready event bus for inter-module communication"""
    
    def __init__(self, max_queue_size: int = 10000, max_workers: int = 4):
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        
        # Priority queues for different event types
        self.queues = {
            EventPriority.CRITICAL: queue.PriorityQueue(maxsize=max_queue_size),
            EventPriority.HIGH: queue.PriorityQueue(maxsize=max_queue_size),
            EventPriority.NORMAL: queue.PriorityQueue(maxsize=max_queue_size),
            EventPriority.LOW: queue.PriorityQueue(maxsize=max_queue_size)
        }
        
        # Event handlers registry
        self.handlers: Dict[str, List[EventHandler]] = {}
        
        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False
        
        # Monitoring
        self.monitor = EventMonitor()
        
        # Threading locks
        self.handlers_lock = threading.RLock()
        self.shutdown_lock = threading.Lock()
        
        # Event processing statistics
        self.stats = {
            "published": 0,
            "processed": 0,
            "failed": 0,
            "retried": 0
        }
        
    def start(self) -> bool:
        """Start the event system with worker threads"""
        try:
            logger.info("Starting event system...")
            
            with self.shutdown_lock:
                if self.running:
                    logger.warning("Event system already running")
                    return True
                
                self.running = True
                
                # Start worker threads
                for i in range(self.max_workers):
                    worker = threading.Thread(
                        target=self._worker_loop,
                        name=f"EventWorker-{i}",
                        daemon=True
                    )
                    worker.start()
                    self.workers.append(worker)
                
                logger.info(f"Event system started with {self.max_workers} workers")
                return True
                
        except Exception as e:
            logger.critical(f"Failed to start event system: {str(e)}", exc_info=True)
            return False
    
    def stop(self) -> bool:
        """Stop the event system gracefully"""
        try:
            logger.info("Stopping event system...")
            
            with self.shutdown_lock:
                if not self.running:
                    logger.warning("Event system not running")
                    return True
                
                self.running = False
                
                # Signal workers to stop
                for q in self.queues.values():
                    q.put(None)  # Sentinel value
                
                # Wait for workers to finish
                for worker in self.workers:
                    worker.join(timeout=5)
                    if worker.is_alive():
                        logger.warning(f"Worker thread {worker.name} did not shutdown gracefully")
                
                self.workers.clear()
                logger.info("Event system stopped")
                return True
                
        except Exception as e:
            logger.critical(f"Failed to stop event system: {str(e)}", exc_info=True)
            return False
    
    def publish(self, event_type: str, data: Dict[str, Any] = None, 
                source: str = "", priority: EventPriority = EventPriority.NORMAL,
                correlation_id: Optional[str] = None) -> bool:
        """Publish an event to the event bus"""
        try:
            if not self.running:
                logger.warning("Event system not running, cannot publish event")
                return False
            
            event = Event(
                event_type=event_type,
                data=data or {},
                source=source,
                priority=priority,
                correlation_id=correlation_id
            )
            
            # Add to appropriate queue
            queue_obj = self.queues[priority]
            
            try:
                queue_obj.put((event.timestamp, event), block=False)
                self.stats["published"] += 1
                
                # Update monitor
                self.monitor.update_queue_size(sum(q.qsize() for q in self.queues.values()))
                
                logger.debug(f"Published event: {event_type} from {source}")
                return True
                
            except queue.Full:
                logger.error(f"Event queue full for priority {priority.name}, dropping event")
                self.monitor.record_error(event, "Queue full")
                return False
                
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {str(e)}", exc_info=True)
            return False
    
    def subscribe(self, event_type: str, handler: Callable, 
                  priority_filter: Optional[EventPriority] = None,
                  source_filter: Optional[str] = None) -> bool:
        """Subscribe to events with optional filters"""
        try:
            with self.handlers_lock:
                if event_type not in self.handlers:
                    self.handlers[event_type] = []
                
                event_handler = EventHandler(
                    event_type=event_type,
                    handler=handler,
                    priority_filter=priority_filter,
                    source_filter=source_filter
                )
                
                self.handlers[event_type].append(event_handler)
                self.monitor.update_handler_count(
                    sum(len(handlers) for handlers in self.handlers.values())
                )
                
                logger.debug(f"Subscribed to event: {event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to subscribe to {event_type}: {str(e)}", exc_info=True)
            return False
    
    def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """Unsubscribe from events"""
        try:
            with self.handlers_lock:
                if event_type not in self.handlers:
                    return True
                
                # Find and remove handler
                self.handlers[event_type] = [
                    h for h in self.handlers[event_type] 
                    if h.handler != handler
                ]
                
                # Clean up empty lists
                if not self.handlers[event_type]:
                    del self.handlers[event_type]
                
                self.monitor.update_handler_count(
                    sum(len(handlers) for handlers in self.handlers.values())
                )
                
                logger.debug(f"Unsubscribed from event: {event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {event_type}: {str(e)}", exc_info=True)
            return False
    
    def _worker_loop(self):
        """Main worker loop for processing events"""
        logger.debug(f"Worker thread {threading.current_thread().name} started")
        
        while self.running:
            event = None
            try:
                # Get next event (prioritized)
                event = self._get_next_event()
                if event is None:  # Shutdown signal
                    break
                
                # Process event
                self._process_event(event)
                
            except Exception as e:
                if event:
                    logger.error(f"Error processing event {event.event_type}: {str(e)}", 
                               exc_info=True)
                    self.monitor.record_error(event, str(e))
                else:
                    logger.error(f"Worker error: {str(e)}", exc_info=True)
        
        logger.debug(f"Worker thread {threading.current_thread().name} stopped")
    
    def _get_next_event(self) -> Optional[Event]:
        """Get next event from priority queues"""
        # Check queues in priority order
        for priority in [EventPriority.CRITICAL, EventPriority.HIGH, 
                        EventPriority.NORMAL, EventPriority.LOW]:
            try:
                timestamp, event = self.queues[priority].get(block=False)
                self.queues[priority].task_done()
                return event
            except queue.Empty:
                continue
        
        # No events available, wait briefly
        time.sleep(0.01)
        return None
    
    def _process_event(self, event: Event):
        """Process a single event with all matching handlers"""
        start_time = time.time()
        success = True
        
        try:
            # Get matching handlers
            with self.handlers_lock:
                matching_handlers = []
                if event.event_type in self.handlers:
                    matching_handlers = [
                        h for h in self.handlers[event.event_type] 
                        if h.matches(event)
                    ]
            
            if not matching_handlers:
                logger.debug(f"No handlers for event: {event.event_type}")
                return
            
            # Execute handlers
            for handler in matching_handlers:
                if not handler.execute(event):
                    success = False
                    self.stats["failed"] += 1
            
            self.stats["processed"] += 1
            
        except Exception as e:
            success = False
            self.stats["failed"] += 1
            logger.error(f"Event processing failed: {str(e)}", exc_info=True)
        
        finally:
            # Record metrics
            processing_time = time.time() - start_time
            self.monitor.record_event(event, processing_time, success)
            self.monitor.update_queue_size(sum(q.qsize() for q in self.queues.values()))
    
    def process_events(self):
        """Process any pending events (for non-threaded scenarios)"""
        while self.running:
            event = self._get_next_event()
            if event is None:
                break
            self._process_event(event)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics"""
        return {
            **self.stats,
            "queue_sizes": {
                priority.name: q.qsize() 
                for priority, q in self.queues.items()
            },
            "handler_count": sum(len(handlers) for handlers in self.handlers.values()),
            "running": self.running
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Get event system health status"""
        return self.monitor.get_health_status()
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while self.running:
                try:
                    health = self.get_health()
                    if health["status"] != "healthy":
                        logger.warning(f"Event system health degraded: {health['health_score']}")
                    
                    # Check for queue buildup
                    total_queue_size = sum(q.qsize() for q in self.queues.values())
                    if total_queue_size > self.max_queue_size * 0.8:
                        logger.warning(f"Event queue filling up: {total_queue_size}/{self.max_queue_size}")
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Monitor error: {str(e)}", exc_info=True)
        
        monitor_thread = threading.Thread(
            target=monitor_loop,
            name="EventMonitor",
            daemon=True
        )
        monitor_thread.start()
        logger.info("Event system monitoring started")