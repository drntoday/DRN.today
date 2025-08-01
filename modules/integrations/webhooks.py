#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Integrations - Webhooks Module
Production-Ready Implementation
"""

import asyncio
import logging
import json
import time
import hmac
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import aiohttp
from aiohttp import web, web_response
import backoff
import base64
import re

# Core system imports
from engine.orchestrator import BaseModule
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# Initialize webhooks logger
logger = logging.getLogger(__name__)

@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    uuid: str
    name: str
    url: str
    events: List[str] = field(default_factory=list)
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True
    retry_count: int = 3
    timeout_seconds: int = 30
    created_at: float = field(default_factory=time.time)
    last_triggered: float = 0.0
    failure_count: int = 0
    cooldown_until: float = 0.0

@dataclass
class WebhookEvent:
    """Webhook event data structure"""
    uuid: str
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "drn.today"
    timestamp: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3
    next_attempt: float = field(default_factory=time.time)
    status: str = "pending"  # "pending", "sent", "failed"
    error: Optional[str] = None

@dataclass
class WebhookDelivery:
    """Webhook delivery result"""
    uuid: str
    event_uuid: str
    endpoint_uuid: str
    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    attempt: int = 1
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

class WebhooksConfig:
    """Configuration for the webhooks module"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.integrations_config = config_dict.get("integrations", {})
        self.security_config = config_dict.get("security", {})
        
        # Webhook server settings
        self.server_host = self.integrations_config.get("webhook_host", "localhost")
        self.server_port = self.integrations_config.get("webhook_port", 8080)
        self.server_path = self.integrations_config.get("webhook_path", "/webhook")
        self.max_payload_size = self.integrations_config.get("max_payload_size", 1048576)  # 1MB
        
        # Security settings
        self.require_signature = self.security_config.get("webhook_require_signature", True)
        self.signature_algorithm = self.security_config.get("webhook_signature_algorithm", "sha256")
        self.allowed_origins = self.security_config.get("webhook_allowed_origins", [])
        
        # Processing settings
        self.max_concurrent_deliveries = self.integrations_config.get("max_concurrent_deliveries", 10)
        self.retry_delay_seconds = self.integrations_config.get("retry_delay_seconds", 60)
        self.max_retry_attempts = self.integrations_config.get("max_retry_attempts", 3)
        self.event_retention_days = self.integrations_config.get("event_retention_days", 30)

class WebhooksModule(BaseModule):
    """Production-ready webhooks module for integrations"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Dict[str, Any]):
        super().__init__(name, event_bus, storage, license_manager, config)
        self.config = WebhooksConfig(config)
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.pending_events: asyncio.Queue = asyncio.Queue()
        self.delivery_semaphore = asyncio.Semaphore(self.config.max_concurrent_deliveries)
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.session_stats = {
            "events_received": 0,
            "events_sent": 0,
            "events_failed": 0,
            "endpoints_active": 0,
            "deliveries_retried": 0,
            "signature_validations": 0,
            "signature_failures": 0
        }
        
    def _setup_event_handlers(self):
        """Setup event handlers for webhook events"""
        self.event_bus.subscribe("webhook.register", self._handle_register_request)
        self.event_bus.subscribe("webhook.unregister", self._handle_unregister_request)
        self.event_bus.subscribe("webhook.trigger", self._handle_trigger_request)
        self.event_bus.subscribe("webhook.status", self._handle_status_request)
        
        # Subscribe to all system events for webhook triggering
        self.event_bus.subscribe("*", self._handle_system_event)
        
    def _validate_requirements(self):
        """Validate module requirements"""
        # Load existing endpoints from storage
        self._load_endpoints()
        
    async def _start_services(self):
        """Start webhook services"""
        # Start webhook server
        await self._start_webhook_server()
        
        # Start delivery worker
        asyncio.create_task(self._delivery_worker())
        
        # Start cleanup worker
        asyncio.create_task(self._cleanup_worker())
        
        logger.info("Webhooks module services started successfully")
    
    async def _stop_services(self):
        """Stop webhook services"""
        # Stop webhook server
        await self._stop_webhook_server()
        
        logger.info("Webhooks module services stopped")
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up old events
        self._cleanup_old_events()
        
        # Check endpoint cooldowns
        current_time = time.time()
        for endpoint in self.endpoints.values():
            if endpoint.cooldown_until > current_time:
                endpoint.is_active = False
            else:
                endpoint.is_active = True
        
        # Log session stats
        logger.debug(f"Webhooks stats: {self.session_stats}")
    
    async def _handle_register_request(self, event_type: str, data: Dict[str, Any]):
        """Handle webhook registration requests"""
        try:
            endpoint_data = data.get("endpoint")
            if not endpoint_data:
                logger.warning("Invalid register request: missing endpoint data")
                return
            
            # Register endpoint
            endpoint = self._register_endpoint(endpoint_data)
            if endpoint:
                self.event_bus.publish("webhook.registered", {
                    "endpoint_uuid": endpoint.uuid,
                    "name": endpoint.name
                })
            
        except Exception as e:
            logger.error(f"Error handling register request: {str(e)}", exc_info=True)
    
    async def _handle_unregister_request(self, event_type: str, data: Dict[str, Any]):
        """Handle webhook unregistration requests"""
        try:
            endpoint_uuid = data.get("endpoint_uuid")
            if not endpoint_uuid:
                logger.warning("Invalid unregister request: missing endpoint UUID")
                return
            
            # Unregister endpoint
            if self._unregister_endpoint(endpoint_uuid):
                self.event_bus.publish("webhook.unregistered", {
                    "endpoint_uuid": endpoint_uuid
                })
            
        except Exception as e:
            logger.error(f"Error handling unregister request: {str(e)}", exc_info=True)
    
    async def _handle_trigger_request(self, event_type: str, data: Dict[str, Any]):
        """Handle webhook trigger requests"""
        try:
            event_type = data.get("event_type")
            event_data = data.get("data", {})
            
            if not event_type:
                logger.warning("Invalid trigger request: missing event type")
                return
            
            # Trigger webhook event
            await self._trigger_webhook_event(event_type, event_data)
            
        except Exception as e:
            logger.error(f"Error handling trigger request: {str(e)}", exc_info=True)
    
    async def _handle_status_request(self, event_type: str, data: Dict[str, Any]):
        """Handle status requests"""
        status = {
            "endpoints": len(self.endpoints),
            "active_endpoints": len([e for e in self.endpoints.values() if e.is_active]),
            "pending_events": self.pending_events.qsize(),
            "session_stats": self.session_stats
        }
        self.event_bus.publish("webhooks.status.response", status)
    
    async def _handle_system_event(self, event_type: str, data: Dict[str, Any]):
        """Handle all system events for webhook triggering"""
        # Skip webhook events to prevent loops
        if event_type.startswith("webhook."):
            return
        
        # Trigger webhook event
        await self._trigger_webhook_event(event_type, data)
    
    def _register_endpoint(self, endpoint_data: Dict[str, Any]) -> Optional[WebhookEndpoint]:
        """Register a new webhook endpoint"""
        try:
            # Create endpoint
            endpoint = WebhookEndpoint(
                uuid=str(uuid.uuid4()),
                name=endpoint_data.get("name"),
                url=endpoint_data.get("url"),
                events=endpoint_data.get("events", []),
                secret=endpoint_data.get("secret"),
                headers=endpoint_data.get("headers", {}),
                retry_count=endpoint_data.get("retry_count", 3),
                timeout_seconds=endpoint_data.get("timeout_seconds", 30)
            )
            
            # Validate endpoint
            if not self._validate_endpoint(endpoint):
                logger.error(f"Invalid webhook endpoint: {endpoint.name}")
                return None
            
            # Add to endpoints
            self.endpoints[endpoint.uuid] = endpoint
            
            # Save to storage
            self._save_endpoint_to_storage(endpoint)
            
            logger.info(f"Registered webhook endpoint: {endpoint.name}")
            return endpoint
            
        except Exception as e:
            logger.error(f"Error registering webhook endpoint: {str(e)}", exc_info=True)
            return None
    
    def _unregister_endpoint(self, endpoint_uuid: str) -> bool:
        """Unregister a webhook endpoint"""
        try:
            if endpoint_uuid not in self.endpoints:
                logger.warning(f"Endpoint not found: {endpoint_uuid}")
                return False
            
            endpoint = self.endpoints[endpoint_uuid]
            
            # Remove from endpoints
            del self.endpoints[endpoint_uuid]
            
            # Remove from storage
            self._remove_endpoint_from_storage(endpoint_uuid)
            
            logger.info(f"Unregistered webhook endpoint: {endpoint.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering webhook endpoint: {str(e)}", exc_info=True)
            return False
    
    def _validate_endpoint(self, endpoint: WebhookEndpoint) -> bool:
        """Validate webhook endpoint configuration"""
        try:
            # Check required fields
            if not all([endpoint.name, endpoint.url, endpoint.events]):
                return False
            
            # Validate URL format
            if not re.match(r'^https?://', endpoint.url):
                return False
            
            # Validate events
            if not isinstance(endpoint.events, list) or not endpoint.events:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Endpoint validation failed: {str(e)}", exc_info=True)
            return False
    
    def _save_endpoint_to_storage(self, endpoint: WebhookEndpoint):
        """Save endpoint to storage"""
        try:
            endpoint_data = {
                "uuid": endpoint.uuid,
                "name": endpoint.name,
                "url": endpoint.url,
                "events": endpoint.events,
                "secret": endpoint.secret,
                "headers": endpoint.headers,
                "is_active": endpoint.is_active,
                "retry_count": endpoint.retry_count,
                "timeout_seconds": endpoint.timeout_seconds,
                "created_at": endpoint.created_at,
                "last_triggered": endpoint.last_triggered,
                "failure_count": endpoint.failure_count,
                "cooldown_until": endpoint.cooldown_until
            }
            
            self.storage.save_lead({
                "uuid": endpoint.uuid,
                "source": "webhook_endpoint",
                "name": endpoint.name,
                "raw_content": json.dumps(endpoint_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving endpoint to storage: {str(e)}", exc_info=True)
    
    def _remove_endpoint_from_storage(self, endpoint_uuid: str):
        """Remove endpoint from storage"""
        try:
            self.storage.delete_lead(endpoint_uuid)
        except Exception as e:
            logger.error(f"Error removing endpoint from storage: {str(e)}", exc_info=True)
    
    def _load_endpoints(self):
        """Load endpoints from storage"""
        try:
            # Query storage for webhook endpoints
            endpoints_data = self.storage.query_leads({
                "source": "webhook_endpoint",
                "category": "system"
            })
            
            for endpoint_data in endpoints_data:
                try:
                    endpoint = WebhookEndpoint(
                        uuid=endpoint_data.get("uuid"),
                        name=endpoint_data.get("name"),
                        url=endpoint_data.get("url"),
                        events=endpoint_data.get("events", []),
                        secret=endpoint_data.get("secret"),
                        headers=endpoint_data.get("headers", {}),
                        is_active=endpoint_data.get("is_active", True),
                        retry_count=endpoint_data.get("retry_count", 3),
                        timeout_seconds=endpoint_data.get("timeout_seconds", 30),
                        created_at=endpoint_data.get("created_at", time.time()),
                        last_triggered=endpoint_data.get("last_triggered", 0.0),
                        failure_count=endpoint_data.get("failure_count", 0),
                        cooldown_until=endpoint_data.get("cooldown_until", 0.0)
                    )
                    
                    self.endpoints[endpoint.uuid] = endpoint
                    
                except Exception as e:
                    logger.error(f"Error loading endpoint: {str(e)}", exc_info=True)
            
            logger.info(f"Loaded {len(self.endpoints)} webhook endpoints")
            
        except Exception as e:
            logger.error(f"Error loading endpoints: {str(e)}", exc_info=True)
    
    async def _trigger_webhook_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger a webhook event"""
        try:
            # Create event
            event = WebhookEvent(
                uuid=str(uuid.uuid4()),
                event_type=event_type,
                data=event_data
            )
            
            # Add to pending queue
            await self.pending_events.put(event)
            
            # Update stats
            self.session_stats["events_received"] += 1
            
            logger.debug(f"Triggered webhook event: {event_type}")
            
        except Exception as e:
            logger.error(f"Error triggering webhook event: {str(e)}", exc_info=True)
    
    async def _delivery_worker(self):
        """Worker for processing webhook deliveries"""
        while True:
            try:
                # Get event from queue
                event = await self.pending_events.get()
                
                # Process event
                await self._process_event(event)
                
                # Mark task as done
                self.pending_events.task_done()
                
            except Exception as e:
                logger.error(f"Error in delivery worker: {str(e)}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_event(self, event: WebhookEvent):
        """Process a webhook event"""
        try:
            # Find matching endpoints
            matching_endpoints = [
                endpoint for endpoint in self.endpoints.values()
                if endpoint.is_active and event.event_type in endpoint.events
            ]
            
            if not matching_endpoints:
                logger.debug(f"No matching endpoints for event: {event.event_type}")
                return
            
            # Deliver to each endpoint
            for endpoint in matching_endpoints:
                await self._deliver_to_endpoint(event, endpoint)
                
        except Exception as e:
            logger.error(f"Error processing event: {str(e)}", exc_info=True)
    
    async def _deliver_to_endpoint(self, event: WebhookEvent, endpoint: WebhookEndpoint):
        """Deliver webhook event to an endpoint"""
        async with self.delivery_semaphore:
            start_time = time.time()
            
            try:
                # Check cooldown
                if endpoint.cooldown_until > time.time():
                    logger.debug(f"Endpoint {endpoint.name} is on cooldown")
                    return
                
                # Prepare payload
                payload = {
                    "uuid": event.uuid,
                    "event_type": event.event_type,
                    "data": event.data,
                    "source": event.source,
                    "timestamp": event.timestamp
                }
                
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "DRN.today-Webhook/1.0"
                }
                
                # Add custom headers
                headers.update(endpoint.headers)
                
                # Add signature if secret is provided
                if endpoint.secret:
                    signature = self._generate_signature(payload, endpoint.secret)
                    headers["X-DRN-Signature"] = signature
                
                # Send webhook
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint.url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
                    ) as response:
                        response_body = await response.text()
                        
                        # Create delivery record
                        delivery = WebhookDelivery(
                            uuid=str(uuid.uuid4()),
                            event_uuid=event.uuid,
                            endpoint_uuid=endpoint.uuid,
                            success=response.status < 400,
                            status_code=response.status,
                            response_body=response_body,
                            attempt=event.attempts + 1,
                            duration=time.time() - start_time
                        )
                        
                        # Save delivery
                        self._save_delivery(delivery)
                        
                        # Update endpoint stats
                        endpoint.last_triggered = time.time()
                        
                        if delivery.success:
                            endpoint.failure_count = 0
                            self.session_stats["events_sent"] += 1
                            logger.debug(f"Webhook delivered successfully to {endpoint.name}")
                        else:
                            endpoint.failure_count += 1
                            self.session_stats["events_failed"] += 1
                            
                            # Put endpoint on cooldown if too many failures
                            if endpoint.failure_count >= 5:
                                endpoint.cooldown_until = time.time() + 300  # 5 minutes
                                endpoint.is_active = False
                            
                            logger.warning(f"Webhook delivery failed to {endpoint.name}: {response.status}")
                
            except Exception as e:
                # Create failed delivery record
                delivery = WebhookDelivery(
                    uuid=str(uuid.uuid4()),
                    event_uuid=event.uuid,
                    endpoint_uuid=endpoint.uuid,
                    success=False,
                    error=str(e),
                    attempt=event.attempts + 1,
                    duration=time.time() - start_time
                )
                
                # Save delivery
                self._save_delivery(delivery)
                
                # Update endpoint stats
                endpoint.failure_count += 1
                self.session_stats["events_failed"] += 1
                
                logger.error(f"Error delivering webhook to {endpoint.name}: {str(e)}", exc_info=True)
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate webhook signature"""
        try:
            # Convert payload to JSON string
            payload_str = json.dumps(payload, sort_keys=True)
            
            # Generate signature
            signature = hmac.new(
                secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).digest()
            
            # Return base64 encoded signature
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            logger.error(f"Error generating signature: {str(e)}", exc_info=True)
            return ""
    
    def _save_delivery(self, delivery: WebhookDelivery):
        """Save delivery record to storage"""
        try:
            delivery_data = {
                "uuid": delivery.uuid,
                "event_uuid": delivery.event_uuid,
                "endpoint_uuid": delivery.endpoint_uuid,
                "success": delivery.success,
                "status_code": delivery.status_code,
                "response_body": delivery.response_body,
                "error": delivery.error,
                "attempt": delivery.attempt,
                "duration": delivery.duration,
                "timestamp": delivery.timestamp
            }
            
            self.storage.save_lead({
                "uuid": delivery.uuid,
                "source": "webhook_delivery",
                "name": f"Delivery {delivery.uuid}",
                "raw_content": json.dumps(delivery_data),
                "category": "system"
            })
            
        except Exception as e:
            logger.error(f"Error saving delivery: {str(e)}", exc_info=True)
    
    async def _start_webhook_server(self):
        """Start the webhook server"""
        try:
            # Create web application
            app = web.Application()
            
            # Add routes
            app.router.add_post(self.config.server_path, self._handle_incoming_webhook)
            app.router.add_get("/webhooks", self._handle_webhooks_list)
            app.router.add_get("/webhooks/{endpoint_id}", self._handle_webhook_detail)
            
            # Create runner
            self.runner = web.AppRunner(app)
            await self.runner.setup()
            
            # Create site
            self.site = web.TCPSite(
                self.runner,
                self.config.server_host,
                self.config.server_port
            )
            
            # Start site
            await self.site.start()
            
            logger.info(f"Webhook server started on {self.config.server_host}:{self.config.server_port}")
            
        except Exception as e:
            logger.error(f"Error starting webhook server: {str(e)}", exc_info=True)
            raise
    
    async def _stop_webhook_server(self):
        """Stop the webhook server"""
        try:
            if self.site:
                await self.site.stop()
            
            if self.runner:
                await self.runner.cleanup()
            
            logger.info("Webhook server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping webhook server: {str(e)}", exc_info=True)
    
    async def _handle_incoming_webhook(self, request: web.Request):
        """Handle incoming webhook requests"""
        try:
            # Check content type
            content_type = request.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                return web.json_response(
                    {"error": "Invalid content type"},
                    status=400
                )
            
            # Check content length
            content_length = int(request.headers.get("Content-Length", 0))
            if content_length > self.config.max_payload_size:
                return web.json_response(
                    {"error": "Payload too large"},
                    status=413
                )
            
            # Parse payload
            try:
                payload = await request.json()
            except json.JSONDecodeError:
                return web.json_response(
                    {"error": "Invalid JSON"},
                    status=400
                )
            
            # Validate signature if required
            if self.config.require_signature:
                signature = request.headers.get("X-DRN-Signature")
                if not signature:
                    return web.json_response(
                        {"error": "Missing signature"},
                        status=401
                    )
                
                # Find endpoint by URL (simplified)
                endpoint = None
                for ep in self.endpoints.values():
                    if ep.url == str(request.url):
                        endpoint = ep
                        break
                
                if not endpoint or not endpoint.secret:
                    return web.json_response(
                        {"error": "Endpoint not found"},
                        status=404
                    )
                
                # Verify signature
                expected_signature = self._generate_signature(payload, endpoint.secret)
                if not hmac.compare_digest(signature, expected_signature):
                    self.session_stats["signature_failures"] += 1
                    return web.json_response(
                        {"error": "Invalid signature"},
                        status=401
                    )
                
                self.session_stats["signature_validations"] += 1
            
            # Process webhook event
            event_type = payload.get("event_type")
            event_data = payload.get("data", {})
            
            if event_type:
                await self._trigger_webhook_event(event_type, event_data)
            
            return web.json_response({"status": "ok"})
            
        except Exception as e:
            logger.error(f"Error handling incoming webhook: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
    
    async def _handle_webhooks_list(self, request: web.Request):
        """Handle list webhooks request"""
        try:
            webhooks_list = []
            for endpoint in self.endpoints.values():
                webhooks_list.append({
                    "uuid": endpoint.uuid,
                    "name": endpoint.name,
                    "url": endpoint.url,
                    "events": endpoint.events,
                    "is_active": endpoint.is_active,
                    "created_at": endpoint.created_at
                })
            
            return web.json_response({"webhooks": webhooks_list})
            
        except Exception as e:
            logger.error(f"Error handling webhooks list: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
    
    async def _handle_webhook_detail(self, request: web.Request):
        """Handle webhook detail request"""
        try:
            endpoint_id = request.match_info["endpoint_id"]
            
            if endpoint_id not in self.endpoints:
                return web.json_response(
                    {"error": "Webhook not found"},
                    status=404
                )
            
            endpoint = self.endpoints[endpoint_id]
            
            return web.json_response({
                "uuid": endpoint.uuid,
                "name": endpoint.name,
                "url": endpoint.url,
                "events": endpoint.events,
                "is_active": endpoint.is_active,
                "headers": endpoint.headers,
                "retry_count": endpoint.retry_count,
                "timeout_seconds": endpoint.timeout_seconds,
                "created_at": endpoint.created_at,
                "last_triggered": endpoint.last_triggered,
                "failure_count": endpoint.failure_count
            })
            
        except Exception as e:
            logger.error(f"Error handling webhook detail: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
    
    async def _cleanup_worker(self):
        """Worker for cleanup tasks"""
        while True:
            try:
                # Perform cleanup
                self._perform_maintenance()
                
                # Sleep for next cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    def _cleanup_old_events(self):
        """Clean up old webhook events and deliveries"""
        try:
            cutoff_time = time.time() - (self.config.event_retention_days * 86400)
            
            # In a real implementation, this would query and delete old records
            # For demo, we'll just log
            logger.debug(f"Cleaning up events older than {cutoff_time}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old events: {str(e)}", exc_info=True)
    
    async def register_webhook(self, name: str, url: str, events: List[str], **kwargs) -> Dict[str, Any]:
        """Public method to register a webhook"""
        endpoint_data = {
            "name": name,
            "url": url,
            "events": events,
            **kwargs
        }
        
        endpoint = self._register_endpoint(endpoint_data)
        if endpoint:
            return {
                "endpoint_uuid": endpoint.uuid,
                "name": endpoint.name,
                "status": "registered"
            }
        else:
            return {
                "status": "failed",
                "error": "Invalid webhook configuration"
            }
    
    async def unregister_webhook(self, endpoint_uuid: str) -> Dict[str, Any]:
        """Public method to unregister a webhook"""
        if self._unregister_endpoint(endpoint_uuid):
            return {
                "endpoint_uuid": endpoint_uuid,
                "status": "unregistered"
            }
        else:
            return {
                "endpoint_uuid": endpoint_uuid,
                "status": "failed",
                "error": "Webhook not found"
            }
    
    async def trigger_webhook(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Public method to trigger a webhook event"""
        await self._trigger_webhook_event(event_type, data)
        
        return {
            "event_type": event_type,
            "status": "triggered"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get webhooks module statistics"""
        return {
            "session_stats": self.session_stats,
            "endpoints": len(self.endpoints),
            "active_endpoints": len([e for e in self.endpoints.values() if e.is_active]),
            "pending_events": self.pending_events.qsize(),
            "server_running": self.site is not None
        }
