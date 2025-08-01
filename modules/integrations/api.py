# modules/integrations/api.py

import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import jwt
from fastapi import FastAPI, HTTPException, Depends, status, Security, BackgroundTasks
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from starlette.requests import Request
from starlette.responses import JSONResponse

from engine.storage import Storage
from engine.license import LicenseManager
from modules.lead_enrichment.tagging import DNAStyleTaggingSystem
from modules.lead_enrichment.insights import LeadInsightsEngine
from modules.conversation_mining.monitor import ConversationMonitor
from modules.conversation_mining.classifier import ConversationClassifier
from modules.web_crawlers.self_healing import DOMSelfHealingEngine
from modules.web_crawlers.retry_logic import RetryManager
from modules.email_system.smtp_manager import SMTPManager
from modules.email_system.imap_manager import IMAPManager
from modules.email_system.bounce_detector import BounceDetector
from modules.intent_engine.heatmap import HeatmapGenerator
from modules.intent_engine.recommender import OutreachRecommender
from modules.template_engine.personalizer import TemplatePersonalizer
from modules.template_engine.ab_testing import ABTestingEngine
from ai.nlp import NLPProcessor
from ai.scoring import LeadScorer


class APIEndpoint(Enum):
    LEADS = "/leads"
    CAMPAIGNS = "/campaigns"
    EMAILS = "/emails"
    TEMPLATES = "/templates"
    REPORTS = "/reports"
    INTEGRATIONS = "/integrations"
    WEBHOOKS = "/webhooks"
    ANALYTICS = "/analytics"
    HEALTH = "/health"


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIKey:
    key: str
    name: str
    permissions: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True


@dataclass
class Webhook:
    id: str
    url: str
    events: List[str]
    secret: str
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    is_active: bool = True


class LeadCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    company: Optional[str] = Field(None, max_length=100)
    job_title: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    industry: Optional[str] = Field(None, max_length=50)
    source: Optional[str] = Field(None, max_length=50)
    metadata: Optional[Dict] = None


class LeadUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    company: Optional[str] = Field(None, max_length=100)
    job_title: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    industry: Optional[str] = Field(None, max_length=50)
    source: Optional[str] = Field(None, max_length=50)
    metadata: Optional[Dict] = None


class CampaignCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    template_id: str = Field(..., min_length=1)
    target_audience: Optional[Dict] = None
    schedule: Optional[Dict] = None
    metadata: Optional[Dict] = None


class EmailSendRequest(BaseModel):
    campaign_id: str = Field(..., min_length=1)
    lead_ids: List[str] = Field(..., min_items=1)
    template_id: Optional[str] = None
    personalization_level: Optional[str] = Field("advanced", regex=r"^(basic|intermediate|advanced|hyper_personalized)$")
    schedule_at: Optional[datetime] = None


class WebhookCreateRequest(BaseModel):
    url: str = Field(..., regex=r'^https?://.+')
    events: List[str] = Field(..., min_items=1)
    secret: Optional[str] = Field(None, min_length=16)


class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    permissions: List[str] = Field(..., min_items=1)
    expires_in_days: Optional[int] = Field(365, ge=1, le=3650)


class IntegrationAPI:
    def __init__(self, storage: Storage, license_manager: LicenseManager,
                 tagging_system: DNAStyleTaggingSystem,
                 insights_engine: LeadInsightsEngine,
                 conversation_monitor: ConversationMonitor,
                 conversation_classifier: ConversationClassifier,
                 self_healing_engine: DOMSelfHealingEngine,
                 retry_manager: RetryManager,
                 smtp_manager: SMTPManager,
                 imap_manager: IMAPManager,
                 bounce_detector: BounceDetector,
                 heatmap_generator: HeatmapGenerator,
                 recommender: OutreachRecommender,
                 template_personalizer: TemplatePersonalizer,
                 ab_testing_engine: ABTestingEngine,
                 nlp_processor: NLPProcessor,
                 lead_scorer: LeadScorer):
        self.storage = storage
        self.license_manager = license_manager
        self.tagging_system = tagging_system
        self.insights_engine = insights_engine
        self.conversation_monitor = conversation_monitor
        self.conversation_classifier = conversation_classifier
        self.self_healing_engine = self_healing_engine
        self.retry_manager = retry_manager
        self.smtp_manager = smtp_manager
        self.imap_manager = imap_manager
        self.bounce_detector = bounce_detector
        self.heatmap_generator = heatmap_generator
        self.recommender = recommender
        self.template_personalizer = template_personalizer
        self.ab_testing_engine = ab_testing_engine
        self.nlp_processor = nlp_processor
        self.lead_scorer = lead_scorer
        
        self.logger = logging.getLogger("integration_api")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="DRN.today API",
            description="Enterprise-grade lead generation and outreach platform API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load API keys and webhooks
        self.api_keys: Dict[str, APIKey] = {}
        self.webhooks: Dict[str, Webhook] = {}
        self._load_api_keys()
        self._load_webhooks()
        
        # Set up authentication
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
        
        # Set up rate limiting
        self.rate_limits = {
            "default": 100,  # requests per minute
            "leads": 50,
            "campaigns": 30,
            "emails": 20,
            "analytics": 10
        }
        
        # Set up request tracking
        self.request_counts = {}
        
        # Register routes
        self._register_routes()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            permissions TEXT,
            created_at TEXT,
            expires_at TEXT,
            last_used TEXT,
            is_active INTEGER DEFAULT 1
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS webhooks (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            events TEXT,
            secret TEXT,
            created_at TEXT,
            last_triggered TEXT,
            is_active INTEGER DEFAULT 1
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT,
            endpoint TEXT,
            method TEXT,
            status_code INTEGER,
            response_time REAL,
            timestamp TEXT,
            ip_address TEXT,
            user_agent TEXT
        )
        """)

    def _load_api_keys(self):
        """Load API keys from storage"""
        for row in self.storage.query("SELECT * FROM api_keys WHERE is_active = 1"):
            api_key = APIKey(
                key=row['key'],
                name=row['name'],
                permissions=json.loads(row['permissions']),
                created_at=datetime.fromisoformat(row['created_at']),
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
                is_active=bool(row['is_active'])
            )
            self.api_keys[api_key.key] = api_key

    def _load_webhooks(self):
        """Load webhooks from storage"""
        for row in self.storage.query("SELECT * FROM webhooks WHERE is_active = 1"):
            webhook = Webhook(
                id=row['id'],
                url=row['url'],
                events=json.loads(row['events']),
                secret=row['secret'],
                created_at=datetime.fromisoformat(row['created_at']),
                last_triggered=datetime.fromisoformat(row['last_triggered']) if row['last_triggered'] else None,
                is_active=bool(row['is_active'])
            )
            self.webhooks[webhook.id] = webhook

    def _register_routes(self):
        """Register API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # Authentication endpoints
        @self.app.post("/token")
        async def create_token(api_key: str = Security(self.api_key_header)):
            if not api_key or api_key not in self.api_keys:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            
            key_data = self.api_keys[api_key]
            
            # Check if key is expired
            if key_data.expires_at and key_data.expires_at < datetime.now():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key expired"
                )
            
            # Update last used
            key_data.last_used = datetime.now()
            self.storage.execute(
                "UPDATE api_keys SET last_used = ? WHERE key = ?",
                (key_data.last_used.isoformat(), api_key)
            )
            
            # Create JWT token
            token_data = {
                "sub": api_key,
                "name": key_data.name,
                "permissions": key_data.permissions,
                "exp": datetime.utcnow() + timedelta(hours=1)
            }
            
            token = jwt.encode(token_data, "secret", algorithm="HS256")
            
            return {"access_token": token, "token_type": "bearer"}
        
        # API key management
        @self.app.post("/api-keys")
        async def create_api_key(
            request: APIKeyCreateRequest,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "api_keys:create")
            
            # Generate new API key
            new_key = f"drn_{secrets.token_urlsafe(32)}"
            
            # Calculate expiration date
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.now() + timedelta(days=request.expires_in_days)
            
            # Create API key object
            api_key_obj = APIKey(
                key=new_key,
                name=request.name,
                permissions=request.permissions,
                expires_at=expires_at
            )
            
            # Save to database
            self.storage.execute(
                """
                INSERT INTO api_keys 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_key,
                    request.name,
                    json.dumps(request.permissions),
                    api_key_obj.created_at.isoformat(),
                    expires_at.isoformat() if expires_at else None,
                    None,
                    1
                )
            )
            
            # Add to in-memory cache
            self.api_keys[new_key] = api_key_obj
            
            return {
                "key": new_key,
                "name": request.name,
                "permissions": request.permissions,
                "expires_at": expires_at.isoformat() if expires_at else None
            }
        
        @self.app.get("/api-keys")
        async def list_api_keys(
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "api_keys:read")
            
            return [
                {
                    "name": key.name,
                    "permissions": key.permissions,
                    "created_at": key.created_at.isoformat(),
                    "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                    "last_used": key.last_used.isoformat() if key.last_used else None
                }
                for key in self.api_keys.values()
            ]
        
        @self.app.delete("/api-keys/{key_name}")
        async def delete_api_key(
            key_name: str,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "api_keys:delete")
            
            # Find the key by name
            key_to_delete = None
            for key in self.api_keys.values():
                if key.name == key_name:
                    key_to_delete = key
                    break
            
            if not key_to_delete:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="API key not found"
                )
            
            # Deactivate in database
            self.storage.execute(
                "UPDATE api_keys SET is_active = 0 WHERE key = ?",
                (key_to_delete.key,)
            )
            
            # Remove from in-memory cache
            del self.api_keys[key_to_delete.key]
            
            return {"status": "success"}
        
        # Lead management
        @self.app.post("/leads")
        async def create_lead(
            request: LeadCreateRequest,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "leads:create")
            
            # Create lead in database
            lead_id = f"lead_{int(datetime.now().timestamp())}"
            
            self.storage.execute(
                """
                INSERT INTO leads 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead_id,
                    request.name,
                    request.email,
                    request.company,
                    request.job_title,
                    request.phone,
                    request.industry,
                    request.source,
                    json.dumps(request.metadata) if request.metadata else None,
                    datetime.now().isoformat(),
                    None  # score will be calculated separately
                )
            )
            
            # Calculate lead score
            lead_data = {
                "name": request.name,
                "email": request.email,
                "company": request.company,
                "job_title": request.job_title,
                "industry": request.industry,
                "source": request.source
            }
            
            score = self.lead_scorer.calculate_lead_score(lead_data)
            self.storage.execute(
                "UPDATE leads SET score = ? WHERE id = ?",
                (score, lead_id)
            )
            
            # Trigger webhooks
            self._trigger_webhooks("lead.created", {
                "lead_id": lead_id,
                "name": request.name,
                "email": request.email
            })
            
            return {"lead_id": lead_id, "score": score}
        
        @self.app.get("/leads/{lead_id}")
        async def get_lead(
            lead_id: str,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "leads:read")
            
            # Get lead from database
            row = self.storage.query(
                "SELECT * FROM leads WHERE id = ?",
                (lead_id,)
            ).fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Lead not found"
                )
            
            lead = {
                "id": row['id'],
                "name": row['name'],
                "email": row['email'],
                "company": row['company'],
                "job_title": row['job_title'],
                "phone": row['phone'],
                "industry": row['industry'],
                "source": row['source'],
                "metadata": json.loads(row['metadata']) if row['metadata'] else None,
                "created_at": row['created_at'],
                "score": row['score']
            }
            
            return lead
        
        @self.app.put("/leads/{lead_id}")
        async def update_lead(
            lead_id: str,
            request: LeadUpdateRequest,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "leads:update")
            
            # Check if lead exists
            existing = self.storage.query(
                "SELECT id FROM leads WHERE id = ?",
                (lead_id,)
            ).fetchone()
            
            if not existing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Lead not found"
                )
            
            # Build update query
            updates = {}
            if request.name is not None:
                updates["name"] = request.name
            if request.email is not None:
                updates["email"] = request.email
            if request.company is not None:
                updates["company"] = request.company
            if request.job_title is not None:
                updates["job_title"] = request.job_title
            if request.phone is not None:
                updates["phone"] = request.phone
            if request.industry is not None:
                updates["industry"] = request.industry
            if request.source is not None:
                updates["source"] = request.source
            if request.metadata is not None:
                updates["metadata"] = json.dumps(request.metadata)
            
            # Execute update
            if updates:
                set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                values = list(updates.values()) + [lead_id]
                
                self.storage.execute(
                    f"UPDATE leads SET {set_clause} WHERE id = ?",
                    values
                )
            
            # Trigger webhooks
            self._trigger_webhooks("lead.updated", {
                "lead_id": lead_id,
                "updates": updates
            })
            
            return {"status": "success"}
        
        @self.app.delete("/leads/{lead_id}")
        async def delete_lead(
            lead_id: str,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "leads:delete")
            
            # Check if lead exists
            existing = self.storage.query(
                "SELECT id FROM leads WHERE id = ?",
                (lead_id,)
            ).fetchone()
            
            if not existing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Lead not found"
                )
            
            # Delete lead
            self.storage.execute(
                "DELETE FROM leads WHERE id = ?",
                (lead_id,)
            )
            
            # Trigger webhooks
            self._trigger_webhooks("lead.deleted", {
                "lead_id": lead_id
            })
            
            return {"status": "success"}
        
        @self.app.get("/leads")
        async def list_leads(
            skip: int = 0,
            limit: int = 100,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "leads:read")
            
            # Get leads from database
            leads = []
            for row in self.storage.query(
                "SELECT * FROM leads ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, skip)
            ):
                lead = {
                    "id": row['id'],
                    "name": row['name'],
                    "email": row['email'],
                    "company": row['company'],
                    "job_title": row['job_title'],
                    "industry": row['industry'],
                    "source": row['source'],
                    "created_at": row['created_at'],
                    "score": row['score']
                }
                leads.append(lead)
            
            return leads
        
        # Campaign management
        @self.app.post("/campaigns")
        async def create_campaign(
            request: CampaignCreateRequest,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "campaigns:create")
            
            # Create campaign in database
            campaign_id = f"campaign_{int(datetime.now().timestamp())}"
            
            self.storage.execute(
                """
                INSERT INTO campaigns 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    request.name,
                    request.description,
                    request.template_id,
                    json.dumps(request.target_audience) if request.target_audience else None,
                    json.dumps(request.schedule) if request.schedule else None,
                    datetime.now().isoformat()
                )
            )
            
            # Trigger webhooks
            self._trigger_webhooks("campaign.created", {
                "campaign_id": campaign_id,
                "name": request.name
            })
            
            return {"campaign_id": campaign_id}
        
        @self.app.get("/campaigns/{campaign_id}")
        async def get_campaign(
            campaign_id: str,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "campaigns:read")
            
            # Get campaign from database
            row = self.storage.query(
                "SELECT * FROM campaigns WHERE id = ?",
                (campaign_id,)
            ).fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Campaign not found"
                )
            
            campaign = {
                "id": row['id'],
                "name": row['name'],
                "description": row['description'],
                "template_id": row['template_id'],
                "target_audience": json.loads(row['target_audience']) if row['target_audience'] else None,
                "schedule": json.loads(row['schedule']) if row['schedule'] else None,
                "created_at": row['created_at']
            }
            
            return campaign
        
        @self.app.get("/campaigns")
        async def list_campaigns(
            skip: int = 0,
            limit: int = 100,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "campaigns:read")
            
            # Get campaigns from database
            campaigns = []
            for row in self.storage.query(
                "SELECT * FROM campaigns ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, skip)
            ):
                campaign = {
                    "id": row['id'],
                    "name": row['name'],
                    "description": row['description'],
                    "template_id": row['template_id'],
                    "created_at": row['created_at']
                }
                campaigns.append(campaign)
            
            return campaigns
        
        # Email sending
        @self.app.post("/emails/send")
        async def send_email(
            request: EmailSendRequest,
            background_tasks: BackgroundTasks,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "emails:send")
            
            # Validate campaign exists
            campaign = self.storage.query(
                "SELECT * FROM campaigns WHERE id = ?",
                (request.campaign_id,)
            ).fetchone()
            
            if not campaign:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Campaign not found"
                )
            
            # Get leads
            leads = []
            for lead_id in request.lead_ids:
                lead = self.storage.query(
                    "SELECT * FROM leads WHERE id = ?",
                    (lead_id,)
                ).fetchone()
                
                if not lead:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Lead not found: {lead_id}"
                    )
                
                leads.append({
                    "id": lead['id'],
                    "name": lead['name'],
                    "email": lead['email'],
                    "company": lead['company'],
                    "job_title": lead['job_title']
                })
            
            # Get template
            template_id = request.template_id or campaign['template_id']
            template = self.storage.query(
                "SELECT * FROM templates WHERE id = ?",
                (template_id,)
            ).fetchone()
            
            if not template:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Template not found"
                )
            
            # Send emails in background
            background_tasks.add_task(
                self._send_emails_task,
                request.campaign_id,
                leads,
                template,
                request.personalization_level,
                request.schedule_at
            )
            
            return {
                "status": "queued",
                "campaign_id": request.campaign_id,
                "lead_count": len(leads)
            }
        
        # Webhook management
        @self.app.post("/webhooks")
        async def create_webhook(
            request: WebhookCreateRequest,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "webhooks:create")
            
            # Generate secret if not provided
            secret = request.secret or secrets.token_urlsafe(32)
            
            # Create webhook
            webhook_id = f"webhook_{int(datetime.now().timestamp())}"
            webhook = Webhook(
                id=webhook_id,
                url=request.url,
                events=request.events,
                secret=secret
            )
            
            # Save to database
            self.storage.execute(
                """
                INSERT INTO webhooks 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    webhook_id,
                    request.url,
                    json.dumps(request.events),
                    secret,
                    webhook.created_at.isoformat(),
                    None,
                    1
                )
            )
            
            # Add to in-memory cache
            self.webhooks[webhook_id] = webhook
            
            return {
                "webhook_id": webhook_id,
                "url": request.url,
                "events": request.events,
                "secret": secret
            }
        
        @self.app.get("/webhooks")
        async def list_webhooks(
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "webhooks:read")
            
            return [
                {
                    "id": webhook.id,
                    "url": webhook.url,
                    "events": webhook.events,
                    "created_at": webhook.created_at.isoformat(),
                    "last_triggered": webhook.last_triggered.isoformat() if webhook.last_triggered else None
                }
                for webhook in self.webhooks.values()
            ]
        
        @self.app.delete("/webhooks/{webhook_id}")
        async def delete_webhook(
            webhook_id: str,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "webhooks:delete")
            
            # Check if webhook exists
            if webhook_id not in self.webhooks:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Webhook not found"
                )
            
            # Deactivate in database
            self.storage.execute(
                "UPDATE webhooks SET is_active = 0 WHERE id = ?",
                (webhook_id,)
            )
            
            # Remove from in-memory cache
            del self.webhooks[webhook_id]
            
            return {"status": "success"}
        
        # Analytics
        @self.app.get("/analytics/leads")
        async def get_lead_analytics(
            days: int = 30,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "analytics:read")
            
            since = datetime.now() - timedelta(days=days)
            
            # Get lead counts by source
            source_counts = {}
            for row in self.storage.query(
                "SELECT source, COUNT(*) as count FROM leads WHERE created_at >= ? GROUP BY source",
                (since.isoformat(),)
            ):
                source_counts[row['source'] or "Unknown"] = row['count']
            
            # Get lead counts by industry
            industry_counts = {}
            for row in self.storage.query(
                "SELECT industry, COUNT(*) as count FROM leads WHERE created_at >= ? GROUP BY industry",
                (since.isoformat(),)
            ):
                industry_counts[row['industry'] or "Unknown"] = row['count']
            
            # Get average lead score
            avg_score = self.storage.query(
                "SELECT AVG(score) as avg FROM leads WHERE created_at >= ?",
                (since.isoformat(),)
            ).fetchone()[0] or 0
            
            return {
                "total_leads": sum(source_counts.values()),
                "source_distribution": source_counts,
                "industry_distribution": industry_counts,
                "average_score": avg_score,
                "period_days": days
            }
        
        @self.app.get("/analytics/campaigns")
        async def get_campaign_analytics(
            days: int = 30,
            api_key: str = Security(self.api_key_header)
        ):
            self._check_permission(api_key, "analytics:read")
            
            since = datetime.now() - timedelta(days=days)
            
            # Get campaign performance
            campaigns = []
            for row in self.storage.query(
                """
                SELECT c.id, c.name, 
                       COUNT(e.id) as emails_sent,
                       SUM(CASE WHEN e.event_type = 'open' THEN 1 ELSE 0 END) as opens,
                       SUM(CASE WHEN e.event_type = 'click' THEN 1 ELSE 0 END) as clicks
                FROM campaigns c
                LEFT JOIN email_events e ON c.id = e.campaign_id AND e.timestamp >= ?
                GROUP BY c.id, c.name
                """,
                (since.isoformat(),)
            ):
                open_rate = (row['opens'] / row['emails_sent']) if row['emails_sent'] > 0 else 0
                click_rate = (row['clicks'] / row['emails_sent']) if row['emails_sent'] > 0 else 0
                
                campaigns.append({
                    "campaign_id": row['id'],
                    "name": row['name'],
                    "emails_sent": row['emails_sent'],
                    "opens": row['opens'],
                    "clicks": row['clicks'],
                    "open_rate": open_rate,
                    "click_rate": click_rate
                })
            
            return {
                "campaigns": campaigns,
                "period_days": days
            }
        
        # Middleware for authentication and rate limiting
        @self.app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip authentication for health check
            if request.url.path == "/health":
                return await call_next(request)
            
            # Get API key from header
            api_key = request.headers.get("X-API-Key")
            
            # Check API key
            if not api_key or api_key not in self.api_keys:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid API key"}
                )
            
            key_data = self.api_keys[api_key]
            
            # Check if key is expired
            if key_data.expires_at and key_data.expires_at < datetime.now():
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "API key expired"}
                )
            
            # Update last used
            key_data.last_used = datetime.now()
            self.storage.execute(
                "UPDATE api_keys SET last_used = ? WHERE key = ?",
                (key_data.last_used.isoformat(), api_key)
            )
            
            # Check rate limit
            endpoint = self._get_endpoint_category(request.url.path)
            rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
            
            # Get current minute
            now = datetime.now()
            minute_key = f"{api_key}:{now.strftime('%Y-%m-%d %H:%M')}"
            
            if minute_key not in self.request_counts:
                self.request_counts[minute_key] = 0
            
            self.request_counts[minute_key] += 1
            
            if self.request_counts[minute_key] > rate_limit:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded"}
                )
            
            # Process request
            start_time = datetime.now()
            response = await call_next(request)
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Log request
            self.storage.execute(
                """
                INSERT INTO api_logs 
                (api_key, endpoint, method, status_code, response_time, timestamp, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    api_key,
                    request.url.path,
                    request.method,
                    response.status_code,
                    response_time,
                    now.isoformat(),
                    request.client.host,
                    request.headers.get("user-agent", "")
                )
            )
            
            return response

    def _check_permission(self, api_key: str, permission: str):
        """Check if API key has required permission"""
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        key_data = self.api_keys[api_key]
        
        if permission not in key_data.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

    def _get_endpoint_category(self, path: str) -> str:
        """Get the category of an endpoint for rate limiting"""
        if path.startswith("/leads"):
            return "leads"
        elif path.startswith("/campaigns"):
            return "campaigns"
        elif path.startswith("/emails"):
            return "emails"
        elif path.startswith("/analytics"):
            return "analytics"
        else:
            return "default"

    def _trigger_webhooks(self, event: str, data: Dict):
        """Trigger webhooks for a specific event"""
        import aiohttp
        
        async def trigger_webhook(webhook: Webhook):
            try:
                # Prepare payload
                payload = {
                    "event": event,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add signature if secret is provided
                headers = {"Content-Type": "application/json"}
                if webhook.secret:
                    import hmac
                    import hashlib
                    
                    signature = hmac.new(
                        webhook.secret.encode(),
                        json.dumps(payload).encode(),
                        hashlib.sha256
                    ).hexdigest()
                    
                    headers["X-Webhook-Signature"] = f"sha256={signature}"
                
                # Send webhook
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook.url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            # Update last triggered
                            webhook.last_triggered = datetime.now()
                            self.storage.execute(
                                "UPDATE webhooks SET last_triggered = ? WHERE id = ?",
                                (webhook.last_triggered.isoformat(), webhook.id)
                            )
                        else:
                            self.logger.error(f"Webhook failed: {response.status}")
            except Exception as e:
                self.logger.error(f"Error triggering webhook: {str(e)}")
        
        # Find webhooks for this event
        for webhook in self.webhooks.values():
            if event in webhook.events:
                # Trigger webhook in background
                import asyncio
                asyncio.create_task(trigger_webhook(webhook))

    async def _send_emails_task(
        self,
        campaign_id: str,
        leads: List[Dict],
        template: Dict,
        personalization_level: str,
        schedule_at: Optional[datetime]
    ):
        """Background task to send emails"""
        # Wait until scheduled time if needed
        if schedule_at and schedule_at > datetime.now():
            await asyncio.sleep((schedule_at - datetime.now()).total_seconds())
        
        # Send emails
        for lead in leads:
            try:
                # Personalize template
                personalized_content = self.template_personalizer.personalize_template(
                    template['content'],
                    lead['id'],
                    lead,
                    level=self.template_personalizer.PersonalizationLevel(personalization_level)
                )
                
                # Send email
                await self.smtp_manager.send_email(
                    to_email=lead['email'],
                    subject=template['subject'],
                    content=personalized_content,
                    campaign_id=campaign_id,
                    lead_id=lead['id']
                )
                
                # Log email sent
                self.storage.execute(
                    """
                    INSERT INTO email_events 
                    (campaign_id, lead_id, event_type, timestamp)
                    VALUES (?, ?, 'sent', ?)
                    """,
                    (campaign_id, lead['id'], datetime.now().isoformat())
                )
            except Exception as e:
                self.logger.error(f"Error sending email to {lead['email']}: {str(e)}")
                
                # Log email failed
                self.storage.execute(
                    """
                    INSERT INTO email_events 
                    (campaign_id, lead_id, event_type, timestamp, metadata)
                    VALUES (?, ?, 'failed', ?, ?)
                    """,
                    (campaign_id, lead['id'], datetime.now().isoformat(), json.dumps({"error": str(e)}))
                )

    def get_app(self) -> FastAPI:
        """Get the FastAPI app instance"""
        return self.app