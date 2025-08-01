# modules/integrations/sync.py

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

import aiohttp
import pandas as pd
from aiohttp import ClientSession, ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from engine.SecureStorage import SecureStorage
from engine.license import LicenseManager
from ai.nlp import NLPProcessor


class SyncPlatform(Enum):
    NOTION = "notion"
    AIRTABLE = "airtable"
    SLACK = "slack"
    HUBSPOT = "hubspot"
    SALESFORCE = "salesforce"
    PIPEDRIVE = "pipedrive"
    ZOHO = "zoho"
    CUSTOM = "custom"


class SyncDirection(Enum):
    IMPORT = "import"  # From platform to DRN
    EXPORT = "export"  # From DRN to platform
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class SyncEntityType(Enum):
    LEADS = "leads"
    CAMPAIGNS = "campaigns"
    EMAILS = "emails"
    TEMPLATES = "templates"
    TAGS = "tags"
    ACTIVITIES = "activities"


@dataclass
class SyncConfig:
    id: str
    platform: SyncPlatform
    direction: SyncDirection
    entities: List[SyncEntityType]
    auth_config: Dict
    sync_interval: int = 3600  # seconds
    is_active: bool = True
    last_sync: Optional[datetime] = None
    next_sync: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SyncOperation:
    id: str
    config_id: str
    entity_type: SyncEntityType
    entity_id: str
    operation: str  # "create", "update", "delete"
    status: SyncStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SyncResult:
    config_id: str
    platform: SyncPlatform
    entity_type: SyncEntityType
    status: SyncStatus
    total_records: int
    success_count: int
    failed_count: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class SyncManager:
    def __init__(self, SecureStorage: SecureStorage, license_manager: LicenseManager, nlp_processor: NLPProcessor):
        self.SecureStorage = SecureStorage
        self.license_manager = license_manager
        self.nlp = nlp_processor
        self.logger = logging.getLogger("sync_manager")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load sync configurations
        self.sync_configs: Dict[str, SyncConfig] = {}
        self._load_sync_configs()
        
        # Active sync operations
        self.active_syncs: Dict[str, asyncio.Task] = {}
        
        # Rate limiting
        self.rate_limits = {
            SyncPlatform.NOTION: {"requests": 10, "period": 60},  # 10 requests per minute
            SyncPlatform.AIRTABLE: {"requests": 5, "period": 60},   # 5 requests per second
            SyncPlatform.SLACK: {"requests": 1, "period": 1},     # 1 request per second
            SyncPlatform.HUBSPOT: {"requests": 10, "period": 10},  # 10 requests per 10 seconds
            SyncPlatform.SALESFORCE: {"requests": 5, "period": 5}, # 5 requests per 5 seconds
            SyncPlatform.PIPEDRIVE: {"requests": 10, "period": 10}, # 10 requests per 10 seconds
            SyncPlatform.ZOHO: {"requests": 10, "period": 60},    # 10 requests per minute
            SyncPlatform.CUSTOM: {"requests": 10, "period": 60}   # 10 requests per minute
        }
        
        # Request tracking for rate limiting
        self.request_counts: Dict[str, List[float]] = {}
        
        # Platform-specific field mappings
        self.field_mappings = self._initialize_field_mappings()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS sync_configs (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            direction TEXT NOT NULL,
            entities TEXT NOT NULL,
            auth_config TEXT NOT NULL,
            sync_interval INTEGER DEFAULT 3600,
            is_active INTEGER DEFAULT 1,
            last_sync TEXT,
            next_sync TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS sync_operations (
            id TEXT PRIMARY KEY,
            config_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0,
            metadata TEXT,
            FOREIGN KEY (config_id) REFERENCES sync_configs (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS sync_results (
            id TEXT PRIMARY KEY,
            config_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            status TEXT NOT NULL,
            total_records INTEGER,
            success_count INTEGER,
            failed_count INTEGER,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            errors TEXT,
            metadata TEXT,
            FOREIGN KEY (config_id) REFERENCES sync_configs (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS sync_field_mappings (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            drn_field TEXT NOT NULL,
            platform_field TEXT NOT NULL,
            transform_type TEXT DEFAULT 'direct',
            transform_config TEXT,
            created_at TEXT
        )
        """)

    def _initialize_field_mappings(self) -> Dict:
        """Initialize default field mappings for each platform"""
        return {
            SyncPlatform.NOTION: {
                SyncEntityType.LEADS: {
                    "name": "Name",
                    "email": "Email",
                    "company": "Company",
                    "job_title": "Job Title",
                    "phone": "Phone",
                    "industry": "Industry",
                    "source": "Source",
                    "score": "Score"
                }
            },
            SyncPlatform.AIRTABLE: {
                SyncEntityType.LEADS: {
                    "name": "Name",
                    "email": "Email",
                    "company": "Company",
                    "job_title": "Job Title",
                    "phone": "Phone",
                    "industry": "Industry",
                    "source": "Source",
                    "score": "Score"
                }
            },
            SyncPlatform.HUBSPOT: {
                SyncEntityType.LEADS: {
                    "name": "firstname",
                    "email": "email",
                    "company": "company",
                    "job_title": "jobtitle",
                    "phone": "phone",
                    "industry": "industry",
                    "source": "source",
                    "score": "hs_lead_score"
                }
            },
            SyncPlatform.SALESFORCE: {
                SyncEntityType.LEADS: {
                    "name": "FirstName",
                    "email": "Email",
                    "company": "Company",
                    "job_title": "Title",
                    "phone": "Phone",
                    "industry": "Industry",
                    "source": "LeadSource",
                    "score": "Lead_Score__c"
                }
            },
            SyncPlatform.PIPEDRIVE: {
                SyncEntityType.LEADS: {
                    "name": "name",
                    "email": "email",
                    "company": "org_name",
                    "job_title": "job_title",
                    "phone": "phone",
                    "industry": "industry",
                    "source": "source",
                    "score": "score"
                }
            },
            SyncPlatform.ZOHO: {
                SyncEntityType.LEADS: {
                    "name": "First_Name",
                    "email": "Email",
                    "company": "Company",
                    "job_title": "Title",
                    "phone": "Phone",
                    "industry": "Industry",
                    "source": "Lead_Source",
                    "score": "Lead_Score"
                }
            }
        }

    def _load_sync_configs(self):
        """Load sync configurations from SecureStorage"""
        for row in self.SecureStorage.query("SELECT * FROM sync_configs WHERE is_active = 1"):
            config = SyncConfig(
                id=row['id'],
                platform=SyncPlatform(row['platform']),
                direction=SyncDirection(row['direction']),
                entities=[SyncEntityType(e) for e in json.loads(row['entities'])],
                auth_config=json.loads(row['auth_config']),
                sync_interval=row['sync_interval'],
                is_active=bool(row['is_active']),
                last_sync=datetime.fromisoformat(row['last_sync']) if row['last_sync'] else None,
                next_sync=datetime.fromisoformat(row['next_sync']) if row['next_sync'] else None,
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
            self.sync_configs[config.id] = config

    def create_sync_config(self, platform: SyncPlatform, direction: SyncDirection,
                          entities: List[SyncEntityType], auth_config: Dict,
                          sync_interval: int = 3600) -> SyncConfig:
        """Create a new sync configuration"""
        config_id = f"sync_{uuid.uuid4().hex}"
        
        config = SyncConfig(
            id=config_id,
            platform=platform,
            direction=direction,
            entities=entities,
            auth_config=auth_config,
            sync_interval=sync_interval
        )
        
        # Save to database
        self.SecureStorage.execute(
            """
            INSERT INTO sync_configs 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.id,
                config.platform.value,
                config.direction.value,
                json.dumps([e.value for e in config.entities]),
                json.dumps(config.auth_config),
                config.sync_interval,
                1 if config.is_active else 0,
                config.last_sync.isoformat() if config.last_sync else None,
                config.next_sync.isoformat() if config.next_sync else None,
                config.created_at.isoformat(),
                config.updated_at.isoformat()
            )
        )
        
        # Add to in-memory cache
        self.sync_configs[config_id] = config
        
        # Schedule initial sync
        if config.is_active:
            self._schedule_sync(config)
        
        self.logger.info(f"Created sync config: {config_id} for {platform.value}")
        return config

    def update_sync_config(self, config_id: str, **kwargs) -> Optional[SyncConfig]:
        """Update a sync configuration"""
        if config_id not in self.sync_configs:
            return None
        
        config = self.sync_configs[config_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.updated_at = datetime.now()
        
        # Update in database
        self.SecureStorage.execute(
            """
            UPDATE sync_configs 
            SET platform = ?, direction = ?, entities = ?, auth_config = ?, 
                sync_interval = ?, is_active = ?, last_sync = ?, next_sync = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                config.platform.value,
                config.direction.value,
                json.dumps([e.value for e in config.entities]),
                json.dumps(config.auth_config),
                config.sync_interval,
                1 if config.is_active else 0,
                config.last_sync.isoformat() if config.last_sync else None,
                config.next_sync.isoformat() if config.next_sync else None,
                config.updated_at.isoformat(),
                config_id
            )
        )
        
        # Reschedule if needed
        if config.is_active and config_id in self.active_syncs:
            # Cancel existing sync task
            self.active_syncs[config_id].cancel()
            del self.active_syncs[config_id]
            
            # Schedule new sync
            self._schedule_sync(config)
        
        self.logger.info(f"Updated sync config: {config_id}")
        return config

    def delete_sync_config(self, config_id: str) -> bool:
        """Delete a sync configuration"""
        if config_id not in self.sync_configs:
            return False
        
        # Cancel active sync if running
        if config_id in self.active_syncs:
            self.active_syncs[config_id].cancel()
            del self.active_syncs[config_id]
        
        # Deactivate in database
        self.SecureStorage.execute(
            "UPDATE sync_configs SET is_active = 0 WHERE id = ?",
            (config_id,)
        )
        
        # Remove from in-memory cache
        del self.sync_configs[config_id]
        
        self.logger.info(f"Deleted sync config: {config_id}")
        return True

    def _schedule_sync(self, config: SyncConfig):
        """Schedule a sync operation"""
        if not config.is_active:
            return
        
        # Calculate next sync time
        if config.last_sync:
            next_sync = config.last_sync + timedelta(seconds=config.sync_interval)
        else:
            next_sync = datetime.now()
        
        config.next_sync = next_sync
        
        # Update in database
        self.SecureStorage.execute(
            "UPDATE sync_configs SET next_sync = ? WHERE id = ?",
            (next_sync.isoformat(), config.id)
        )
        
        # Schedule task
        delay = (next_sync - datetime.now()).total_seconds()
        if delay < 0:
            delay = 0
        
        task = asyncio.create_task(self._run_sync_after_delay(config, delay))
        self.active_syncs[config.id] = task

    async def _run_sync_after_delay(self, config: SyncConfig, delay: float):
        """Run sync after a delay"""
        await asyncio.sleep(delay)
        await self.run_sync(config.id)

    async def run_sync(self, config_id: str) -> SyncResult:
        """Run a sync operation for a configuration"""
        if config_id not in self.sync_configs:
            raise ValueError(f"Sync config not found: {config_id}")
        
        config = self.sync_configs[config_id]
        
        if not config.is_active:
            raise ValueError(f"Sync config is not active: {config_id}")
        
        # Check if sync is already running
        if config_id in self.active_syncs and not self.active_syncs[config_id].done():
            raise ValueError(f"Sync is already running for config: {config_id}")
        
        # Create sync result
        result = SyncResult(
            config_id=config_id,
            platform=config.platform,
            entity_type=config.entities[0],  # For now, sync one entity type at a time
            status=SyncStatus.RUNNING,
            total_records=0,
            success_count=0,
            failed_count=0,
            started_at=datetime.now()
        )
        
        try:
            # Run sync based on direction
            if config.direction == SyncDirection.IMPORT:
                result = await self._import_data(config, result)
            elif config.direction == SyncDirection.EXPORT:
                result = await self._export_data(config, result)
            elif config.direction == SyncDirection.BIDIRECTIONAL:
                result = await self._bidirectional_sync(config, result)
            
            # Update config timestamps
            config.last_sync = datetime.now()
            config.next_sync = config.last_sync + timedelta(seconds=config.sync_interval)
            
            self.SecureStorage.execute(
                """
                UPDATE sync_configs 
                SET last_sync = ?, next_sync = ? 
                WHERE id = ?
                """,
                (
                    config.last_sync.isoformat(),
                    config.next_sync.isoformat(),
                    config_id
                )
            )
            
            # Schedule next sync
            self._schedule_sync(config)
            
        except Exception as e:
            result.status = SyncStatus.FAILED
            result.errors.append(str(e))
            self.logger.error(f"Sync failed for config {config_id}: {str(e)}")
        
        finally:
            result.completed_at = datetime.now()
            
            # Save result to database
            self._save_sync_result(result)
            
            # Remove from active syncs
            if config_id in self.active_syncs:
                del self.active_syncs[config_id]
        
        return result

    async def _import_data(self, config: SyncConfig, result: SyncResult) -> SyncResult:
        """Import data from platform to DRN"""
        platform = config.platform
        entity_type = config.entities[0]  # For now, sync one entity type at a time
        
        # Get platform-specific handler
        handler = self._get_platform_handler(platform)
        
        # Fetch data from platform
        platform_data = await handler.fetch_data(config.auth_config, entity_type)
        
        if not platform_data:
            result.status = SyncStatus.SUCCESS
            return result
        
        # Transform data
        transformed_data = self._transform_data(platform, entity_type, platform_data, "import")
        
        # Import to DRN
        success_count = 0
        failed_count = 0
        
        for record in transformed_data:
            try:
                if entity_type == SyncEntityType.LEADS:
                    await self._import_lead(record)
                    success_count += 1
                elif entity_type == SyncEntityType.CAMPAIGNS:
                    await self._import_campaign(record)
                    success_count += 1
                elif entity_type == SyncEntityType.EMAILS:
                    await self._import_email(record)
                    success_count += 1
                # Add more entity types as needed
            except Exception as e:
                failed_count += 1
                result.errors.append(f"Failed to import {record.get('id', 'unknown')}: {str(e)}")
        
        result.total_records = len(transformed_data)
        result.success_count = success_count
        result.failed_count = failed_count
        result.status = SyncStatus.SUCCESS if failed_count == 0 else SyncStatus.PARTIAL
        
        return result

    async def _export_data(self, config: SyncConfig, result: SyncResult) -> SyncResult:
        """Export data from DRN to platform"""
        platform = config.platform
        entity_type = config.entities[0]  # For now, sync one entity type at a time
        
        # Get platform-specific handler
        handler = self._get_platform_handler(platform)
        
        # Fetch data from DRN
        drn_data = await self._fetch_drn_data(entity_type)
        
        if not drn_data:
            result.status = SyncStatus.SUCCESS
            return result
        
        # Transform data
        transformed_data = self._transform_data(platform, entity_type, drn_data, "export")
        
        # Export to platform
        success_count = 0
        failed_count = 0
        
        for record in transformed_data:
            try:
                await handler.push_data(config.auth_config, entity_type, record)
                success_count += 1
            except Exception as e:
                failed_count += 1
                result.errors.append(f"Failed to export {record.get('id', 'unknown')}: {str(e)}")
        
        result.total_records = len(transformed_data)
        result.success_count = success_count
        result.failed_count = failed_count
        result.status = SyncStatus.SUCCESS if failed_count == 0 else SyncStatus.PARTIAL
        
        return result

    async def _bidirectional_sync(self, config: SyncConfig, result: SyncResult) -> SyncResult:
        """Perform bidirectional sync between DRN and platform"""
        # Import data from platform to DRN
        import_result = await self._import_data(config, result)
        
        # Export data from DRN to platform
        export_result = await self._export_data(config, result)
        
        # Combine results
        result.total_records = import_result.total_records + export_result.total_records
        result.success_count = import_result.success_count + export_result.success_count
        result.failed_count = import_result.failed_count + export_result.failed_count
        result.errors.extend(import_result.errors)
        result.errors.extend(export_result.errors)
        
        # Determine overall status
        if result.failed_count == 0:
            result.status = SyncStatus.SUCCESS
        elif result.success_count > 0:
            result.status = SyncStatus.PARTIAL
        else:
            result.status = SyncStatus.FAILED
        
        return result

    def _get_platform_handler(self, platform: SyncPlatform):
        """Get platform-specific handler"""
        if platform == SyncPlatform.NOTION:
            return NotionHandler(self.rate_limits, self.request_counts)
        elif platform == SyncPlatform.AIRTABLE:
            return AirtableHandler(self.rate_limits, self.request_counts)
        elif platform == SyncPlatform.SLACK:
            return SlackHandler(self.rate_limits, self.request_counts)
        elif platform == SyncPlatform.HUBSPOT:
            return HubSpotHandler(self.rate_limits, self.request_counts)
        elif platform == SyncPlatform.SALESFORCE:
            return SalesforceHandler(self.rate_limits, self.request_counts)
        elif platform == SyncPlatform.PIPEDRIVE:
            return PipedriveHandler(self.rate_limits, self.request_counts)
        elif platform == SyncPlatform.ZOHO:
            return ZohoHandler(self.rate_limits, self.request_counts)
        elif platform == SyncPlatform.CUSTOM:
            return CustomHandler(self.rate_limits, self.request_counts)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def _transform_data(self, platform: SyncPlatform, entity_type: SyncEntityType,
                        data: List[Dict], direction: str) -> List[Dict]:
        """Transform data between DRN and platform formats"""
        transformed = []
        
        # Get field mappings
        mappings = self._get_field_mappings(platform, entity_type, direction)
        
        for record in data:
            transformed_record = {}
            
            # Apply field mappings
            for drn_field, platform_field in mappings.items():
                if drn_field in record:
                    if direction == "import":
                        transformed_record[drn_field] = record[platform_field]
                    else:  # export
                        transformed_record[platform_field] = record[drn_field]
            
            # Apply transformations
            transformed_record = self._apply_transformations(
                platform, entity_type, transformed_record, direction
            )
            
            transformed.append(transformed_record)
        
        return transformed

    def _get_field_mappings(self, platform: SyncPlatform, entity_type: SyncEntityType, direction: str) -> Dict:
        """Get field mappings for a platform and entity type"""
        # Get default mappings
        default_mappings = self.field_mappings.get(platform, {}).get(entity_type, {})
        
        # Get custom mappings from database
        custom_mappings = {}
        for row in self.SecureStorage.query(
            """
            SELECT drn_field, platform_field, transform_type, transform_config 
            FROM sync_field_mappings 
            WHERE platform = ? AND entity_type = ?
            """,
            (platform.value, entity_type.value)
        ):
            custom_mappings[row['drn_field']] = row['platform_field']
        
        # Use custom mappings if available, otherwise use defaults
        mappings = custom_mappings if custom_mappings else default_mappings
        
        # Reverse mapping for export
        if direction == "export":
            return {v: k for k, v in mappings.items()}
        
        return mappings

    def _apply_transformations(self, platform: SyncPlatform, entity_type: SyncEntityType,
                              record: Dict, direction: str) -> Dict:
        """Apply transformations to record data"""
        # Get transformations from database
        transformations = {}
        for row in self.SecureStorage.query(
            """
            SELECT drn_field, transform_type, transform_config 
            FROM sync_field_mappings 
            WHERE platform = ? AND entity_type = ? AND transform_type != 'direct'
            """,
            (platform.value, entity_type.value)
        ):
            transformations[row['drn_field']] = {
                "type": row['transform_type'],
                "config": json.loads(row['transform_config']) if row['transform_config'] else {}
            }
        
        # Apply transformations
        for field, transform in transformations.items():
            if field in record:
                transform_type = transform["type"]
                transform_config = transform["config"]
                
                if transform_type == "date_format":
                    # Format date according to config
                    date_format = transform_config.get("format", "%Y-%m-%d")
                    if isinstance(record[field], str):
                        try:
                            record[field] = datetime.strptime(record[field], date_format).isoformat()
                        except ValueError:
                            pass
                elif transform_type == "value_map":
                    # Map values according to config
                    value_map = transform_config.get("map", {})
                    if record[field] in value_map:
                        record[field] = value_map[record[field]]
                elif transform_type == "concatenate":
                    # Concatenate multiple fields
                    fields = transform_config.get("fields", [])
                    separator = transform_config.get("separator", " ")
                    values = [record.get(f, "") for f in fields]
                    record[field] = separator.join(values)
                elif transform_type == "split":
                    # Split field into multiple fields
                    separator = transform_config.get("separator", " ")
                    fields = transform_config.get("fields", [])
                    value = str(record.get(field, ""))
                    parts = value.split(separator)
                    for i, f in enumerate(fields):
                        if i < len(parts):
                            record[f] = parts[i]
                    del record[field]
        
        return record

    async def _import_lead(self, lead_data: Dict):
        """Import a lead record into DRN"""
        lead_id = lead_data.get("id")
        
        # Check if lead already exists
        existing = self.SecureStorage.query(
            "SELECT id FROM leads WHERE id = ? OR email = ?",
            (lead_id, lead_data.get("email"))
        ).fetchone()
        
        if existing:
            # Update existing lead
            update_fields = []
            update_values = []
            
            for field, value in lead_data.items():
                if field != "id":
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
            
            if update_fields:
                update_values.append(existing['id'])
                self.SecureStorage.execute(
                    f"UPDATE leads SET {', '.join(update_fields)} WHERE id = ?",
                    update_values
                )
        else:
            # Create new lead
            lead_id = f"lead_{uuid.uuid4().hex}"
            self.SecureStorage.execute(
                """
                INSERT INTO leads 
                (id, name, email, company, job_title, phone, industry, source, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead_id,
                    lead_data.get("name"),
                    lead_data.get("email"),
                    lead_data.get("company"),
                    lead_data.get("job_title"),
                    lead_data.get("phone"),
                    lead_data.get("industry"),
                    lead_data.get("source"),
                    json.dumps(lead_data.get("metadata", {})),
                    datetime.now().isoformat()
                )
            )

    async def _import_campaign(self, campaign_data: Dict):
        """Import a campaign record into DRN"""
        campaign_id = campaign_data.get("id")
        
        # Check if campaign already exists
        existing = self.SecureStorage.query(
            "SELECT id FROM campaigns WHERE id = ?",
            (campaign_id,)
        ).fetchone()
        
        if existing:
            # Update existing campaign
            update_fields = []
            update_values = []
            
            for field, value in campaign_data.items():
                if field != "id":
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
            
            if update_fields:
                update_values.append(existing['id'])
                self.SecureStorage.execute(
                    f"UPDATE campaigns SET {', '.join(update_fields)} WHERE id = ?",
                    update_values
                )
        else:
            # Create new campaign
            campaign_id = f"campaign_{uuid.uuid4().hex}"
            self.SecureStorage.execute(
                """
                INSERT INTO campaigns 
                (id, name, description, template_id, target_audience, schedule, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    campaign_data.get("name"),
                    campaign_data.get("description"),
                    campaign_data.get("template_id"),
                    json.dumps(campaign_data.get("target_audience", {})),
                    json.dumps(campaign_data.get("schedule", {})),
                    json.dumps(campaign_data.get("metadata", {})),
                    datetime.now().isoformat()
                )
            )

    async def _import_email(self, email_data: Dict):
        """Import an email record into DRN"""
        email_id = email_data.get("id")
        
        # Check if email already exists
        existing = self.SecureStorage.query(
            "SELECT id FROM email_events WHERE id = ?",
            (email_id,)
        ).fetchone()
        
        if not existing:
            # Create new email event
            self.SecureStorage.execute(
                """
                INSERT INTO email_events 
                (id, campaign_id, lead_id, event_type, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    email_id,
                    email_data.get("campaign_id"),
                    email_data.get("lead_id"),
                    email_data.get("event_type", "sent"),
                    email_data.get("timestamp", datetime.now().isoformat()),
                    json.dumps(email_data.get("metadata", {}))
                )
            )

    async def _fetch_drn_data(self, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from DRN for export"""
        data = []
        
        if entity_type == SyncEntityType.LEADS:
            for row in self.SecureStorage.query("SELECT * FROM leads"):
                data.append({
                    "id": row['id'],
                    "name": row['name'],
                    "email": row['email'],
                    "company": row['company'],
                    "job_title": row['job_title'],
                    "phone": row['phone'],
                    "industry": row['industry'],
                    "source": row['source'],
                    "score": row['score'],
                    "metadata": json.loads(row['metadata']) if row['metadata'] else {}
                })
        elif entity_type == SyncEntityType.CAMPAIGNS:
            for row in self.SecureStorage.query("SELECT * FROM campaigns"):
                data.append({
                    "id": row['id'],
                    "name": row['name'],
                    "description": row['description'],
                    "template_id": row['template_id'],
                    "target_audience": json.loads(row['target_audience']) if row['target_audience'] else {},
                    "schedule": json.loads(row['schedule']) if row['schedule'] else {},
                    "metadata": json.loads(row['metadata']) if row['metadata'] else {}
                })
        elif entity_type == SyncEntityType.EMAILS:
            for row in self.SecureStorage.query("SELECT * FROM email_events"):
                data.append({
                    "id": row['id'],
                    "campaign_id": row['campaign_id'],
                    "lead_id": row['lead_id'],
                    "event_type": row['event_type'],
                    "timestamp": row['timestamp'],
                    "metadata": json.loads(row['metadata']) if row['metadata'] else {}
                })
        
        return data

    def _save_sync_result(self, result: SyncResult):
        """Save sync result to database"""
        self.SecureStorage.execute(
            """
            INSERT INTO sync_results 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"result_{uuid.uuid4().hex}",
                result.config_id,
                result.platform.value,
                result.entity_type.value,
                result.status.value,
                result.total_records,
                result.success_count,
                result.failed_count,
                result.started_at.isoformat(),
                result.completed_at.isoformat() if result.completed_at else None,
                json.dumps(result.errors),
                json.dumps(result.metadata)
            )
        )

    def get_sync_configs(self) -> List[SyncConfig]:
        """Get all sync configurations"""
        return list(self.sync_configs.values())

    def get_sync_results(self, config_id: str = None, days: int = 7) -> List[SyncResult]:
        """Get sync results"""
        results = []
        
        query = "SELECT * FROM sync_results WHERE started_at >= ?"
        params = [(datetime.now() - timedelta(days=days)).isoformat()]
        
        if config_id:
            query += " AND config_id = ?"
            params.append(config_id)
        
        query += " ORDER BY started_at DESC"
        
        for row in self.SecureStorage.query(query, params):
            result = SyncResult(
                config_id=row['config_id'],
                platform=SyncPlatform(row['platform']),
                entity_type=SyncEntityType(row['entity_type']),
                status=SyncStatus(row['status']),
                total_records=row['total_records'],
                success_count=row['success_count'],
                failed_count=row['failed_count'],
                started_at=datetime.fromisoformat(row['started_at']),
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                errors=json.loads(row['errors']) if row['errors'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            results.append(result)
        
        return results

    def get_sync_stats(self, days: int = 30) -> Dict:
        """Get sync statistics"""
        since = datetime.now() - timedelta(days=days)
        
        # Get sync counts by platform
        platform_counts = {platform.value: 0 for platform in SyncPlatform}
        total_syncs = 0
        
        for row in self.SecureStorage.query(
            "SELECT platform, COUNT(*) as count FROM sync_results WHERE started_at >= ? GROUP BY platform",
            (since.isoformat(),)
        ):
            platform_counts[row['platform']] = row['count']
            total_syncs += row['count']
        
        # Get success rate
        success_count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM sync_results WHERE status = 'success' AND started_at >= ?",
            (since.isoformat(),)
        ).fetchone()[0] or 0
        
        success_rate = (success_count / total_syncs) if total_syncs > 0 else 0
        
        # Get average records per sync
        avg_records = self.SecureStorage.query(
            "SELECT AVG(total_records) FROM sync_results WHERE started_at >= ?",
            (since.isoformat(),)
        ).fetchone()[0] or 0
        
        return {
            "total_syncs": total_syncs,
            "platform_distribution": platform_counts,
            "success_rate": success_rate,
            "average_records_per_sync": avg_records
        }

    def add_field_mapping(self, platform: SyncPlatform, entity_type: SyncEntityType,
                         drn_field: str, platform_field: str, transform_type: str = "direct",
                         transform_config: Dict = None):
        """Add a custom field mapping"""
        mapping_id = f"mapping_{uuid.uuid4().hex}"
        
        self.SecureStorage.execute(
            """
            INSERT INTO sync_field_mappings 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                mapping_id,
                platform.value,
                entity_type.value,
                drn_field,
                platform_field,
                transform_type,
                json.dumps(transform_config) if transform_config else None,
                datetime.now().isoformat()
            )
        )
        
        self.logger.info(f"Added field mapping: {drn_field} -> {platform_field} for {platform.value}")

    def remove_field_mapping(self, mapping_id: str):
        """Remove a field mapping"""
        self.SecureStorage.execute(
            "DELETE FROM sync_field_mappings WHERE id = ?",
            (mapping_id,)
        )
        
        self.logger.info(f"Removed field mapping: {mapping_id}")


# Platform-specific handlers
class BasePlatformHandler:
    def __init__(self, rate_limits: Dict, request_counts: Dict):
        self.rate_limits = rate_limits
        self.request_counts = request_counts
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    async def _make_request(self, method: str, url: str, auth_config: Dict, 
                           data: Dict = None, headers: Dict = None) -> Dict:
        """Make an HTTP request with rate limiting"""
        platform = self.__class__.__name__.lower().replace("handler", "")
        
        # Apply rate limiting
        await self._check_rate_limit(platform)
        
        # Prepare headers
        if headers is None:
            headers = {}
        
        # Add authentication headers
        if "api_key" in auth_config:
            headers["Authorization"] = f"Bearer {auth_config['api_key']}"
        elif "token" in auth_config:
            headers["Authorization"] = f"Token {auth_config['token']}"
        
        # Make request
        timeout = ClientTimeout(total=30)
        async with ClientSession(timeout=timeout) as session:
            if method.upper() == "GET":
                async with session.get(url, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == "POST":
                async with session.post(url, json=data, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == "PUT":
                async with session.put(url, json=data, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == "DELETE":
                async with session.delete(url, headers=headers) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

    async def _handle_response(self, response) -> Dict:
        """Handle HTTP response"""
        if response.status == 200:
            return await response.json()
        elif response.status == 429:
            # Rate limited
            retry_after = int(response.headers.get("Retry-After", 60))
            await asyncio.sleep(retry_after)
            raise Exception("Rate limited, retrying...")
        else:
            error_text = await response.text()
            raise Exception(f"HTTP {response.status}: {error_text}")

    async def _check_rate_limit(self, platform: str):
        """Check and apply rate limiting"""
        now = time.time()
        
        # Initialize request count tracking
        if platform not in self.request_counts:
            self.request_counts[platform] = []
        
        # Remove old requests
        period = self.rate_limits[platform]["period"]
        self.request_counts[platform] = [
            req_time for req_time in self.request_counts[platform]
            if now - req_time < period
        ]
        
        # Check if we've exceeded the limit
        if len(self.request_counts[platform]) >= self.rate_limits[platform]["requests"]:
            # Calculate wait time
            oldest_request = min(self.request_counts[platform])
            wait_time = period - (now - oldest_request)
            
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached for {platform}, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_counts[platform].append(now)

    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from platform (to be implemented by subclasses)"""
        raise NotImplementedError

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to platform (to be implemented by subclasses)"""
        raise NotImplementedError


class NotionHandler(BasePlatformHandler):
    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from Notion"""
        # Placeholder implementation
        self.logger.info("Fetching data from Notion")
        return []

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to Notion"""
        # Placeholder implementation
        self.logger.info("Pushing data to Notion")


class AirtableHandler(BasePlatformHandler):
    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from Airtable"""
        # Placeholder implementation
        self.logger.info("Fetching data from Airtable")
        return []

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to Airtable"""
        # Placeholder implementation
        self.logger.info("Pushing data to Airtable")


class SlackHandler(BasePlatformHandler):
    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from Slack"""
        # Placeholder implementation
        self.logger.info("Fetching data from Slack")
        return []

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to Slack"""
        # Placeholder implementation
        self.logger.info("Pushing data to Slack")


class HubSpotHandler(BasePlatformHandler):
    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from HubSpot"""
        # Placeholder implementation
        self.logger.info("Fetching data from HubSpot")
        return []

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to HubSpot"""
        # Placeholder implementation
        self.logger.info("Pushing data to HubSpot")


class SalesforceHandler(BasePlatformHandler):
    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from Salesforce"""
        # Placeholder implementation
        self.logger.info("Fetching data from Salesforce")
        return []

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to Salesforce"""
        # Placeholder implementation
        self.logger.info("Pushing data to Salesforce")


class PipedriveHandler(BasePlatformHandler):
    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from Pipedrive"""
        # Placeholder implementation
        self.logger.info("Fetching data from Pipedrive")
        return []

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to Pipedrive"""
        # Placeholder implementation
        self.logger.info("Pushing data to Pipedrive")


class ZohoHandler(BasePlatformHandler):
    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from Zoho"""
        # Placeholder implementation
        self.logger.info("Fetching data from Zoho")
        return []

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to Zoho"""
        # Placeholder implementation
        self.logger.info("Pushing data to Zoho")


class CustomHandler(BasePlatformHandler):
    async def fetch_data(self, auth_config: Dict, entity_type: SyncEntityType) -> List[Dict]:
        """Fetch data from custom platform"""
        # Placeholder implementation
        self.logger.info("Fetching data from custom platform")
        return []

    async def push_data(self, auth_config: Dict, entity_type: SyncEntityType, data: Dict):
        """Push data to custom platform"""
        # Placeholder implementation
        self.logger.info("Pushing data to custom platform")
