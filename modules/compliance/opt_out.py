# modules/compliance/opt_out.py

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from engine.storage import SecureStorage
from ai.nlp import NLPProcessor


class OptOutType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SMS = "sms"
    MAIL = "mail"
    ALL = "all"


class OptOutStatus(Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


class RegulationType(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    LGPD = "lgpd"
    PDPA = "pdpa"
    NONE = "none"


@dataclass
class OptOutRequest:
    id: str
    identifier: str  # email, phone number, etc.
    opt_out_type: OptOutType
    status: OptOutStatus
    source: str  # web, email, api, etc.
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class DataRetentionPolicy:
    id: str
    name: str
    description: str
    entity_type: str  # leads, campaigns, etc.
    retention_period_days: int
    regulation_type: RegulationType
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class OptOutManager:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.logger = logging.getLogger("opt_out_manager")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load opt-out requests and retention policies
        self.opt_out_requests: Dict[str, OptOutRequest] = {}
        self.retention_policies: Dict[str, DataRetentionPolicy] = {}
        self._load_opt_out_data()
        
        # Default retention periods (in days)
        self.default_retention_periods = {
            RegulationType.GDPR: 30,
            RegulationType.CCPA: 24,
            RegulationType.LGPD: 30,
            RegulationType.PDPA: 30,
            RegulationType.NONE: 365
        }

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS opt_out_requests (
            id TEXT PRIMARY KEY,
            identifier TEXT NOT NULL,
            opt_out_type TEXT NOT NULL,
            status TEXT NOT NULL,
            source TEXT NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            timestamp TEXT NOT NULL,
            processed_at TEXT,
            expires_at TEXT,
            metadata TEXT
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS data_retention_policies (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            entity_type TEXT NOT NULL,
            retention_period_days INTEGER NOT NULL,
            regulation_type TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS compliance_audit_log (
            id TEXT PRIMARY KEY,
            action TEXT NOT NULL,
            entity_type TEXT,
            entity_id TEXT,
            user_id TEXT,
            timestamp TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            user_agent TEXT
        )
        """)

    def _load_opt_out_data(self):
        """Load opt-out requests and retention policies from SecureStorage"""
        # Load opt-out requests
        for row in self.SecureStorage.query("SELECT * FROM opt_out_requests"):
            request = OptOutRequest(
                id=row['id'],
                identifier=row['identifier'],
                opt_out_type=OptOutType(row['opt_out_type']),
                status=OptOutStatus(row['status']),
                source=row['source'],
                ip_address=row['ip_address'],
                user_agent=row['user_agent'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                processed_at=datetime.fromisoformat(row['processed_at']) if row['processed_at'] else None,
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            self.opt_out_requests[request.id] = request
        
        # Load retention policies
        for row in self.SecureStorage.query("SELECT * FROM data_retention_policies WHERE is_active = 1"):
            policy = DataRetentionPolicy(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                entity_type=row['entity_type'],
                retention_period_days=row['retention_period_days'],
                regulation_type=RegulationType(row['regulation_type']),
                is_active=bool(row['is_active']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
            self.retention_policies[policy.id] = policy

    def create_opt_out_request(self, identifier: str, opt_out_type: OptOutType,
                             source: str, ip_address: str = None,
                             user_agent: str = None,
                             expires_in_days: int = None) -> OptOutRequest:
        """Create a new opt-out request"""
        request_id = f"optout_{uuid.uuid4().hex}"
        
        # Calculate expiration date if provided
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        request = OptOutRequest(
            id=request_id,
            identifier=identifier,
            opt_out_type=opt_out_type,
            status=OptOutStatus.PENDING,
            source=source,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
        
        # Save to database
        self._save_opt_out_request(request)
        
        # Process the opt-out request
        self.process_opt_out_request(request_id)
        
        self.logger.info(f"Created opt-out request: {request_id} for {identifier}")
        return request

    def _save_opt_out_request(self, request: OptOutRequest):
        """Save an opt-out request to database"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO opt_out_requests 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.id,
                request.identifier,
                request.opt_out_type.value,
                request.status.value,
                request.source,
                request.ip_address,
                request.user_agent,
                request.timestamp.isoformat(),
                request.processed_at.isoformat() if request.processed_at else None,
                request.expires_at.isoformat() if request.expires_at else None,
                json.dumps(request.metadata)
            )
        )
        
        # Update in-memory cache
        self.opt_out_requests[request.id] = request

    def process_opt_out_request(self, request_id: str) -> bool:
        """Process an opt-out request"""
        if request_id not in self.opt_out_requests:
            return False
        
        request = self.opt_out_requests[request_id]
        
        try:
            # Determine regulation type based on identifier
            regulation_type = self._determine_regulation_type(request.identifier)
            
            # Apply opt-out based on type
            if request.opt_out_type == OptOutType.EMAIL:
                self._apply_email_opt_out(request.identifier)
            elif request.opt_out_type == OptOutType.PHONE:
                self._apply_phone_opt_out(request.identifier)
            elif request.opt_out_type == OptOutType.SMS:
                self._apply_sms_opt_out(request.identifier)
            elif request.opt_out_type == OptOutType.MAIL:
                self._apply_mail_opt_out(request.identifier)
            elif request.opt_out_type == OptOutType.ALL:
                self._apply_all_opt_out(request.identifier)
            
            # Update request status
            request.status = OptOutStatus.PROCESSED
            request.processed_at = datetime.now()
            request.metadata["regulation_type"] = regulation_type.value
            
            self._save_opt_out_request(request)
            
            # Log compliance action
            self._log_compliance_action(
                "opt_out_processed",
                "opt_out_request",
                request_id,
                None,
                {
                    "identifier": request.identifier,
                    "opt_out_type": request.opt_out_type.value,
                    "regulation_type": regulation_type.value
                },
                request.ip_address,
                request.user_agent
            )
            
            self.logger.info(f"Processed opt-out request: {request_id}")
            return True
            
        except Exception as e:
            request.status = OptOutStatus.FAILED
            request.metadata["error"] = str(e)
            self._save_opt_out_request(request)
            
            self.logger.error(f"Failed to process opt-out request {request_id}: {str(e)}")
            return False

    def _determine_regulation_type(self, identifier: str) -> RegulationType:
        """Determine the applicable regulation type based on identifier"""
        # In a real implementation, this would use IP geolocation or other methods
        # For now, we'll use a simple heuristic based on email domain or phone country code
        
        if "@" in identifier:
            # Email address
            domain = identifier.split("@")[1].lower()
            
            # EU domains
            eu_domains = [".de", ".fr", ".it", ".es", ".nl", ".be", ".at", ".pt", ".se", ".dk", ".fi", ".pl", ".cz", ".hu", ".ro", ".bg", ".hr", ".si", ".sk", ".lt", ".lv", ".ee", ".mt", ".cy", ".lu"]
            if any(domain.endswith(tld) for tld in eu_domains):
                return RegulationType.GDPR
            
            # California domains (simplified)
            if ".ca." in domain or domain.endswith(".ca"):
                return RegulationType.CCPA
            
            # Brazil domains
            if ".br" in domain:
                return RegulationType.LGPD
            
            # Thailand domains
            if ".th" in domain:
                return RegulationType.PDPA
        
        elif "+" in identifier:
            # Phone number
            country_code = identifier.split("+")[1].split(" ")[0]
            
            # EU country codes
            eu_country_codes = ["43", "32", "359", "385", "353", "30", "49", "36", "352", "372", "358", "33", "350", "371", "423", "31", "48", "40", "421", "386", "34", "46", "44"]
            if country_code in eu_country_codes:
                return RegulationType.GDPR
            
            # US (California)
            if country_code == "1":
                return RegulationType.CCPA
            
            # Brazil
            if country_code == "55":
                return RegulationType.LGPD
            
            # Thailand
            if country_code == "66":
                return RegulationType.PDPA
        
        return RegulationType.NONE

    def _apply_email_opt_out(self, email: str):
        """Apply email opt-out"""
        # Update leads table
        self.SecureStorage.execute(
            "UPDATE leads SET email_opt_out = 1 WHERE email = ?",
            (email,)
        )
        
        # Update any campaign-specific opt-out tables
        self.SecureStorage.execute(
            "UPDATE email_campaign_recipients SET opt_out = 1 WHERE email = ?",
            (email,)
        )

    def _apply_phone_opt_out(self, phone: str):
        """Apply phone opt-out"""
        # Update leads table
        self.SecureStorage.execute(
            "UPDATE leads SET phone_opt_out = 1 WHERE phone = ?",
            (phone,)
        )

    def _apply_sms_opt_out(self, phone: str):
        """Apply SMS opt-out"""
        # Update leads table
        self.SecureStorage.execute(
            "UPDATE leads SET sms_opt_out = 1 WHERE phone = ?",
            (phone,)
        )

    def _apply_mail_opt_out(self, address: str):
        """Apply mail opt-out"""
        # In a real implementation, this would update a physical mail opt-out list
        # For now, we'll just log it
        self.logger.info(f"Applied mail opt-out for address: {address}")

    def _apply_all_opt_out(self, identifier: str):
        """Apply opt-out for all communication types"""
        if "@" in identifier:
            self._apply_email_opt_out(identifier)
        elif "+" in identifier or identifier.isdigit():
            self._apply_phone_opt_out(identifier)
            self._apply_sms_opt_out(identifier)
        else:
            # Assume it's an address
            self._apply_mail_opt_out(identifier)

    def is_opted_out(self, identifier: str, opt_out_type: OptOutType = None) -> bool:
        """Check if an identifier has opted out"""
        # Check specific opt-out type
        if opt_out_type:
            if opt_out_type == OptOutType.EMAIL:
                result = self.SecureStorage.query(
                    "SELECT 1 FROM leads WHERE email = ? AND email_opt_out = 1",
                    (identifier,)
                ).fetchone()
                return bool(result)
            elif opt_out_type == OptOutType.PHONE:
                result = self.SecureStorage.query(
                    "SELECT 1 FROM leads WHERE phone = ? AND phone_opt_out = 1",
                    (identifier,)
                ).fetchone()
                return bool(result)
            elif opt_out_type == OptOutType.SMS:
                result = self.SecureStorage.query(
                    "SELECT 1 FROM leads WHERE phone = ? AND sms_opt_out = 1",
                    (identifier,)
                ).fetchone()
                return bool(result)
        
        # Check for any opt-out
        result = self.SecureStorage.query(
            "SELECT 1 FROM opt_out_requests WHERE identifier = ? AND status = ?",
            (identifier, OptOutStatus.PROCESSED.value)
        ).fetchone()
        
        return bool(result)

    def get_opt_out_requests(self, identifier: str = None, opt_out_type: OptOutType = None,
                           status: OptOutStatus = None, days: int = 30) -> List[OptOutRequest]:
        """Get opt-out requests with filtering"""
        requests = []
        
        # Build query
        sql = "SELECT * FROM opt_out_requests WHERE 1=1"
        params = []
        
        # Add filters
        if identifier:
            sql += " AND identifier = ?"
            params.append(identifier)
        
        if opt_out_type:
            sql += " AND opt_out_type = ?"
            params.append(opt_out_type.value)
        
        if status:
            sql += " AND status = ?"
            params.append(status.value)
        
        # Add time filter
        if days:
            since = datetime.now() - timedelta(days=days)
            sql += " AND timestamp >= ?"
            params.append(since.isoformat())
        
        sql += " ORDER BY timestamp DESC"
        
        # Execute query
        for row in self.SecureStorage.query(sql, params):
            request = OptOutRequest(
                id=row['id'],
                identifier=row['identifier'],
                opt_out_type=OptOutType(row['opt_out_type']),
                status=OptOutStatus(row['status']),
                source=row['source'],
                ip_address=row['ip_address'],
                user_agent=row['user_agent'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                processed_at=datetime.fromisoformat(row['processed_at']) if row['processed_at'] else None,
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            requests.append(request)
        
        return requests

    def create_retention_policy(self, name: str, description: str, entity_type: str,
                              retention_period_days: int, regulation_type: RegulationType) -> DataRetentionPolicy:
        """Create a new data retention policy"""
        policy_id = f"policy_{uuid.uuid4().hex}"
        
        policy = DataRetentionPolicy(
            id=policy_id,
            name=name,
            description=description,
            entity_type=entity_type,
            retention_period_days=retention_period_days,
            regulation_type=regulation_type
        )
        
        # Save to database
        self.SecureStorage.execute(
            """
            INSERT INTO data_retention_policies 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                policy.id,
                policy.name,
                policy.description,
                policy.entity_type,
                policy.retention_period_days,
                policy.regulation_type.value,
                1 if policy.is_active else 0,
                policy.created_at.isoformat(),
                policy.updated_at.isoformat() if policy.updated_at else None
            )
        )
        
        # Add to in-memory cache
        self.retention_policies[policy.id] = policy
        
        self.logger.info(f"Created retention policy: {policy_id} - {name}")
        return policy

    def get_retention_policy(self, entity_type: str, regulation_type: RegulationType = None) -> Optional[DataRetentionPolicy]:
        """Get a retention policy for an entity type"""
        # Try to find a specific policy
        if regulation_type:
            for policy in self.retention_policies.values():
                if (policy.entity_type == entity_type and 
                    policy.regulation_type == regulation_type and 
                    policy.is_active):
                    return policy
        
        # Try to find a general policy for the entity type
        for policy in self.retention_policies.values():
            if policy.entity_type == entity_type and policy.is_active:
                return policy
        
        return None

    def apply_retention_policies(self):
        """Apply data retention policies to expire old data"""
        self.logger.info("Applying data retention policies")
        
        # Process each entity type
        entity_types = ["leads", "campaigns", "email_events", "opt_out_requests"]
        
        for entity_type in entity_types:
            # Get applicable policies
            policies = []
            for policy in self.retention_policies.values():
                if policy.entity_type == entity_type and policy.is_active:
                    policies.append(policy)
            
            if not policies:
                # Use default retention period
                retention_days = self.default_retention_periods.get(RegulationType.NONE, 365)
            else:
                # Use the shortest retention period from applicable policies
                retention_days = min(p.retention_period_days for policy in policies)
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Apply retention policy
            if entity_type == "leads":
                self._apply_lead_retention(cutoff_date)
            elif entity_type == "campaigns":
                self._apply_campaign_retention(cutoff_date)
            elif entity_type == "email_events":
                self._apply_email_event_retention(cutoff_date)
            elif entity_type == "opt_out_requests":
                self._apply_opt_out_request_retention(cutoff_date)
            
            self.logger.info(f"Applied {retention_days}-day retention policy for {entity_type}")

    def _apply_lead_retention(self, cutoff_date: datetime):
        """Apply retention policy to leads"""
        # Archive or delete old leads
        # In a real implementation, this might move data to an archive table
        # For now, we'll just log the action
        
        count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM leads WHERE created_at < ?",
            (cutoff_date.isoformat(),)
        ).fetchone()[0]
        
        if count > 0:
            self.logger.info(f"Retention policy would affect {count} leads created before {cutoff_date}")
            
            # In a real implementation:
            # self.SecureStorage.execute("DELETE FROM leads WHERE created_at < ?", (cutoff_date.isoformat(),))
            
            # Log compliance action
            self._log_compliance_action(
                "data_retention_applied",
                "leads",
                None,
                None,
                {
                    "cutoff_date": cutoff_date.isoformat(),
                    "affected_records": count
                },
                None,
                None
            )

    def _apply_campaign_retention(self, cutoff_date: datetime):
        """Apply retention policy to campaigns"""
        count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM campaigns WHERE created_at < ?",
            (cutoff_date.isoformat(),)
        ).fetchone()[0]
        
        if count > 0:
            self.logger.info(f"Retention policy would affect {count} campaigns created before {cutoff_date}")
            
            # Log compliance action
            self._log_compliance_action(
                "data_retention_applied",
                "campaigns",
                None,
                None,
                {
                    "cutoff_date": cutoff_date.isoformat(),
                    "affected_records": count
                },
                None,
                None
            )

    def _apply_email_event_retention(self, cutoff_date: datetime):
        """Apply retention policy to email events"""
        count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM email_events WHERE timestamp < ?",
            (cutoff_date.isoformat(),)
        ).fetchone()[0]
        
        if count > 0:
            self.logger.info(f"Retention policy would affect {count} email events before {cutoff_date}")
            
            # Log compliance action
            self._log_compliance_action(
                "data_retention_applied",
                "email_events",
                None,
                None,
                {
                    "cutoff_date": cutoff_date.isoformat(),
                    "affected_records": count
                },
                None,
                None
            )

    def _apply_opt_out_request_retention(self, cutoff_date: datetime):
        """Apply retention policy to opt-out requests"""
        count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM opt_out_requests WHERE timestamp < ?",
            (cutoff_date.isoformat(),)
        ).fetchone()[0]
        
        if count > 0:
            self.logger.info(f"Retention policy would affect {count} opt-out requests before {cutoff_date}")
            
            # Log compliance action
            self._log_compliance_action(
                "data_retention_applied",
                "opt_out_requests",
                None,
                None,
                {
                    "cutoff_date": cutoff_date.isoformat(),
                    "affected_records": count
                },
                None,
                None
            )

    def _log_compliance_action(self, action: str, entity_type: str, entity_id: str,
                             user_id: str, details: Dict, ip_address: str = None,
                             user_agent: str = None):
        """Log a compliance action to the audit log"""
        log_id = f"log_{uuid.uuid4().hex}"
        
        self.SecureStorage.execute(
            """
            INSERT INTO compliance_audit_log 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                action,
                entity_type,
                entity_id,
                user_id,
                datetime.now().isoformat(),
                json.dumps(details),
                ip_address,
                user_agent
            )
        )

    def get_compliance_audit_log(self, entity_type: str = None, days: int = 30) -> List[Dict]:
        """Get compliance audit log entries"""
        logs = []
        
        # Build query
        sql = "SELECT * FROM compliance_audit_log WHERE 1=1"
        params = []
        
        # Add filters
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        
        # Add time filter
        if days:
            since = datetime.now() - timedelta(days=days)
            sql += " AND timestamp >= ?"
            params.append(since.isoformat())
        
        sql += " ORDER BY timestamp DESC"
        
        # Execute query
        for row in self.SecureStorage.query(sql, params):
            log = {
                "id": row['id'],
                "action": row['action'],
                "entity_type": row['entity_type'],
                "entity_id": row['entity_id'],
                "user_id": row['user_id'],
                "timestamp": row['timestamp'],
                "details": json.loads(row['details']) if row['details'] else {},
                "ip_address": row['ip_address'],
                "user_agent": row['user_agent']
            }
            logs.append(log)
        
        return logs

    def generate_compliance_report(self, days: int = 30) -> Dict:
        """Generate a compliance report"""
        since = datetime.now() - timedelta(days=days)
        
        # Get opt-out statistics
        opt_out_stats = {}
        for opt_out_type in OptOutType:
            count = self.SecureStorage.query(
                "SELECT COUNT(*) FROM opt_out_requests WHERE opt_out_type = ? AND timestamp >= ?",
                (opt_out_type.value, since.isoformat())
            ).fetchone()[0]
            opt_out_stats[opt_out_type.value] = count
        
        # Get retention statistics
        retention_stats = {}
        for entity_type in ["leads", "campaigns", "email_events"]:
            policy = self.get_retention_policy(entity_type)
            if policy:
                retention_stats[entity_type] = {
                    "policy_name": policy.name,
                    "retention_days": policy.retention_period_days,
                    "regulation": policy.regulation_type.value
                }
        
        # Get audit log statistics
        audit_stats = {}
        for action in ["opt_out_processed", "data_retention_applied"]:
            count = self.SecureStorage.query(
                "SELECT COUNT(*) FROM compliance_audit_log WHERE action = ? AND timestamp >= ?",
                (action, since.isoformat())
            ).fetchone()[0]
            audit_stats[action] = count
        
        return {
            "report_period_days": days,
            "opt_out_statistics": opt_out_stats,
            "retention_policies": retention_stats,
            "audit_statistics": audit_stats,
            "generated_at": datetime.now().isoformat()
        }

    def export_opt_out_list(self, format: str = "csv") -> str:
        """Export the opt-out list in the specified format"""
        # Get all opted-out identifiers
        opted_out = set()
        
        for row in self.SecureStorage.query("SELECT identifier FROM opt_out_requests WHERE status = ?", (OptOutStatus.PROCESSED.value,)):
            opted_out.add(row['identifier'])
        
        # Also check leads table for opt-out flags
        for row in self.SecureStorage.query("SELECT email FROM leads WHERE email_opt_out = 1"):
            opted_out.add(row['email'])
        
        for row in self.SecureStorage.query("SELECT phone FROM leads WHERE phone_opt_out = 1"):
            opted_out.add(row['phone'])
        
        for row in self.SecureStorage.query("SELECT phone FROM leads WHERE sms_opt_out = 1"):
            opted_out.add(row['phone'])
        
        # Prepare data
        data = [{"identifier": identifier} for identifier in opted_out]
        
        if format.lower() == "csv":
            import pandas as pd
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        elif format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def process_expired_opt_out_requests(self):
        """Process expired opt-out requests"""
        now = datetime.now()
        
        # Find expired requests
        expired_requests = []
        for request in self.opt_out_requests.values():
            if (request.expires_at and 
                request.expires_at < now and 
                request.status == OptOutStatus.PROCESSED):
                expired_requests.append(request)
        
        # Process each expired request
        for request in expired_requests:
            request.status = OptOutStatus.EXPIRED
            self._save_opt_out_request(request)
            
            # Log compliance action
            self._log_compliance_action(
                "opt_out_expired",
                "opt_out_request",
                request.id,
                None,
                {
                    "identifier": request.identifier,
                    "opt_out_type": request.opt_out_type.value,
                    "expired_at": request.expires_at.isoformat()
                },
                None,
                None
            )
            
            # Remove opt-out flags
            if request.opt_out_type == OptOutType.EMAIL:
                self.SecureStorage.execute(
                    "UPDATE leads SET email_opt_out = 0 WHERE email = ?",
                    (request.identifier,)
                )
            elif request.opt_out_type == OptOutType.PHONE:
                self.SecureStorage.execute(
                    "UPDATE leads SET phone_opt_out = 0 WHERE phone = ?",
                    (request.identifier,)
                )
            elif request.opt_out_type == OptOutType.SMS:
                self.SecureStorage.execute(
                    "UPDATE leads SET sms_opt_out = 0 WHERE phone = ?",
                    (request.identifier,)
                )
        
        if expired_requests:
            self.logger.info(f"Processed {len(expired_requests)} expired opt-out requests")
