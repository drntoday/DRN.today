# modules/compliance/retention.py

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from engine.SecureStorage import SecureStorage
from ai.nlp import NLPProcessor


class RetentionAction(Enum):
    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    FLAG = "flag"


class RegulationType(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    LGPD = "lgpd"
    PDPA = "pdpa"
    NONE = "none"


class RetentionStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RetentionPolicy:
    id: str
    name: str
    description: str
    entity_type: str  # leads, campaigns, email_events, etc.
    retention_period_days: int
    action: RetentionAction
    regulation_type: RegulationType
    conditions: Dict = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    last_applied: Optional[datetime] = None


@dataclass
class RetentionJob:
    id: str
    policy_id: str
    status: RetentionStatus
    scheduled_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    affected_records: int = 0
    processed_records: int = 0
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetentionAudit:
    id: str
    policy_id: str
    job_id: str
    action: str
    entity_type: str
    entity_id: str
    timestamp: datetime
    details: Dict = field(default_factory=dict)
    user_id: Optional[str] = None


class DataRetentionManager:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.logger = logging.getLogger("data_retention_manager")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load retention policies
        self.policies: Dict[str, RetentionPolicy] = {}
        self._load_policies()
        
        # Active retention jobs
        self.active_jobs: Dict[str, RetentionJob] = {}
        self._load_active_jobs()
        
        # Start background scheduler
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Default retention periods by regulation
        self.default_retention_periods = {
            RegulationType.GDPR: {
                "leads": 30,
                "campaigns": 365,
                "email_events": 30,
                "opt_out_requests": 1825  # 5 years
            },
            RegulationType.CCPA: {
                "leads": 24,
                "campaigns": 365,
                "email_events": 24,
                "opt_out_requests": 1825
            },
            RegulationType.LGPD: {
                "leads": 30,
                "campaigns": 365,
                "email_events": 30,
                "opt_out_requests": 1825
            },
            RegulationType.PDPA: {
                "leads": 30,
                "campaigns": 365,
                "email_events": 30,
                "opt_out_requests": 1825
            },
            RegulationType.NONE: {
                "leads": 365,
                "campaigns": 730,
                "email_events": 90,
                "opt_out_requests": 2555  # 7 years
            }
        }

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS retention_policies (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            entity_type TEXT NOT NULL,
            retention_period_days INTEGER NOT NULL,
            action TEXT NOT NULL,
            regulation_type TEXT NOT NULL,
            conditions TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            last_applied TEXT
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS retention_jobs (
            id TEXT PRIMARY KEY,
            policy_id TEXT NOT NULL,
            status TEXT NOT NULL,
            scheduled_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            affected_records INTEGER DEFAULT 0,
            processed_records INTEGER DEFAULT 0,
            error_message TEXT,
            metadata TEXT,
            FOREIGN KEY (policy_id) REFERENCES retention_policies (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS retention_audit (
            id TEXT PRIMARY KEY,
            policy_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT,
            user_id TEXT,
            FOREIGN KEY (policy_id) REFERENCES retention_policies (id),
            FOREIGN KEY (job_id) REFERENCES retention_jobs (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS data_archive (
            id TEXT PRIMARY KEY,
            original_table TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            data TEXT NOT NULL,
            archived_at TEXT NOT NULL,
            retention_policy_id TEXT,
            scheduled_deletion_at TEXT,
            FOREIGN KEY (retention_policy_id) REFERENCES retention_policies (id)
        )
        """)

    def _load_policies(self):
        """Load retention policies from SecureStorage"""
        for row in self.SecureStorage.query("SELECT * FROM retention_policies WHERE is_active = 1"):
            policy = RetentionPolicy(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                entity_type=row['entity_type'],
                retention_period_days=row['retention_period_days'],
                action=RetentionAction(row['action']),
                regulation_type=RegulationType(row['regulation_type']),
                conditions=json.loads(row['conditions']) if row['conditions'] else {},
                is_active=bool(row['is_active']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                last_applied=datetime.fromisoformat(row['last_applied']) if row['last_applied'] else None
            )
            self.policies[policy.id] = policy

    def _load_active_jobs(self):
        """Load active retention jobs from SecureStorage"""
        for row in self.SecureStorage.query("SELECT * FROM retention_jobs WHERE status IN (?, ?)", (RetentionStatus.PROCESSING.value, RetentionStatus.ACTIVE.value)):
            job = RetentionJob(
                id=row['id'],
                policy_id=row['policy_id'],
                status=RetentionStatus(row['status']),
                scheduled_at=datetime.fromisoformat(row['scheduled_at']),
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                affected_records=row['affected_records'],
                processed_records=row['processed_records'],
                error_message=row['error_message'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            self.active_jobs[job.id] = job

    def create_policy(self, name: str, description: str, entity_type: str,
                    retention_period_days: int, action: RetentionAction,
                    regulation_type: RegulationType, conditions: Dict = None) -> RetentionPolicy:
        """Create a new retention policy"""
        policy_id = f"policy_{uuid.uuid4().hex}"
        
        policy = RetentionPolicy(
            id=policy_id,
            name=name,
            description=description,
            entity_type=entity_type,
            retention_period_days=retention_period_days,
            action=action,
            regulation_type=regulation_type,
            conditions=conditions or {}
        )
        
        # Save to database
        self._save_policy(policy)
        
        self.logger.info(f"Created retention policy: {policy_id} - {name}")
        return policy

    def _save_policy(self, policy: RetentionPolicy):
        """Save a retention policy to database"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO retention_policies 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                policy.id,
                policy.name,
                policy.description,
                policy.entity_type,
                policy.retention_period_days,
                policy.action.value,
                policy.regulation_type.value,
                json.dumps(policy.conditions),
                1 if policy.is_active else 0,
                policy.created_at.isoformat(),
                policy.updated_at.isoformat() if policy.updated_at else None,
                policy.last_applied.isoformat() if policy.last_applied else None
            )
        )
        
        # Update in-memory cache
        self.policies[policy.id] = policy

    def update_policy(self, policy_id: str, **kwargs) -> Optional[RetentionPolicy]:
        """Update a retention policy"""
        if policy_id not in self.policies:
            return None
        
        policy = self.policies[policy_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        policy.updated_at = datetime.now()
        
        # Save to database
        self._save_policy(policy)
        
        self.logger.info(f"Updated retention policy: {policy_id}")
        return policy

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a retention policy"""
        if policy_id not in self.policies:
            return False
        
        # Deactivate in database
        self.SecureStorage.execute(
            "UPDATE retention_policies SET is_active = 0 WHERE id = ?",
            (policy_id,)
        )
        
        # Remove from in-memory cache
        del self.policies[policy_id]
        
        self.logger.info(f"Deleted retention policy: {policy_id}")
        return True

    def get_policies(self, entity_type: str = None, regulation_type: RegulationType = None) -> List[RetentionPolicy]:
        """Get retention policies with optional filtering"""
        policies = []
        
        for policy in self.policies.values():
            if entity_type and policy.entity_type != entity_type:
                continue
            
            if regulation_type and policy.regulation_type != regulation_type:
                continue
            
            policies.append(policy)
        
        return policies

    def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get a retention policy by ID"""
        return self.policies.get(policy_id)

    def schedule_retention_job(self, policy_id: str, run_at: datetime = None) -> RetentionJob:
        """Schedule a retention job for a policy"""
        if policy_id not in self.policies:
            raise ValueError(f"Policy not found: {policy_id}")
        
        policy = self.policies[policy_id]
        
        # Default to run now if not specified
        if not run_at:
            run_at = datetime.now()
        
        job_id = f"job_{uuid.uuid4().hex}"
        
        job = RetentionJob(
            id=job_id,
            policy_id=policy_id,
            status=RetentionStatus.ACTIVE,
            scheduled_at=run_at
        )
        
        # Save to database
        self.SecureStorage.execute(
            """
            INSERT INTO retention_jobs 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.policy_id,
                job.status.value,
                job.scheduled_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.affected_records,
                job.processed_records,
                job.error_message,
                json.dumps(job.metadata)
            )
        )
        
        # Add to active jobs
        self.active_jobs[job.id] = job
        
        self.logger.info(f"Scheduled retention job: {job_id} for policy {policy_id}")
        return job

    def _scheduler_loop(self):
        """Background scheduler loop"""
        while True:
            try:
                # Check for jobs that need to run
                now = datetime.now()
                jobs_to_run = []
                
                for job_id, job in list(self.active_jobs.items()):
                    if (job.status == RetentionStatus.ACTIVE and 
                        job.scheduled_at <= now):
                        jobs_to_run.append(job)
                
                # Run jobs
                for job in jobs_to_run:
                    try:
                        self._run_retention_job(job.id)
                    except Exception as e:
                        self.logger.error(f"Error running retention job {job.id}: {str(e)}")
                
                # Sleep for a minute
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in retention scheduler: {str(e)}")
                time.sleep(60)

    def _run_retention_job(self, job_id: str):
        """Run a retention job"""
        if job_id not in self.active_jobs:
            return
        
        job = self.active_jobs[job_id]
        
        # Get policy
        policy = self.policies.get(job.policy_id)
        if not policy:
            job.status = RetentionStatus.FAILED
            job.error_message = "Policy not found"
            self._save_job(job)
            return
        
        try:
            # Update job status
            job.status = RetentionStatus.PROCESSING
            job.started_at = datetime.now()
            self._save_job(job)
            
            # Get affected records
            cutoff_date = datetime.now() - timedelta(days=policy.retention_period_days)
            affected_records = self._get_affected_records(policy, cutoff_date)
            job.affected_records = len(affected_records)
            
            # Process each record
            processed_count = 0
            for record in affected_records:
                try:
                    self._process_record(policy, record, job)
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing record {record.get('id', 'unknown')}: {str(e)}")
            
            # Update job
            job.processed_records = processed_count
            job.status = RetentionStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Update policy
            policy.last_applied = datetime.now()
            self._save_policy(policy)
            
            self.logger.info(f"Completed retention job {job_id}: processed {processed_count}/{job.affected_records} records")
            
        except Exception as e:
            job.status = RetentionStatus.FAILED
            job.error_message = str(e)
            self.logger.error(f"Failed to run retention job {job_id}: {str(e)}")
        
        finally:
            self._save_job(job)
            
            # Remove from active jobs if completed
            if job.status in [RetentionStatus.COMPLETED, RetentionStatus.FAILED]:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]

    def _save_job(self, job: RetentionJob):
        """Save a retention job to database"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO retention_jobs 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.policy_id,
                job.status.value,
                job.scheduled_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.affected_records,
                job.processed_records,
                job.error_message,
                json.dumps(job.metadata)
            )
        )
        
        # Update in-memory cache
        self.active_jobs[job.id] = job

    def _get_affected_records(self, policy: RetentionPolicy, cutoff_date: datetime) -> List[Dict]:
        """Get records affected by a retention policy"""
        records = []
        
        if policy.entity_type == "leads":
            for row in self.SecureStorage.query(
                "SELECT id, name, email, company, created_at FROM leads WHERE created_at < ?",
                (cutoff_date.isoformat(),)
            ):
                records.append({
                    "id": row['id'],
                    "name": row['name'],
                    "email": row['email'],
                    "company": row['company'],
                    "created_at": row['created_at']
                })
        
        elif policy.entity_type == "campaigns":
            for row in self.SecureStorage.query(
                "SELECT id, name, created_at FROM campaigns WHERE created_at < ?",
                (cutoff_date.isoformat(),)
            ):
                records.append({
                    "id": row['id'],
                    "name": row['name'],
                    "created_at": row['created_at']
                })
        
        elif policy.entity_type == "email_events":
            for row in self.SecureStorage.query(
                "SELECT id, campaign_id, lead_id, event_type, timestamp FROM email_events WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            ):
                records.append({
                    "id": row['id'],
                    "campaign_id": row['campaign_id'],
                    "lead_id": row['lead_id'],
                    "event_type": row['event_type'],
                    "timestamp": row['timestamp']
                })
        
        elif policy.entity_type == "opt_out_requests":
            for row in self.SecureStorage.query(
                "SELECT id, identifier, opt_out_type, timestamp FROM opt_out_requests WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            ):
                records.append({
                    "id": row['id'],
                    "identifier": row['identifier'],
                    "opt_out_type": row['opt_out_type'],
                    "timestamp": row['timestamp']
                })
        
        return records

    def _process_record(self, policy: RetentionPolicy, record: Dict, job: RetentionJob):
        """Process a single record according to the retention policy"""
        # Check conditions
        if not self._check_conditions(policy, record):
            return
        
        # Apply action
        if policy.action == RetentionAction.DELETE:
            self._delete_record(policy, record, job)
        elif policy.action == RetentionAction.ARCHIVE:
            self._archive_record(policy, record, job)
        elif policy.action == RetentionAction.ANONYMIZE:
            self._anonymize_record(policy, record, job)
        elif policy.action == RetentionAction.FLAG:
            self._flag_record(policy, record, job)

    def _check_conditions(self, policy: RetentionPolicy, record: Dict) -> bool:
        """Check if a record meets the policy conditions"""
        conditions = policy.conditions
        
        # Check field conditions
        if "fields" in conditions:
            for field, condition in conditions["fields"].items():
                if field not in record:
                    return False
                
                if "equals" in condition:
                    if record[field] != condition["equals"]:
                        return False
                
                if "contains" in condition:
                    if condition["contains"] not in str(record[field]):
                        return False
                
                if "not_contains" in condition:
                    if condition["not_contains"] in str(record[field]):
                        return False
        
        # Check custom conditions
        if "custom" in conditions:
            for condition in conditions["custom"]:
                if condition["type"] == "sql":
                    # Execute custom SQL condition
                    result = self.SecureStorage.query(
                        condition["query"],
                        (record["id"],)
                    ).fetchone()
                    
                    if not result or not result[0]:
                        return False
        
        return True

    def _delete_record(self, policy: RetentionPolicy, record: Dict, job: RetentionJob):
        """Delete a record"""
        entity_id = record["id"]
        
        if policy.entity_type == "leads":
            self.SecureStorage.execute("DELETE FROM leads WHERE id = ?", (entity_id,))
        elif policy.entity_type == "campaigns":
            self.SecureStorage.execute("DELETE FROM campaigns WHERE id = ?", (entity_id,))
        elif policy.entity_type == "email_events":
            self.SecureStorage.execute("DELETE FROM email_events WHERE id = ?", (entity_id,))
        elif policy.entity_type == "opt_out_requests":
            self.SecureStorage.execute("DELETE FROM opt_out_requests WHERE id = ?", (entity_id,))
        
        # Log audit
        self._log_audit(
            policy.id,
            job.id,
            "delete",
            policy.entity_type,
            entity_id,
            {"policy_name": policy.name}
        )

    def _archive_record(self, policy: RetentionPolicy, record: Dict, job: RetentionJob):
        """Archive a record"""
        entity_id = record["id"]
        
        # Get record data
        if policy.entity_type == "leads":
            record_data = self.SecureStorage.query(
                "SELECT * FROM leads WHERE id = ?",
                (entity_id,)
            ).fetchone()
        elif policy.entity_type == "campaigns":
            record_data = self.SecureStorage.query(
                "SELECT * FROM campaigns WHERE id = ?",
                (entity_id,)
            ).fetchone()
        elif policy.entity_type == "email_events":
            record_data = self.SecureStorage.query(
                "SELECT * FROM email_events WHERE id = ?",
                (entity_id,)
            ).fetchone()
        elif policy.entity_type == "opt_out_requests":
            record_data = self.SecureStorage.query(
                "SELECT * FROM opt_out_requests WHERE id = ?",
                (entity_id,)
            ).fetchone()
        
        if not record_data:
            return
        
        # Convert to dict
        record_dict = dict(record_data)
        
        # Calculate scheduled deletion date (e.g., 7 years from now)
        scheduled_deletion = datetime.now() + timedelta(days=2555)
        
        # Archive record
        archive_id = f"archive_{uuid.uuid4().hex}"
        self.SecureStorage.execute(
            """
            INSERT INTO data_archive 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                archive_id,
                policy.entity_type,
                entity_id,
                json.dumps(record_dict),
                datetime.now().isoformat(),
                policy.id,
                scheduled_deletion.isoformat()
            )
        )
        
        # Delete original record
        self._delete_record(policy, record, job)
        
        # Log audit
        self._log_audit(
            policy.id,
            job.id,
            "archive",
            policy.entity_type,
            entity_id,
            {
                "policy_name": policy.name,
                "archive_id": archive_id,
                "scheduled_deletion": scheduled_deletion.isoformat()
            }
        )

    def _anonymize_record(self, policy: RetentionPolicy, record: Dict, job: RetentionJob):
        """Anonymize a record"""
        entity_id = record["id"]
        
        if policy.entity_type == "leads":
            # Anonymize PII fields
            self.SecureStorage.execute(
                """
                UPDATE leads 
                SET name = ?, email = ?, phone = ?, company = ? 
                WHERE id = ?
                """,
                (
                    f"Anonymous {entity_id[:8]}",
                    f"anon-{entity_id[:8]}@anon.com",
                    None,
                    None,
                    entity_id
                )
            )
        
        # Log audit
        self._log_audit(
            policy.id,
            job.id,
            "anonymize",
            policy.entity_type,
            entity_id,
            {"policy_name": policy.name}
        )

    def _flag_record(self, policy: RetentionPolicy, record: Dict, job: RetentionJob):
        """Flag a record for review"""
        entity_id = record["id"]
        
        # Add flag metadata
        if "metadata" not in record:
            record["metadata"] = {}
        
        record["metadata"]["retention_flag"] = {
            "policy_id": policy.id,
            "policy_name": policy.name,
            "flagged_at": datetime.now().isoformat()
        }
        
        # Update record with flag
        if policy.entity_type == "leads":
            self.SecureStorage.execute(
                "UPDATE leads SET metadata = ? WHERE id = ?",
                (json.dumps(record["metadata"]), entity_id)
            )
        elif policy.entity_type == "campaigns":
            self.SecureStorage.execute(
                "UPDATE campaigns SET metadata = ? WHERE id = ?",
                (json.dumps(record["metadata"]), entity_id)
            )
        
        # Log audit
        self._log_audit(
            policy.id,
            job.id,
            "flag",
            policy.entity_type,
            entity_id,
            {"policy_name": policy.name}
        )

    def _log_audit(self, policy_id: str, job_id: str, action: str, 
                 entity_type: str, entity_id: str, details: Dict):
        """Log a retention audit entry"""
        audit_id = f"audit_{uuid.uuid4().hex}"
        
        self.SecureStorage.execute(
            """
            INSERT INTO retention_audit 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                policy_id,
                job_id,
                action,
                entity_type,
                entity_id,
                datetime.now().isoformat(),
                json.dumps(details)
            )
        )

    def get_retention_jobs(self, status: RetentionStatus = None, days: int = 30) -> List[RetentionJob]:
        """Get retention jobs with optional filtering"""
        jobs = []
        
        # Build query
        sql = "SELECT * FROM retention_jobs WHERE 1=1"
        params = []
        
        # Add filters
        if status:
            sql += " AND status = ?"
            params.append(status.value)
        
        # Add time filter
        if days:
            since = datetime.now() - timedelta(days=days)
            sql += " AND scheduled_at >= ?"
            params.append(since.isoformat())
        
        sql += " ORDER BY scheduled_at DESC"
        
        # Execute query
        for row in self.SecureStorage.query(sql, params):
            job = RetentionJob(
                id=row['id'],
                policy_id=row['policy_id'],
                status=RetentionStatus(row['status']),
                scheduled_at=datetime.fromisoformat(row['scheduled_at']),
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                affected_records=row['affected_records'],
                processed_records=row['processed_records'],
                error_message=row['error_message'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            jobs.append(job)
        
        return jobs

    def get_audit_log(self, policy_id: str = None, days: int = 30) -> List[RetentionAudit]:
        """Get retention audit log entries"""
        audits = []
        
        # Build query
        sql = "SELECT * FROM retention_audit WHERE 1=1"
        params = []
        
        # Add filters
        if policy_id:
            sql += " AND policy_id = ?"
            params.append(policy_id)
        
        # Add time filter
        if days:
            since = datetime.now() - timedelta(days=days)
            sql += " AND timestamp >= ?"
            params.append(since.isoformat())
        
        sql += " ORDER BY timestamp DESC"
        
        # Execute query
        for row in self.SecureStorage.query(sql, params):
            audit = RetentionAudit(
                id=row['id'],
                policy_id=row['policy_id'],
                job_id=row['job_id'],
                action=row['action'],
                entity_type=row['entity_type'],
                entity_id=row['entity_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                details=json.loads(row['details']) if row['details'] else {},
                user_id=row['user_id']
            )
            audits.append(audit)
        
        return audits

    def apply_default_policies(self):
        """Apply default retention policies based on regulations"""
        for regulation_type, periods in self.default_retention_periods.items():
            for entity_type, retention_days in periods.items():
                # Check if policy already exists
                existing = None
                for policy in self.policies.values():
                    if (policy.entity_type == entity_type and 
                        policy.regulation_type == regulation_type):
                        existing = policy
                        break
                
                if not existing:
                    # Create default policy
                    self.create_policy(
                        name=f"Default {entity_type} retention ({regulation_type.value})",
                        description=f"Default retention policy for {entity_type} under {regulation_type.value}",
                        entity_type=entity_type,
                        retention_period_days=retention_days,
                        action=RetentionAction.DELETE,
                        regulation_type=regulation_type
                    )
                    
                    # Schedule job to run immediately
                    policy = list(self.policies.values())[-1]
                    self.schedule_retention_job(policy.id)
        
        self.logger.info("Applied default retention policies")

    def generate_retention_report(self, days: int = 30) -> Dict:
        """Generate a retention report"""
        since = datetime.now() - timedelta(days=days)
        
        # Get job statistics
        job_stats = {}
        for status in RetentionStatus:
            count = self.SecureStorage.query(
                "SELECT COUNT(*) FROM retention_jobs WHERE status = ? AND scheduled_at >= ?",
                (status.value, since.isoformat())
            ).fetchone()[0]
            job_stats[status.value] = count
        
        # Get policy statistics
        policy_stats = {}
        for policy in self.policies.values():
            policy_stats[policy.id] = {
                "name": policy.name,
                "entity_type": policy.entity_type,
                "regulation": policy.regulation_type.value,
                "retention_days": policy.retention_period_days,
                "action": policy.action.value,
                "last_applied": policy.last_applied.isoformat() if policy.last_applied else None
            }
        
        # Get audit statistics
        audit_stats = {}
        for action in ["delete", "archive", "anonymize", "flag"]:
            count = self.SecureStorage.query(
                "SELECT COUNT(*) FROM retention_audit WHERE action = ? AND timestamp >= ?",
                (action, since.isoformat())
            ).fetchone()[0]
            audit_stats[action] = count
        
        # Get archive statistics
        archive_count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM data_archive WHERE archived_at >= ?",
            (since.isoformat(),)
        ).fetchone()[0]
        
        return {
            "report_period_days": days,
            "job_statistics": job_stats,
            "policies": policy_stats,
            "audit_statistics": audit_stats,
            "archived_records": archive_count,
            "generated_at": datetime.now().isoformat()
        }

    def cleanup_expired_archives(self):
        """Clean up expired archived records"""
        now = datetime.now()
        
        # Get expired archives
        expired_archives = []
        for row in self.SecureStorage.query(
            "SELECT id, original_table, entity_id FROM data_archive WHERE scheduled_deletion_at < ?",
            (now.isoformat(),)
        ):
            expired_archives.append({
                "id": row['id'],
                "original_table": row['original_table'],
                "entity_id": row['entity_id']
            })
        
        # Delete expired archives
        for archive in expired_archives:
            self.SecureStorage.execute(
                "DELETE FROM data_archive WHERE id = ?",
                (archive['id'],)
            )
            
            self.logger.info(f"Deleted expired archive: {archive['id']}")
        
        if expired_archives:
            self.logger.info(f"Cleaned up {len(expired_archives)} expired archives")
