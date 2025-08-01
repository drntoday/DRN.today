# modules/email_system/bounce_detector.py

import email
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from engine.storage import Storage
from ai.nlp import NLPProcessor


class BounceType(Enum):
    HARD_BOUNCE = "hard_bounce"
    SOFT_BOUNCE = "soft_bounce"
    DNS_FAILURE = "dns_failure"
    MAILBOX_FULL = "mailbox_full"
    BLOCKED = "blocked"
    SPAM_REJECTED = "spam_rejected"
    UNKNOWN = "unknown"


@dataclass
class BounceAnalysis:
    email_id: str
    bounce_type: BounceType
    bounce_reason: str
    confidence: float
    diagnostic_code: Optional[str] = None
    recipient: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_headers: Dict = field(default_factory=dict)


class BounceDetector:
    def __init__(self, storage: Storage, nlp_processor: NLPProcessor):
        self.storage = storage
        self.nlp = nlp_processor
        self.logger = logging.getLogger("bounce_detector")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Initialize bounce patterns
        self._initialize_bounce_patterns()
        
        # Initialize ML model
        self._initialize_bounce_classifier()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS bounce_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            pattern TEXT NOT NULL,
            bounce_type TEXT NOT NULL,
            description TEXT,
            active INTEGER DEFAULT 1
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS bounce_analyses (
            id TEXT PRIMARY KEY,
            email_id TEXT NOT NULL,
            bounce_type TEXT NOT NULL,
            bounce_reason TEXT NOT NULL,
            confidence REAL,
            diagnostic_code TEXT,
            recipient TEXT,
            timestamp TEXT,
            raw_headers TEXT,
            FOREIGN KEY (email_id) REFERENCES email_replies (id)
        )
        """)

    def _initialize_bounce_patterns(self):
        """Initialize bounce detection patterns"""
        # Load default patterns if table is empty
        pattern_count = self.storage.query("SELECT COUNT(*) FROM bounce_patterns").fetchone()[0]
        
        if pattern_count == 0:
            default_patterns = [
                # Hard bounce patterns
                ("regex", r"user.*unknown", "hard_bounce", "User does not exist"),
                ("regex", r"recipient.*not.*found", "hard_bounce", "Recipient not found"),
                ("regex", r"no.*such.*user", "hard_bounce", "No such user"),
                ("regex", r"invalid.*recipient", "hard_bounce", "Invalid recipient"),
                ("regex", r"account.*disabled", "hard_bounce", "Account disabled"),
                ("regex", r"domain.*not.*found", "hard_bounce", "Domain not found"),
                ("regex", r"550.*5\.1\.1", "hard_bounce", "SMTP 550 5.1.1 error"),
                
                # Soft bounce patterns
                ("regex", r"mailbox.*full", "soft_bounce", "Mailbox full"),
                ("regex", r"over.*quota", "soft_bounce", "Over quota"),
                ("regex", r"exceeded.*storage", "soft_bounce", "Exceeded storage limit"),
                ("regex", r"temporarily.*unavailable", "soft_bounce", "Temporarily unavailable"),
                ("regex", r"421.*4\.2\.1", "soft_bounce", "SMTP 421 4.2.1 error"),
                
                # DNS failure patterns
                ("regex", r"DNS.*failure", "dns_failure", "DNS resolution failed"),
                ("regex", r"domain.*error", "dns_failure", "Domain error"),
                ("regex", r"host.*unknown", "dns_failure", "Host unknown"),
                ("regex", r"550.*5\.4\.4", "dns_failure", "SMTP 550 5.4.4 error"),
                
                # Blocked patterns
                ("regex", r"blocked", "blocked", "Message blocked"),
                ("regex", r"blacklisted", "blocked", "IP blacklisted"),
                ("regex", r"rejected.*policy", "blocked", "Rejected by policy"),
                ("regex", r"550.*5\.7\.1", "blocked", "SMTP 550 5.7.1 error"),
                
                # Spam rejected patterns
                ("regex", r"spam", "spam_rejected", "Marked as spam"),
                ("regex", r"bulk.*rejected", "spam_rejected", "Bulk email rejected"),
                ("regex", r"content.*rejected", "spam_rejected", "Content rejected"),
                ("regex", r"550.*5\.7\.0", "spam_rejected", "SMTP 550 5.7.0 error"),
                
                # Diagnostic code patterns
                ("diagnostic_code", r"5\.1\.1", "hard_bounce", "Bad destination mailbox"),
                ("diagnostic_code", r"5\.1\.2", "hard_bounce", "Bad destination system"),
                ("diagnostic_code", r"5\.1\.3", "hard_bounce", "Bad destination mailbox address syntax"),
                ("diagnostic_code", r"5\.2\.1", "soft_bounce", "Mailbox disabled"),
                ("diagnostic_code", r"5\.2\.2", "soft_bounce", "Mailbox full"),
                ("diagnostic_code", r"5\.2\.3", "soft_bounce", "Message length exceeds administrative limit"),
                ("diagnostic_code", r"5\.3\.0", "soft_bounce", "Other or undefined mail system status"),
                ("diagnostic_code", r"5\.4\.0", "dns_failure", "Persistent transient failure"),
                ("diagnostic_code", r"5\.4\.1", "dns_failure", "No answer from host"),
                ("diagnostic_code", r"5\.4\.2", "dns_failure", "Bad connection"),
                ("diagnostic_code", r"5\.4\.3", "dns_failure", "Directory server failure"),
                ("diagnostic_code", r"5\.4\.4", "dns_failure", "Unable to route"),
                ("diagnostic_code", r"5\.4\.5", "dns_failure", "Mail system congestion"),
                ("diagnostic_code", r"5\.4\.6", "dns_failure", "Routing loop detected"),
                ("diagnostic_code", r"5\.4\.7", "dns_failure", "Delivery time expired"),
                ("diagnostic_code", r"5\.7\.1", "blocked", "Delivery not authorized"),
                ("diagnostic_code", r"5\.7\.0", "spam_rejected", "Other security status")
            ]
            
            for pattern_type, pattern, bounce_type, description in default_patterns:
                self.storage.execute(
                    "INSERT INTO bounce_patterns (pattern_type, pattern, bounce_type, description) VALUES (?, ?, ?, ?)",
                    (pattern_type, pattern, bounce_type, description)
                )
        
        # Load patterns from database
        self.patterns = {
            "regex": [],
            "diagnostic_code": []
        }
        
        for row in self.storage.query("SELECT * FROM bounce_patterns WHERE active = 1"):
            self.patterns[row['pattern_type']].append({
                "pattern": row['pattern'],
                "bounce_type": BounceType(row['bounce_type']),
                "description": row['description']
            })

    def _initialize_bounce_classifier(self):
        """Initialize the ML-based bounce classifier"""
        # Create a simple pipeline for bounce classification
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        # In a real implementation, we would load a pre-trained model
        # For now, we'll rely primarily on rule-based detection

    def detect_bounce(self, email_reply) -> Tuple[BounceType, str]:
        """Detect bounce type and reason from an email reply"""
        # Extract text content for analysis
        text_content = self._extract_text_content(email_reply)
        
        # Extract diagnostic codes
        diagnostic_codes = self._extract_diagnostic_codes(email_reply)
        
        # Extract recipient
        recipient = self._extract_recipient(email_reply)
        
        # Rule-based detection
        rule_based_result = self._rule_based_detection(text_content, diagnostic_codes)
        
        # ML-based detection (if rule-based is inconclusive)
        if rule_based_result[1] < 0.7:
            ml_result = self._ml_based_detection(text_content)
            if ml_result[1] > rule_based_result[1]:
                rule_based_result = ml_result
        
        # Create bounce analysis
        bounce_type, confidence = rule_based_result
        bounce_reason = self._generate_bounce_reason(bounce_type, text_content, diagnostic_codes)
        
        # Log the analysis
        analysis = BounceAnalysis(
            email_id=email_reply.id,
            bounce_type=bounce_type,
            bounce_reason=bounce_reason,
            confidence=confidence,
            diagnostic_code=", ".join(diagnostic_codes) if diagnostic_codes else None,
            recipient=recipient,
            raw_headers=self._extract_headers(email_reply)
        )
        
        self._save_bounce_analysis(analysis)
        
        return bounce_type, bounce_reason

    def _extract_text_content(self, email_reply) -> str:
        """Extract text content from email reply"""
        text_parts = []
        
        # Add subject
        if email_reply.subject:
            text_parts.append(email_reply.subject)
        
        # Add body text
        if email_reply.body_text:
            text_parts.append(email_reply.body_text)
        
        # Add body HTML (stripped of tags)
        if email_reply.body_html:
            # Simple HTML tag removal
            html_text = re.sub(r'<[^>]+>', '', email_reply.body_html)
            text_parts.append(html_text)
        
        return " ".join(text_parts)

    def _extract_diagnostic_codes(self, email_reply) -> List[str]:
        """Extract SMTP diagnostic codes from email"""
        diagnostic_codes = []
        
        # Common diagnostic code patterns
        code_patterns = [
            r'\b\d{3}\s*\.\d{1,3}\.\d{1,3}\b',  # SMTP enhanced status codes (e.g., 5.1.1)
            r'\b\d{3}\b',  # Simple SMTP status codes (e.g., 550)
            r'\[#[0-9]+\]',  # Exchange diagnostic codes (e.g., [#5.1.0])
            r'Diagnostic-Code:\s*([^;\n]+)',  # Diagnostic-Code header
            r'Status:\s*([^;\n]+)'  # Status header
        ]
        
        text = self._extract_text_content(email_reply)
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            diagnostic_codes.extend(matches)
        
        # Remove duplicates
        return list(set(diagnostic_codes))

    def _extract_recipient(self, email_reply) -> Optional[str]:
        """Extract the recipient email from bounce message"""
        # Common patterns for recipient email in bounce messages
        recipient_patterns = [
            r'Recipient:\s*([^\s;]+)',
            r'For:\s*([^\s;]+)',
            r'To:\s*([^\s;]+)',
            r'Original-Recipient:\s*([^\s;]+)',
            r'Final-Recipient:\s*([^\s;]+)',
            r'delivery to the following recipients failed:\s*([^\n]+)'
        ]
        
        text = self._extract_text_content(email_reply)
        
        for pattern in recipient_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                recipient = match.group(1).strip()
                # Validate email format
                if re.match(r'^[^@]+@[^@]+\.[^@]+$', recipient):
                    return recipient
        
        return None

    def _extract_headers(self, email_reply) -> Dict[str, str]:
        """Extract headers from email reply"""
        # In a real implementation, we would parse the raw email headers
        # For now, we'll return a simplified version
        return {
            "message_id": email_reply.message_id,
            "in_reply_to": email_reply.in_reply_to,
            "from": email_reply.from_email,
            "to": email_reply.to_email,
            "subject": email_reply.subject
        }

    def _rule_based_detection(self, text: str, diagnostic_codes: List[str]) -> Tuple[BounceType, float]:
        """Detect bounce type using rule-based patterns"""
        # Check diagnostic codes first (most reliable)
        for code in diagnostic_codes:
            for pattern in self.patterns["diagnostic_code"]:
                if re.search(pattern["pattern"], code, re.IGNORECASE):
                    return pattern["bounce_type"], 0.95
        
        # Check text patterns
        for pattern in self.patterns["regex"]:
            if re.search(pattern["pattern"], text, re.IGNORECASE):
                return pattern["bounce_type"], 0.85
        
        # Default to unknown
        return BounceType.UNKNOWN, 0.3

    def _ml_based_detection(self, text: str) -> Tuple[BounceType, float]:
        """Detect bounce type using ML classification"""
        # In a real implementation, we would use the trained classifier
        # For now, we'll use a simple keyword-based approach
        
        text_lower = text.lower()
        
        # Simple scoring based on keywords
        scores = {
            BounceType.HARD_BOUNCE: 0,
            BounceType.SOFT_BOUNCE: 0,
            BounceType.DNS_FAILURE: 0,
            BounceType.BLOCKED: 0,
            BounceType.SPAM_REJECTED: 0,
            BounceType.UNKNOWN: 0
        }
        
        # Hard bounce keywords
        hard_bounce_keywords = ["unknown", "not found", "invalid", "doesn't exist", "disabled"]
        for keyword in hard_bounce_keywords:
            if keyword in text_lower:
                scores[BounceType.HARD_BOUNCE] += 1
        
        # Soft bounce keywords
        soft_bounce_keywords = ["full", "quota", "temporarily", "delayed", "retry"]
        for keyword in soft_bounce_keywords:
            if keyword in text_lower:
                scores[BounceType.SOFT_BOUNCE] += 1
        
        # DNS failure keywords
        dns_keywords = ["dns", "domain", "host", "resolution"]
        for keyword in dns_keywords:
            if keyword in text_lower:
                scores[BounceType.DNS_FAILURE] += 1
        
        # Blocked keywords
        blocked_keywords = ["blocked", "blacklisted", "rejected", "policy"]
        for keyword in blocked_keywords:
            if keyword in text_lower:
                scores[BounceType.BLOCKED] += 1
        
        # Spam rejected keywords
        spam_keywords = ["spam", "bulk", "content", "junk"]
        for keyword in spam_keywords:
            if keyword in text_lower:
                scores[BounceType.SPAM_REJECTED] += 1
        
        # Find the highest score
        max_score = max(scores.values())
        if max_score > 0:
            bounce_type = max(scores, key=scores.get)
            confidence = min(max_score / 3, 0.8)  # Normalize confidence
            return bounce_type, confidence
        
        return BounceType.UNKNOWN, 0.3

    def _generate_bounce_reason(self, bounce_type: BounceType, text: str, diagnostic_codes: List[str]) -> str:
        """Generate a human-readable bounce reason"""
        # Default reasons for each bounce type
        default_reasons = {
            BounceType.HARD_BOUNCE: "The recipient's email address is invalid or does not exist",
            BounceType.SOFT_BOUNCE: "The recipient's mailbox is temporarily unavailable or full",
            BounceType.DNS_FAILURE: "The recipient's domain could not be resolved",
            BounceType.BLOCKED: "The message was blocked by the recipient's mail server",
            BounceType.SPAM_REJECTED: "The message was rejected as spam",
            BounceType.UNKNOWN: "The bounce reason could not be determined"
        }
        
        reason = default_reasons.get(bounce_type, "Unknown bounce reason")
        
        # Add diagnostic code if available
        if diagnostic_codes:
            reason += f" (Diagnostic code: {', '.join(diagnostic_codes)})"
        
        return reason

    def _save_bounce_analysis(self, analysis: BounceAnalysis):
        """Save bounce analysis to database"""
        self.storage.execute(
            """
            INSERT OR REPLACE INTO bounce_analyses 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis.email_id,
                analysis.bounce_type.value,
                analysis.bounce_reason,
                analysis.confidence,
                analysis.diagnostic_code,
                analysis.recipient,
                analysis.timestamp.isoformat(),
                json.dumps(analysis.raw_headers)
            )
        )

    def get_bounce_analysis(self, email_id: str) -> Optional[BounceAnalysis]:
        """Get bounce analysis for a specific email"""
        row = self.storage.query(
            "SELECT * FROM bounce_analyses WHERE email_id = ?",
            (email_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return BounceAnalysis(
            email_id=row['email_id'],
            bounce_type=BounceType(row['bounce_type']),
            bounce_reason=row['bounce_reason'],
            confidence=row['confidence'],
            diagnostic_code=row['diagnostic_code'],
            recipient=row['recipient'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            raw_headers=json.loads(row['raw_headers']) if row['raw_headers'] else {}
        )

    def get_bounce_stats(self, days: int = 30) -> Dict:
        """Get bounce statistics for the specified time period"""
        since = datetime.now() - pd.Timedelta(days=days)
        
        # Get bounce counts by type
        bounce_counts = {bounce_type.value: 0 for bounce_type in BounceType}
        total_bounces = 0
        
        for row in self.storage.query(
            "SELECT bounce_type, COUNT(*) as count FROM bounce_analyses WHERE timestamp >= ? GROUP BY bounce_type",
            (since.isoformat(),)
        ):
            bounce_counts[row['bounce_type']] = row['count']
            total_bounces += row['count']
        
        # Get average confidence by type
        confidence_stats = {}
        for row in self.storage.query(
            "SELECT bounce_type, AVG(confidence) as avg_confidence FROM bounce_analyses WHERE timestamp >= ? GROUP BY bounce_type",
            (since.isoformat(),)
        ):
            confidence_stats[row['bounce_type']] = row['avg_confidence']
        
        # Get top bounce reasons
        top_reasons = self.storage.query(
            "SELECT bounce_reason, COUNT(*) as count FROM bounce_analyses WHERE timestamp >= ? GROUP BY bounce_reason ORDER BY count DESC LIMIT 5",
            (since.isoformat(),)
        ).fetchall()
        
        # Calculate bounce rate
        total_emails = self.storage.query(
            "SELECT COUNT(*) FROM email_replies WHERE received_at >= ?",
            (since.isoformat(),)
        ).fetchone()[0]
        
        bounce_rate = (total_bounces / total_emails) if total_emails > 0 else 0
        
        return {
            "total_bounces": total_bounces,
            "bounce_distribution": bounce_counts,
            "confidence_stats": confidence_stats,
            "top_bounce_reasons": [{"reason": row['bounce_reason'], "count": row['count']} for row in top_reasons],
            "bounce_rate": bounce_rate,
            "total_emails": total_emails
        }

    def export_bounce_data(self, days: int = 30, format: str = "csv") -> str:
        """Export bounce data in specified format"""
        since = datetime.now() - pd.Timedelta(days=days)
        
        # Get bounce data
        bounce_data = []
        for row in self.storage.query(
            "SELECT * FROM bounce_analyses WHERE timestamp >= ? ORDER BY timestamp DESC",
            (since.isoformat(),)
        ):
            bounce_data.append({
                "email_id": row['email_id'],
                "bounce_type": row['bounce_type'],
                "bounce_reason": row['bounce_reason'],
                "confidence": row['confidence'],
                "diagnostic_code": row['diagnostic_code'],
                "recipient": row['recipient'],
                "timestamp": row['timestamp']
            })
        
        if format.lower() == "csv":
            df = pd.DataFrame(bounce_data)
            return df.to_csv(index=False)
        elif format.lower() == "json":
            return json.dumps(bounce_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def add_bounce_pattern(self, pattern_type: str, pattern: str, bounce_type: BounceType, description: str = ""):
        """Add a new bounce detection pattern"""
        self.storage.execute(
            "INSERT INTO bounce_patterns (pattern_type, pattern, bounce_type, description) VALUES (?, ?, ?, ?)",
            (pattern_type, pattern, bounce_type.value, description)
        )
        
        # Update patterns in memory
        if pattern_type in self.patterns:
            self.patterns[pattern_type].append({
                "pattern": pattern,
                "bounce_type": bounce_type,
                "description": description
            })
        else:
            self.patterns[pattern_type] = [{
                "pattern": pattern,
                "bounce_type": bounce_type,
                "description": description
            }]
        
        self.logger.info(f"Added new {pattern_type} pattern for {bounce_type.value}")

    def remove_bounce_pattern(self, pattern_id: int):
        """Remove a bounce detection pattern"""
        self.storage.execute(
            "UPDATE bounce_patterns SET active = 0 WHERE id = ?",
            (pattern_id,)
        )
        
        # Reload patterns
        self._initialize_bounce_patterns()
        
        self.logger.info(f"Removed bounce pattern with ID {pattern_id}")

    def get_bounce_patterns(self, pattern_type: str = None) -> List[Dict]:
        """Get bounce detection patterns"""
        if pattern_type:
            return self.patterns.get(pattern_type, [])
        else:
            all_patterns = []
            for patterns in self.patterns.values():
                all_patterns.extend(patterns)
            return all_patterns