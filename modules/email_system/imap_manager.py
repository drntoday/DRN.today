# modules/email_system/imap_manager.py

import email
import imaplib
import logging
import re
import time
from datetime import datetime, timedelta
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import quopri
import base64

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from engine.SecureStorage import SecureStorage
from ai.nlp import NLPProcessor
from ai.scoring import LeadScorer
from modules.email_system.bounce_detector import BounceDetector


class ReplyType(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    SPAM = "spam"
    BOUNCE = "bounce"
    OUT_OF_OFFICE = "out_of_office"
    UNKNOWN = "unknown"


@dataclass
class EmailReply:
    id: str
    message_id: str
    in_reply_to: str
    from_email: str
    to_email: str
    subject: str
    body_text: str
    body_html: str
    received_at: datetime
    reply_type: ReplyType = ReplyType.UNKNOWN
    confidence: float = 0.0
    lead_id: Optional[str] = None
    campaign_id: Optional[str] = None
    processed: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class IMAPAccount:
    id: str
    email: str
    server: str
    port: int
    username: str
    password: str
    use_ssl: bool = True
    last_checked: Optional[datetime] = None
    active: bool = True
    folder: str = "INBOX"


class IMAPManager:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor, 
                 scorer: LeadScorer, bounce_detector: BounceDetector):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.scorer = scorer
        self.bounce_detector = bounce_detector
        self.logger = logging.getLogger("imap_manager")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize tables
        self._initialize_tables()
        
        # Load IMAP accounts
        self.imap_accounts: Dict[str, IMAPAccount] = {}
        self._load_imap_accounts()
        
        # Initialize reply classifier
        self.reply_classifier = self._initialize_reply_classifier()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS imap_accounts (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            server TEXT NOT NULL,
            port INTEGER NOT NULL,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            use_ssl INTEGER DEFAULT 1,
            last_checked TEXT,
            active INTEGER DEFAULT 1,
            folder TEXT DEFAULT 'INBOX'
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS email_replies (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            in_reply_to TEXT NOT NULL,
            from_email TEXT NOT NULL,
            to_email TEXT NOT NULL,
            subject TEXT NOT NULL,
            body_text TEXT,
            body_html TEXT,
            received_at TEXT NOT NULL,
            reply_type TEXT NOT NULL,
            confidence REAL,
            lead_id TEXT,
            campaign_id TEXT,
            processed INTEGER DEFAULT 0,
            metadata TEXT,
            FOREIGN KEY (lead_id) REFERENCES leads (id),
            FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS bounce_events (
            id TEXT PRIMARY KEY,
            email_reply_id TEXT,
            bounce_type TEXT,
            bounce_reason TEXT,
            original_recipient TEXT,
            timestamp TEXT,
            FOREIGN KEY (email_reply_id) REFERENCES email_replies (id)
        )
        """)

    def _load_imap_accounts(self):
        """Load IMAP accounts from SecureStorage"""
        for row in self.SecureStorage.query("SELECT * FROM imap_accounts WHERE active = 1"):
            account = IMAPAccount(
                id=row['id'],
                email=row['email'],
                server=row['server'],
                port=row['port'],
                username=row['username'],
                password=row['password'],
                use_ssl=bool(row['use_ssl']),
                last_checked=datetime.fromisoformat(row['last_checked']) if row['last_checked'] else None,
                active=bool(row['active']),
                folder=row['folder']
            )
            self.imap_accounts[account.id] = account

    def _initialize_reply_classifier(self):
        """Initialize the reply classification model"""
        # Create a simple pipeline for reply classification
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        # In a real implementation, we would load a pre-trained model
        # For now, we'll create a default model
        return pipeline

    def add_imap_account(self, email: str, server: str, port: int, 
                        username: str, password: str, use_ssl: bool = True,
                        folder: str = "INBOX") -> IMAPAccount:
        """Add a new IMAP account"""
        account_id = f"imap_{int(time.time())}"
        account = IMAPAccount(
            id=account_id,
            email=email,
            server=server,
            port=port,
            username=username,
            password=password,
            use_ssl=use_ssl,
            folder=folder
        )
        
        self.SecureStorage.execute(
            """
            INSERT INTO imap_accounts 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id, email, server, port, username, password,
                1 if use_ssl else 0, None, 1, folder
            )
        )
        
        self.imap_accounts[account_id] = account
        self.logger.info(f"Added IMAP account: {email}")
        return account

    def remove_imap_account(self, account_id: str):
        """Remove an IMAP account"""
        if account_id in self.imap_accounts:
            del self.imap_accounts[account_id]
            self.SecureStorage.execute(
                "UPDATE imap_accounts SET active = 0 WHERE id = ?",
                (account_id,)
            )
            self.logger.info(f"Removed IMAP account: {account_id}")

    def connect_to_imap(self, account: IMAPAccount) -> Optional[imaplib.IMAP4_SSL]:
        """Connect to an IMAP server"""
        try:
            if account.use_ssl:
                imap = imaplib.IMAP4_SSL(account.server, account.port)
            else:
                imap = imaplib.IMAP4(account.server, account.port)
            
            imap.login(account.username, account.password)
            return imap
        except Exception as e:
            self.logger.error(f"Failed to connect to IMAP server for {account.email}: {str(e)}")
            return None

    def fetch_emails(self, account: IMAPAccount, since: Optional[datetime] = None,
                    limit: int = 100) -> List[EmailReply]:
        """Fetch emails from an IMAP account"""
        imap = self.connect_to_imap(account)
        if not imap:
            return []
        
        try:
            # Select folder
            imap.select(account.folder)
            
            # Search criteria
            criteria = []
            if since:
                criteria.append(f'SINCE "{since.strftime("%d-%b-%Y")}"')
            
            search_criteria = ' '.join(criteria) if criteria else 'ALL'
            
            # Search for emails
            status, messages = imap.search(None, search_criteria)
            if status != 'OK':
                self.logger.error(f"IMAP search failed for {account.email}")
                return []
            
            # Get message IDs
            message_ids = messages[0].split()
            if limit and len(message_ids) > limit:
                message_ids = message_ids[-limit:]  # Get the most recent
            
            # Fetch emails
            replies = []
            for msg_id in message_ids:
                try:
                    # Fetch the email
                    status, msg_data = imap.fetch(msg_id, '(RFC822)')
                    if status != 'OK':
                        continue
                    
                    # Parse the email
                    raw_email = msg_data[0][1]
                    email_message = email.message_from_bytes(raw_email)
                    
                    # Extract email data
                    reply = self._parse_email(email_message, account.id)
                    if reply:
                        replies.append(reply)
                except Exception as e:
                    self.logger.error(f"Error processing email {msg_id}: {str(e)}")
            
            # Update last checked time
            account.last_checked = datetime.now()
            self.SecureStorage.execute(
                "UPDATE imap_accounts SET last_checked = ? WHERE id = ?",
                (account.last_checked.isoformat(), account.id)
            )
            
            return replies
        finally:
            try:
                imap.close()
                imap.logout()
            except:
                pass

    def _parse_email(self, email_message: email.message.Message, account_id: str) -> Optional[EmailReply]:
        """Parse an email message into an EmailReply object"""
        try:
            # Get basic headers
            message_id = email_message.get('Message-ID', '')
            in_reply_to = email_message.get('In-Reply-To', '')
            from_email = self._decode_email_header(email_message.get('From', ''))
            to_email = self._decode_email_header(email_message.get('To', ''))
            subject = self._decode_email_header(email_message.get('Subject', ''))
            
            # Parse received date
            date_str = email_message.get('Date', '')
            try:
                received_at = parsedate_to_datetime(date_str)
            except:
                received_at = datetime.now()
            
            # Extract body
            body_text = ""
            body_html = ""
            
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    if "attachment" not in content_disposition:
                        if content_type == "text/plain":
                            body_text = self._decode_email_body(part.get_payload(decode=True))
                        elif content_type == "text/html":
                            body_html = self._decode_email_body(part.get_payload(decode=True))
            else:
                content_type = email_message.get_content_type()
                if content_type == "text/plain":
                    body_text = self._decode_email_body(email_message.get_payload(decode=True))
                elif content_type == "text/html":
                    body_html = self._decode_email_body(email_message.get_payload(decode=True))
            
            # Create reply object
            reply_id = f"reply_{int(time.time())}_{hash(message_id) % 10000}"
            reply = EmailReply(
                id=reply_id,
                message_id=message_id,
                in_reply_to=in_reply_to,
                from_email=from_email,
                to_email=to_email,
                subject=subject,
                body_text=body_text,
                body_html=body_html,
                received_at=received_at
            )
            
            return reply
        except Exception as e:
            self.logger.error(f"Error parsing email: {str(e)}")
            return None

    def _decode_email_header(self, header: str) -> str:
        """Decode an email header"""
        if not header:
            return ""
        
        decoded_parts = decode_header(header)
        decoded_header = ""
        
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                if encoding:
                    decoded_header += part.decode(encoding)
                else:
                    decoded_header += part.decode('utf-8', errors='ignore')
            else:
                decoded_header += part
        
        return decoded_header

    def _decode_email_body(self, body: bytes) -> str:
        """Decode an email body"""
        try:
            # Try UTF-8 first
            return body.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Try ISO-8859-1
                return body.decode('iso-8859-1')
            except UnicodeDecodeError:
                # Try other common encodings
                for encoding in ['windows-1252', 'latin1']:
                    try:
                        return body.decode(encoding)
                    except UnicodeDecodeError:
                        pass
                
                # Fallback to ignoring errors
                return body.decode('utf-8', errors='ignore')

    def process_replies(self, replies: List[EmailReply]) -> List[EmailReply]:
        """Process a list of email replies"""
        processed_replies = []
        
        for reply in replies:
            try:
                # Classify reply type
                reply_type, confidence = self._classify_reply(reply)
                reply.reply_type = reply_type
                reply.confidence = confidence
                
                # Check if it's a bounce
                if reply_type == ReplyType.BOUNCE:
                    bounce_type, bounce_reason = self.bounce_detector.detect_bounce(reply)
                    
                    # Record bounce event
                    self.SecureStorage.execute(
                        """
                        INSERT INTO bounce_events 
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            f"bounce_{int(time.time())}",
                            reply.id,
                            bounce_type.value,
                            bounce_reason,
                            reply.to_email,
                            datetime.now().isoformat()
                        )
                    )
                    
                    # Remove lead from active list
                    if reply.lead_id:
                        self._handle_bounce(reply.lead_id, bounce_type)
                
                # Save reply to database
                self._save_reply(reply)
                
                # Update lead score based on reply
                if reply.lead_id and reply_type in [ReplyType.POSITIVE, ReplyType.NEGATIVE]:
                    self._update_lead_score(reply.lead_id, reply_type, confidence)
                
                reply.processed = True
                processed_replies.append(reply)
                
            except Exception as e:
                self.logger.error(f"Error processing reply {reply.id}: {str(e)}")
        
        return processed_replies

    def _classify_reply(self, reply: EmailReply) -> Tuple[ReplyType, float]:
        """Classify an email reply using ML and rules"""
        # Combine subject and body for analysis
        text = f"{reply.subject} {reply.body_text}".lower()
        
        # First check for obvious patterns
        if self._is_bounce(text):
            return ReplyType.BOUNCE, 0.95
        elif self._is_out_of_office(text):
            return ReplyType.OUT_OF_OFFICE, 0.9
        elif self._is_spam(text):
            return ReplyType.SPAM, 0.85
        
        # Use ML model for classification
        try:
            # In a real implementation, we would use the trained model
            # For now, we'll use simple keyword matching
            positive_keywords = ["interested", "yes", "please", "thank you", "great", "love", "excited", "let's", "schedule", "meeting"]
            negative_keywords = ["not interested", "no", "unsubscribe", "remove", "stop", "never", "don't", "won't", "can't"]
            
            positive_score = sum(1 for keyword in positive_keywords if keyword in text)
            negative_score = sum(1 for keyword in negative_keywords if keyword in text)
            
            if positive_score > negative_score:
                return ReplyType.POSITIVE, min(positive_score / len(positive_keywords), 1.0)
            elif negative_score > positive_score:
                return ReplyType.NEGATIVE, min(negative_score / len(negative_keywords), 1.0)
            else:
                return ReplyType.NEUTRAL, 0.5
        except Exception as e:
            self.logger.error(f"Error classifying reply: {str(e)}")
            return ReplyType.UNKNOWN, 0.0

    def _is_bounce(self, text: str) -> bool:
        """Check if an email is a bounce message"""
        bounce_patterns = [
            r"delivery.*failed",
            r"undeliverable",
            r"returned.*mail",
            r"delivery.*error",
            r"mail.*delivery.*failed",
            r"message.*rejected",
            r"recipient.*not.*found",
            r"mailbox.*unavailable",
            r"user.*unknown",
            r"account.*disabled"
        ]
        
        for pattern in bounce_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_out_of_office(self, text: str) -> bool:
        """Check if an email is an out-of-office reply"""
        ooo_patterns = [
            r"out.*of.*office",
            r"out.*of.*the.*office",
            r"on.*vacation",
            r"on.*leave",
            r"away.*from.*desk",
            r"automatic.*reply",
            r"auto.*reply",
            r"out.*until",
            r"return.*on"
        ]
        
        for pattern in ooo_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_spam(self, text: str) -> bool:
        """Check if an email is spam"""
        spam_patterns = [
            r"click.*here",
            r"limited.*time",
            r"act.*now",
            r"free.*gift",
            r"congratulations",
            r"winner",
            r"urgent",
            r"viagra",
            r"cialis",
            r"lottery",
            r"million.*dollars"
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _save_reply(self, reply: EmailReply):
        """Save an email reply to the database"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO email_replies 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reply.id,
                reply.message_id,
                reply.in_reply_to,
                reply.from_email,
                reply.to_email,
                reply.subject,
                reply.body_text,
                reply.body_html,
                reply.received_at.isoformat(),
                reply.reply_type.value,
                reply.confidence,
                reply.lead_id,
                reply.campaign_id,
                1 if reply.processed else 0,
                json.dumps(reply.metadata)
            )
        )

    def _handle_bounce(self, lead_id: str, bounce_type: BounceType):
        """Handle a bounced email by updating the lead"""
        # Mark lead as bounced
        self.SecureStorage.execute(
            "UPDATE leads SET status = 'bounced', bounced_at = ? WHERE id = ?",
            (datetime.now().isoformat(), lead_id)
        )
        
        # Update lead score
        self.scorer.update_lead_score(lead_id, -20)  # Significant penalty for bounce
        
        self.logger.info(f"Marked lead {lead_id} as bounced due to {bounce_type.value}")

    def _update_lead_score(self, lead_id: str, reply_type: ReplyType, confidence: float):
        """Update a lead's score based on reply type"""
        score_change = 0
        
        if reply_type == ReplyType.POSITIVE:
            score_change = 10 * confidence
        elif reply_type == ReplyType.NEGATIVE:
            score_change = -5 * confidence
        
        if score_change != 0:
            self.scorer.update_lead_score(lead_id, score_change)
            self.logger.info(f"Updated lead {lead_id} score by {score_change} based on {reply_type.value} reply")

    def check_all_accounts(self, since: Optional[datetime] = None, 
                          limit: int = 100) -> List[EmailReply]:
        """Check all IMAP accounts for new emails"""
        all_replies = []
        
        for account in self.imap_accounts.values():
            if account.active:
                replies = self.fetch_emails(account, since, limit)
                all_replies.extend(replies)
        
        # Process all replies
        processed_replies = self.process_replies(all_replies)
        
        return processed_replies

    def get_replies_for_campaign(self, campaign_id: str) -> List[EmailReply]:
        """Get all replies for a specific campaign"""
        replies = []
        
        for row in self.SecureStorage.query(
            "SELECT * FROM email_replies WHERE campaign_id = ? ORDER BY received_at DESC",
            (campaign_id,)
        ):
            reply = EmailReply(
                id=row['id'],
                message_id=row['message_id'],
                in_reply_to=row['in_reply_to'],
                from_email=row['from_email'],
                to_email=row['to_email'],
                subject=row['subject'],
                body_text=row['body_text'],
                body_html=row['body_html'],
                received_at=datetime.fromisoformat(row['received_at']),
                reply_type=ReplyType(row['reply_type']),
                confidence=row['confidence'],
                lead_id=row['lead_id'],
                campaign_id=row['campaign_id'],
                processed=bool(row['processed']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            replies.append(reply)
        
        return replies

    def get_reply_stats(self, days: int = 30) -> Dict:
        """Get reply statistics for the specified time period"""
        since = datetime.now() - timedelta(days=days)
        
        # Get reply counts by type
        reply_counts = {reply_type.value: 0 for reply_type in ReplyType}
        total_replies = 0
        
        for row in self.SecureStorage.query(
            "SELECT reply_type, COUNT(*) as count FROM email_replies WHERE received_at >= ? GROUP BY reply_type",
            (since.isoformat(),)
        ):
            reply_counts[row['reply_type']] = row['count']
            total_replies += row['count']
        
        # Get bounce counts
        bounce_counts = {}
        for row in self.SecureStorage.query(
            "SELECT bounce_type, COUNT(*) as count FROM bounce_events WHERE timestamp >= ? GROUP BY bounce_type",
            (since.isoformat(),)
        ):
            bounce_counts[row['bounce_type']] = row['count']
        
        # Calculate reply rate
        # In a real implementation, we would get the total emails sent
        total_sent = self.SecureStorage.query(
            "SELECT COUNT(*) FROM email_events WHERE event_type = 'sent' AND timestamp >= ?",
            (since.isoformat(),)
        ).fetchone()[0]
        
        reply_rate = (total_replies / total_sent) if total_sent > 0 else 0
        
        return {
            "total_replies": total_replies,
            "reply_distribution": reply_counts,
            "bounce_distribution": bounce_counts,
            "reply_rate": reply_rate,
            "total_sent": total_sent
        }

    def export_replies(self, campaign_id: str = None, format: str = "csv") -> str:
        """Export email replies in specified format"""
        query = "SELECT * FROM email_replies"
        params = []
        
        if campaign_id:
            query += " WHERE campaign_id = ?"
            params.append(campaign_id)
        
        query += " ORDER BY received_at DESC"
        
        replies = []
        for row in self.SecureStorage.query(query, params):
            replies.append({
                "id": row['id'],
                "message_id": row['message_id'],
                "from_email": row['from_email'],
                "to_email": row['to_email'],
                "subject": row['subject'],
                "body_text": row['body_text'],
                "received_at": row['received_at'],
                "reply_type": row['reply_type'],
                "confidence": row['confidence'],
                "lead_id": row['lead_id'],
                "campaign_id": row['campaign_id']
            })
        
        if format.lower() == "csv":
            df = pd.DataFrame(replies)
            return df.to_csv(index=False)
        elif format.lower() == "json":
            return json.dumps(replies, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
