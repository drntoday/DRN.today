# modules/marketplace/community.py

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from engine.storage import Storage
from ai.nlp import NLPProcessor


class PostType(Enum):
    QUESTION = "question"
    DISCUSSION = "discussion"
    ANNOUNCEMENT = "announcement"
    TUTORIAL = "tutorial"
    SHOWCASE = "showcase"
    FEEDBACK = "feedback"
    HELP = "help"


class PostStatus(Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    LOCKED = "locked"
    DELETED = "deleted"
    FLAGGED = "flagged"


class ModerationAction(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    DELETE = "delete"
    LOCK = "lock"
    FLAG = "flag"
    BAN_USER = "ban_user"


class UserRole(Enum):
    MEMBER = "member"
    MODERATOR = "moderator"
    ADMIN = "admin"
    CONTRIBUTOR = "contributor"


@dataclass
class CommunityPost:
    id: str
    title: str
    content: str
    author_id: str
    author_name: str
    post_type: PostType
    status: PostStatus
    category: str
    tags: List[str]
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    is_pinned: bool = False
    is_featured: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Comment:
    id: str
    post_id: str
    author_id: str
    author_name: str
    content: str
    parent_id: Optional[str] = None  # For nested comments
    like_count: int = 0
    is_accepted_answer: bool = False
    is_flagged: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


@dataclass
class UserProfile:
    id: str
    username: str
    email: str
    display_name: str
    bio: str
    avatar_url: Optional[str] = None
    role: UserRole = UserRole.MEMBER
    reputation: int = 0
    post_count: int = 0
    comment_count: int = 0
    helpful_count: int = 0
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    is_banned: bool = False
    preferences: Dict = field(default_factory=dict)


@dataclass
class ModerationQueueItem:
    id: str
    content_type: str  # "post" or "comment"
    content_id: str
    reason: str
    reporter_id: str
    reporter_name: str
    description: str
    status: str = "pending"  # pending, reviewed, resolved
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    action_taken: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class CommunityGuideline:
    id: str
    title: str
    description: str
    category: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class CommunityManager:
    def __init__(self, storage: Storage, nlp_processor: NLPProcessor):
        self.storage = storage
        self.nlp = nlp_processor
        self.logger = logging.getLogger("community_manager")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load community data
        self.posts: Dict[str, CommunityPost] = {}
        self.comments: Dict[str, Comment] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.moderation_queue: Dict[str, ModerationQueueItem] = {}
        self.guidelines: Dict[str, CommunityGuideline] = {}
        
        self._load_community_data()
        
        # Community categories
        self.categories = [
            "General Discussion",
            "Help & Support",
            "Feature Requests",
            "Bug Reports",
            "Showcase",
            "Tutorials",
            "Announcements",
            "Marketplace",
            "Integrations",
            "Best Practices"
        ]
        
        # Content moderation patterns
        self.moderation_patterns = {
            "spam": [
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                r"buy now", r"limited time", r"click here", r"free trial"
            ],
            "inappropriate": [
                r"\b(fuck|shit|ass|bitch|cunt|dick|piss|cock|pussy|tits|asshole)\b",
                r"\b(kill|murder|suicide|rape|abuse)\b"
            ],
            "harassment": [
                r"\b(stupid|idiot|moron|retard|dumb)\b",
                r"you suck", r"go die", r"kill yourself"
            ]
        }
        
        # Reputation rules
        self.reputation_rules = {
            "post_created": 5,
            "comment_created": 2,
            "post_liked": 1,
            "comment_liked": 1,
            "answer_accepted": 10,
            "post_flagged": -2,
            "comment_flagged": -1,
            "moderation_action": -5
        }

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS community_posts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            author_id TEXT NOT NULL,
            author_name TEXT NOT NULL,
            post_type TEXT NOT NULL,
            status TEXT NOT NULL,
            category TEXT NOT NULL,
            tags TEXT,
            view_count INTEGER DEFAULT 0,
            like_count INTEGER DEFAULT 0,
            comment_count INTEGER DEFAULT 0,
            is_pinned INTEGER DEFAULT 0,
            is_featured INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_activity TEXT NOT NULL,
            metadata TEXT
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS community_comments (
            id TEXT PRIMARY KEY,
            post_id TEXT NOT NULL,
            author_id TEXT NOT NULL,
            author_name TEXT NOT NULL,
            content TEXT NOT NULL,
            parent_id TEXT,
            like_count INTEGER DEFAULT 0,
            is_accepted_answer INTEGER DEFAULT 0,
            is_flagged INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            FOREIGN KEY (post_id) REFERENCES community_posts (id)
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS community_user_profiles (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            display_name TEXT NOT NULL,
            bio TEXT,
            avatar_url TEXT,
            role TEXT NOT NULL,
            reputation INTEGER DEFAULT 0,
            post_count INTEGER DEFAULT 0,
            comment_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            joined_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            is_banned INTEGER DEFAULT 0,
            preferences TEXT
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS community_moderation_queue (
            id TEXT PRIMARY KEY,
            content_type TEXT NOT NULL,
            content_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            reporter_id TEXT NOT NULL,
            reporter_name TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            resolved_by TEXT,
            action_taken TEXT,
            notes TEXT
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS community_guidelines (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS community_post_likes (
            post_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (post_id, user_id),
            FOREIGN KEY (post_id) REFERENCES community_posts (id)
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS community_comment_likes (
            comment_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (comment_id, user_id),
            FOREIGN KEY (comment_id) REFERENCES community_comments (id)
        )
        """)

    def _load_community_data(self):
        """Load community data from storage"""
        # Load posts
        for row in self.storage.query("SELECT * FROM community_posts WHERE status != 'deleted'"):
            post = CommunityPost(
                id=row['id'],
                title=row['title'],
                content=row['content'],
                author_id=row['author_id'],
                author_name=row['author_name'],
                post_type=PostType(row['post_type']),
                status=PostStatus(row['status']),
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                view_count=row['view_count'],
                like_count=row['like_count'],
                comment_count=row['comment_count'],
                is_pinned=bool(row['is_pinned']),
                is_featured=bool(row['is_featured']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                last_activity=datetime.fromisoformat(row['last_activity']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            self.posts[post.id] = post
        
        # Load comments
        for row in self.storage.query("SELECT * FROM community_comments"):
            comment = Comment(
                id=row['id'],
                post_id=row['post_id'],
                author_id=row['author_id'],
                author_name=row['author_name'],
                content=row['content'],
                parent_id=row['parent_id'],
                like_count=row['like_count'],
                is_accepted_answer=bool(row['is_accepted_answer']),
                is_flagged=bool(row['is_flagged']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
            self.comments[comment.id] = comment
        
        # Load user profiles
        for row in self.storage.query("SELECT * FROM community_user_profiles WHERE is_banned = 0"):
            profile = UserProfile(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                display_name=row['display_name'],
                bio=row['bio'],
                avatar_url=row['avatar_url'],
                role=UserRole(row['role']),
                reputation=row['reputation'],
                post_count=row['post_count'],
                comment_count=row['comment_count'],
                helpful_count=row['helpful_count'],
                joined_at=datetime.fromisoformat(row['joined_at']),
                last_active=datetime.fromisoformat(row['last_active']),
                is_banned=bool(row['is_banned']),
                preferences=json.loads(row['preferences']) if row['preferences'] else {}
            )
            self.user_profiles[profile.id] = profile
        
        # Load moderation queue
        for row in self.storage.query("SELECT * FROM community_moderation_queue WHERE status = 'pending'"):
            item = ModerationQueueItem(
                id=row['id'],
                content_type=row['content_type'],
                content_id=row['content_id'],
                reason=row['reason'],
                reporter_id=row['reporter_id'],
                reporter_name=row['reporter_name'],
                description=row['description'],
                status=row['status'],
                created_at=datetime.fromisoformat(row['created_at']),
                resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                resolved_by=row['resolved_by'],
                action_taken=row['action_taken'],
                notes=row['notes']
            )
            self.moderation_queue[item.id] = item
        
        # Load guidelines
        for row in self.storage.query("SELECT * FROM community_guidelines WHERE is_active = 1"):
            guideline = CommunityGuideline(
                id=row['id'],
                title=row['title'],
                description=row['description'],
                category=row['category'],
                is_active=bool(row['is_active']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
            self.guidelines[guideline.id] = guideline

    def create_user_profile(self, username: str, email: str, display_name: str, 
                           bio: str = "", role: UserRole = UserRole.MEMBER) -> UserProfile:
        """Create a new user profile"""
        user_id = f"user_{uuid.uuid4().hex}"
        
        profile = UserProfile(
            id=user_id,
            username=username,
            email=email,
            display_name=display_name,
            bio=bio,
            role=role
        )
        
        # Save to database
        self.storage.execute(
            """
            INSERT INTO community_user_profiles 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.id,
                profile.username,
                profile.email,
                profile.display_name,
                profile.bio,
                profile.avatar_url,
                profile.role.value,
                profile.reputation,
                profile.post_count,
                profile.comment_count,
                profile.helpful_count,
                profile.joined_at.isoformat(),
                profile.last_active.isoformat(),
                1 if profile.is_banned else 0,
                json.dumps(profile.preferences)
            )
        )
        
        # Add to in-memory cache
        self.user_profiles[user_id] = profile
        
        self.logger.info(f"Created user profile: {username}")
        return profile

    def update_user_profile(self, user_id: str, **kwargs) -> Optional[UserProfile]:
        """Update a user profile"""
        if user_id not in self.user_profiles:
            return None
        
        profile = self.user_profiles[user_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.last_active = datetime.now()
        
        # Save to database
        self.storage.execute(
            """
            UPDATE community_user_profiles 
            SET display_name = ?, bio = ?, avatar_url = ?, role = ?, 
                last_active = ?, preferences = ?
            WHERE id = ?
            """,
            (
                profile.display_name,
                profile.bio,
                profile.avatar_url,
                profile.role.value,
                profile.last_active.isoformat(),
                json.dumps(profile.preferences),
                user_id
            )
        )
        
        self.logger.info(f"Updated user profile: {profile.username}")
        return profile

    def create_post(self, title: str, content: str, author_id: str, author_name: str,
                   post_type: PostType, category: str, tags: List[str] = None) -> CommunityPost:
        """Create a new community post"""
        post_id = f"post_{uuid.uuid4().hex}"
        
        post = CommunityPost(
            id=post_id,
            title=title,
            content=content,
            author_id=author_id,
            author_name=author_name,
            post_type=post_type,
            status=PostStatus.PUBLISHED,
            category=category,
            tags=tags or []
        )
        
        # Check for content violations
        violations = self._check_content_violations(content)
        if violations:
            post.status = PostStatus.FLAGGED
            self._add_to_moderation_queue("post", post_id, "auto_flag", "system", 
                                         "Auto-flagged for content violations", 
                                         f"Violations: {', '.join(violations)}")
        
        # Save to database
        self._save_post(post)
        
        # Update user stats
        if author_id in self.user_profiles:
            self.user_profiles[author_id].post_count += 1
            self.user_profiles[author_id].reputation += self.reputation_rules["post_created"]
            self._save_user_profile(self.user_profiles[author_id])
        
        self.logger.info(f"Created post: {post_id} - {title}")
        return post

    def _save_post(self, post: CommunityPost):
        """Save a post to database"""
        self.storage.execute(
            """
            INSERT OR REPLACE INTO community_posts 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                post.id,
                post.title,
                post.content,
                post.author_id,
                post.author_name,
                post.post_type.value,
                post.status.value,
                post.category,
                json.dumps(post.tags),
                post.view_count,
                post.like_count,
                post.comment_count,
                1 if post.is_pinned else 0,
                1 if post.is_featured else 0,
                post.created_at.isoformat(),
                post.updated_at.isoformat(),
                post.last_activity.isoformat(),
                json.dumps(post.metadata)
            )
        )
        
        # Update in-memory cache
        self.posts[post.id] = post

    def _save_user_profile(self, profile: UserProfile):
        """Save a user profile to database"""
        self.storage.execute(
            """
            UPDATE community_user_profiles 
            SET reputation = ?, post_count = ?, comment_count = ?, 
                helpful_count = ?, last_active = ?, preferences = ?
            WHERE id = ?
            """,
            (
                profile.reputation,
                profile.post_count,
                profile.comment_count,
                profile.helpful_count,
                profile.last_active.isoformat(),
                json.dumps(profile.preferences),
                profile.id
            )
        )
        
        # Update in-memory cache
        self.user_profiles[profile.id] = profile

    def create_comment(self, post_id: str, author_id: str, author_name: str,
                     content: str, parent_id: str = None) -> Comment:
        """Create a new comment on a post"""
        comment_id = f"comment_{uuid.uuid4().hex}"
        
        comment = Comment(
            id=comment_id,
            post_id=post_id,
            author_id=author_id,
            author_name=author_name,
            content=content,
            parent_id=parent_id
        )
        
        # Check for content violations
        violations = self._check_content_violations(content)
        if violations:
            comment.is_flagged = True
            self._add_to_moderation_queue("comment", comment_id, "auto_flag", "system", 
                                         "Auto-flagged for content violations", 
                                         f"Violations: {', '.join(violations)}")
        
        # Save to database
        self._save_comment(comment)
        
        # Update post stats
        if post_id in self.posts:
            self.posts[post_id].comment_count += 1
            self.posts[post_id].last_activity = datetime.now()
            self._save_post(self.posts[post_id])
        
        # Update user stats
        if author_id in self.user_profiles:
            self.user_profiles[author_id].comment_count += 1
            self.user_profiles[author_id].reputation += self.reputation_rules["comment_created"]
            self._save_user_profile(self.user_profiles[author_id])
        
        self.logger.info(f"Created comment: {comment_id} on post {post_id}")
        return comment

    def _save_comment(self, comment: Comment):
        """Save a comment to database"""
        self.storage.execute(
            """
            INSERT OR REPLACE INTO community_comments 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comment.id,
                comment.post_id,
                comment.author_id,
                comment.author_name,
                comment.content,
                comment.parent_id,
                comment.like_count,
                1 if comment.is_accepted_answer else 0,
                1 if comment.is_flagged else 0,
                comment.created_at.isoformat(),
                comment.updated_at.isoformat() if comment.updated_at else None
            )
        )
        
        # Update in-memory cache
        self.comments[comment.id] = comment

    def _check_content_violations(self, content: str) -> List[str]:
        """Check content for violations"""
        violations = []
        content_lower = content.lower()
        
        for violation_type, patterns in self.moderation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    violations.append(violation_type)
        
        return violations

    def _add_to_moderation_queue(self, content_type: str, content_id: str, reason: str,
                                reporter_id: str, reporter_name: str, 
                                description: str, notes: str = None):
        """Add an item to the moderation queue"""
        item_id = f"mod_{uuid.uuid4().hex}"
        
        item = ModerationQueueItem(
            id=item_id,
            content_type=content_type,
            content_id=content_id,
            reason=reason,
            reporter_id=reporter_id,
            reporter_name=reporter_name,
            description=description,
            notes=notes
        )
        
        # Save to database
        self.storage.execute(
            """
            INSERT INTO community_moderation_queue 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.content_type,
                item.content_id,
                item.reason,
                item.reporter_id,
                item.reporter_name,
                item.description,
                item.status,
                item.created_at.isoformat(),
                item.resolved_at.isoformat() if item.resolved_at else None,
                item.resolved_by,
                item.action_taken,
                item.notes
            )
        )
        
        # Add to in-memory cache
        self.moderation_queue[item.id] = item

    def get_post(self, post_id: str) -> Optional[CommunityPost]:
        """Get a post by ID"""
        return self.posts.get(post_id)

    def get_comment(self, comment_id: str) -> Optional[Comment]:
        """Get a comment by ID"""
        return self.comments.get(comment_id)

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile by ID"""
        return self.user_profiles.get(user_id)

    def get_user_profile_by_username(self, username: str) -> Optional[UserProfile]:
        """Get a user profile by username"""
        for profile in self.user_profiles.values():
            if profile.username == username:
                return profile
        return None

    def get_posts(self, category: str = None, post_type: PostType = None,
                 status: PostStatus = PostStatus.PUBLISHED, 
                 sort_by: str = "last_activity", sort_order: str = "desc",
                 limit: int = 20, offset: int = 0) -> List[CommunityPost]:
        """Get posts with filtering and sorting"""
        # Build query
        sql = "SELECT * FROM community_posts WHERE 1=1"
        params = []
        
        # Add filters
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        if post_type:
            sql += " AND post_type = ?"
            params.append(post_type.value)
        
        if status:
            sql += " AND status = ?"
            params.append(status.value)
        
        # Add sorting
        valid_sort_fields = ["created_at", "updated_at", "last_activity", "like_count", "comment_count", "view_count"]
        if sort_by in valid_sort_fields:
            sql += f" ORDER BY {sort_by}"
        else:
            sql += " ORDER BY last_activity"
        
        if sort_order.lower() == "asc":
            sql += " ASC"
        else:
            sql += " DESC"
        
        # Add pagination
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        # Execute query
        posts = []
        for row in self.storage.query(sql, params):
            post = CommunityPost(
                id=row['id'],
                title=row['title'],
                content=row['content'],
                author_id=row['author_id'],
                author_name=row['author_name'],
                post_type=PostType(row['post_type']),
                status=PostStatus(row['status']),
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                view_count=row['view_count'],
                like_count=row['like_count'],
                comment_count=row['comment_count'],
                is_pinned=bool(row['is_pinned']),
                is_featured=bool(row['is_featured']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                last_activity=datetime.fromisoformat(row['last_activity']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            posts.append(post)
        
        return posts

    def get_comments(self, post_id: str, parent_id: str = None) -> List[Comment]:
        """Get comments for a post"""
        comments = []
        
        sql = "SELECT * FROM community_comments WHERE post_id = ?"
        params = [post_id]
        
        if parent_id:
            sql += " AND parent_id = ?"
            params.append(parent_id)
        else:
            sql += " AND parent_id IS NULL"
        
        sql += " ORDER BY created_at"
        
        for row in self.storage.query(sql, params):
            comment = Comment(
                id=row['id'],
                post_id=row['post_id'],
                author_id=row['author_id'],
                author_name=row['author_name'],
                content=row['content'],
                parent_id=row['parent_id'],
                like_count=row['like_count'],
                is_accepted_answer=bool(row['is_accepted_answer']),
                is_flagged=bool(row['is_flagged']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
            comments.append(comment)
        
        return comments

    def like_post(self, post_id: str, user_id: str) -> bool:
        """Like a post"""
        if post_id not in self.posts:
            return False
        
        # Check if already liked
        existing = self.storage.query(
            "SELECT * FROM community_post_likes WHERE post_id = ? AND user_id = ?",
            (post_id, user_id)
        ).fetchone()
        
        if existing:
            return False
        
        # Add like
        self.storage.execute(
            "INSERT INTO community_post_likes VALUES (?, ?, ?)",
            (post_id, user_id, datetime.now().isoformat())
        )
        
        # Update post stats
        self.posts[post_id].like_count += 1
        self._save_post(self.posts[post_id])
        
        # Update user stats
        if user_id in self.user_profiles:
            self.user_profiles[user_id].reputation += self.reputation_rules["post_liked"]
            self._save_user_profile(self.user_profiles[user_id])
        
        self.logger.info(f"User {user_id} liked post {post_id}")
        return True

    def unlike_post(self, post_id: str, user_id: str) -> bool:
        """Unlike a post"""
        if post_id not in self.posts:
            return False
        
        # Check if liked
        existing = self.storage.query(
            "SELECT * FROM community_post_likes WHERE post_id = ? AND user_id = ?",
            (post_id, user_id)
        ).fetchone()
        
        if not existing:
            return False
        
        # Remove like
        self.storage.execute(
            "DELETE FROM community_post_likes WHERE post_id = ? AND user_id = ?",
            (post_id, user_id)
        )
        
        # Update post stats
        self.posts[post_id].like_count = max(0, self.posts[post_id].like_count - 1)
        self._save_post(self.posts[post_id])
        
        # Update user stats
        if user_id in self.user_profiles:
            self.user_profiles[user_id].reputation = max(0, self.user_profiles[user_id].reputation - self.reputation_rules["post_liked"])
            self._save_user_profile(self.user_profiles[user_id])
        
        self.logger.info(f"User {user_id} unliked post {post_id}")
        return True

    def like_comment(self, comment_id: str, user_id: str) -> bool:
        """Like a comment"""
        if comment_id not in self.comments:
            return False
        
        # Check if already liked
        existing = self.storage.query(
            "SELECT * FROM community_comment_likes WHERE comment_id = ? AND user_id = ?",
            (comment_id, user_id)
        ).fetchone()
        
        if existing:
            return False
        
        # Add like
        self.storage.execute(
            "INSERT INTO community_comment_likes VALUES (?, ?, ?)",
            (comment_id, user_id, datetime.now().isoformat())
        )
        
        # Update comment stats
        self.comments[comment_id].like_count += 1
        self._save_comment(self.comments[comment_id])
        
        # Update user stats
        if user_id in self.user_profiles:
            self.user_profiles[user_id].reputation += self.reputation_rules["comment_liked"]
            self._save_user_profile(self.user_profiles[user_id])
        
        self.logger.info(f"User {user_id} liked comment {comment_id}")
        return True

    def unlike_comment(self, comment_id: str, user_id: str) -> bool:
        """Unlike a comment"""
        if comment_id not in self.comments:
            return False
        
        # Check if liked
        existing = self.storage.query(
            "SELECT * FROM community_comment_likes WHERE comment_id = ? AND user_id = ?",
            (comment_id, user_id)
        ).fetchone()
        
        if not existing:
            return False
        
        # Remove like
        self.storage.execute(
            "DELETE FROM community_comment_likes WHERE comment_id = ? AND user_id = ?",
            (comment_id, user_id)
        )
        
        # Update comment stats
        self.comments[comment_id].like_count = max(0, self.comments[comment_id].like_count - 1)
        self._save_comment(self.comments[comment_id])
        
        # Update user stats
        if user_id in self.user_profiles:
            self.user_profiles[user_id].reputation = max(0, self.user_profiles[user_id].reputation - self.reputation_rules["comment_liked"])
            self._save_user_profile(self.user_profiles[user_id])
        
        self.logger.info(f"User {user_id} unliked comment {comment_id}")
        return True

    def accept_answer(self, comment_id: str, user_id: str) -> bool:
        """Accept a comment as the answer to a question"""
        if comment_id not in self.comments:
            return False
        
        comment = self.comments[comment_id]
        post_id = comment.post_id
        
        if post_id not in self.posts:
            return False
        
        # Check if user is the post author
        if self.posts[post_id].author_id != user_id:
            return False
        
        # Unaccept any previously accepted answers
        self.storage.execute(
            "UPDATE community_comments SET is_accepted_answer = 0 WHERE post_id = ?",
            (post_id,)
        )
        
        # Accept this answer
        comment.is_accepted_answer = True
        self._save_comment(comment)
        
        # Update user stats
        if comment.author_id in self.user_profiles:
            self.user_profiles[comment.author_id].reputation += self.reputation_rules["answer_accepted"]
            self.user_profiles[comment.author_id].helpful_count += 1
            self._save_user_profile(self.user_profiles[comment.author_id])
        
        self.logger.info(f"Comment {comment_id} accepted as answer for post {post_id}")
        return True

    def flag_content(self, content_type: str, content_id: str, reporter_id: str, 
                   reporter_name: str, reason: str, description: str) -> bool:
        """Flag content for moderation"""
        # Add to moderation queue
        self._add_to_moderation_queue(content_type, content_id, reason, reporter_id, 
                                     reporter_name, description)
        
        # Update content status
        if content_type == "post" and content_id in self.posts:
            self.posts[content_id].status = PostStatus.FLAGGED
            self._save_post(self.posts[content_id])
        elif content_type == "comment" and content_id in self.comments:
            self.comments[content_id].is_flagged = True
            self._save_comment(self.comments[content_id])
        
        # Update reporter stats
        if reporter_id in self.user_profiles:
            if content_type == "post":
                self.user_profiles[reporter_id].reputation += self.reputation_rules["post_flagged"]
            else:
                self.user_profiles[reporter_id].reputation += self.reputation_rules["comment_flagged"]
            self._save_user_profile(self.user_profiles[reporter_id])
        
        self.logger.info(f"Content {content_type}:{content_id} flagged by {reporter_id}")
        return True

    def get_moderation_queue(self, status: str = "pending") -> List[ModerationQueueItem]:
        """Get items from the moderation queue"""
        items = []
        
        for row in self.storage.query(
            "SELECT * FROM community_moderation_queue WHERE status = ? ORDER BY created_at ASC",
            (status,)
        ):
            item = ModerationQueueItem(
                id=row['id'],
                content_type=row['content_type'],
                content_id=row['content_id'],
                reason=row['reason'],
                reporter_id=row['reporter_id'],
                reporter_name=row['reporter_name'],
                description=row['description'],
                status=row['status'],
                created_at=datetime.fromisoformat(row['created_at']),
                resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                resolved_by=row['resolved_by'],
                action_taken=row['action_taken'],
                notes=row['notes']
            )
            items.append(item)
        
        return items

    def moderate_content(self, item_id: str, action: ModerationAction, 
                       moderator_id: str, notes: str = None) -> bool:
        """Take moderation action on content"""
        if item_id not in self.moderation_queue:
            return False
        
        item = self.moderation_queue[item_id]
        
        # Update moderation queue item
        item.status = "resolved"
        item.resolved_at = datetime.now()
        item.resolved_by = moderator_id
        item.action_taken = action.value
        item.notes = notes
        
        self.storage.execute(
            """
            UPDATE community_moderation_queue 
            SET status = ?, resolved_at = ?, resolved_by = ?, action_taken = ?, notes = ?
            WHERE id = ?
            """,
            (
                item.status,
                item.resolved_at.isoformat(),
                item.resolved_by,
                item.action_taken,
                item.notes,
                item_id
            )
        )
        
        # Apply action
        if action == ModerationAction.DELETE:
            if item.content_type == "post" and item.content_id in self.posts:
                self.posts[item.content_id].status = PostStatus.DELETED
                self._save_post(self.posts[item.content_id])
            elif item.content_type == "comment" and item.content_id in self.comments:
                # Mark comment as deleted (we don't actually delete to preserve history)
                self.comments[item.content_id].is_flagged = True
                self._save_comment(self.comments[item.content_id])
        
        elif action == ModerationAction.LOCK:
            if item.content_type == "post" and item.content_id in self.posts:
                self.posts[item.content_id].status = PostStatus.LOCKED
                self._save_post(self.posts[item.content_id])
        
        elif action == ModerationAction.BAN_USER:
            # Get user ID from content
            user_id = None
            if item.content_type == "post" and item.content_id in self.posts:
                user_id = self.posts[item.content_id].author_id
            elif item.content_type == "comment" and item.content_id in self.comments:
                user_id = self.comments[item.content_id].author_id
            
            if user_id and user_id in self.user_profiles:
                self.user_profiles[user_id].is_banned = True
                self._save_user_profile(self.user_profiles[user_id])
        
        # Update moderator stats
        if moderator_id in self.user_profiles:
            self.user_profiles[moderator_id].reputation += self.reputation_rules["moderation_action"]
            self._save_user_profile(self.user_profiles[moderator_id])
        
        self.logger.info(f"Moderation action {action.value} taken on {item.content_type}:{item.content_id}")
        return True

    def get_guidelines(self) -> List[CommunityGuideline]:
        """Get community guidelines"""
        return list(self.guidelines.values())

    def create_guideline(self, title: str, description: str, category: str) -> CommunityGuideline:
        """Create a new community guideline"""
        guideline_id = f"guideline_{uuid.uuid4().hex}"
        
        guideline = CommunityGuideline(
            id=guideline_id,
            title=title,
            description=description,
            category=category
        )
        
        # Save to database
        self.storage.execute(
            """
            INSERT INTO community_guidelines 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                guideline.id,
                guideline.title,
                guideline.description,
                guideline.category,
                1 if guideline.is_active else 0,
                guideline.created_at.isoformat(),
                guideline.updated_at.isoformat() if guideline.updated_at else None
            )
        )
        
        # Add to in-memory cache
        self.guidelines[guideline.id] = guideline
        
        self.logger.info(f"Created guideline: {title}")
        return guideline

    def get_community_stats(self) -> Dict:
        """Get community statistics"""
        # Total posts
        total_posts = self.storage.query(
            "SELECT COUNT(*) FROM community_posts WHERE status = 'published'"
        ).fetchone()[0]
        
        # Total comments
        total_comments = self.storage.query(
            "SELECT COUNT(*) FROM community_comments"
        ).fetchone()[0]
        
        # Total users
        total_users = self.storage.query(
            "SELECT COUNT(*) FROM community_user_profiles WHERE is_banned = 0"
        ).fetchone()[0]
        
        # Posts by category
        category_counts = {}
        for row in self.storage.query(
            "SELECT category, COUNT(*) as count FROM community_posts WHERE status = 'published' GROUP BY category"
        ):
            category_counts[row['category']] = row['count']
        
        # Posts by type
        type_counts = {}
        for row in self.storage.query(
            "SELECT post_type, COUNT(*) as count FROM community_posts WHERE status = 'published' GROUP BY post_type"
        ):
            type_counts[row['post_type']] = row['count']
        
        # Top contributors
        top_contributors = []
        for row in self.storage.query(
            """
            SELECT author_id, author_name, COUNT(*) as post_count
            FROM community_posts 
            WHERE status = 'published' 
            GROUP BY author_id 
            ORDER BY post_count DESC 
            LIMIT 5
            """
        ):
            top_contributors.append({
                "user_id": row['author_id'],
                "name": row['author_name'],
                "post_count": row['post_count']
            })
        
        return {
            "total_posts": total_posts,
            "total_comments": total_comments,
            "total_users": total_users,
            "category_distribution": category_counts,
            "type_distribution": type_counts,
            "top_contributors": top_contributors
        }

    def search_posts(self, query: str, category: str = None, limit: int = 20) -> List[CommunityPost]:
        """Search posts by content"""
        # Use NLP to search for relevant posts
        results = []
        
        for post in self.posts.values():
            if post.status != PostStatus.PUBLISHED:
                continue
            
            if category and post.category != category:
                continue
            
            # Calculate relevance score
            title_score = self.nlp.calculate_similarity(query, post.title)
            content_score = self.nlp.calculate_similarity(query, post.content)
            
            # Combined score
            combined_score = (title_score * 0.6) + (content_score * 0.4)
            
            if combined_score > 0.3:  # Minimum relevance threshold
                results.append((post, combined_score))
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return [post for post, score in results[:limit]]

    def get_trending_posts(self, days: int = 7, limit: int = 10) -> List[CommunityPost]:
        """Get trending posts based on engagement"""
        since = datetime.now() - timedelta(days=days)
        
        # Get posts with high engagement
        trending_posts = []
        for row in self.storage.query(
            """
            SELECT * FROM community_posts 
            WHERE status = 'published' AND created_at >= ?
            ORDER BY (like_count * 2 + comment_count) DESC, created_at DESC
            LIMIT ?
            """,
            (since.isoformat(), limit)
        ):
            post = CommunityPost(
                id=row['id'],
                title=row['title'],
                content=row['content'],
                author_id=row['author_id'],
                author_name=row['author_name'],
                post_type=PostType(row['post_type']),
                status=PostStatus(row['status']),
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                view_count=row['view_count'],
                like_count=row['like_count'],
                comment_count=row['comment_count'],
                is_pinned=bool(row['is_pinned']),
                is_featured=bool(row['is_featured']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                last_activity=datetime.fromisoformat(row['last_activity']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            trending_posts.append(post)
        
        return trending_posts