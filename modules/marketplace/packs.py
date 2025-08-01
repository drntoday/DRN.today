# modules/marketplace/packs.py

import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

import pandas as pd
from fastapi import UploadFile
from sqlalchemy import desc

from engine.storage import Storage
from engine.license import LicenseManager
from ai.nlp import NLPProcessor


class PackType(Enum):
    ADAPTER = "adapter"
    BLUEPRINT = "blueprint"
    TEMPLATE = "template"
    ENRICHMENT = "enrichment"
    WORKFLOW = "workflow"


class PackStatus(Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class PurchaseStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class LeadPack:
    id: str
    name: str
    description: str
    pack_type: PackType
    version: str
    author: str
    author_id: str
    category: str
    tags: List[str]
    price: float = 0.0
    rating: float = 0.0
    download_count: int = 0
    purchase_count: int = 0
    status: PackStatus = PackStatus.DRAFT
    file_path: Optional[str] = None
    preview_url: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    compatibility: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    published_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PackReview:
    id: str
    pack_id: str
    user_id: str
    rating: int
    title: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


@dataclass
class PackPurchase:
    id: str
    pack_id: str
    user_id: str
    price: float
    status: PurchaseStatus
    transaction_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class MarketplaceEngine:
    def __init__(self, storage: Storage, license_manager: LicenseManager, nlp_processor: NLPProcessor):
        self.storage = storage
        self.license_manager = license_manager
        self.nlp = nlp_processor
        self.logger = logging.getLogger("marketplace_engine")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load packs from storage
        self.packs: Dict[str, LeadPack] = {}
        self._load_packs()
        
        # Set up file storage
        self.pack_storage_path = Path("resources/marketplace/packs")
        self.pack_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Categories and tags
        self.categories = [
            "Lead Generation",
            "Email Outreach",
            "Social Media",
            "Web Scraping",
            "Data Enrichment",
            "Analytics",
            "Automation",
            "Integration"
        ]
        
        # Popular tags
        self.popular_tags = [
            "b2b", "saas", "technology", "healthcare", "finance",
            "ecommerce", "real estate", "education", "marketing",
            "sales", "hr", "recruiting", "customer support"
        ]

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS lead_packs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            pack_type TEXT NOT NULL,
            version TEXT NOT NULL,
            author TEXT NOT NULL,
            author_id TEXT NOT NULL,
            category TEXT NOT NULL,
            tags TEXT,
            price REAL DEFAULT 0.0,
            rating REAL DEFAULT 0.0,
            download_count INTEGER DEFAULT 0,
            purchase_count INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            file_path TEXT,
            preview_url TEXT,
            requirements TEXT,
            compatibility TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            published_at TEXT,
            metadata TEXT
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS pack_reviews (
            id TEXT PRIMARY KEY,
            pack_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            FOREIGN KEY (pack_id) REFERENCES lead_packs (id)
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS pack_purchases (
            id TEXT PRIMARY KEY,
            pack_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL,
            transaction_id TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (pack_id) REFERENCES lead_packs (id)
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS pack_downloads (
            id TEXT PRIMARY KEY,
            pack_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            download_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (pack_id) REFERENCES lead_packs (id)
        )
        """)

    def _load_packs(self):
        """Load lead packs from storage"""
        for row in self.storage.query("SELECT * FROM lead_packs"):
            pack = LeadPack(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                pack_type=PackType(row['pack_type']),
                version=row['version'],
                author=row['author'],
                author_id=row['author_id'],
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                price=row['price'],
                rating=row['rating'],
                download_count=row['download_count'],
                purchase_count=row['purchase_count'],
                status=PackStatus(row['status']),
                file_path=row['file_path'],
                preview_url=row['preview_url'],
                requirements=json.loads(row['requirements']) if row['requirements'] else [],
                compatibility=json.loads(row['compatibility']) if row['compatibility'] else [],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            self.packs[pack.id] = pack

    def create_pack(self, name: str, description: str, pack_type: PackType,
                   version: str, author: str, author_id: str, category: str,
                   tags: List[str], price: float = 0.0, requirements: List[str] = None,
                   compatibility: List[str] = None, metadata: Dict = None) -> LeadPack:
        """Create a new lead pack"""
        pack_id = f"pack_{uuid.uuid4().hex}"
        
        pack = LeadPack(
            id=pack_id,
            name=name,
            description=description,
            pack_type=pack_type,
            version=version,
            author=author,
            author_id=author_id,
            category=category,
            tags=tags,
            price=price,
            requirements=requirements or [],
            compatibility=compatibility or [],
            metadata=metadata or {}
        )
        
        # Save to database
        self._save_pack(pack)
        
        self.logger.info(f"Created lead pack: {pack_id} - {name}")
        return pack

    def update_pack(self, pack_id: str, **kwargs) -> Optional[LeadPack]:
        """Update a lead pack"""
        if pack_id not in self.packs:
            return None
        
        pack = self.packs[pack_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(pack, key):
                setattr(pack, key, value)
        
        pack.updated_at = datetime.now()
        
        # Save to database
        self._save_pack(pack)
        
        self.logger.info(f"Updated lead pack: {pack_id}")
        return pack

    def delete_pack(self, pack_id: str) -> bool:
        """Delete a lead pack"""
        if pack_id not in self.packs:
            return False
        
        # Delete from database
        self.storage.execute("DELETE FROM lead_packs WHERE id = ?", (pack_id,))
        
        # Delete from in-memory cache
        del self.packs[pack_id]
        
        # Delete file if exists
        pack = self.packs.get(pack_id)
        if pack and pack.file_path and os.path.exists(pack.file_path):
            os.remove(pack.file_path)
        
        self.logger.info(f"Deleted lead pack: {pack_id}")
        return True

    def _save_pack(self, pack: LeadPack):
        """Save a lead pack to database"""
        self.storage.execute(
            """
            INSERT OR REPLACE INTO lead_packs 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pack.id,
                pack.name,
                pack.description,
                pack.pack_type.value,
                pack.version,
                pack.author,
                pack.author_id,
                pack.category,
                json.dumps(pack.tags),
                pack.price,
                pack.rating,
                pack.download_count,
                pack.purchase_count,
                pack.status.value,
                pack.file_path,
                pack.preview_url,
                json.dumps(pack.requirements),
                json.dumps(pack.compatibility),
                pack.created_at.isoformat(),
                pack.updated_at.isoformat(),
                pack.published_at.isoformat() if pack.published_at else None,
                json.dumps(pack.metadata)
            )
        )
        
        # Update in-memory cache
        self.packs[pack.id] = pack

    def upload_pack_file(self, pack_id: str, file: UploadFile) -> bool:
        """Upload a file for a lead pack"""
        if pack_id not in self.packs:
            return False
        
        pack = self.packs[pack_id]
        
        # Create pack directory if it doesn't exist
        pack_dir = self.pack_storage_path / pack_id
        pack_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = pack_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update pack
        pack.file_path = str(file_path)
        self._save_pack(pack)
        
        self.logger.info(f"Uploaded file for pack {pack_id}: {file.filename}")
        return True

    def download_pack(self, pack_id: str, user_id: str) -> Optional[str]:
        """Download a lead pack file"""
        if pack_id not in self.packs:
            return None
        
        pack = self.packs[pack_id]
        
        # Check if user has purchased the pack or if it's free
        if pack.price > 0:
            purchase = self.storage.query(
                "SELECT * FROM pack_purchases WHERE pack_id = ? AND user_id = ? AND status = ?",
                (pack_id, user_id, PurchaseStatus.COMPLETED.value)
            ).fetchone()
            
            if not purchase:
                return None
        
        # Check if file exists
        if not pack.file_path or not os.path.exists(pack.file_path):
            return None
        
        # Create temporary copy
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"{pack_id}_{pack.name.replace(' ', '_')}.zip")
        
        # Create ZIP file
        with zipfile.ZipFile(temp_path, 'w') as zipf:
            zipf.write(pack.file_path, os.path.basename(pack.file_path))
        
        # Record download
        self._record_download(pack_id, user_id, temp_path)
        
        # Update download count
        pack.download_count += 1
        self._save_pack(pack)
        
        self.logger.info(f"Downloaded pack {pack_id} for user {user_id}")
        return temp_path

    def _record_download(self, pack_id: str, user_id: str, download_path: str):
        """Record a pack download"""
        download_id = f"dl_{uuid.uuid4().hex}"
        
        self.storage.execute(
            """
            INSERT INTO pack_downloads 
            VALUES (?, ?, ?, ?)
            """,
            (
                download_id,
                pack_id,
                user_id,
                download_path,
                datetime.now().isoformat()
            )
        )

    def purchase_pack(self, pack_id: str, user_id: str, transaction_id: str = None) -> Optional[PackPurchase]:
        """Purchase a lead pack"""
        if pack_id not in self.packs:
            return None
        
        pack = self.packs[pack_id]
        
        # Check if already purchased
        existing = self.storage.query(
            "SELECT * FROM pack_purchases WHERE pack_id = ? AND user_id = ? AND status = ?",
            (pack_id, user_id, PurchaseStatus.COMPLETED.value)
        ).fetchone()
        
        if existing:
            return None
        
        # Create purchase record
        purchase_id = f"pur_{uuid.uuid4().hex}"
        purchase = PackPurchase(
            id=purchase_id,
            pack_id=pack_id,
            user_id=user_id,
            price=pack.price,
            status=PurchaseStatus.PENDING,
            transaction_id=transaction_id
        )
        
        # Save to database
        self.storage.execute(
            """
            INSERT INTO pack_purchases 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                purchase.id,
                purchase.pack_id,
                purchase.user_id,
                purchase.price,
                purchase.status.value,
                purchase.transaction_id,
                purchase.created_at.isoformat()
            )
        )
        
        # In a real implementation, we would process payment here
        # For now, we'll mark as completed
        purchase.status = PurchaseStatus.COMPLETED
        purchase.completed_at = datetime.now()
        
        self.storage.execute(
            """
            UPDATE pack_purchases 
            SET status = ?, completed_at = ? 
            WHERE id = ?
            """,
            (
                purchase.status.value,
                purchase.completed_at.isoformat(),
                purchase.id
            )
        )
        
        # Update pack purchase count
        pack.purchase_count += 1
        self._save_pack(pack)
        
        self.logger.info(f"Purchased pack {pack_id} for user {user_id}")
        return purchase

    def add_review(self, pack_id: str, user_id: str, rating: int, title: str, content: str) -> Optional[PackReview]:
        """Add a review for a lead pack"""
        if pack_id not in self.packs:
            return None
        
        # Check if user already reviewed this pack
        existing = self.storage.query(
            "SELECT * FROM pack_reviews WHERE pack_id = ? AND user_id = ?",
            (pack_id, user_id)
        ).fetchone()
        
        if existing:
            # Update existing review
            review_id = existing['id']
            self.storage.execute(
                """
                UPDATE pack_reviews 
                SET rating = ?, title = ?, content = ?, updated_at = ? 
                WHERE id = ?
                """,
                (
                    rating,
                    title,
                    content,
                    datetime.now().isoformat(),
                    review_id
                )
            )
        else:
            # Create new review
            review_id = f"rev_{uuid.uuid4().hex}"
            self.storage.execute(
                """
                INSERT INTO pack_reviews 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    pack_id,
                    user_id,
                    rating,
                    title,
                    content,
                    datetime.now().isoformat(),
                    None
                )
            )
        
        # Update pack rating
        self._update_pack_rating(pack_id)
        
        self.logger.info(f"Added review for pack {pack_id} by user {user_id}")
        return self.get_review(review_id)

    def _update_pack_rating(self, pack_id: str):
        """Update the average rating for a pack"""
        # Calculate average rating
        result = self.storage.query(
            "SELECT AVG(rating) as avg_rating FROM pack_reviews WHERE pack_id = ?",
            (pack_id,)
        ).fetchone()
        
        avg_rating = result['avg_rating'] or 0.0
        
        # Update pack
        if pack_id in self.packs:
            self.packs[pack_id].rating = avg_rating
            self._save_pack(self.packs[pack_id])

    def get_review(self, review_id: str) -> Optional[PackReview]:
        """Get a review by ID"""
        row = self.storage.query(
            "SELECT * FROM pack_reviews WHERE id = ?",
            (review_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return PackReview(
            id=row['id'],
            pack_id=row['pack_id'],
            user_id=row['user_id'],
            rating=row['rating'],
            title=row['title'],
            content=row['content'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )

    def get_pack_reviews(self, pack_id: str) -> List[PackReview]:
        """Get all reviews for a pack"""
        reviews = []
        
        for row in self.storage.query(
            "SELECT * FROM pack_reviews WHERE pack_id = ? ORDER BY created_at DESC",
            (pack_id,)
        ):
            review = PackReview(
                id=row['id'],
                pack_id=row['pack_id'],
                user_id=row['user_id'],
                rating=row['rating'],
                title=row['title'],
                content=row['content'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
            reviews.append(review)
        
        return reviews

    def search_packs(self, query: str = "", category: str = "", pack_type: PackType = None,
                    tags: List[str] = None, author: str = "", min_rating: float = 0.0,
                    max_price: float = None, status: PackStatus = PackStatus.PUBLISHED,
                    sort_by: str = "created_at", sort_order: str = "desc",
                    limit: int = 20, offset: int = 0) -> List[LeadPack]:
        """Search for lead packs"""
        # Build query
        sql = "SELECT * FROM lead_packs WHERE 1=1"
        params = []
        
        # Add filters
        if query:
            sql += " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        if pack_type:
            sql += " AND pack_type = ?"
            params.append(pack_type.value)
        
        if tags:
            for tag in tags:
                sql += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        
        if author:
            sql += " AND author = ?"
            params.append(author)
        
        if min_rating > 0:
            sql += " AND rating >= ?"
            params.append(min_rating)
        
        if max_price is not None:
            sql += " AND price <= ?"
            params.append(max_price)
        
        if status:
            sql += " AND status = ?"
            params.append(status.value)
        
        # Add sorting
        valid_sort_fields = ["name", "rating", "price", "download_count", "purchase_count", "created_at", "updated_at"]
        if sort_by in valid_sort_fields:
            sql += f" ORDER BY {sort_by}"
        else:
            sql += " ORDER BY created_at"
        
        if sort_order.lower() == "asc":
            sql += " ASC"
        else:
            sql += " DESC"
        
        # Add pagination
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        # Execute query
        packs = []
        for row in self.storage.query(sql, params):
            pack = LeadPack(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                pack_type=PackType(row['pack_type']),
                version=row['version'],
                author=row['author'],
                author_id=row['author_id'],
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                price=row['price'],
                rating=row['rating'],
                download_count=row['download_count'],
                purchase_count=row['purchase_count'],
                status=PackStatus(row['status']),
                file_path=row['file_path'],
                preview_url=row['preview_url'],
                requirements=json.loads(row['requirements']) if row['requirements'] else [],
                compatibility=json.loads(row['compatibility']) if row['compatibility'] else [],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            packs.append(pack)
        
        return packs

    def get_pack(self, pack_id: str) -> Optional[LeadPack]:
        """Get a lead pack by ID"""
        return self.packs.get(pack_id)

    def get_user_packs(self, user_id: str, include_purchased: bool = False) -> List[LeadPack]:
        """Get packs created by a user"""
        packs = []
        
        # Get packs created by user
        for row in self.storage.query(
            "SELECT * FROM lead_packs WHERE author_id = ? ORDER BY created_at DESC",
            (user_id,)
        ):
            pack = LeadPack(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                pack_type=PackType(row['pack_type']),
                version=row['version'],
                author=row['author'],
                author_id=row['author_id'],
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                price=row['price'],
                rating=row['rating'],
                download_count=row['download_count'],
                purchase_count=row['purchase_count'],
                status=PackStatus(row['status']),
                file_path=row['file_path'],
                preview_url=row['preview_url'],
                requirements=json.loads(row['requirements']) if row['requirements'] else [],
                compatibility=json.loads(row['compatibility']) if row['compatibility'] else [],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            packs.append(pack)
        
        # Get packs purchased by user
        if include_purchased:
            purchased_pack_ids = set()
            for row in self.storage.query(
                "SELECT pack_id FROM pack_purchases WHERE user_id = ? AND status = ?",
                (user_id, PurchaseStatus.COMPLETED.value)
            ):
                purchased_pack_ids.add(row['pack_id'])
            
            for pack_id in purchased_pack_ids:
                if pack_id in self.packs:
                    packs.append(self.packs[pack_id])
        
        return packs

    def get_popular_packs(self, limit: int = 10, days: int = 30) -> List[LeadPack]:
        """Get popular packs based on downloads and purchases"""
        since = datetime.now() - timedelta(days=days)
        
        # Get pack IDs sorted by popularity
        popular_pack_ids = []
        for row in self.storage.query(
            """
            SELECT p.id, 
                   (p.download_count + p.purchase_count) as popularity_score
            FROM lead_packs p
            WHERE p.status = ? AND p.created_at >= ?
            ORDER BY popularity_score DESC
            LIMIT ?
            """,
            (PackStatus.PUBLISHED.value, since.isoformat(), limit)
        ):
            popular_pack_ids.append(row['id'])
        
        # Get pack objects
        popular_packs = []
        for pack_id in popular_pack_ids:
            if pack_id in self.packs:
                popular_packs.append(self.packs[pack_id])
        
        return popular_packs

    def get_recommended_packs(self, user_id: str, limit: int = 5) -> List[LeadPack]:
        """Get recommended packs for a user"""
        # Get user's purchase history
        purchased_categories = set()
        purchased_tags = set()
        
        for row in self.storage.query(
            """
            SELECT p.category, p.tags
            FROM pack_purchases pp
            JOIN lead_packs p ON pp.pack_id = p.id
            WHERE pp.user_id = ? AND pp.status = ?
            """,
            (user_id, PurchaseStatus.COMPLETED.value)
        ):
            purchased_categories.add(row['category'])
            tags = json.loads(row['tags']) if row['tags'] else []
            purchased_tags.update(tags)
        
        # Get packs with similar categories or tags
        recommended_packs = []
        
        # Exclude packs already purchased
        purchased_pack_ids = set()
        for row in self.storage.query(
            "SELECT pack_id FROM pack_purchases WHERE user_id = ? AND status = ?",
            (user_id, PurchaseStatus.COMPLETED.value)
        ):
            purchased_pack_ids.add(row['pack_id'])
        
        # Find packs with matching categories
        for category in purchased_categories:
            for row in self.storage.query(
                """
                SELECT * FROM lead_packs 
                WHERE category = ? AND status = ? AND id NOT IN ({})
                ORDER BY rating DESC, download_count DESC
                LIMIT ?
                """.format(','.join(['?'] * len(purchased_pack_ids))),
                [category, PackStatus.PUBLISHED.value] + list(purchased_pack_ids) + [limit]
            ):
                pack = LeadPack(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    pack_type=PackType(row['pack_type']),
                    version=row['version'],
                    author=row['author'],
                    author_id=row['author_id'],
                    category=row['category'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    price=row['price'],
                    rating=row['rating'],
                    download_count=row['download_count'],
                    purchase_count=row['purchase_count'],
                    status=PackStatus(row['status']),
                    file_path=row['file_path'],
                    preview_url=row['preview_url'],
                    requirements=json.loads(row['requirements']) if row['requirements'] else [],
                    compatibility=json.loads(row['compatibility']) if row['compatibility'] else [],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                recommended_packs.append(pack)
        
        # Find packs with matching tags
        for tag in purchased_tags:
            for row in self.storage.query(
                """
                SELECT * FROM lead_packs 
                WHERE tags LIKE ? AND status = ? AND id NOT IN ({})
                ORDER BY rating DESC, download_count DESC
                LIMIT ?
                """.format(','.join(['?'] * len(purchased_pack_ids))),
                [f"%{tag}%", PackStatus.PUBLISHED.value] + list(purchased_pack_ids) + [limit]
            ):
                pack = LeadPack(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    pack_type=PackType(row['pack_type']),
                    version=row['version'],
                    author=row['author'],
                    author_id=row['author_id'],
                    category=row['category'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    price=row['price'],
                    rating=row['rating'],
                    download_count=row['download_count'],
                    purchase_count=row['purchase_count'],
                    status=PackStatus(row['status']),
                    file_path=row['file_path'],
                    preview_url=row['preview_url'],
                    requirements=json.loads(row['requirements']) if row['requirements'] else [],
                    compatibility=json.loads(row['compatibility']) if row['compatibility'] else [],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                recommended_packs.append(pack)
        
        # Remove duplicates and limit
        unique_packs = {}
        for pack in recommended_packs:
            if pack.id not in unique_packs:
                unique_packs[pack.id] = pack
        
        return list(unique_packs.values())[:limit]

    def get_marketplace_stats(self) -> Dict:
        """Get marketplace statistics"""
        # Total packs
        total_packs = self.storage.query(
            "SELECT COUNT(*) FROM lead_packs"
        ).fetchone()[0]
        
        # Published packs
        published_packs = self.storage.query(
            "SELECT COUNT(*) FROM lead_packs WHERE status = ?",
            (PackStatus.PUBLISHED.value,)
        ).fetchone()[0]
        
        # Total downloads
        total_downloads = self.storage.query(
            "SELECT SUM(download_count) FROM lead_packs"
        ).fetchone()[0] or 0
        
        # Total purchases
        total_purchases = self.storage.query(
            "SELECT COUNT(*) FROM pack_purchases WHERE status = ?",
            (PurchaseStatus.COMPLETED.value,)
        ).fetchone()[0]
        
        # Total revenue
        total_revenue = self.storage.query(
            "SELECT SUM(price) FROM pack_purchases WHERE status = ?",
            (PurchaseStatus.COMPLETED.value,)
        ).fetchone()[0] or 0
        
        # Packs by category
        category_counts = {}
        for row in self.storage.query(
            "SELECT category, COUNT(*) as count FROM lead_packs WHERE status = ? GROUP BY category",
            (PackStatus.PUBLISHED.value,)
        ):
            category_counts[row['category']] = row['count']
        
        # Packs by type
        type_counts = {}
        for row in self.storage.query(
            "SELECT pack_type, COUNT(*) as count FROM lead_packs WHERE status = ? GROUP BY pack_type",
            (PackStatus.PUBLISHED.value,)
        ):
            type_counts[row['pack_type']] = row['count']
        
        return {
            "total_packs": total_packs,
            "published_packs": published_packs,
            "total_downloads": total_downloads,
            "total_purchases": total_purchases,
            "total_revenue": total_revenue,
            "category_distribution": category_counts,
            "type_distribution": type_counts
        }

    def submit_for_review(self, pack_id: str) -> bool:
        """Submit a pack for review"""
        if pack_id not in self.packs:
            return False
        
        pack = self.packs[pack_id]
        pack.status = PackStatus.PENDING_REVIEW
        self._save_pack(pack)
        
        self.logger.info(f"Submitted pack {pack_id} for review")
        return True

    def approve_pack(self, pack_id: str) -> bool:
        """Approve a pack"""
        if pack_id not in self.packs:
            return False
        
        pack = self.packs[pack_id]
        pack.status = PackStatus.APPROVED
        pack.published_at = datetime.now()
        self._save_pack(pack)
        
        self.logger.info(f"Approved pack {pack_id}")
        return True

    def reject_pack(self, pack_id: str, reason: str) -> bool:
        """Reject a pack"""
        if pack_id not in self.packs:
            return False
        
        pack = self.packs[pack_id]
        pack.status = PackStatus.REJECTED
        pack.metadata["rejection_reason"] = reason
        self._save_pack(pack)
        
        self.logger.info(f"Rejected pack {pack_id}: {reason}")
        return True

    def publish_pack(self, pack_id: str) -> bool:
        """Publish a pack"""
        if pack_id not in self.packs:
            return False
        
        pack = self.packs[pack_id]
        if pack.status != PackStatus.APPROVED:
            return False
        
        pack.status = PackStatus.PUBLISHED
        pack.published_at = datetime.now()
        self._save_pack(pack)
        
        self.logger.info(f"Published pack {pack_id}")
        return True

    def archive_pack(self, pack_id: str) -> bool:
        """Archive a pack"""
        if pack_id not in self.packs:
            return False
        
        pack = self.packs[pack_id]
        pack.status = PackStatus.ARCHIVED
        self._save_pack(pack)
        
        self.logger.info(f"Archived pack {pack_id}")
        return True

    def get_pending_reviews(self) -> List[LeadPack]:
        """Get packs pending review"""
        pending_packs = []
        
        for row in self.storage.query(
            "SELECT * FROM lead_packs WHERE status = ? ORDER BY created_at ASC",
            (PackStatus.PENDING_REVIEW.value,)
        ):
            pack = LeadPack(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                pack_type=PackType(row['pack_type']),
                version=row['version'],
                author=row['author'],
                author_id=row['author_id'],
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                price=row['price'],
                rating=row['rating'],
                download_count=row['download_count'],
                purchase_count=row['purchase_count'],
                status=PackStatus(row['status']),
                file_path=row['file_path'],
                preview_url=row['preview_url'],
                requirements=json.loads(row['requirements']) if row['requirements'] else [],
                compatibility=json.loads(row['compatibility']) if row['compatibility'] else [],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            pending_packs.append(pack)
        
        return pending_packs