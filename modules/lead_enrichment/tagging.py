# modules/lead_enrichment/tagging.py

import json
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from engine.storage import Storage
from ai.nlp import NLPProcessor
from ai.scoring import LeadScorer


class TagType(Enum):
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    FIRMAGGRAPHIC = "firmagraphic"
    TECHNOGRAPHIC = "technographic"
    CUSTOM = "custom"


@dataclass
class Tag:
    id: str
    name: str
    description: str
    type: TagType
    keywords: List[str] = field(default_factory=list)
    rules: Dict = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Persona:
    id: str
    name: str
    description: str
    tag_combinations: List[List[str]]  # OR logic between groups, AND within groups
    created_at: datetime = field(default_factory=datetime.now)


class DNAStyleTaggingSystem:
    def __init__(self, storage: Storage, nlp_processor: NLPProcessor, scorer: LeadScorer):
        self.storage = storage
        self.nlp = nlp_processor
        self.scorer = scorer
        self.tags: Dict[str, Tag] = {}
        self.personas: Dict[str, Persona] = {}
        self.lead_tags: Dict[str, Set[str]] = {}  # lead_id -> set of tag_ids
        self._initialize_tables()
        self._load_system_data()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            type TEXT NOT NULL,
            keywords TEXT,
            rules TEXT,
            weight REAL DEFAULT 1.0,
            created_at TEXT,
            updated_at TEXT
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS personas (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            tag_combinations TEXT,
            created_at TEXT
        )
        """)

        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS lead_tags (
            lead_id TEXT,
            tag_id TEXT,
            source TEXT,
            confidence REAL,
            created_at TEXT,
            PRIMARY KEY (lead_id, tag_id)
        )
        """)

    def _load_system_data(self):
        """Load tags and personas from storage"""
        # Load tags
        for row in self.storage.query("SELECT * FROM tags"):
            tag = Tag(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                type=TagType(row['type']),
                keywords=json.loads(row['keywords']) if row['keywords'] else [],
                rules=json.loads(row['rules']) if row['rules'] else {},
                weight=row['weight'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
            self.tags[tag.id] = tag

        # Load personas
        for row in self.storage.query("SELECT * FROM personas"):
            persona = Persona(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                tag_combinations=json.loads(row['tag_combinations']),
                created_at=datetime.fromisoformat(row['created_at'])
            )
            self.personas[persona.id] = persona

        # Load lead-tag associations
        for row in self.storage.query("SELECT * FROM lead_tags"):
            lead_id = row['lead_id']
            tag_id = row['tag_id']
            if lead_id not in self.lead_tags:
                self.lead_tags[lead_id] = set()
            self.lead_tags[lead_id].add(tag_id)

    def create_tag(self, name: str, description: str, tag_type: TagType,
                  keywords: List[str] = None, rules: Dict = None,
                  weight: float = 1.0) -> Tag:
        """Create a new tag"""
        tag_id = f"tag_{int(datetime.now().timestamp())}"
        tag = Tag(
            id=tag_id,
            name=name,
            description=description,
            type=tag_type,
            keywords=keywords or [],
            rules=rules or {},
            weight=weight
        )
        self.tags[tag_id] = tag
        
        self.storage.execute(
            "INSERT INTO tags VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tag_id, name, description, tag_type.value,
                json.dumps(keywords), json.dumps(rules), weight,
                tag.created_at.isoformat(), tag.updated_at.isoformat()
            )
        )
        return tag

    def create_persona(self, name: str, description: str,
                      tag_combinations: List[List[str]]) -> Persona:
        """Create a new persona with tag combinations"""
        persona_id = f"persona_{int(datetime.now().timestamp())}"
        persona = Persona(
            id=persona_id,
            name=name,
            description=description,
            tag_combinations=tag_combinations
        )
        self.personas[persona_id] = persona
        
        self.storage.execute(
            "INSERT INTO personas VALUES (?, ?, ?, ?, ?)",
            (
                persona_id, name, description,
                json.dumps(tag_combinations), persona.created_at.isoformat()
            )
        )
        return persona

    def apply_tags_to_lead(self, lead_id: str, lead_data: Dict) -> Dict[str, float]:
        """Apply all relevant tags to a lead and return tag scores"""
        tag_scores = {}
        text_content = self._extract_text_content(lead_data)
        
        for tag_id, tag in self.tags.items():
            score = 0.0
            
            # Keyword-based matching
            if tag.keywords:
                score += self._calculate_keyword_score(text_content, tag.keywords) * tag.weight
            
            # Rule-based matching
            if tag.rules:
                score += self._evaluate_rules(lead_data, tag.rules) * tag.weight
            
            # NLP-based semantic matching
            if tag.type in [TagType.DEMOGRAPHIC, TagType.BEHAVIORAL]:
                score += self._calculate_semantic_score(text_content, tag.description) * tag.weight
            
            if score > 0.3:  # Minimum confidence threshold
                tag_scores[tag_id] = min(score, 1.0)
                self._assign_tag_to_lead(lead_id, tag_id, score, "automatic")
        
        return tag_scores

    def _extract_text_content(self, lead_data: Dict) -> str:
        """Extract all text content from lead data for analysis"""
        text_fields = [
            lead_data.get('name', ''),
            lead_data.get('job_title', ''),
            lead_data.get('company', ''),
            lead_data.get('bio', ''),
            lead_data.get('industry', ''),
            ' '.join(lead_data.get('skills', [])),
            ' '.join(lead_data.get('keywords', []))
        ]
        return ' '.join(filter(None, text_fields))

    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword matching score"""
        if not keywords:
            return 0.0
            
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(matches / len(keywords), 1.0)

    def _evaluate_rules(self, lead_data: Dict, rules: Dict) -> float:
        """Evaluate rule-based conditions"""
        score = 0.0
        
        for field, condition in rules.items():
            if field not in lead_data:
                continue
                
            value = lead_data[field]
            
            if 'equals' in condition:
                if value == condition['equals']:
                    score += 0.5
            elif 'contains' in condition:
                if condition['contains'].lower() in str(value).lower():
                    score += 0.7
            elif 'range' in condition:
                min_val, max_val = condition['range']
                if min_val <= value <= max_val:
                    score += 0.6
            elif 'in' in condition:
                if value in condition['in']:
                    score += 0.8
        
        return min(score, 1.0)

    def _calculate_semantic_score(self, text: str, description: str) -> float:
        """Calculate semantic similarity using NLP"""
        if not text or not description:
            return 0.0
            
        # Use TF-IDF for semantic similarity
        corpus = [text, description]
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        vectors = vectorizer.toarray()
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        return cosine_sim

    def _assign_tag_to_lead(self, lead_id: str, tag_id: str, confidence: float, source: str):
        """Assign a tag to a lead and persist"""
        if lead_id not in self.lead_tags:
            self.lead_tags[lead_id] = set()
            
        if tag_id not in self.lead_tags[lead_id]:
            self.lead_tags[lead_id].add(tag_id)
            
            self.storage.execute(
                "INSERT OR REPLACE INTO lead_tags VALUES (?, ?, ?, ?, ?)",
                (
                    lead_id, tag_id, source,
                    confidence, datetime.now().isoformat()
                )
            )

    def get_lead_tags(self, lead_id: str) -> List[Tag]:
        """Get all tags assigned to a lead"""
        if lead_id not in self.lead_tags:
            return []
            
        return [self.tags[tag_id] for tag_id in self.lead_tags[lead_id] if tag_id in self.tags]

    def get_leads_by_persona(self, persona_id: str) -> List[str]:
        """Get all lead IDs that match a persona"""
        if persona_id not in self.personas:
            return []
            
        persona = self.personas[persona_id]
        matching_leads = []
        
        for lead_id, tag_ids in self.lead_tags.items():
            # Check if lead matches any of the tag combinations (OR logic)
            for combination in persona.tag_combinations:
                # Check if lead has all tags in the combination (AND logic)
                if all(tag_id in tag_ids for tag_id in combination):
                    matching_leads.append(lead_id)
                    break  # No need to check other combinations
        
        return matching_leads

    def get_persona_insights(self, persona_id: str) -> Dict:
        """Get insights about leads matching a persona"""
        lead_ids = self.get_leads_by_persona(persona_id)
        if not lead_ids:
            return {}
            
        # Aggregate data from leads
        lead_data = []
        for lead_id in lead_ids:
            # In a real implementation, we'd fetch full lead data from storage
            lead_data.append({"id": lead_id})  # Simplified for example
            
        # Calculate insights using NLP and scoring
        insights = {
            "lead_count": len(lead_ids),
            "avg_score": self.scorer.calculate_average_score(lead_ids),
            "top_keywords": self.nlp.extract_top_keywords(lead_data),
            "industry_distribution": self._calculate_industry_distribution(lead_data),
            "seniority_distribution": self._calculate_seniority_distribution(lead_data)
        }
        
        return insights

    def _calculate_industry_distribution(self, leads: List[Dict]) -> Dict[str, float]:
        """Calculate industry distribution for persona leads"""
        industries = [lead.get("industry", "Unknown") for lead in leads]
        industry_counts = pd.Series(industries).value_counts(normalize=True).to_dict()
        return industry_counts

    def _calculate_seniority_distribution(self, leads: List[Dict]) -> Dict[str, float]:
        """Calculate seniority distribution for persona leads"""
        seniority_levels = []
        for lead in leads:
            title = lead.get("job_title", "").lower()
            if any(word in title for word in ["chief", "vp", "head"]):
                seniority_levels.append("Executive")
            elif any(word in title for word in ["director", "manager"]):
                seniority_levels.append("Management")
            elif any(word in title for word in ["senior", "lead"]):
                seniority_levels.append("Senior")
            else:
                seniority_levels.append("Junior")
                
        seniority_counts = pd.Series(seniority_levels).value_counts(normalize=True).to_dict()
        return seniority_counts

    def update_tag_weights(self, tag_updates: Dict[str, float]):
        """Update tag weights based on performance data"""
        for tag_id, new_weight in tag_updates.items():
            if tag_id in self.tags:
                self.tags[tag_id].weight = new_weight
                self.tags[tag_id].updated_at = datetime.now()
                
                self.storage.execute(
                    "UPDATE tags SET weight = ?, updated_at = ? WHERE id = ?",
                    (new_weight, datetime.now().isoformat(), tag_id)
                )

    def bulk_tag_leads(self, lead_ids: List[str]):
        """Apply tags to a batch of leads efficiently"""
        for lead_id in lead_ids:
            # In a real implementation, we'd fetch full lead data from storage
            lead_data = {"id": lead_id}  # Simplified for example
            self.apply_tags_to_lead(lead_id, lead_data)

    def export_tag_data(self, format: str = "json") -> str:
        """Export all tag data in specified format"""
        data = {
            "tags": [
                {
                    "id": tag.id,
                    "name": tag.name,
                    "description": tag.description,
                    "type": tag.type.value,
                    "keywords": tag.keywords,
                    "rules": tag.rules,
                    "weight": tag.weight
                }
                for tag in self.tags.values()
            ],
            "personas": [
                {
                    "id": persona.id,
                    "name": persona.name,
                    "description": persona.description,
                    "tag_combinations": persona.tag_combinations
                }
                for persona in self.personas.values()
            ]
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        elif format.lower() == "csv":
            # Convert to CSV format
            df = pd.DataFrame([tag.__dict__ for tag in self.tags.values()])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")