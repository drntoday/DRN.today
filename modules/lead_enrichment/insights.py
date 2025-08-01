# modules/lead_enrichment/insights.py

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from engine.storage import Storage
from ai.nlp import NLPProcessor
from ai.scoring import LeadScorer


class JobSeniority(Enum):
    ENTRY = "Entry Level"
    JUNIOR = "Junior"
    MID = "Mid-Level"
    SENIOR = "Senior"
    EXECUTIVE = "Executive"
    C_LEVEL = "C-Level"


class BudgetRange(Enum):
    MICRO = "$0-$10K"
    SMALL = "$10K-$50K"
    MEDIUM = "$50K-$100K"
    LARGE = "$100K-$500K"
    ENTERPRISE = "$500K+"


class UrgencyLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class AuthorityLevel(Enum):
    INDIVIDUAL = "Individual Contributor"
    TEAM_LEAD = "Team Lead"
    MANAGER = "Manager"
    DIRECTOR = "Director"
    VP = "VP"
    C_LEVEL = "C-Level"


@dataclass
class LeadInsights:
    lead_id: str
    job_seniority: JobSeniority
    budget_range: BudgetRange
    urgency_level: UrgencyLevel
    industry_fit_score: float  # 0.0 to 1.0
    authority_level: AuthorityLevel
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class LeadInsightsEngine:
    def __init__(self, storage: Storage, nlp_processor: NLPProcessor, scorer: LeadScorer):
        self.storage = storage
        self.nlp = nlp_processor
        self.scorer = scorer
        self._initialize_tables()
        self._load_reference_data()
        
        # Industry benchmarks for budget estimation
        self.industry_budget_benchmarks = {
            "Technology": {"small": 50000, "medium": 200000, "large": 1000000},
            "Healthcare": {"small": 75000, "medium": 300000, "large": 1500000},
            "Finance": {"small": 100000, "medium": 500000, "large": 2000000},
            "Retail": {"small": 30000, "medium": 150000, "large": 750000},
            "Manufacturing": {"small": 60000, "medium": 250000, "large": 1200000},
            "Education": {"small": 20000, "medium": 80000, "large": 400000},
            "Consulting": {"small": 40000, "medium": 180000, "large": 900000},
            "Real Estate": {"small": 35000, "medium": 160000, "large": 800000},
            "Other": {"small": 25000, "medium": 100000, "large": 500000}
        }
        
        # Company size multipliers
        self.company_size_multipliers = {
            "1-10": 0.5,
            "11-50": 0.8,
            "51-200": 1.2,
            "201-500": 1.8,
            "501-1000": 2.5,
            "1001-5000": 4.0,
            "5001-10000": 6.0,
            "10001+": 10.0
        }
        
        # Seniority keywords for classification
        self.seniority_keywords = {
            JobSeniority.ENTRY: ["intern", "trainee", "assistant", "junior", "associate", "coordinator"],
            JobSeniority.JUNIOR: ["junior", "associate", "analyst", "specialist"],
            JobSeniority.MID: ["mid", "senior", "lead", "principal", "experienced"],
            JobSeniority.SENIOR: ["senior", "lead", "principal", "expert", "staff"],
            JobSeniority.EXECUTIVE: ["director", "head", "vp", "vice president", "chief"],
            JobSeniority.C_LEVEL: ["ceo", "cto", "cfo", "coo", "cmo", "cio", "cso", "chief"]
        }
        
        # Authority level mapping
        self.authority_mapping = {
            JobSeniority.ENTRY: AuthorityLevel.INDIVIDUAL,
            JobSeniority.JUNIOR: AuthorityLevel.INDIVIDUAL,
            JobSeniority.MID: AuthorityLevel.INDIVIDUAL,
            JobSeniority.SENIOR: AuthorityLevel.TEAM_LEAD,
            JobSeniority.EXECUTIVE: AuthorityLevel.MANAGER,
            JobSeniority.C_LEVEL: AuthorityLevel.C_LEVEL
        }
        
        # Urgency detection keywords
        self.urgency_keywords = {
            UrgencyLevel.CRITICAL: ["urgent", "immediately", "asap", "emergency", "critical", "deadline"],
            UrgencyLevel.HIGH: ["soon", "quickly", "fast", "need", "required", "priority"],
            UrgencyLevel.MEDIUM: ["considering", "looking", "exploring", "planning"],
            UrgencyLevel.LOW: ["future", "later", "maybe", "potential", "research"]
        }

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.storage.execute("""
        CREATE TABLE IF NOT EXISTS lead_insights (
            lead_id TEXT PRIMARY KEY,
            job_seniority TEXT,
            budget_range TEXT,
            urgency_level TEXT,
            industry_fit_score REAL,
            authority_level TEXT,
            confidence_scores TEXT,
            last_updated TEXT
        )
        """)

    def _load_reference_data(self):
        """Load any reference data from storage"""
        # In a real implementation, we might load industry benchmarks, etc.
        pass

    def generate_insights(self, lead_id: str, lead_data: Dict) -> LeadInsights:
        """Generate all insights for a lead"""
        # Extract relevant data
        job_title = lead_data.get("job_title", "").lower()
        company_industry = lead_data.get("industry", "Other")
        company_size = lead_data.get("company_size", "1-10")
        bio = lead_data.get("bio", "")
        communication_history = lead_data.get("communication_history", [])
        
        # Generate individual insights
        job_seniority, seniority_confidence = self._infer_job_seniority(job_title, bio)
        budget_range, budget_confidence = self._estimate_budget_range(
            company_industry, company_size, job_seniority
        )
        urgency_level, urgency_confidence = self._detect_urgency(communication_history)
        industry_fit_score = self._calculate_industry_fit(lead_data)
        authority_level = self._determine_authority_level(job_seniority, lead_data)
        
        # Create insights object
        insights = LeadInsights(
            lead_id=lead_id,
            job_seniority=job_seniority,
            budget_range=budget_range,
            urgency_level=urgency_level,
            industry_fit_score=industry_fit_score,
            authority_level=authority_level,
            confidence_scores={
                "job_seniority": seniority_confidence,
                "budget_range": budget_confidence,
                "urgency_level": urgency_confidence,
                "industry_fit": industry_fit_score,
                "authority_level": 0.9  # High confidence as it's derived from seniority
            }
        )
        
        # Save to storage
        self._save_insights(insights)
        
        return insights

    def _infer_job_seniority(self, job_title: str, bio: str) -> Tuple[JobSeniority, float]:
        """Infer job seniority from title and bio using NLP"""
        text = f"{job_title} {bio}".lower()
        
        # Use NLP to extract entities and sentiment
        entities = self.nlp.extract_entities(text)
        sentiment = self.nlp.analyze_sentiment(text)
        
        # Rule-based classification with keyword matching
        seniority_scores = {}
        for seniority, keywords in self.seniority_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            seniority_scores[seniority] = score / len(keywords)
        
        # Use TinyBERT for semantic classification
        bert_scores = self.nlp.classify_text(text, list(self.seniority_keywords.keys()))
        
        # Combine rule-based and BERT scores
        combined_scores = {}
        for seniority in JobSeniority:
            rule_score = seniority_scores.get(seniority, 0)
            bert_score = bert_scores.get(seniority.value, 0)
            combined_scores[seniority] = (rule_score * 0.4) + (bert_score * 0.6)
        
        # Normalize scores
        max_score = max(combined_scores.values()) if combined_scores else 0
        if max_score > 0:
            normalized_scores = {k: v/max_score for k, v in combined_scores.items()}
        else:
            normalized_scores = {k: 0.2 for k in JobSeniority}  # Default low confidence
        
        # Get the top seniority
        top_seniority = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[top_seniority]
        
        # Adjust confidence based on sentiment
        if sentiment > 0.7:  # Very positive sentiment might indicate higher seniority
            confidence = min(confidence * 1.1, 1.0)
        elif sentiment < 0.3:  # Negative sentiment might indicate lower seniority
            confidence = max(confidence * 0.9, 0.1)
        
        return top_seniority, confidence

    def _estimate_budget_range(self, industry: str, company_size: str, 
                              seniority: JobSeniority) -> Tuple[BudgetRange, float]:
        """Estimate budget range based on industry, company size, and seniority"""
        # Get industry benchmarks
        industry_data = self.industry_budget_benchmarks.get(industry, self.industry_budget_benchmarks["Other"])
        
        # Get company size multiplier
        size_multiplier = self.company_size_multipliers.get(company_size, 1.0)
        
        # Seniority multiplier
        seniority_multipliers = {
            JobSeniority.ENTRY: 0.3,
            JobSeniority.JUNIOR: 0.5,
            JobSeniority.MID: 0.8,
            JobSeniority.SENIOR: 1.2,
            JobSeniority.EXECUTIVE: 2.0,
            JobSeniority.C_LEVEL: 5.0
        }
        seniority_multiplier = seniority_multipliers.get(seniority, 1.0)
        
        # Calculate estimated budget
        base_budget = industry_data["medium"]  # Use medium as baseline
        estimated_budget = base_budget * size_multiplier * seniority_multiplier
        
        # Map to budget range
        if estimated_budget < 10000:
            budget_range = BudgetRange.MICRO
        elif estimated_budget < 50000:
            budget_range = BudgetRange.SMALL
        elif estimated_budget < 100000:
            budget_range = BudgetRange.MEDIUM
        elif estimated_budget < 500000:
            budget_range = BudgetRange.LARGE
        else:
            budget_range = BudgetRange.ENTERPRISE
        
        # Calculate confidence based on data quality
        confidence = 0.7  # Base confidence
        
        # Increase confidence if we have good industry and size data
        if industry != "Other" and company_size != "1-10":
            confidence += 0.2
        
        # Adjust based on seniority confidence
        if seniority in [JobSeniority.EXECUTIVE, JobSeniority.C_LEVEL]:
            confidence += 0.1
        
        return budget_range, min(confidence, 1.0)

    def _detect_urgency(self, communication_history: List[Dict]) -> Tuple[UrgencyLevel, float]:
        """Detect urgency level from communication history"""
        if not communication_history:
            return UrgencyLevel.MEDIUM, 0.5  # Default medium urgency
        
        # Combine all communication text
        all_text = " ".join([msg.get("content", "") for msg in communication_history]).lower()
        
        # Count urgency keywords
        urgency_counts = {level: 0 for level in UrgencyLevel}
        for level, keywords in self.urgency_keywords.items():
            for keyword in keywords:
                urgency_counts[level] += all_text.count(keyword)
        
        # Normalize counts
        total_keywords = sum(urgency_counts.values())
        if total_keywords > 0:
            urgency_scores = {level: count/total_keywords for level, count in urgency_counts.items()}
        else:
            urgency_scores = {level: 0.25 for level in UrgencyLevel}  # Equal distribution
        
        # Get top urgency level
        top_urgency = max(urgency_scores, key=urgency_scores.get)
        confidence = urgency_scores[top_urgency]
        
        # Analyze sentiment for additional signals
        sentiment = self.nlp.analyze_sentiment(all_text)
        if sentiment < 0.3:  # Negative sentiment might indicate urgency
            confidence = min(confidence * 1.2, 1.0)
            if top_urgency != UrgencyLevel.CRITICAL:
                # Upgrade urgency if sentiment is negative
                levels = list(UrgencyLevel)
                current_index = levels.index(top_urgency)
                if current_index > 0:
                    top_urgency = levels[current_index - 1]
        
        return top_urgency, confidence

    def _calculate_industry_fit(self, lead_data: Dict) -> float:
        """Calculate how well a lead fits the target industry"""
        # In a real implementation, we would have target industries defined
        target_industries = ["Technology", "Software", "SaaS", "FinTech", "HealthTech"]
        
        lead_industry = lead_data.get("industry", "")
        company_description = lead_data.get("company_description", "")
        job_title = lead_data.get("job_title", "")
        
        # Check exact industry match
        if lead_industry in target_industries:
            return 1.0
        
        # Check for partial matches in description and title
        text = f"{lead_industry} {company_description} {job_title}".lower()
        
        # Calculate similarity to target industries
        similarities = []
        for industry in target_industries:
            similarity = self.nlp.calculate_similarity(text, industry.lower())
            similarities.append(similarity)
        
        # Return the maximum similarity
        return max(similarities) if similarities else 0.0

    def _determine_authority_level(self, seniority: JobSeniority, lead_data: Dict) -> AuthorityLevel:
        """Determine authority level based on seniority and additional factors"""
        # Base authority from seniority
        authority = self.authority_mapping.get(seniority, AuthorityLevel.INDIVIDUAL)
        
        # Check for additional authority indicators
        job_title = lead_data.get("job_title", "").lower()
        company_size = lead_data.get("company_size", "1-10")
        
        # Upgrade authority for certain keywords
        if any(keyword in job_title for keyword in ["decision maker", "approver", "signatory"]):
            if authority == AuthorityLevel.MANAGER:
                authority = AuthorityLevel.DIRECTOR
            elif authority == AuthorityLevel.DIRECTOR:
                authority = AuthorityLevel.VP
        
        # Adjust based on company size
        if company_size in ["1-10", "11-50"]:
            # In smaller companies, people often have broader authority
            if authority == AuthorityLevel.TEAM_LEAD:
                authority = AuthorityLevel.MANAGER
            elif authority == AuthorityLevel.MANAGER:
                authority = AuthorityLevel.DIRECTOR
        
        return authority

    def _save_insights(self, insights: LeadInsights):
        """Save insights to storage"""
        self.storage.execute(
            """
            INSERT OR REPLACE INTO lead_insights 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                insights.lead_id,
                insights.job_seniority.value,
                insights.budget_range.value,
                insights.urgency_level.value,
                insights.industry_fit_score,
                insights.authority_level.value,
                json.dumps(insights.confidence_scores),
                insights.last_updated.isoformat()
            )
        )

    def get_insights(self, lead_id: str) -> Optional[LeadInsights]:
        """Retrieve insights for a lead"""
        row = self.storage.query(
            "SELECT * FROM lead_insights WHERE lead_id = ?",
            (lead_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return LeadInsights(
            lead_id=row['lead_id'],
            job_seniority=JobSeniority(row['job_seniority']),
            budget_range=BudgetRange(row['budget_range']),
            urgency_level=UrgencyLevel(row['urgency_level']),
            industry_fit_score=row['industry_fit_score'],
            authority_level=AuthorityLevel(row['authority_level']),
            confidence_scores=json.loads(row['confidence_scores']),
            last_updated=datetime.fromisoformat(row['last_updated'])
        )

    def update_insights(self, lead_id: str, lead_data: Dict):
        """Update insights for a lead with new data"""
        insights = self.generate_insights(lead_id, lead_data)
        return insights

    def bulk_generate_insights(self, lead_ids: List[str]) -> Dict[str, LeadInsights]:
        """Generate insights for multiple leads efficiently"""
        results = {}
        for lead_id in lead_ids:
            # In a real implementation, we'd fetch lead data from storage
            lead_data = {"id": lead_id}  # Simplified for example
            insights = self.generate_insights(lead_id, lead_data)
            results[lead_id] = insights
        return results

    def get_insights_summary(self, lead_ids: List[str]) -> Dict:
        """Get a summary of insights across multiple leads"""
        insights_list = []
        for lead_id in lead_ids:
            insights = self.get_insights(lead_id)
            if insights:
                insights_list.append(insights)
        
        if not insights_list:
            return {}
        
        # Calculate summary statistics
        summary = {
            "total_leads": len(insights_list),
            "job_seniority_distribution": self._calculate_distribution(
                [i.job_seniority.value for i in insights_list]
            ),
            "budget_range_distribution": self._calculate_distribution(
                [i.budget_range.value for i in insights_list]
            ),
            "urgency_level_distribution": self._calculate_distribution(
                [i.urgency_level.value for i in insights_list]
            ),
            "authority_level_distribution": self._calculate_distribution(
                [i.authority_level.value for i in insights_list]
            ),
            "avg_industry_fit": np.mean([i.industry_fit_score for i in insights_list]),
            "high_urgency_count": sum(1 for i in insights_list if i.urgency_level == UrgencyLevel.CRITICAL),
            "high_authority_count": sum(1 for i in insights_list if i.authority_level in [AuthorityLevel.VP, AuthorityLevel.C_LEVEL])
        }
        
        return summary

    def _calculate_distribution(self, values: List[str]) -> Dict[str, float]:
        """Calculate percentage distribution of categorical values"""
        from collections import Counter
        counts = Counter(values)
        total = len(values)
        return {k: v/total for k, v in counts.items()}