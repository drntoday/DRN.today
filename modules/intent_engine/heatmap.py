# modules/intent_engine/heatmap.py

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from engine.storage import SecureStorage
from ai.nlp import NLPProcessor
from ai.scoring import LeadScorer


class EngagementType(Enum):
    OPEN = "open"
    CLICK = "click"
    REPLY = "reply"
    SCROLL = "scroll"
    DWELL = "dwell"
    FORWARD = "forward"
    UNSUBSCRIBE = "unsubscribe"
    SPAM_REPORT = "spam_report"


@dataclass
class EngagementEvent:
    id: str
    lead_id: str
    campaign_id: str
    engagement_type: EngagementType
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
    value: float = 1.0  # For scroll depth, dwell time, etc.


@dataclass
class HeatmapData:
    id: str
    campaign_id: str
    industry: str
    engagement_type: EngagementType
    data: Dict[str, float]  # Time slot -> engagement value
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class HeatmapGenerator:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor, scorer: LeadScorer):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.scorer = scorer
        self.logger = logging.getLogger("heatmap_generator")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Set up visualization style
        self._setup_visualization_style()
        
        # Time slots for heatmap (24 hours in 1-hour slots)
        self.time_slots = [f"{h:02d}:00" for h in range(24)]
        
        # Engagement weights for scoring
        self.engagement_weights = {
            EngagementType.OPEN: 1.0,
            EngagementType.CLICK: 2.0,
            EngagementType.REPLY: 5.0,
            EngagementType.SCROLL: 1.5,
            EngagementType.DWELL: 1.2,
            EngagementType.FORWARD: 3.0,
            EngagementType.UNSUBSCRIBE: -5.0,
            EngagementType.SPAM_REPORT: -10.0
        }

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS engagement_events (
            id TEXT PRIMARY KEY,
            lead_id TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            engagement_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            value REAL DEFAULT 1.0,
            FOREIGN KEY (lead_id) REFERENCES leads (id),
            FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS heatmap_data (
            id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL,
            industry TEXT NOT NULL,
            engagement_type TEXT NOT NULL,
            data TEXT NOT NULL,
            recommendations TEXT,
            created_at TEXT
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS engagement_patterns (
            id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL,
            industry TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            pattern_data TEXT NOT NULL,
            confidence REAL,
            created_at TEXT
        )
        """)

    def _setup_visualization_style(self):
        """Set up matplotlib and seaborn styles for heatmaps"""
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12
        plt.rcParams["legend.fontsize"] = 12

    def track_engagement(self, lead_id: str, campaign_id: str, 
                       engagement_type: EngagementType, 
                       value: float = 1.0, 
                       metadata: Dict = None) -> str:
        """Track an engagement event"""
        event_id = f"eng_{int(datetime.now().timestamp())}"
        
        self.SecureStorage.execute(
            """
            INSERT INTO engagement_events 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                lead_id,
                campaign_id,
                engagement_type.value,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None,
                value
            )
        )
        
        self.logger.debug(f"Tracked engagement: {engagement_type.value} for lead {lead_id}")
        return event_id

    def generate_heatmap(self, campaign_id: str, industry: str = None, 
                        engagement_type: EngagementType = None,
                        days: int = 30) -> HeatmapData:
        """Generate a heatmap for a campaign"""
        # Get engagement data
        engagement_data = self._get_engagement_data(campaign_id, industry, engagement_type, days)
        
        if not engagement_data:
            self.logger.warning(f"No engagement data found for campaign {campaign_id}")
            return None
        
        # Aggregate data by time slots
        heatmap_matrix = self._aggregate_by_time_slots(engagement_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(engagement_data, heatmap_matrix)
        
        # Create heatmap data object
        heatmap_id = f"heatmap_{campaign_id}_{int(datetime.now().timestamp())}"
        heatmap_data = HeatmapData(
            id=heatmap_id,
            campaign_id=campaign_id,
            industry=industry or "All",
            engagement_type=engagement_type or EngagementType.OPEN,
            data=heatmap_matrix,
            recommendations=recommendations
        )
        
        # Save to database
        self._save_heatmap_data(heatmap_data)
        
        return heatmap_data

    def _get_engagement_data(self, campaign_id: str, industry: str = None,
                           engagement_type: EngagementType = None,
                           days: int = 30) -> List[EngagementEvent]:
        """Get engagement events from database"""
        since = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT * FROM engagement_events 
        WHERE campaign_id = ? AND timestamp >= ?
        """
        params = [campaign_id, since.isoformat()]
        
        if industry:
            query += " AND lead_id IN (SELECT id FROM leads WHERE industry = ?)"
            params.append(industry)
        
        if engagement_type:
            query += " AND engagement_type = ?"
            params.append(engagement_type.value)
        
        query += " ORDER BY timestamp"
        
        events = []
        for row in self.SecureStorage.query(query, params):
            event = EngagementEvent(
                id=row['id'],
                lead_id=row['lead_id'],
                campaign_id=row['campaign_id'],
                engagement_type=EngagementType(row['engagement_type']),
                timestamp=datetime.fromisoformat(row['timestamp']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                value=row['value']
            )
            events.append(event)
        
        return events

    def _aggregate_by_time_slots(self, events: List[EngagementEvent]) -> Dict[str, float]:
        """Aggregate engagement events by time slots"""
        # Initialize time slot values
        time_slot_values = {slot: 0.0 for slot in self.time_slots}
        
        # Aggregate events
        for event in events:
            hour = event.timestamp.hour
            time_slot = f"{hour:02d}:00"
            
            # Apply engagement weight
            weight = self.engagement_weights.get(event.engagement_type, 1.0)
            weighted_value = event.value * weight
            
            time_slot_values[time_slot] += weighted_value
        
        # Normalize values
        max_value = max(time_slot_values.values()) if time_slot_values.values() else 1.0
        if max_value > 0:
            time_slot_values = {k: v / max_value for k, v in time_slot_values.items()}
        
        return time_slot_values

    def _generate_recommendations(self, events: List[EngagementEvent], 
                                heatmap_matrix: Dict[str, float]) -> List[str]:
        """Generate AI-based recommendations using TinyBERT"""
        recommendations = []
        
        # Find peak engagement times
        peak_times = sorted(heatmap_matrix.items(), key=lambda x: x[1], reverse=True)[:3]
        if peak_times and peak_times[0][1] > 0.5:
            recommendations.append(f"Peak engagement time: {peak_times[0][0]}")
        
        # Analyze engagement patterns
        engagement_counts = {}
        for event in events:
            engagement_counts[event.engagement_type] = engagement_counts.get(event.engagement_type, 0) + 1
        
        # Check for high reply rates
        if engagement_counts.get(EngagementType.REPLY, 0) > len(events) * 0.1:
            recommendations.append("High reply rate detected - consider follow-up sequence")
        
        # Check for high unsubscribe rates
        if engagement_counts.get(EngagementType.UNSUBSCRIBE, 0) > len(events) * 0.05:
            recommendations.append("High unsubscribe rate - review email content and frequency")
        
        # Check for low engagement
        total_weighted = sum(self.engagement_weights.get(e.engagement_type, 1.0) * e.value for e in events)
        if total_weighted < len(events) * 1.5:
            recommendations.append("Low overall engagement - consider subject line optimization")
        
        # Use NLP to analyze engagement patterns
        if events:
            # Create engagement summary text
            engagement_summary = self._create_engagement_summary(events)
            
            # Use TinyBERT to generate insights
            insights = self.nlp.analyze_text(engagement_summary)
            
            if insights.get("sentiment") == "negative":
                recommendations.append("Negative engagement patterns detected - review campaign strategy")
            
            # Extract key topics
            topics = self.nlp.extract_key_topics(engagement_summary)
            if topics:
                recommendations.append(f"Key engagement topics: {', '.join(topics[:3])}")
        
        return recommendations
