# modules/intent_engine/recommender.py

import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from engine.storage import SecureStorage
from ai.nlp import NLPProcessor
from ai.scoring import LeadScorer


class RecommendationType(Enum):
    EMAIL_FOLLOWUP = "email_followup"
    PHONE_CALL = "phone_call"
    PERSONALIZED_CONTENT = "personalized_content"
    DEMO_REQUEST = "demo_request"
    MEETING_SCHEDULING = "meeting_scheduling"
    CASE_STUDY = "case_study"
    WEBINAR_INVITE = "webinar_invite"
    DISCOUNT_OFFER = "discount_offer"
    NO_ACTION = "no_action"


@dataclass
class EngagementSignal:
    lead_id: str
    campaign_id: str
    signal_type: str  # e.g., "email_open", "link_click", "form_submit"
    signal_value: float  # e.g., duration, scroll depth, number of clicks
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class Recommendation:
    id: str
    lead_id: str
    campaign_id: str
    recommendation_type: RecommendationType
    confidence: float
    reasoning: str
    priority: int  # 1-5, 5 being highest
    suggested_content: Optional[str] = None
    suggested_timing: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


class OutreachRecommender:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor, scorer: LeadScorer):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.scorer = scorer
        self.logger = logging.getLogger("outreach_recommender")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load or train models
        self._load_or_train_models()
        
        # Recommendation rules
        self.recommendation_rules = self._initialize_recommendation_rules()
        
        # Signal weights for scoring
        self.signal_weights = {
            "email_open": 1.0,
            "link_click": 2.0,
            "form_submit": 3.0,
            "page_visit": 1.5,
            "content_download": 2.5,
            "webinar_attend": 3.0,
            "demo_request": 4.0,
            "meeting_scheduled": 5.0,
            "reply_received": 4.5,
            "unsubscribe": -5.0,
            "spam_report": -10.0
        }

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS engagement_signals (
            id TEXT PRIMARY KEY,
            lead_id TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            signal_value REAL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (lead_id) REFERENCES leads (id),
            FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY,
            lead_id TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            confidence REAL,
            reasoning TEXT,
            priority INTEGER,
            suggested_content TEXT,
            suggested_timing TEXT,
            created_at TEXT,
            expires_at TEXT,
            FOREIGN KEY (lead_id) REFERENCES leads (id),
            FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_feedback (
            id TEXT PRIMARY KEY,
            recommendation_id TEXT NOT NULL,
            feedback_type TEXT NOT NULL,  # 'positive', 'negative', 'neutral'
            feedback_text TEXT,
            timestamp TEXT,
            FOREIGN KEY (recommendation_id) REFERENCES recommendations (id)
        )
        """)

    def _load_or_train_models(self):
        """Load trained models if available, otherwise train new ones"""
        model_path = Path("ai/models/recommender/outreach_recommender.pkl")
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.recommendation_model = model_data['model']
                    self.label_encoder = model_data['label_encoder']
                    self.vectorizer = model_data['vectorizer']
                    self.scaler = model_data['scaler']
                self.logger.info("Loaded existing recommendation model")
                return
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
        
        # Train new model
        self.logger.info("Training new recommendation model")
        self._train_models()

    def _train_models(self):
        """Train the recommendation models"""
        # Get training data
        training_data = self._get_training_data()
        
        if not training_data:
            self.logger.warning("No training data available, using default model")
            self._create_default_model()
            return
        
        # Prepare features and labels
        features = []
        labels = []
        
        for record in training_data:
            # Extract features
            feature_vector = self._extract_features(record['signals'])
            features.append(feature_vector)
            labels.append(record['outcome'])
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.recommendation_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.recommendation_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.recommendation_model.predict(X_test_scaled)
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        self.logger.info(f"Model trained with accuracy: {report['accuracy']:.2f}")
        self.logger.info(f"Model F1 scores: {report['macro avg']['f1-score']:.2f}")
        
        # Save model
        self._save_model()

    def _get_training_data(self) -> List[Dict]:
        """Get training data from database"""
        # In a real implementation, we would query historical data
        # For now, we'll return an empty list to use the default model
        return []

    def _create_default_model(self):
        """Create a default rule-based model"""
        self.recommendation_model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([r.value for r in RecommendationType])
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.scaler = StandardScaler()
        self.logger.info("Created default rule-based model")

    def _save_model(self):
        """Save the trained model to disk"""
        model_path = Path("ai/models/recommender/outreach_recommender.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.recommendation_model,
            'label_encoder': self.label_encoder,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Saved recommendation model to {model_path}")

    def _initialize_recommendation_rules(self) -> Dict[RecommendationType, Dict]:
        """Initialize recommendation rules"""
        return {
            RecommendationType.EMAIL_FOLLOWUP: {
                "conditions": [
                    {"signal": "email_open", "min_value": 1, "timeframe": 7},
                    {"signal": "link_click", "min_value": 0, "timeframe": 7}
                ],
                "priority": 3,
                "confidence_boost": 0.2
            },
            RecommendationType.PHONE_CALL: {
                "conditions": [
                    {"signal": "email_open", "min_value": 3, "timeframe": 14},
                    {"signal": "link_click", "min_value": 2, "timeframe": 14}
                ],
                "priority": 4,
                "confidence_boost": 0.3
            },
            RecommendationType.PERSONALIZED_CONTENT: {
                "conditions": [
                    {"signal": "page_visit", "min_value": 2, "timeframe": 7}
                ],
                "priority": 2,
                "confidence_boost": 0.15
            },
            RecommendationType.DEMO_REQUEST: {
                "conditions": [
                    {"signal": "link_click", "min_value": 3, "timeframe": 10},
                    {"signal": "page_visit", "min_value": 5, "timeframe": 10}
                ],
                "priority": 5,
                "confidence_boost": 0.4
            },
            RecommendationType.MEETING_SCHEDULING: {
                "conditions": [
                    {"signal": "demo_request", "min_value": 1, "timeframe": 7}
                ],
                "priority": 5,
                "confidence_boost": 0.5
            },
            RecommendationType.CASE_STUDY: {
                "conditions": [
                    {"signal": "page_visit", "min_value": 3, "timeframe": 14},
                    {"signal": "content_download", "min_value": 1, "timeframe": 14}
                ],
                "priority": 3,
                "confidence_boost": 0.25
            },
            RecommendationType.WEBINAR_INVITE: {
                "conditions": [
                    {"signal": "email_open", "min_value": 2, "timeframe": 21}
                ],
                "priority": 2,
                "confidence_boost": 0.1
            },
            RecommendationType.DISCOUNT_OFFER: {
                "conditions": [
                    {"signal": "demo_request", "min_value": 1, "timeframe": 30},
                    {"signal": "meeting_scheduled", "min_value": 0, "timeframe": 30}
                ],
                "priority": 4,
                "confidence_boost": 0.35
            },
            RecommendationType.NO_ACTION: {
                "conditions": [
                    {"signal": "unsubscribe", "min_value": 1, "timeframe": 30},
                    {"signal": "spam_report", "min_value": 1, "timeframe": 30}
                ],
                "priority": 1,
                "confidence_boost": 0.6
            }
        }

    def track_engagement(self, lead_id: str, campaign_id: str, 
                        signal_type: str, signal_value: float = 1.0,
                        metadata: Dict = None) -> str:
        """Track an engagement signal"""
        signal_id = f"sig_{int(datetime.now().timestamp())}"
        
        self.SecureStorage.execute(
            """
            INSERT INTO engagement_signals 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                lead_id,
                campaign_id,
                signal_type,
                signal_value,
                json.dumps(metadata) if metadata else None
            )
        )
        
        self.logger.debug(f"Tracked signal: {signal_type} for lead {lead_id}")
        return signal_id

    def generate_recommendations(self, lead_id: str, campaign_id: str = None) -> List[Recommendation]:
        """Generate recommendations for a lead based on engagement signals"""
        # Get recent engagement signals
        signals = self._get_recent_signals(lead_id, campaign_id)
        
        if not signals:
            return [self._create_default_recommendation(lead_id, campaign_id)]
        
        # Extract features from signals
        features = self._extract_features(signals)
        
        # Get rule-based recommendations
        rule_based_recs = self._get_rule_based_recommendations(signals)
        
        # Get ML-based recommendations if model is available
        ml_recs = []
        if self.recommendation_model is not None:
            ml_recs = self._get_ml_based_recommendations(features, lead_id, campaign_id)
        
        # Combine and rank recommendations
        all_recs = rule_based_recs + ml_recs
        
        # Remove duplicates and sort by priority
        unique_recs = {}
        for rec in all_recs:
            key = (rec.lead_id, rec.recommendation_type)
            if key not in unique_recs or rec.confidence > unique_recs[key].confidence:
                unique_recs[key] = rec
        
        # Sort by priority and confidence
        sorted_recs = sorted(
            unique_recs.values(),
            key=lambda x: (x.priority, x.confidence),
            reverse=True
        )
        
        # Save recommendations to database
        for rec in sorted_recs:
            self._save_recommendation(rec)
        
        return sorted_recs

    def _get_recent_signals(self, lead_id: str, campaign_id: str = None, 
                           days: int = 30) -> List[EngagementSignal]:
        """Get recent engagement signals for a lead"""
        since = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT * FROM engagement_signals 
        WHERE lead_id = ? AND timestamp >= ?
        """
        params = [lead_id, since.isoformat()]
        
        if campaign_id:
            query += " AND campaign_id = ?"
            params.append(campaign_id)
        
        query += " ORDER BY timestamp DESC"
        
        signals = []
        for row in self.SecureStorage.query(query, params):
            signal = EngagementSignal(
                lead_id=row['lead_id'],
                campaign_id=row['campaign_id'],
                signal_type=row['signal_type'],
                signal_value=row['signal_value'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            signals.append(signal)
        
        return signals

    def _extract_features(self, signals: List[EngagementSignal]) -> np.ndarray:
        """Extract feature vector from engagement signals"""
        # Count signals by type
        signal_counts = {}
        for signal in signals:
            signal_counts[signal.signal_type] = signal_counts.get(signal.signal_type, 0) + 1
        
        # Calculate weighted engagement score
        engagement_score = 0
        for signal_type, count in signal_counts.items():
            weight = self.signal_weights.get(signal_type, 1.0)
            engagement_score += count * weight
        
        # Calculate recency score (more recent signals get higher weight)
        recency_score = 0
        for signal in signals:
            days_ago = (datetime.now() - signal.timestamp).days
            recency_weight = max(0, 1 - days_ago / 30)  # Linear decay over 30 days
            signal_weight = self.signal_weights.get(signal.signal_type, 1.0)
            recency_score += signal_weight * recency_weight
        
        # Calculate diversity score (number of different signal types)
        diversity_score = len(signal_counts)
        
        # Calculate frequency score (signals per day)
        days_span = max(1, (datetime.now() - min(s.timestamp for s in signals)).days)
        frequency_score = len(signals) / days_span
        
        # Create feature vector
        features = np.array([
            engagement_score,
            recency_score,
            diversity_score,
            frequency_score,
            signal_counts.get("email_open", 0),
            signal_counts.get("link_click", 0),
            signal_counts.get("form_submit", 0),
            signal_counts.get("page_visit", 0),
            signal_counts.get("content_download", 0),
            signal_counts.get("demo_request", 0),
            signal_counts.get("meeting_scheduled", 0),
            signal_counts.get("reply_received", 0),
            signal_counts.get("unsubscribe", 0),
            signal_counts.get("spam_report", 0)
        ])
        
        return features

    def _get_rule_based_recommendations(self, signals: List[EngagementSignal]) -> List[Recommendation]:
        """Get recommendations based on predefined rules"""
        recommendations = []
        
        # Count signals by type and timeframe
        signal_counts = {}
        for signal in signals:
            days_ago = (datetime.now() - signal.timestamp).days
            
            for rec_type, rule in self.recommendation_rules.items():
                for condition in rule["conditions"]:
                    if (signal.signal_type == condition["signal"] and 
                        days_ago <= condition["timeframe"]):
                        
                        key = (rec_type, condition["timeframe"])
                        if key not in signal_counts:
                            signal_counts[key] = 0
                        signal_counts[key] += signal.signal_value
        
        # Check which rules are satisfied
        for rec_type, rule in self.recommendation_rules.items():
            conditions_met = 0
            total_conditions = len(rule["conditions"])
            
            for condition in rule["conditions"]:
                key = (rec_type, condition["timeframe"])
                count = signal_counts.get(key, 0)
                
                if count >= condition["min_value"]:
                    conditions_met += 1
            
            # If all conditions are met, create recommendation
            if conditions_met == total_conditions:
                confidence = 0.5 + (rule["confidence_boost"] * (conditions_met / total_conditions))
                
                rec = Recommendation(
                    id=f"rec_{int(datetime.now().timestamp())}_{rec_type.value}",
                    lead_id=signals[0].lead_id if signals else "unknown",
                    campaign_id=signals[0].campaign_id if signals else None,
                    recommendation_type=rec_type,
                    confidence=min(confidence, 1.0),
                    reasoning=f"Based on {conditions_met} conditions met: {', '.join([c['signal'] for c in rule['conditions']])}",
                    priority=rule["priority"],
                    suggested_timing=self._calculate_suggested_timing(rec_type)
                )
                
                recommendations.append(rec)
        
        return recommendations

    def _get_ml_based_recommendations(self, features: np.ndarray, lead_id: str, 
                                    campaign_id: str) -> List[Recommendation]:
        """Get recommendations using the ML model"""
        if self.recommendation_model is None:
            return []
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction probabilities
            probabilities = self.recommendation_model.predict_proba(features_scaled)[0]
            
            # Get top recommendations
            top_indices = np.argsort(probabilities)[-3:][::-1]  # Top 3
            
            recommendations = []
            for idx in top_indices:
                if probabilities[idx] > 0.3:  # Minimum confidence threshold
                    rec_type = RecommendationType(self.label_encoder.inverse_transform([idx])[0])
                    
                    rec = Recommendation(
                        id=f"rec_{int(datetime.now().timestamp())}_{rec_type.value}",
                        lead_id=lead_id,
                        campaign_id=campaign_id,
                        recommendation_type=rec_type,
                        confidence=float(probabilities[idx]),
                        reasoning="Based on ML analysis of engagement patterns",
                        priority=self._get_priority_for_type(rec_type),
                        suggested_timing=self._calculate_suggested_timing(rec_type)
                    )
                    
                    recommendations.append(rec)
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating ML recommendations: {str(e)}")
            return []

    def _get_priority_for_type(self, rec_type: RecommendationType) -> int:
        """Get priority level for a recommendation type"""
        priorities = {
            RecommendationType.EMAIL_FOLLOWUP: 3,
            RecommendationType.PHONE_CALL: 4,
            RecommendationType.PERSONALIZED_CONTENT: 2,
            RecommendationType.DEMO_REQUEST: 5,
            RecommendationType.MEETING_SCHEDULING: 5,
            RecommendationType.CASE_STUDY: 3,
            RecommendationType.WEBINAR_INVITE: 2,
            RecommendationType.DISCOUNT_OFFER: 4,
            RecommendationType.NO_ACTION: 1
        }
        return priorities.get(rec_type, 3)

    def _calculate_suggested_timing(self, rec_type: RecommendationType) -> Optional[datetime]:
        """Calculate suggested timing for a recommendation"""
        # Simple timing logic based on recommendation type
        if rec_type == RecommendationType.EMAIL_FOLLOWUP:
            return datetime.now() + timedelta(days=1)
        elif rec_type == RecommendationType.PHONE_CALL:
            # Suggest business hours (9 AM - 5 PM)
            now = datetime.now()
            if now.hour < 9:
                return now.replace(hour=9, minute=0)
            elif now.hour >= 17:
                return (now + timedelta(days=1)).replace(hour=9, minute=0)
            else:
                return now + timedelta(hours=1)
        elif rec_type == RecommendationType.PERSONALIZED_CONTENT:
            return datetime.now() + timedelta(days=2)
        elif rec_type == RecommendationType.DEMO_REQUEST:
            return datetime.now() + timedelta(days=3)
        elif rec_type == RecommendationType.MEETING_SCHEDULING:
            return datetime.now() + timedelta(days=1)
        elif rec_type == RecommendationType.CASE_STUDY:
            return datetime.now() + timedelta(days=2)
        elif rec_type == RecommendationType.WEBINAR_INVITE:
            return datetime.now() + timedelta(days=7)
        elif rec_type == RecommendationType.DISCOUNT_OFFER:
            return datetime.now() + timedelta(days=5)
        else:
            return None

    def _create_default_recommendation(self, lead_id: str, campaign_id: str) -> Recommendation:
        """Create a default recommendation when no signals are available"""
        return Recommendation(
            id=f"rec_{int(datetime.now().timestamp())}_default",
            lead_id=lead_id,
            campaign_id=campaign_id,
            recommendation_type=RecommendationType.EMAIL_FOLLOWUP,
            confidence=0.3,
            reasoning="Default recommendation due to lack of engagement data",
            priority=2,
            suggested_timing=datetime.now() + timedelta(days=1)
        )

    def _save_recommendation(self, recommendation: Recommendation):
        """Save a recommendation to the database"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO recommendations 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recommendation.id,
                recommendation.lead_id,
                recommendation.campaign_id,
                recommendation.recommendation_type.value,
                recommendation.confidence,
                recommendation.reasoning,
                recommendation.priority,
                recommendation.suggested_content,
                recommendation.suggested_timing.isoformat() if recommendation.suggested_timing else None,
                recommendation.created_at.isoformat(),
                recommendation.expires_at.isoformat() if recommendation.expires_at else None
            )
        )

    def record_feedback(self, recommendation_id: str, feedback_type: str, 
                       feedback_text: str = None):
        """Record feedback on a recommendation"""
        feedback_id = f"fb_{int(datetime.now().timestamp())}"
        
        self.SecureStorage.execute(
            """
            INSERT INTO recommendation_feedback 
            VALUES (?, ?, ?, ?)
            """,
            (
                feedback_id,
                recommendation_id,
                feedback_type,
                feedback_text
            )
        )
        
        self.logger.info(f"Recorded {feedback_type} feedback for recommendation {recommendation_id}")
        
        # Retrain model periodically based on feedback
        self._check_retrain_needed()

    def _check_retrain_needed(self):
        """Check if model retraining is needed"""
        # Count feedback received since last model update
        count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM recommendation_feedback WHERE timestamp > ?",
            (datetime.now().isoformat(),)  # This would be the last model update time
        ).fetchone()[0]
        
        # Retrain if we have 50+ new feedback entries
        if count >= 50:
            self.logger.info("Retraining model with new feedback data")
            self._train_models()

    def get_recommendations_for_lead(self, lead_id: str, active_only: bool = True) -> List[Recommendation]:
        """Get recommendations for a specific lead"""
        query = "SELECT * FROM recommendations WHERE lead_id = ?"
        params = [lead_id]
        
        if active_only:
            query += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(datetime.now().isoformat())
        
        query += " ORDER BY priority DESC, confidence DESC"
        
        recommendations = []
        for row in self.SecureStorage.query(query, params):
            rec = Recommendation(
                id=row['id'],
                lead_id=row['lead_id'],
                campaign_id=row['campaign_id'],
                recommendation_type=RecommendationType(row['recommendation_type']),
                confidence=row['confidence'],
                reasoning=row['reasoning'],
                priority=row['priority'],
                suggested_content=row['suggested_content'],
                suggested_timing=datetime.fromisoformat(row['suggested_timing']) if row['suggested_timing'] else None,
                created_at=datetime.fromisoformat(row['created_at']),
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
            )
            recommendations.append(rec)
        
        return recommendations

    def get_recommendation_stats(self, days: int = 30) -> Dict:
        """Get recommendation statistics for the specified time period"""
        since = datetime.now() - timedelta(days=days)
        
        # Get recommendation counts by type
        rec_counts = {rec_type.value: 0 for rec_type in RecommendationType}
        total_recs = 0
        
        for row in self.SecureStorage.query(
            "SELECT recommendation_type, COUNT(*) as count FROM recommendations WHERE created_at >= ? GROUP BY recommendation_type",
            (since.isoformat(),)
        ):
            rec_counts[row['recommendation_type']] = row['count']
            total_recs += row['count']
        
        # Get feedback counts
        feedback_counts = {}
        for row in self.SecureStorage.query(
            "SELECT feedback_type, COUNT(*) as count FROM recommendation_feedback WHERE timestamp >= ? GROUP BY feedback_type",
            (since.isoformat(),)
        ):
            feedback_counts[row['feedback_type']] = row['count']
        
        # Calculate average confidence
        avg_confidence = self.SecureStorage.query(
            "SELECT AVG(confidence) FROM recommendations WHERE created_at >= ?",
            (since.isoformat(),)
        ).fetchone()[0] or 0
        
        # Calculate acceptance rate (positive feedback / total feedback)
        positive_feedback = feedback_counts.get("positive", 0)
        total_feedback = sum(feedback_counts.values())
        acceptance_rate = positive_feedback / total_feedback if total_feedback > 0 else 0
        
        return {
            "total_recommendations": total_recs,
            "recommendation_distribution": rec_counts,
            "feedback_distribution": feedback_counts,
            "average_confidence": avg_confidence,
            "acceptance_rate": acceptance_rate,
            "total_feedback": total_feedback
        }

    def bulk_generate_recommendations(self, lead_ids: List[str]) -> Dict[str, List[Recommendation]]:
        """Generate recommendations for multiple leads efficiently"""
        results = {}
        
        for lead_id in lead_ids:
            recommendations = self.generate_recommendations(lead_id)
            results[lead_id] = recommendations
        
        return results
