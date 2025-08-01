#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
AI Lead Scoring Engine
Production-Ready Implementation
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Initialize scoring logger
logger = logging.getLogger(__name__)

@dataclass
class LeadScore:
    """Lead scoring result data structure"""
    uuid: str
    lead_uuid: str
    conversion_probability: float
    campaign_responsiveness: float
    overall_score: float
    confidence: float
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    recommendation: str = "contact"
    scoring_model: str = "ensemble"
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScoringFeatures:
    """Feature set for lead scoring"""
    # Basic features
    source_quality: float = 0.0
    lead_age: float = 0.0
    engagement_score: float = 0.0
    
    # Demographic features
    job_seniority_score: float = 0.0
    company_size_score: float = 0.0
    industry_fit_score: float = 0.0
    location_score: float = 0.0
    
    # Behavioral features
    email_open_rate: float = 0.0
    click_through_rate: float = 0.0
    website_visits: float = 0.0
    content_engagement: float = 0.0
    
    # Firmographic features
    company_revenue: float = 0.0
    funding_stage: float = 0.0
    employee_count: float = 0.0
    tech_stack_fit: float = 0.0
    
    # Intent features
    buying_intent_score: float = 0.0
    urgency_score: float = 0.0
    budget_score: float = 0.0
    authority_score: float = 0.0
    
    # Text features (from TinyBERT)
    text_embedding: np.ndarray = field(default_factory=lambda: np.zeros(768))
    sentiment_score: float = 0.0
    keyword_relevance: float = 0.0

class LeadScorer:
    """Production-ready lead scoring engine with ML models"""
    
    def __init__(self, model_path: str = "ai/models"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.tinybert_tokenizer: Optional[AutoTokenizer] = None
        self.tinybert_model: Optional[AutoModel] = None
        self.sklearn_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        
        # Feature names
        self.feature_names = [
            "source_quality", "lead_age", "engagement_score",
            "job_seniority_score", "company_size_score", "industry_fit_score", "location_score",
            "email_open_rate", "click_through_rate", "website_visits", "content_engagement",
            "company_revenue", "funding_stage", "employee_count", "tech_stack_fit",
            "buying_intent_score", "urgency_score", "budget_score", "authority_score"
        ]
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all ML models and preprocessing components"""
        try:
            # Load TinyBERT model
            tinybert_path = self.model_path / "tinybert"
            if tinybert_path.exists():
                self.tinybert_tokenizer = AutoTokenizer.from_pretrained(tinybert_path)
                self.tinybert_model = AutoModel.from_pretrained(tinybert_path)
                self.tinybert_model.to(self.device)
                self.tinybert_model.eval()
                logger.info("TinyBERT model loaded successfully")
            
            # Load Scikit-learn models
            sklearn_path = self.model_path / "scikit"
            if sklearn_path.exists():
                # Load main scoring model
                model_file = sklearn_path / "lead_scorer.pkl"
                if model_file.exists():
                    self.sklearn_models["main"] = joblib.load(model_file)
                
                # Load campaign responsiveness model
                campaign_model_file = sklearn_path / "campaign_responsiveness.pkl"
                if campaign_model_file.exists():
                    self.sklearn_models["campaign"] = joblib.load(campaign_model_file)
                
                # Load scalers
                scaler_file = sklearn_path / "scaler.pkl"
                if scaler_file.exists():
                    self.scalers["main"] = joblib.load(scaler_file)
                
                # Load encoders
                encoder_files = {
                    "source": sklearn_path / "source_encoder.pkl",
                    "industry": sklearn_path / "industry_encoder.pkl",
                    "location": sklearn_path / "location_encoder.pkl"
                }
                
                for name, file_path in encoder_files.items():
                    if file_path.exists():
                        self.encoders[name] = joblib.load(file_path)
                
                # Load TF-IDF vectorizer
                tfidf_file = sklearn_path / "tfidf_vectorizer.pkl"
                if tfidf_file.exists():
                    self.tfidf_vectorizer = joblib.load(tfidf_file)
                
                logger.info("Scikit-learn models loaded successfully")
            
            logger.info("All scoring models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise
    
    def extract_features(self, lead_data: Dict[str, Any]) -> ScoringFeatures:
        """Extract features from lead data"""
        features = ScoringFeatures()
        
        try:
            # Basic features
            features.source_quality = self._calculate_source_quality(lead_data.get("source", ""))
            features.lead_age = self._calculate_lead_age(lead_data.get("created_at", time.time()))
            features.engagement_score = self._calculate_engagement_score(lead_data)
            
            # Demographic features
            features.job_seniority_score = self._calculate_job_seniority_score(lead_data)
            features.company_size_score = self._calculate_company_size_score(lead_data)
            features.industry_fit_score = self._calculate_industry_fit_score(lead_data)
            features.location_score = self._calculate_location_score(lead_data)
            
            # Behavioral features
            features.email_open_rate = lead_data.get("email_open_rate", 0.0)
            features.click_through_rate = lead_data.get("click_through_rate", 0.0)
            features.website_visits = lead_data.get("website_visits", 0.0)
            features.content_engagement = lead_data.get("content_engagement", 0.0)
            
            # Firmographic features
            features.company_revenue = self._normalize_revenue(lead_data.get("company_revenue", 0))
            features.funding_stage = self._normalize_funding_stage(lead_data.get("funding_stage", ""))
            features.employee_count = self._normalize_employee_count(lead_data.get("employee_count", 0))
            features.tech_stack_fit = self._calculate_tech_stack_fit(lead_data)
            
            # Intent features
            features.buying_intent_score = lead_data.get("buying_intent_score", 0.0)
            features.urgency_score = lead_data.get("urgency_score", 0.0)
            features.budget_score = lead_data.get("budget_score", 0.0)
            features.authority_score = lead_data.get("authority_score", 0.0)
            
            # Text features
            if self.tinybert_model and lead_data.get("raw_content"):
                text_features = self._extract_text_features(lead_data["raw_content"])
                features.text_embedding = text_features["embedding"]
                features.sentiment_score = text_features["sentiment"]
                features.keyword_relevance = text_features["keyword_relevance"]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}", exc_info=True)
            return features
    
    def _calculate_source_quality(self, source: str) -> float:
        """Calculate source quality score"""
        source_scores = {
            "linkedin": 0.9,
            "website": 0.8,
            "referral": 0.7,
            "google": 0.6,
            "social": 0.5,
            "unknown": 0.3
        }
        return source_scores.get(source.lower(), 0.3)
    
    def _calculate_lead_age(self, created_at: float) -> float:
        """Calculate lead age score (newer is better)"""
        age_hours = (time.time() - created_at) / 3600
        if age_hours < 24:
            return 1.0
        elif age_hours < 168:  # 1 week
            return 0.8
        elif age_hours < 720:  # 1 month
            return 0.6
        else:
            return 0.4
    
    def _calculate_engagement_score(self, lead_data: Dict[str, Any]) -> float:
        """Calculate overall engagement score"""
        scores = []
        
        if "email_opens" in lead_data and "emails_sent" in lead_data:
            if lead_data["emails_sent"] > 0:
                scores.append(min(lead_data["email_opens"] / lead_data["emails_sent"], 1.0))
        
        if "clicks" in lead_data and "emails_sent" in lead_data:
            if lead_data["emails_sent"] > 0:
                scores.append(min(lead_data["clicks"] / lead_data["emails_sent"], 1.0))
        
        if "website_visits" in lead_data:
            scores.append(min(lead_data["website_visits"] / 10, 1.0))
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_job_seniority_score(self, lead_data: Dict[str, Any]) -> float:
        """Calculate job seniority score"""
        title = lead_data.get("job_title", "").lower()
        
        seniority_scores = {
            "ceo": 1.0, "cto": 1.0, "cfo": 1.0, "coo": 1.0,
            "vp": 0.9, "vice president": 0.9,
            "director": 0.8,
            "manager": 0.7,
            "senior": 0.6,
            "lead": 0.5,
            "associate": 0.4,
            "analyst": 0.3,
            "intern": 0.2
        }
        
        for keyword, score in seniority_scores.items():
            if keyword in title:
                return score
        
        return 0.5
    
    def _calculate_company_size_score(self, lead_data: Dict[str, Any]) -> float:
        """Calculate company size score"""
        employees = lead_data.get("employee_count", 0)
        
        if employees >= 10000:
            return 1.0
        elif employees >= 1000:
            return 0.9
        elif employees >= 500:
            return 0.8
        elif employees >= 100:
            return 0.7
        elif employees >= 50:
            return 0.6
        elif employees >= 10:
            return 0.5
        else:
            return 0.4
    
    def _calculate_industry_fit_score(self, lead_data: Dict[str, Any]) -> float:
        """Calculate industry fit score"""
        industry = lead_data.get("industry", "").lower()
        
        target_industries = [
            "software", "saas", "technology", "fintech", "healthtech",
            "biotech", "ai", "machine learning", "cloud", "cybersecurity"
        ]
        
        return 1.0 if any(target in industry for target in target_industries) else 0.5
    
    def _calculate_location_score(self, lead_data: Dict[str, Any]) -> float:
        """Calculate location score"""
        location = lead_data.get("location", "").lower()
        
        target_regions = [
            "united states", "canada", "united kingdom", "germany", "france",
            "australia", "japan", "singapore"
        ]
        
        return 1.0 if any(region in location for region in target_regions) else 0.7
    
    def _normalize_revenue(self, revenue: Any) -> float:
        """Normalize company revenue"""
        try:
            if isinstance(revenue, str):
                # Remove currency symbols and commas
                revenue = revenue.replace("$", "").replace("€", "").replace("£", "").replace(",", "")
                revenue = float(revenue)
            
            if revenue >= 1000000000:  # $1B+
                return 1.0
            elif revenue >= 100000000:  # $100M+
                return 0.9
            elif revenue >= 10000000:  # $10M+
                return 0.8
            elif revenue >= 1000000:  # $1M+
                return 0.7
            elif revenue >= 100000:  # $100K+
                return 0.6
            else:
                return 0.5
        except:
            return 0.5
    
    def _normalize_funding_stage(self, stage: str) -> float:
        """Normalize funding stage"""
        stage_scores = {
            "ipo": 1.0,
            "series d": 0.95,
            "series c": 0.9,
            "series b": 0.8,
            "series a": 0.7,
            "seed": 0.6,
            "pre-seed": 0.5,
            "bootstrapped": 0.4,
            "unknown": 0.3
        }
        return stage_scores.get(stage.lower(), 0.3)
    
    def _normalize_employee_count(self, count: Any) -> float:
        """Normalize employee count"""
        try:
            if isinstance(count, str):
                count = int(count.replace(",", ""))
            
            if count >= 10000:
                return 1.0
            elif count >= 1000:
                return 0.9
            elif count >= 500:
                return 0.8
            elif count >= 100:
                return 0.7
            elif count >= 50:
                return 0.6
            elif count >= 10:
                return 0.5
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_tech_stack_fit(self, lead_data: Dict[str, Any]) -> float:
        """Calculate technology stack fit score"""
        tech_stack = lead_data.get("tech_stack", [])
        target_tech = [
            "aws", "azure", "gcp", "python", "javascript", "react",
            "node.js", "docker", "kubernetes", "tensorflow", "pytorch"
        ]
        
        if not tech_stack:
            return 0.5
        
        matches = sum(1 for tech in tech_stack if any(target in tech.lower() for target in target_tech))
        return min(matches / len(target_tech), 1.0)
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract text features using TinyBERT"""
        if not self.tinybert_model or not self.tinybert_tokenizer:
            return {
                "embedding": np.zeros(768),
                "sentiment": 0.5,
                "keyword_relevance": 0.5
            }
        
        try:
            # Tokenize text
            inputs = self.tinybert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.tinybert_model(**inputs)
                # Use CLS token embedding as sentence representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # Calculate sentiment (simplified)
            sentiment = self._calculate_sentiment(text)
            
            # Calculate keyword relevance
            keyword_relevance = self._calculate_keyword_relevance(text)
            
            return {
                "embedding": embeddings,
                "sentiment": sentiment,
                "keyword_relevance": keyword_relevance
            }
            
        except Exception as e:
            logger.error(f"Error extracting text features: {str(e)}", exc_info=True)
            return {
                "embedding": np.zeros(768),
                "sentiment": 0.5,
                "keyword_relevance": 0.5
            }
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score (simplified)"""
        positive_words = ["good", "great", "excellent", "amazing", "interested", "love", "perfect"]
        negative_words = ["bad", "terrible", "awful", "hate", "disappointed", "worst"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5
        
        return (positive_count - negative_count) / total
    
    def _calculate_keyword_relevance(self, text: str) -> float:
        """Calculate keyword relevance score"""
        keywords = [
            "solution", "software", "platform", "tool", "service",
            "interested", "looking", "need", "want", "considering",
            "budget", "pricing", "cost", "quote", "estimate",
            "demo", "trial", "test", "example", "see"
        ]
        
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        return min(matches / len(keywords), 1.0)
    
    def score_lead(self, lead_data: Dict[str, Any]) -> LeadScore:
        """Score a lead using ML models"""
        try:
            # Extract features
            features = self.extract_features(lead_data)
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            # Get predictions
            conversion_prob = self._predict_conversion_probability(feature_vector)
            campaign_resp = self._predict_campaign_responsiveness(feature_vector)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(conversion_prob, campaign_resp, features)
            
            # Get feature contributions
            feature_contributions = self._get_feature_contributions(feature_vector)
            
            # Get recommendation
            recommendation = self._get_recommendation(overall_score, conversion_prob)
            
            # Create score result
            lead_score = LeadScore(
                uuid=str(uuid.uuid4()),
                lead_uuid=lead_data.get("uuid", ""),
                conversion_probability=conversion_prob,
                campaign_responsiveness=campaign_resp,
                overall_score=overall_score,
                confidence=self._calculate_confidence(feature_vector),
                feature_contributions=feature_contributions,
                recommendation=recommendation,
                scoring_model="ensemble",
                metadata={
                    "feature_vector": feature_vector.tolist(),
                    "model_versions": {
                        "tinybert": "1.0",
                        "sklearn": "1.0"
                    }
                }
            )
            
            return lead_score
            
        except Exception as e:
            logger.error(f"Error scoring lead: {str(e)}", exc_info=True)
            # Return default score
            return LeadScore(
                uuid=str(uuid.uuid4()),
                lead_uuid=lead_data.get("uuid", ""),
                conversion_probability=0.5,
                campaign_responsiveness=0.5,
                overall_score=0.5,
                confidence=0.0,
                recommendation="contact",
                scoring_model="fallback",
                metadata={"error": str(e)}
            )
    
    def _prepare_feature_vector(self, features: ScoringFeatures) -> np.ndarray:
        """Prepare feature vector for ML models"""
        try:
            # Get numeric features
            numeric_features = [
                features.source_quality,
                features.lead_age,
                features.engagement_score,
                features.job_seniority_score,
                features.company_size_score,
                features.industry_fit_score,
                features.location_score,
                features.email_open_rate,
                features.click_through_rate,
                features.website_visits,
                features.content_engagement,
                features.company_revenue,
                features.funding_stage,
                features.employee_count,
                features.tech_stack_fit,
                features.buying_intent_score,
                features.urgency_score,
                features.budget_score,
                features.authority_score
            ]
            
            # Add text embedding features
            text_features = features.text_embedding[:20]  # Use first 20 dimensions
            
            # Combine features
            feature_vector = np.concatenate([numeric_features, text_features])
            
            # Scale features if scaler is available
            if "main" in self.scalers:
                feature_vector = self.scalers["main"].transform(feature_vector.reshape(1, -1))[0]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {str(e)}", exc_info=True)
            return np.zeros(40)  # 20 numeric + 20 text features
    
    def _predict_conversion_probability(self, feature_vector: np.ndarray) -> float:
        """Predict conversion probability using ML model"""
        try:
            if "main" in self.sklearn_models:
                model = self.sklearn_models["main"]
                proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
                return float(proba[1])  # Probability of positive class
            else:
                # Simple rule-based fallback
                score = np.mean(feature_vector[:20])  # Average of numeric features
                return float(1 / (1 + np.exp(-score)))  # Sigmoid transformation
                
        except Exception as e:
            logger.error(f"Error predicting conversion probability: {str(e)}", exc_info=True)
            return 0.5
    
    def _predict_campaign_responsiveness(self, feature_vector: np.ndarray) -> float:
        """Predict campaign responsiveness using ML model"""
        try:
            if "campaign" in self.sklearn_models:
                model = self.sklearn_models["campaign"]
                proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
                return float(proba[1])
            else:
                # Use conversion probability as fallback
                return self._predict_conversion_probability(feature_vector)
                
        except Exception as e:
            logger.error(f"Error predicting campaign responsiveness: {str(e)}", exc_info=True)
            return 0.5
    
    def _calculate_overall_score(self, conversion_prob: float, campaign_resp: float, features: ScoringFeatures) -> float:
        """Calculate overall lead score"""
        # Weighted combination of probabilities
        weights = {
            "conversion": 0.6,
            "campaign": 0.3,
            "engagement": 0.1
        }
        
        engagement_score = features.engagement_score
        
        overall_score = (
            weights["conversion"] * conversion_prob +
            weights["campaign"] * campaign_resp +
            weights["engagement"] * engagement_score
        )
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _get_feature_contributions(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Get feature contributions to the score"""
        try:
            if "main" not in self.sklearn_models:
                return {}
            
            model = self.sklearn_models["main"]
            
            # Get feature importances if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                contributions = {}
                for i, importance in enumerate(importances):
                    if i < len(self.feature_names):
                        contributions[self.feature_names[i]] = float(importance * feature_vector[i])
                    else:
                        contributions[f"text_feature_{i-20}"] = float(importance * feature_vector[i])
                
                return contributions
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting feature contributions: {str(e)}", exc_info=True)
            return {}
    
    def _calculate_confidence(self, feature_vector: np.ndarray) -> float:
        """Calculate confidence score for the prediction"""
        try:
            if "main" not in self.sklearn_models:
                return 0.0
            
            model = self.sklearn_models["main"]
            
            # Use prediction probability as confidence
            proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
            confidence = max(proba)  # Maximum probability
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}", exc_info=True)
            return 0.0
    
    def _get_recommendation(self, overall_score: float, conversion_prob: float) -> str:
        """Get recommendation based on score"""
        if overall_score >= 0.8:
            return "contact_immediate"
        elif overall_score >= 0.6:
            return "contact"
        elif overall_score >= 0.4:
            return "nurture"
        else:
            return "disqualify"
    
    def score_leads_batch(self, leads_data: List[Dict[str, Any]]) -> List[LeadScore]:
        """Score multiple leads in batch"""
        scores = []
        for lead_data in leads_data:
            score = self.score_lead(lead_data)
            scores.append(score)
        return scores
    
    def retrain_model(self, training_data: List[Dict[str, Any]], target_column: str = "converted") -> bool:
        """Retrain the ML model with new data"""
        try:
            # Prepare training data
            features_list = []
            targets = []
            
            for data in training_data:
                features = self.extract_features(data)
                feature_vector = self._prepare_feature_vector(features)
                features_list.append(feature_vector)
                targets.append(data.get(target_column, 0))
            
            X = np.array(features_list)
            y = np.array(targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train new model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            logger.info(f"Model retrained. Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
            
            # Save model
            model_path = self.model_path / "scikit" / "lead_scorer.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            
            # Update loaded model
            self.sklearn_models["main"] = model
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}", exc_info=True)
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        try:
            if "main" not in self.sklearn_models:
                return {}
            
            model = self.sklearn_models["main"]
            
            if hasattr(model, 'feature_importances_'):
                importance_dict = {}
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(self.feature_names):
                        importance_dict[self.feature_names[i]] = float(importance)
                    else:
                        importance_dict[f"text_feature_{i-20}"] = float(importance)
                
                return importance_dict
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}", exc_info=True)
            return {}
