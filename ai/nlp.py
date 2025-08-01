#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
Natural Language Processing (NLP) Engine
Production-Ready Implementation
"""

import os
import re
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

# Initialize NLP logger
logger = logging.getLogger(__name__)

@dataclass
class NLPResult:
    """Data structure for NLP processing results"""
    text: str
    entities: Dict[str, List[str]] = field(default_factory=dict)
    sentiment: Dict[str, float] = field(default_factory=dict)
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class NLPProcessor:
    """Production-ready NLP processor with TinyBERT and Scikit-learn models"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.tokenizer: Optional[AutoTokenizer] = None
        self.tinybert_model: Optional[AutoModel] = None
        self.classification_model: Optional[AutoModelForSequenceClassification] = None
        self.spacy_model: Optional[spacy.Language] = None
        
        # Scikit-learn models
        self.intent_classifier: Optional[Pipeline] = None
        self.sentiment_analyzer: Optional[Pipeline] = None
        self.keyword_extractor: Optional[TfidfVectorizer] = None
        
        # Model metadata
        self.model_info = {
            "tinybert_loaded": False,
            "classification_loaded": False,
            "spacy_loaded": False,
            "sklearn_loaded": False
        }
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all NLP models with error handling"""
        try:
            # Load TinyBERT
            self._load_tinybert()
            
            # Load classification model
            self._load_classification_model()
            
            # Load spaCy model
            self._load_spacy()
            
            # Load Scikit-learn models
            self._load_sklearn_models()
            
            logger.info("All NLP models loaded successfully")
            
        except Exception as e:
            logger.critical(f"Failed to load NLP models: {str(e)}", exc_info=True)
            raise
    
    def _load_tinybert(self):
        """Load TinyBERT model and tokenizer"""
        try:
            model_dir = self.model_path / "tinybert"
            if not model_dir.exists():
                logger.warning(f"TinyBERT model directory not found: {model_dir}")
                return
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.tinybert_model = AutoModel.from_pretrained(model_dir)
            self.tinybert_model.to(self.device)
            self.tinybert_model.eval()
            
            self.model_info["tinybert_loaded"] = True
            logger.info("TinyBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TinyBERT: {str(e)}", exc_info=True)
            raise
    
    def _load_classification_model(self):
        """Load TinyBERT classification model"""
        try:
            model_dir = self.model_path / "classification"
            if not model_dir.exists():
                logger.warning(f"Classification model directory not found: {model_dir}")
                return
            
            self.classification_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.classification_model.to(self.device)
            self.classification_model.eval()
            
            self.model_info["classification_loaded"] = True
            logger.info("Classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load classification model: {str(e)}", exc_info=True)
            raise
    
    def _load_spacy(self):
        """Load spaCy model for entity recognition"""
        try:
            # Try to load English model
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
            except OSError:
                # Fallback to basic English model
                self.spacy_model = spacy.blank("en")
                logger.warning("Using basic spaCy model - entity recognition may be limited")
            
            self.model_info["spacy_loaded"] = True
            logger.info("spaCy model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load spaCy: {str(e)}", exc_info=True)
            raise
    
    def _load_sklearn_models(self):
        """Load Scikit-learn models"""
        try:
            sklearn_dir = self.model_path / "scikit"
            if not sklearn_dir.exists():
                logger.warning(f"Scikit-learn models directory not found: {sklearn_dir}")
                return
            
            # Load intent classifier
            intent_path = sklearn_dir / "intent_classifier.pkl"
            if intent_path.exists():
                self.intent_classifier = joblib.load(intent_path)
            
            # Load sentiment analyzer
            sentiment_path = sklearn_dir / "sentiment_analyzer.pkl"
            if sentiment_path.exists():
                self.sentiment_analyzer = joblib.load(sentiment_path)
            
            # Load keyword extractor
            keyword_path = sklearn_dir / "keyword_extractor.pkl"
            if keyword_path.exists():
                self.keyword_extractor = joblib.load(keyword_path)
            
            self.model_info["sklearn_loaded"] = True
            logger.info("Scikit-learn models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Scikit-learn models: {str(e)}", exc_info=True)
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP analysis"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove phone numbers
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}", exc_info=True)
            return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            "PERSON": [],
            "ORG": [],
            "EMAIL": [],
            "PHONE": [],
            "LOCATION": [],
            "COMPANY": []
        }
        
        try:
            if not self.spacy_model:
                return entities
            
            # Process text with spaCy
            doc = self.spacy_model(text)
            
            # Extract standard entities
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            # Extract email addresses
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            entities["EMAIL"].extend(emails)
            
            # Extract phone numbers
            phones = re.findall(r'(\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
            entities["PHONE"].extend(phones)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        sentiment = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
            "compound": 0.0
        }
        
        try:
            # Try TinyBERT first
            if self.classification_model and self.tokenizer:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.classification_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                sentiment["positive"] = float(probs[0])
                sentiment["negative"] = float(probs[1])
                sentiment["neutral"] = float(probs[2])
                sentiment["compound"] = sentiment["positive"] - sentiment["negative"]
                
                return sentiment
            
            # Fallback to Scikit-learn
            elif self.sentiment_analyzer:
                preprocessed = self.preprocess_text(text)
                probs = self.sentiment_analyzer.predict_proba([preprocessed])[0]
                
                sentiment["positive"] = float(probs[0])
                sentiment["negative"] = float(probs[1])
                sentiment["neutral"] = float(probs[2])
                sentiment["compound"] = sentiment["positive"] - sentiment["negative"]
                
                return sentiment
            
            # Simple rule-based fallback
            else:
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
                negative_words = ["bad", "terrible", "awful", "horrible", "poor", "worst"]
                
                words = text.lower().split()
                positive_count = sum(1 for word in words if word in positive_words)
                negative_count = sum(1 for word in words if word in negative_words)
                
                total = positive_count + negative_count
                if total > 0:
                    sentiment["positive"] = positive_count / total
                    sentiment["negative"] = negative_count / total
                    sentiment["neutral"] = 1 - (sentiment["positive"] + sentiment["negative"])
                    sentiment["compound"] = sentiment["positive"] - sentiment["negative"]
                else:
                    sentiment["neutral"] = 1.0
                
                return sentiment
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
            return sentiment
    
    def detect_intent(self, text: str) -> Tuple[str, float]:
        """Detect user intent from text"""
        try:
            # Try TinyBERT first
            if self.classification_model and self.tokenizer:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.classification_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                intent_labels = ["information", "purchase", "support", "complaint", "feedback"]
                intent_idx = np.argmax(probs)
                confidence = float(probs[intent_idx])
                
                return intent_labels[intent_idx], confidence
            
            # Fallback to Scikit-learn
            elif self.intent_classifier:
                preprocessed = self.preprocess_text(text)
                intent = self.intent_classifier.predict([preprocessed])[0]
                probs = self.intent_classifier.predict_proba([preprocessed])[0]
                confidence = float(np.max(probs))
                
                return intent, confidence
            
            # Rule-based fallback
            else:
                text_lower = text.lower()
                
                # Buying signals
                buying_patterns = [
                    r"looking for",
                    r"need.*service",
                    r"recommend.*tool",
                    r"interested in",
                    r"want to buy",
                    r"pricing",
                    r"cost",
                    r"how much"
                ]
                
                for pattern in buying_patterns:
                    if re.search(pattern, text_lower):
                        return "purchase", 0.8
                
                # Support patterns
                support_patterns = [
                    r"help",
                    r"issue",
                    r"problem",
                    r"broken",
                    r"not working",
                    r"error"
                ]
                
                for pattern in support_patterns:
                    if re.search(pattern, text_lower):
                        return "support", 0.8
                
                # Default to information
                return "information", 0.5
                
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}", exc_info=True)
            return "unknown", 0.0
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        try:
            # Try Scikit-learn TF-IDF
            if self.keyword_extractor:
                features = self.keyword_extractor.transform([text])
                feature_names = self.keyword_extractor.get_feature_names_out()
                scores = features.toarray()[0]
                
                # Get top keywords
                top_indices = np.argsort(scores)[-max_keywords:][::-1]
                keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
                
                return keywords
            
            # Fallback to simple frequency-based extraction
            else:
                # Preprocess text
                text = self.preprocess_text(text)
                words = text.split()
                
                # Remove stop words
                words = [word for word in words if word not in STOP_WORDS]
                
                # Count word frequencies
                word_counts = Counter(words)
                
                # Get most common words
                keywords = [word for word, count in word_counts.most_common(max_keywords)]
                
                return keywords
                
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}", exc_info=True)
            return []
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories"""
        category_scores = {category: 0.0 for category in categories}
        
        try:
            # Try TinyBERT first
            if self.classification_model and self.tokenizer:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.classification_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Map probabilities to categories
                for i, category in enumerate(categories):
                    if i < len(probs):
                        category_scores[category] = float(probs[i])
                
                return category_scores
            
            # Fallback to simple keyword matching
            else:
                text_lower = text.lower()
                
                # Define keywords for each category
                category_keywords = {
                    "technology": ["software", "tech", "ai", "ml", "data", "cloud"],
                    "finance": ["money", "investment", "banking", "finance", "payment"],
                    "healthcare": ["health", "medical", "patient", "doctor", "hospital"],
                    "education": ["school", "education", "learning", "student", "teacher"]
                }
                
                for category, keywords in category_keywords.items():
                    if category in category_scores:
                        matches = sum(1 for keyword in keywords if keyword in text_lower)
                        category_scores[category] = matches / len(keywords)
                
                return category_scores
                
        except Exception as e:
            logger.error(f"Error classifying text: {str(e)}", exc_info=True)
            return category_scores
    
    def process_text(self, text: str, categories: List[str] = None) -> NLPResult:
        """Process text with full NLP pipeline"""
        start_time = time.time()
        
        result = NLPResult(text=text)
        
        try:
            # Preprocess text
            preprocessed = self.preprocess_text(text)
            
            # Extract entities
            result.entities = self.extract_entities(text)
            
            # Analyze sentiment
            result.sentiment = self.analyze_sentiment(text)
            
            # Detect intent
            result.intent, result.intent_confidence = self.detect_intent(text)
            
            # Extract keywords
            result.keywords = self.extract_keywords(text)
            
            # Classify text
            if categories:
                category_scores = self.classify_text(text, categories)
                result.categories = [
                    cat for cat, score in category_scores.items() 
                    if score > 0.5
                ]
            
            # Calculate overall score
            result.score = self._calculate_overall_score(result)
            
            # Record processing time
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}", exc_info=True)
            result.processing_time = time.time() - start_time
            return result
    
    def _calculate_overall_score(self, result: NLPResult) -> float:
        """Calculate overall relevance score"""
        score = 0.0
        
        # Entity score
        entity_count = sum(len(entities) for entities in result.entities.values())
        score += min(0.3, entity_count * 0.1)
        
        # Sentiment score
        if result.sentiment["compound"] > 0:
            score += result.sentiment["compound"] * 0.2
        
        # Intent score
        if result.intent == "purchase":
            score += result.intent_confidence * 0.3
        
        # Keyword score
        score += min(0.2, len(result.keywords) * 0.02)
        
        return min(1.0, score)
    
    def train_intent_classifier(self, texts: List[str], labels: List[str]) -> bool:
        """Train intent classifier with new data"""
        try:
            # Preprocess texts
            preprocessed_texts = [self.preprocess_text(text) for text in texts]
            
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', LogisticRegression(max_iter=1000))
            ])
            
            # Train model
            pipeline.fit(preprocessed_texts, labels)
            
            # Save model
            sklearn_dir = self.model_path / "scikit"
            sklearn_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(pipeline, sklearn_dir / "intent_classifier.pkl")
            
            # Update loaded model
            self.intent_classifier = pipeline
            
            logger.info("Intent classifier trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training intent classifier: {str(e)}", exc_info=True)
            return False
    
    def train_sentiment_analyzer(self, texts: List[str], sentiments: List[str]) -> bool:
        """Train sentiment analyzer with new data"""
        try:
            # Preprocess texts
            preprocessed_texts = [self.preprocess_text(text) for text in texts]
            
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', RandomForestClassifier(n_estimators=100))
            ])
            
            # Train model
            pipeline.fit(preprocessed_texts, sentiments)
            
            # Save model
            sklearn_dir = self.model_path / "scikit"
            sklearn_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(pipeline, sklearn_dir / "sentiment_analyzer.pkl")
            
            # Update loaded model
            self.sentiment_analyzer = pipeline
            
            logger.info("Sentiment analyzer trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training sentiment analyzer: {str(e)}", exc_info=True)
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            **self.model_info,
            "device": str(self.device),
            "model_path": str(self.model_path),
            "tinybert_model": str(self.tinybert_model.__class__.__name__) if self.tinybert_model else None,
            "classification_model": str(self.classification_model.__class__.__name__) if self.classification_model else None,
            "spacy_model": str(self.spacy_model.__class__.__name__) if self.spacy_model else None
        }
