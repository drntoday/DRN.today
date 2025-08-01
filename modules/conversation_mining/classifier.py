# modules/conversation_mining/classifier.py

import json
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from engine.SecureStorage import SecureStorage
from ai.nlp import NLPProcessor


class ConversationCategory(Enum):
    QUESTION = "question"
    RECOMMENDATION_REQUEST = "recommendation_request"
    RECOMMENDATION_GIVEN = "recommendation_given"
    COMPLAINT = "complaint"
    PRAISE = "praise"
    INFORMATION_SHARING = "information_sharing"
    BUYING_SIGNAL = "buying_signal"
    COMPETITOR_MENTION = "competitor_mention"
    PROBLEM_DISCUSSION = "problem_discussion"
    GENERAL_DISCUSSION = "general_discussion"
    OTHER = "other"


@dataclass
class ClassificationResult:
    conversation_id: str
    category: ConversationCategory
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    keywords_found: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationClassifier:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.model = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model_path = Path("ai/models/classifier/conversation_classifier.pkl")
        self.category_keywords = self._initialize_category_keywords()
        
        # Set up logging
        self.logger = logging.getLogger("conversation_classifier")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize tables
        self._initialize_tables()
        
        # Load or train model
        self._load_or_train_model()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS conversation_classifications (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL,
            all_scores TEXT,
            keywords_found TEXT,
            processing_time REAL,
            timestamp TEXT,
            FOREIGN KEY (conversation_id) REFERENCES monitored_conversations (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS classification_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            category TEXT NOT NULL,
            source TEXT,
            timestamp TEXT
        )
        """)

    def _initialize_category_keywords(self) -> Dict[ConversationCategory, List[str]]:
        """Initialize keyword patterns for each conversation category"""
        return {
            ConversationCategory.QUESTION: [
                "how", "what", "when", "where", "why", "who", "which", "can", "could", 
                "would", "should", "is", "are", "do", "does", "anyone", "somebody", "help"
            ],
            ConversationCategory.RECOMMENDATION_REQUEST: [
                "recommend", "suggest", "looking for", "need", "seeking", "advice", 
                "opinion", "best", "good", "anyone know", "what do you use"
            ],
            ConversationCategory.RECOMMENDATION_GIVEN: [
                "i recommend", "i suggest", "try", "check out", "you should", 
                "i use", "i've used", "my favorite", "highly recommend"
            ],
            ConversationCategory.COMPLAINT: [
                "problem", "issue", "broken", "doesn't work", "failed", "error", 
                "bug", "frustrated", "disappointed", "terrible", "awful", "hate"
            ],
            ConversationCategory.PRAISE: [
                "love", "great", "awesome", "amazing", "fantastic", "excellent", 
                "perfect", "best", "good", "helpful", "impressed", "thank you"
            ],
            ConversationCategory.INFORMATION_SHARING: [
                "just launched", "new feature", "update", "announcement", "we built", 
                "i created", "check this out", "interesting article", "did you know"
            ],
            ConversationCategory.BUYING_SIGNAL: [
                "looking to buy", "need to purchase", "shopping for", "budget", 
                "pricing", "cost", "quote", "demo", "trial", "subscribe", "sign up"
            ],
            ConversationCategory.COMPETITOR_MENTION: [
                "competitor", "alternative", "vs", "versus", "compared to", 
                "instead of", "better than", "worse than", "switch from"
            ],
            ConversationCategory.PROBLEM_DISCUSSION: [
                "struggling with", "challenge", "difficulty", "obstacle", 
                "how to solve", "facing", "dealing with", "overcome"
            ],
            ConversationCategory.GENERAL_DISCUSSION: [
                "what do you think", "opinion", "thoughts", "discussion", 
                "let's talk", "curious", "anyone else", "i wonder"
            ]
        }

    def _load_or_train_model(self):
        """Load trained model if available, otherwise train a new one"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.label_encoder = model_data['label_encoder']
                    self.vectorizer = model_data['vectorizer']
                self.logger.info("Loaded existing classification model")
                return
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
        
        # Train new model
        self.logger.info("Training new classification model")
        self._train_model()

    def _train_model(self, training_data: Optional[List[Tuple[str, str]]] = None):
        """Train the classification model"""
        # Get training data
        if training_data is None:
            training_data = self._get_training_data()
        
        if not training_data:
            self.logger.warning("No training data available, using default model")
            self._create_default_model()
            return
        
        # Prepare data
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # Encode labels
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        self.logger.info(f"Model trained with accuracy: {report['accuracy']:.2f}")
        self.logger.info(f"Model F1 scores: {report['macro avg']['f1-score']:.2f}")
        
        # Save model
        self._save_model()

    def _get_training_data(self) -> List[Tuple[str, str]]:
        """Get training data from database and default examples"""
        training_data = []
        
        # Get from database
        for row in self.SecureStorage.query("SELECT text, category FROM classification_training_data"):
            training_data.append((row['text'], row['category']))
        
        # Add default examples if not enough data
        if len(training_data) < 100:
            default_examples = self._get_default_training_examples()
            training_data.extend(default_examples)
        
        return training_data

    def _get_default_training_examples(self) -> List[Tuple[str, str]]:
        """Get default training examples for each category"""
        return [
            # Questions
            ("How do I integrate this with my CRM?", "question"),
            ("What's the best way to track conversions?", "question"),
            ("Can anyone explain how this feature works?", "question"),
            ("When will the new update be released?", "question"),
            ("Where can I find documentation?", "question"),
            
            # Recommendation Requests
            ("Can anyone recommend a good email marketing tool?", "recommendation_request"),
            ("Looking for a CRM that works well with e-commerce", "recommendation_request"),
            ("What analytics tool do you use for your startup?", "recommendation_request"),
            ("Need suggestions for project management software", "recommendation_request"),
            ("Anyone know of a good alternative to Salesforce?", "recommendation_request"),
            
            # Recommendations Given
            ("I recommend HubSpot for marketing automation", "recommendation_given"),
            ("You should try Mailchimp for email campaigns", "recommendation_given"),
            ("I've had great success with Pipedrive", "recommendation_given"),
            ("Check out Notion for project management", "recommendation_given"),
            ("Zoho has been a great CRM for our team", "recommendation_given"),
            
            # Complaints
            ("This software keeps crashing on my computer", "complaint"),
            ("The customer support is terrible", "complaint"),
            ("I'm frustrated with the constant bugs", "complaint"),
            "The pricing is way too high for what you get", "complaint"),
            ("The user interface is confusing and hard to navigate", "complaint"),
            
            # Praise
            ("I love this new feature, it's exactly what I needed", "praise"),
            ("The customer support team is amazing", "praise"),
            ("This tool has saved me so much time", "praise"),
            ("Best software I've used this year", "praise"),
            ("The user interface is beautiful and intuitive", "praise"),
            
            # Information Sharing
            ("Just launched our new integration with Slack", "information_sharing"),
            ("We added a new dashboard feature yesterday", "information_sharing"),
            ("Check out this article about marketing automation", "information_sharing"),
            ("Did you know they just released a mobile app?", "information_sharing"),
            ("Our team created a template for campaign tracking", "information_sharing"),
            
            # Buying Signals
            ("Looking to buy a new CRM for our sales team", "buying_signal"),
            ("We need to purchase a marketing automation tool", "buying_signal"),
            ("What's your pricing for enterprise plans?", "buying_signal"),
            ("Can I get a demo of your product?", "buying_signal"),
            ("We're shopping for analytics solutions", "buying_signal"),
            
            # Competitor Mentions
            ("How does this compare to Salesforce?", "competitor_mention"),
            ("We're switching from HubSpot to this", "competitor_mention"),
            ("Is this better than Marketo?", "competitor_mention),
            ("We used Zoho before this", "competitor_mention"),
            ("This seems like a good alternative to Pipedrive", "competitor_mention),
            
            # Problem Discussions
            ("Struggling with lead conversion rates", "problem_discussion"),
            ("How do you handle customer data privacy?", "problem_discussion"),
            ("We're facing challenges with email deliverability", "problem_discussion"),
            ("Dealing with high customer churn rates", "problem_discussion"),
            ("How to overcome sales objections effectively?", "problem_discussion),
            
            # General Discussions
            ("What do you think about the future of marketing automation?", "general_discussion"),
            ("Anyone else excited about AI in sales tools?", "general_discussion"),
            ("Let's discuss the best practices for lead nurturing", "general_discussion"),
            ("Curious about your experiences with remote sales teams", "general_discussion"),
            ("What are your thoughts on inbound vs outbound marketing?", "general_discussion)
        ]

    def _create_default_model(self):
        """Create a default model with basic keyword matching"""
        # Create a simple rule-based classifier as fallback
        self.model = None  # We'll use keyword matching instead
        self.logger.info("Created default rule-based classifier")

    def _save_model(self):
        """Save the trained model to disk"""
        # Create directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'vectorizer': self.vectorizer
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Saved classification model to {self.model_path}")

    def classify(self, conversation_id: str, text: str) -> ClassificationResult:
        """Classify a conversation text into categories"""
        start_time = datetime.now()
        
        # Preprocess text
        processed_text = self.nlp.preprocess_text(text)
        
        # Check for keyword matches
        keyword_matches = self._check_keyword_matches(processed_text)
        
        # Use model if available, otherwise use keyword matching
        if self.model is not None:
            # Get model predictions
            try:
                probabilities = self.model.predict_proba([processed_text])[0]
                classes = self.label_encoder.classes_
                
                # Create scores dictionary
                all_scores = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
                
                # Get top category
                top_category_idx = np.argmax(probabilities)
                top_category = ConversationCategory(classes[top_category_idx])
                confidence = float(probabilities[top_category_idx])
                
                # Combine with keyword scores
                final_scores = self._combine_scores(all_scores, keyword_matches)
                top_category = max(final_scores, key=final_scores.get)
                confidence = final_scores[top_category]
                
            except Exception as e:
                self.logger.error(f"Error using model for classification: {str(e)}")
                # Fall back to keyword matching
                top_category, confidence, final_scores = self._classify_by_keywords(keyword_matches)
        else:
            # Use keyword matching
            top_category, confidence, final_scores = self._classify_by_keywords(keyword_matches)
        
        # Get keywords found
        keywords_found = []
        for category, matches in keyword_matches.items():
            if matches:
                keywords_found.extend(matches)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = ClassificationResult(
            conversation_id=conversation_id,
            category=top_category,
            confidence=confidence,
            all_scores=final_scores,
            keywords_found=keywords_found,
            processing_time=processing_time
        )
        
        # Save to database
        self._save_classification(result)
        
        return result

    def _check_keyword_matches(self, text: str) -> Dict[ConversationCategory, List[str]]:
        """Check for keyword matches in text"""
        matches = {}
        text_lower = text.lower()
        
        for category, keywords in self.category_keywords.items():
            category_matches = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    category_matches.append(keyword)
            
            if category_matches:
                matches[category] = category_matches
        
        return matches

    def _classify_by_keywords(self, keyword_matches: Dict[ConversationCategory, List[str]]) -> Tuple[ConversationCategory, float, Dict[str, float]]:
        """Classify based on keyword matches"""
        scores = {}
        
        # Initialize scores
        for category in ConversationCategory:
            scores[category] = 0.0
        
        # Score based on keyword matches
        for category, matches in keyword_matches.items():
            # More matches = higher score
            scores[category] = len(matches) * 0.2
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 0
        if max_score > 0:
            normalized_scores = {k: v/max_score for k, v in scores.items()}
        else:
            normalized_scores = {k: 0.1 for k in scores}  # Default low confidence
        
        # Get top category
        top_category = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[top_category]
        
        return top_category, confidence, normalized_scores

    def _combine_scores(self, model_scores: Dict[str, float], 
                       keyword_scores: Dict[ConversationCategory, List[str]]) -> Dict[ConversationCategory, float]:
        """Combine model scores with keyword scores"""
        combined_scores = {}
        
        # Initialize with model scores
        for category_str, score in model_scores.items():
            category = ConversationCategory(category_str)
            combined_scores[category] = score
        
        # Add keyword bonuses
        for category, matches in keyword_scores.items():
            bonus = min(len(matches) * 0.1, 0.3)  # Cap bonus at 0.3
            if category in combined_scores:
                combined_scores[category] = min(combined_scores[category] + bonus, 1.0)
            else:
                combined_scores[category] = bonus
        
        # Ensure all categories have scores
        for category in ConversationCategory:
            if category not in combined_scores:
                combined_scores[category] = 0.01  # Minimal score
        
        return combined_scores

    def _save_classification(self, result: ClassificationResult):
        """Save classification result to database"""
        self.SecureStorage.execute(
            """
            INSERT OR REPLACE INTO conversation_classifications 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"class_{result.conversation_id}_{int(result.timestamp.timestamp())}",
                result.conversation_id,
                result.category.value,
                result.confidence,
                json.dumps(result.all_scores),
                json.dumps(result.keywords_found),
                result.processing_time,
                result.timestamp.isoformat()
            )
        )

    def get_classification(self, conversation_id: str) -> Optional[ClassificationResult]:
        """Get classification result for a conversation"""
        row = self.SecureStorage.query(
            "SELECT * FROM conversation_classifications WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT 1",
            (conversation_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return ClassificationResult(
            conversation_id=row['conversation_id'],
            category=ConversationCategory(row['category']),
            confidence=row['confidence'],
            all_scores=json.loads(row['all_scores']),
            keywords_found=json.loads(row['keywords_found']),
            processing_time=row['processing_time'],
            timestamp=datetime.fromisoformat(row['timestamp'])
        )

    def add_training_example(self, text: str, category: str, source: str = "manual"):
        """Add a new training example"""
        self.SecureStorage.execute(
            "INSERT INTO classification_training_data (text, category, source, timestamp) VALUES (?, ?, ?, ?)",
            (text, category, source, datetime.now().isoformat())
        )
        
        # Retrain model periodically or when enough new data is added
        self._check_retrain_needed()

    def _check_retrain_needed(self):
        """Check if model retraining is needed"""
        # Count new training examples since last model update
        count = self.SecureStorage.query(
            "SELECT COUNT(*) FROM classification_training_data WHERE timestamp > ?",
            (datetime.now().isoformat(),)  # This would be the last model update time
        ).fetchone()[0]
        
        # Retrain if we have 50+ new examples
        if count >= 50:
            self.logger.info("Retraining model with new data")
            self._train_model()

    def get_classification_stats(self, days: int = 7) -> Dict:
        """Get classification statistics"""
        since = datetime.now() - pd.Timedelta(days=days)
        
        # Get classification counts
        query = """
        SELECT category, COUNT(*) as count, AVG(confidence) as avg_confidence
        FROM conversation_classifications 
        WHERE timestamp >= ?
        GROUP BY category
        """
        
        stats = {}
        for row in self.SecureStorage.query(query, (since.isoformat(),)):
            stats[row['category']] = {
                "count": row['count'],
                "avg_confidence": row['avg_confidence']
            }
        
        # Get processing time stats
        time_stats = self.SecureStorage.query(
            "SELECT AVG(processing_time) as avg_time, MAX(processing_time) as max_time FROM conversation_classifications WHERE timestamp >= ?",
            (since.isoformat(),)
        ).fetchone()
        
        return {
            "category_distribution": stats,
            "avg_processing_time": time_stats['avg_time'],
            "max_processing_time": time_stats['max_time'],
            "total_classifications": sum(s["count"] for s in stats.values())
        }

    def export_training_data(self, format: str = "csv") -> str:
        """Export training data in specified format"""
        data = []
        for row in self.SecureStorage.query("SELECT text, category, source, timestamp FROM classification_training_data"):
            data.append({
                "text": row['text'],
                "category": row['category'],
                "source": row['source'],
                "timestamp": row['timestamp']
            })
        
        if format.lower() == "csv":
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        elif format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def bulk_classify(self, conversations: List[Tuple[str, str]]) -> List[ClassificationResult]:
        """Classify multiple conversations efficiently"""
        results = []
        for conversation_id, text in conversations:
            result = self.classify(conversation_id, text)
            results.append(result)
        return results
