import os
import sys
import logging
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def download_tinybert_models(model_dir: Path):
    """Download and save TinyBERT models"""
    try:
        logger.info("Downloading TinyBERT model...")
        model_name = "huawei-noah/TinyBERT_General_4L_312D"
        
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info("Saving TinyBERT model...")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        logger.info(f"TinyBERT model saved to {model_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to download TinyBERT: {str(e)}")
        return False

def create_sklearn_models(model_dir: Path):
    """Create and save scikit-learn models"""
    try:
        logger.info("Creating scikit-learn models...")
        
        # Create a simple lead scoring model
        lead_scorer = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', RandomForestClassifier(n_estimators=100))
        ])
        
        # Dummy training data
        X = ["This is a positive lead", "Negative lead example", "Great potential customer"]
        y = [1, 0, 1]  # 1 = good lead, 0 = bad lead
        
        lead_scorer.fit(X, y)
        
        # Create a simple classifier
        classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', RandomForestClassifier(n_estimators=100))
        ])
        
        # Dummy training data
        X_clf = ["Looking to buy", "Need support", "Requesting information"]
        y_clf = ["purchase", "support", "information"]
        
        classifier.fit(X_clf, y_clf)
        
        # Save models
        joblib.dump(lead_scorer, model_dir / "lead_scorer.pkl")
        joblib.dump(classifier, model_dir / "classifier.pkl")
        
        logger.info(f"Scikit-learn models saved to {model_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create scikit-learn models: {str(e)}")
        return False

def download_models():
    """Main function to download all required models"""
    try:
        # Create model directories
        base_dir = Path(__file__).parent.parent
        ai_dir = base_dir / "ai" / "models"
        
        ensure_dir(ai_dir / "tinybert")
        ensure_dir(ai_dir / "scikit")
        
        # Download models
        tinybert_success = download_tinybert_models(ai_dir / "tinybert")
        sklearn_success = create_sklearn_models(ai_dir / "scikit")
        
        if not tinybert_success or not sklearn_success:
            raise Exception("Some models failed to download/create")
            
        logger.info("All models downloaded and created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model download failed: {str(e)}")
        return False

if __name__ == "__main__":
    if download_models():
        sys.exit(0)
    else:
        sys.exit(1)