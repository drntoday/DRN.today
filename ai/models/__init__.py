import os
import logging
import joblib
from typing import Dict, Any, Optional, Union
from pathlib import Path
from transformers import AutoModel

# Configure logging
logger = logging.getLogger(__name__)

# Define model paths
MODEL_BASE_PATH = Path(__file__).parent
TINYBERT_PATH = MODEL_BASE_PATH / "tinybert"
SCIKIT_PATH = MODEL_BASE_PATH / "scikit"

class TinyBERTModel:
    """Wrapper for TinyBERT model loading and predictions"""
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or TINYBERT_PATH
        try:
            self.model = AutoModel.from_pretrained(str(self.model_path))
            logger.info(f"Loaded TinyBERT model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load TinyBERT model: {str(e)}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """Placeholder for actual prediction logic"""
        return {"prediction": "sample_output"}

def load_sklearn_model(model_name: str):
    """Load a scikit-learn model by name"""
    model_path = SCIKIT_PATH / f"{model_name}.pkl"
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded scikit-learn model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load scikit-learn model {model_name}: {str(e)}")
        raise

class ModelRegistry:
    """Central registry for all AI models"""
    
    _models = {
        "tinybert": {
            "base": None,
        },
        "scikit": {
            "classifier": None,
            "lead_scorer": None
        }
    }
    
    @classmethod
    def initialize(cls):
        """Initialize all models"""
        try:
            # Load TinyBERT
            cls._models["tinybert"]["base"] = TinyBERTModel()
            
            # Load scikit models
            for model_name in cls._models["scikit"].keys():
                cls._models["scikit"][model_name] = load_sklearn_model(model_name)
                
            logger.info("All models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    @classmethod
    def get_model(cls, model_type: str, model_name: str) -> Optional[Any]:
        """Get a loaded model instance"""
        return cls._models.get(model_type, {}).get(model_name)

# Initialize models on import
try:
    ModelRegistry.initialize()
except Exception as e:
    logger.warning(f"Partial model initialization: {str(e)}")

__all__ = [
    "TinyBERTModel",
    "load_sklearn_model",
    "ModelRegistry"
]