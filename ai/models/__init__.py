"""
AI Models Package for DRN.today
Provides unified access to all AI/ML models used in the application.
"""

__version__ = "1.0.0"
__author__ = "DRN.today Team"

import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Define model paths
MODEL_BASE_PATH = Path(__file__).parent
TINYBERT_PATH = MODEL_BASE_PATH / "tinybert"
SCIKIT_PATH = MODEL_BASE_PATH / "scikit"

# Ensure model directories exist
TINYBERT_PATH.mkdir(exist_ok=True)
SCIKIT_PATH.mkdir(exist_ok=True)

class ModelRegistry:
    """Central registry for all AI models used in DRN.today"""
    
    _models = {
        "tinybert": {
            "base": None,
            "entity": None,
            "sentiment": None,
            "intent": None,
            "persona": None
        },
        "scikit": {
            "lead_scoring": None,
            "reply_classifier": None,
            "conversion_predictor": None,
            "category_classifier": None
        }
    }
    
    @classmethod
    def register_model(cls, model_type: str, model_name: str, model_instance: Any):
        """Register a model instance in the registry"""
        if model_type not in cls._models:
            cls._models[model_type] = {}
        cls._models[model_type][model_name] = model_instance
        logger.info(f"Registered {model_type}.{model_name} model")
    
    @classmethod
    def get_model(cls, model_type: str, model_name: str) -> Optional[Any]:
        """Get a model instance from the registry"""
        return cls._models.get(model_type, {}).get(model_name)
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, str]]:
        """List all available models"""
        return {
            model_type: list(models.keys())
            for model_type, models in cls._models.items()
        }

def load_tinybert_model(model_name: str) -> Any:
    """Load a TinyBERT model by name"""
    try:
        from .tinybert import load_model
        model = load_model(model_name)
        ModelRegistry.register_model("tinybert", model_name, model)
        return model
    except ImportError as e:
        logger.error(f"Failed to load TinyBERT model {model_name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading TinyBERT model {model_name}: {str(e)}")
        raise

def load_scikit_model(model_name: str) -> Any:
    """Load a Scikit-learn model by name"""
    try:
        from .scikit import load_model
        model = load_model(model_name)
        ModelRegistry.register_model("scikit", model_name, model)
        return model
    except ImportError as e:
        logger.error(f"Failed to load Scikit model {model_name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading Scikit model {model_name}: {str(e)}")
        raise

def get_model_path(model_type: str, model_name: str) -> Path:
    """Get the file path for a specific model"""
    if model_type == "tinybert":
        return TINYBERT_PATH / f"{model_name}.bin"
    elif model_type == "scikit":
        return SCIKIT_PATH / f"{model_name}.pkl"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def verify_model_integrity(model_type: str, model_name: str) -> bool:
    """Verify that a model file exists and is not corrupted"""
    model_path = get_model_path(model_type, model_name)
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return False
    
    # Basic file size check (models should be at least 1MB)
    if model_path.stat().st_size < 1024 * 1024:
        logger.warning(f"Model file too small: {model_path}")
        return False
    
    return True

def initialize_models():
    """Initialize all required models at application startup"""
    logger.info("Initializing AI models...")
    
    # Load TinyBERT models
    tinybert_models = ["base", "entity", "sentiment", "intent", "persona"]
    for model in tinybert_models:
        if verify_model_integrity("tinybert", model):
            try:
                load_tinybert_model(model)
            except Exception as e:
                logger.error(f"Failed to initialize TinyBERT model {model}: {str(e)}")
        else:
            logger.warning(f"Skipping TinyBERT model {model} - file not found or corrupted")
    
    # Load Scikit-learn models
    scikit_models = ["lead_scoring", "reply_classifier", "conversion_predictor", "category_classifier"]
    for model in scikit_models:
        if verify_model_integrity("scikit", model):
            try:
                load_scikit_model(model)
            except Exception as e:
                logger.error(f"Failed to initialize Scikit model {model}: {str(e)}")
        else:
            logger.warning(f"Skipping Scikit model {model} - file not found or corrupted")
    
    logger.info("AI models initialization completed")

def get_model_info(model_type: str, model_name: str) -> Dict[str, Any]:
    """Get metadata about a specific model"""
    model_path = get_model_path(model_type, model_name)
    if not model_path.exists():
        return {"exists": False}
    
    stat = model_path.stat()
    return {
        "exists": True,
        "path": str(model_path),
        "size_bytes": stat.st_size,
        "last_modified": stat.st_mtime,
        "type": model_type,
        "name": model_name
    }

def list_available_models() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """List all available models with their metadata"""
    models = {
        "tinybert": {},
        "scikit": {}
    }
    
    # List TinyBERT models
    for model_file in TINYBERT_PATH.glob("*.bin"):
        model_name = model_file.stem
        models["tinybert"][model_name] = get_model_info("tinybert", model_name)
    
    # List Scikit models
    for model_file in SCIKIT_PATH.glob("*.pkl"):
        model_name = model_file.stem
        models["scikit"][model_name] = get_model_info("scikit", model_name)
    
    return models

# Initialize models when package is imported
try:
    initialize_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    # Continue without models - they can be loaded later

__all__ = [
    "ModelRegistry",
    "load_tinybert_model",
    "load_scikit_model",
    "get_model_path",
    "verify_model_integrity",
    "initialize_models",
    "get_model_info",
    "list_available_models"
]
