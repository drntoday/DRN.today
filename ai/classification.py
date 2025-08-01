#!/usr/bin/env python3
"""
DRN.today - Enterprise-Grade Lead Generation Platform
AI Text Classification Engine
Production-Ready Implementation
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from collections import Counter

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Initialize classification logger
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Classification result data structure"""
    uuid: str
    text: str
    predicted_label: str
    confidence: float
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    model_used: str = "ensemble"
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetrics:
    """Model evaluation metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    classification_report: str = ""

class TextClassifier:
    """Production-ready text classification engine with multiple models"""
    
    def __init__(self, model_path: str = "ai/models"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.tinybert_models: Dict[str, Any] = {}
        self.sklearn_models: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Model configurations
        self.model_configs = {
            "intent": {
                "labels": ["information", "purchase", "support", "complaint", "feedback"],
                "model_type": "tinybert"
            },
            "sentiment": {
                "labels": ["positive", "negative", "neutral"],
                "model_type": "tinybert"
            },
            "category": {
                "labels": ["technology", "healthcare", "finance", "education", "retail"],
                "model_type": "sklearn"
            },
            "urgency": {
                "labels": ["low", "medium", "high", "critical"],
                "model_type": "sklearn"
            }
        }
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all classification models"""
        try:
            # Load TinyBERT models
            tinybert_path = self.model_path / "tinybert"
            if tinybert_path.exists():
                for model_name in os.listdir(tinybert_path):
                    model_dir = tinybert_path / model_name
                    if model_dir.is_dir():
                        try:
                            config = AutoConfig.from_pretrained(model_dir)
                            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                            tokenizer = AutoTokenizer.from_pretrained(model_dir)
                            
                            model.to(self.device)
                            model.eval()
                            
                            self.tinybert_models[model_name] = {
                                "model": model,
                                "tokenizer": tokenizer,
                                "config": config,
                                "labels": config.id2label
                            }
                            
                            logger.info(f"Loaded TinyBERT model: {model_name}")
                            
                        except Exception as e:
                            logger.error(f"Error loading TinyBERT model {model_name}: {str(e)}", exc_info=True)
            
            # Load Scikit-learn models
            sklearn_path = self.model_path / "scikit"
            if sklearn_path.exists():
                for model_file in sklearn_path.glob("*.pkl"):
                    try:
                        model_name = model_file.stem
                        model = joblib.load(model_file)
                        
                        self.sklearn_models[model_name] = model
                        
                        # Load associated vectorizer and scaler
                        vectorizer_file = sklearn_path / f"{model_name}_vectorizer.pkl"
                        if vectorizer_file.exists():
                            self.vectorizers[model_name] = joblib.load(vectorizer_file)
                        
                        scaler_file = sklearn_path / f"{model_name}_scaler.pkl"
                        if scaler_file.exists():
                            self.scalers[model_name] = joblib.load(scaler_file)
                        
                        encoder_file = sklearn_path / f"{model_name}_encoder.pkl"
                        if encoder_file.exists():
                            self.label_encoders[model_name] = joblib.load(encoder_file)
                        
                        logger.info(f"Loaded Scikit-learn model: {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading Scikit-learn model {model_file}: {str(e)}", exc_info=True)
            
            logger.info("All classification models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise
    
    def classify_text(self, text: str, task: str = "intent") -> ClassificationResult:
        """Classify text using the appropriate model"""
        start_time = datetime.now().timestamp()
        
        try:
            # Get model configuration
            if task not in self.model_configs:
                raise ValueError(f"Unknown classification task: {task}")
            
            config = self.model_configs[task]
            model_type = config["model_type"]
            
            # Classify based on model type
            if model_type == "tinybert":
                result = self._classify_with_tinybert(text, task)
            elif model_type == "sklearn":
                result = self._classify_with_sklearn(text, task)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Update result metadata
            result.processing_time = datetime.now().timestamp() - start_time
            result.model_used = model_type
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying text: {str(e)}", exc_info=True)
            return ClassificationResult(
                uuid=str(uuid.uuid4()),
                text=text,
                predicted_label="unknown",
                confidence=0.0,
                model_used="fallback",
                processing_time=datetime.now().timestamp() - start_time,
                metadata={"error": str(e)}
            )
    
    def _classify_with_tinybert(self, text: str, task: str) -> ClassificationResult:
        """Classify text using TinyBERT model"""
        try:
            if task not in self.tinybert_models:
                raise ValueError(f"TinyBERT model not available for task: {task}")
            
            model_info = self.tinybert_models[task]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            labels = model_info["labels"]
            
            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            # Get results
            predicted_id = predictions.item()
            predicted_label = labels[str(predicted_id)]
            confidence = probabilities[0][predicted_id].item()
            
            # Get all probabilities
            all_probabilities = {
                labels[str(i)]: probabilities[0][i].item()
                for i in range(len(labels))
            }
            
            return ClassificationResult(
                uuid=str(uuid.uuid4()),
                text=text,
                predicted_label=predicted_label,
                confidence=confidence,
                all_probabilities=all_probabilities
            )
            
        except Exception as e:
            logger.error(f"Error classifying with TinyBERT: {str(e)}", exc_info=True)
            raise
    
    def _classify_with_sklearn(self, text: str, task: str) -> ClassificationResult:
        """Classify text using Scikit-learn model"""
        try:
            if task not in self.sklearn_models:
                raise ValueError(f"Scikit-learn model not available for task: {task}")
            
            model = self.sklearn_models[task]
            vectorizer = self.vectorizers.get(task)
            scaler = self.scalers.get(task)
            encoder = self.label_encoders.get(task)
            
            # Vectorize text
            if vectorizer:
                text_vector = vectorizer.transform([text])
            else:
                # Fallback to simple TF-IDF
                vectorizer = TfidfVectorizer(max_features=1000)
                text_vector = vectorizer.fit_transform([text])
            
            # Scale features if scaler is available
            if scaler:
                text_vector = scaler.transform(text_vector)
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(text_vector)[0]
                predicted_id = model.predict(text_vector)[0]
                confidence = float(max(probabilities))
            else:
                predicted_id = model.predict(text_vector)[0]
                confidence = 1.0
            
            # Get label
            if encoder:
                predicted_label = encoder.inverse_transform([predicted_id])[0]
            else:
                predicted_label = str(predicted_id)
            
            # Get all probabilities if available
            all_probabilities = {}
            if hasattr(model, 'predict_proba') and encoder:
                for i, prob in enumerate(probabilities):
                    label = encoder.inverse_transform([i])[0]
                    all_probabilities[label] = float(prob)
            
            return ClassificationResult(
                uuid=str(uuid.uuid4()),
                text=text,
                predicted_label=predicted_label,
                confidence=confidence,
                all_probabilities=all_probabilities
            )
            
        except Exception as e:
            logger.error(f"Error classifying with Scikit-learn: {str(e)}", exc_info=True)
            raise
    
    def classify_batch(self, texts: List[str], task: str = "intent") -> List[ClassificationResult]:
        """Classify multiple texts in batch"""
        results = []
        
        try:
            # Get model configuration
            if task not in self.model_configs:
                raise ValueError(f"Unknown classification task: {task}")
            
            config = self.model_configs[task]
            model_type = config["model_type"]
            
            # Classify based on model type
            if model_type == "tinybert":
                results = self._classify_batch_tinybert(texts, task)
            elif model_type == "sklearn":
                results = self._classify_batch_sklearn(texts, task)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying batch: {str(e)}", exc_info=True)
            # Return fallback results
            return [
                ClassificationResult(
                    uuid=str(uuid.uuid4()),
                    text=text,
                    predicted_label="unknown",
                    confidence=0.0,
                    model_used="fallback",
                    metadata={"error": str(e)}
                )
                for text in texts
            ]
    
    def _classify_batch_tinybert(self, texts: List[str], task: str) -> List[ClassificationResult]:
        """Classify batch of texts using TinyBERT"""
        try:
            if task not in self.tinybert_models:
                raise ValueError(f"TinyBERT model not available for task: {task}")
            
            model_info = self.tinybert_models[task]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            labels = model_info["labels"]
            
            # Tokenize inputs
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            # Process results
            results = []
            for i, text in enumerate(texts):
                predicted_id = predictions[i].item()
                predicted_label = labels[str(predicted_id)]
                confidence = probabilities[i][predicted_id].item()
                
                all_probabilities = {
                    labels[str(j)]: probabilities[i][j].item()
                    for j in range(len(labels))
                }
                
                results.append(ClassificationResult(
                    uuid=str(uuid.uuid4()),
                    text=text,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    all_probabilities=all_probabilities
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying batch with TinyBERT: {str(e)}", exc_info=True)
            raise
    
    def _classify_batch_sklearn(self, texts: List[str], task: str) -> List[ClassificationResult]:
        """Classify batch of texts using Scikit-learn"""
        try:
            if task not in self.sklearn_models:
                raise ValueError(f"Scikit-learn model not available for task: {task}")
            
            model = self.sklearn_models[task]
            vectorizer = self.vectorizers.get(task)
            scaler = self.scalers.get(task)
            encoder = self.label_encoders.get(task)
            
            # Vectorize texts
            if vectorizer:
                text_vectors = vectorizer.transform(texts)
            else:
                # Fallback to simple TF-IDF
                vectorizer = TfidfVectorizer(max_features=1000)
                text_vectors = vectorizer.fit_transform(texts)
            
            # Scale features if scaler is available
            if scaler:
                text_vectors = scaler.transform(text_vectors)
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(text_vectors)
                predictions = model.predict(text_vectors)
            else:
                predictions = model.predict(text_vectors)
                probabilities = None
            
            # Process results
            results = []
            for i, text in enumerate(texts):
                predicted_id = predictions[i]
                confidence = float(max(probabilities[i])) if probabilities is not None else 1.0
                
                if encoder:
                    predicted_label = encoder.inverse_transform([predicted_id])[0]
                else:
                    predicted_label = str(predicted_id)
                
                all_probabilities = {}
                if probabilities is not None and encoder:
                    for j, prob in enumerate(probabilities[i]):
                        label = encoder.inverse_transform([j])[0]
                        all_probabilities[label] = float(prob)
                
                results.append(ClassificationResult(
                    uuid=str(uuid.uuid4()),
                    text=text,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    all_probabilities=all_probabilities
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying batch with Scikit-learn: {str(e)}", exc_info=True)
            raise
    
    def train_classifier(self, texts: List[str], labels: List[str], task: str, 
                       model_type: str = "sklearn") -> bool:
        """Train a new classifier"""
        try:
            # Validate inputs
            if len(texts) != len(labels):
                raise ValueError("Number of texts and labels must match")
            
            if task not in self.model_configs:
                raise ValueError(f"Unknown classification task: {task}")
            
            # Prepare data
            if model_type == "tinybert":
                return self._train_tinybert_classifier(texts, labels, task)
            elif model_type == "sklearn":
                return self._train_sklearn_classifier(texts, labels, task)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}", exc_info=True)
            return False
    
    def _train_tinybert_classifier(self, texts: List[str], labels: List[str], task: str) -> bool:
        """Train a TinyBERT classifier"""
        try:
            # Create label mapping
            unique_labels = list(set(labels))
            label2id = {label: i for i, label in enumerate(unique_labels)}
            id2label = {i: label for i, label in enumerate(unique_labels)}
            
            # Create dataset
            dataset = Dataset.from_dict({
                "text": texts,
                "label": [label2id[label] for label in labels]
            })
            
            # Split dataset
            train_test = dataset.train_test_split(test_size=0.2)
            train_dataset = train_test["train"]
            eval_dataset = train_test["test"]
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Tokenize function
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=512
                )
            
            # Tokenize datasets
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(unique_labels),
                id2label=id2label,
                label2id=label2id
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.model_path / "tinybert" / task),
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=str(self.model_path / "logs"),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=self._compute_metrics
            )
            
            # Train model
            trainer.train()
            
            # Save model
            trainer.save_model(str(self.model_path / "tinybert" / task))
            
            # Update loaded models
            self.tinybert_models[task] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": model.config,
                "labels": id2label
            }
            
            logger.info(f"Successfully trained TinyBERT classifier for task: {task}")
            return True
            
        except Exception as e:
            logger.error(f"Error training TinyBERT classifier: {str(e)}", exc_info=True)
            return False
    
    def _train_sklearn_classifier(self, texts: List[str], labels: List[str], task: str) -> bool:
        """Train a Scikit-learn classifier"""
        try:
            # Create label encoder
            encoder = LabelEncoder()
            encoded_labels = encoder.fit_transform(labels)
            
            # Vectorize texts
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            text_vectors = vectorizer.fit_transform(texts)
            
            # Scale features
            scaler = StandardScaler(with_mean=False)
            scaled_vectors = scaler.fit_transform(text_vectors)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_vectors, encoded_labels, test_size=0.2, random_state=42
            )
            
            # Train multiple models and select best
            models = {
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
                "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
                "svm": SVC(probability=True, random_state=42)
            }
            
            best_model = None
            best_score = 0
            best_name = None
            
            for name, model in models.items():
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name
            
            # Train best model on full training set
            best_model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            test_score = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Best model: {best_name} with F1 score: {test_score:.3f}")
            
            # Save model
            model_path = self.model_path / "scikit" / f"{task}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_path)
            
            # Save vectorizer and scaler
            vectorizer_path = self.model_path / "scikit" / f"{task}_vectorizer.pkl"
            joblib.dump(vectorizer, vectorizer_path)
            
            scaler_path = self.model_path / "scikit" / f"{task}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            
            # Save encoder
            encoder_path = self.model_path / "scikit" / f"{task}_encoder.pkl"
            joblib.dump(encoder, encoder_path)
            
            # Update loaded models
            self.sklearn_models[task] = best_model
            self.vectorizers[task] = vectorizer
            self.scalers[task] = scaler
            self.label_encoders[task] = encoder
            
            logger.info(f"Successfully trained Scikit-learn classifier for task: {task}")
            return True
            
        except Exception as e:
            logger.error(f"Error training Scikit-learn classifier: {str(e)}", exc_info=True)
            return False
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for TinyBERT training"""
        labels = eval_pred.label_ids
        preds = eval_pred.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate_model(self, task: str, test_texts: List[str], test_labels: List[str]) -> ModelMetrics:
        """Evaluate model performance"""
        try:
            # Get predictions
            results = self.classify_batch(test_texts, task)
            
            # Extract predictions and true labels
            predictions = [result.predicted_label for result in results]
            true_labels = test_labels
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')
            
            # Calculate confusion matrix
            label_set = sorted(set(true_labels + predictions))
            cm = confusion_matrix(true_labels, predictions, labels=label_set)
            
            # Generate classification report
            report = classification_report(true_labels, predictions, target_names=label_set)
            
            return ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=cm.tolist(),
                classification_report=report
            )
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            return ModelMetrics()
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available classification tasks"""
        return list(self.model_configs.keys())
    
    def get_model_info(self, task: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            if task not in self.model_configs:
                return {}
            
            config = self.model_configs[task]
            model_type = config["model_type"]
            
            info = {
                "task": task,
                "model_type": model_type,
                "labels": config["labels"]
            }
            
            if model_type == "tinybert" and task in self.tinybert_models:
                model_info = self.tinybert_models[task]
                info.update({
                    "model_architecture": str(model_info["config"].architectures[0]),
                    "num_parameters": sum(p.numel() for p in model_info["model"].parameters()),
                    "max_sequence_length": model_info["config"].max_position_embeddings
                })
            elif model_type == "sklearn" and task in self.sklearn_models:
                model = self.sklearn_models[task]
                info.update({
                    "model_class": str(model.__class__.__name__),
                    "feature_importance": hasattr(model, 'feature_importances_')
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}", exc_info=True)
            return {}