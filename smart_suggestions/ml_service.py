# smart_suggestions/ml_service.py
"""
Machine Learning Service for Smart Suggestions Enhancement
=========================================================
Integrates ML models with your existing suggestion engine to provide
predictive, personalized, and continuously improving suggestions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import joblib
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sqlalchemy.orm import Session
from db import crud, models
from smart_suggestions.suggestion_engine import SmartSuggestion, PortfolioContext, MarketContext

@dataclass
class MLPrediction:
    """ML prediction result"""
    suggestion_relevance: float
    user_engagement_probability: float
    confidence_score: float
    feature_importance: Dict[str, float]
    model_version: str

@dataclass
class UserBehaviorPattern:
    """User behavior analysis result"""
    preferred_categories: List[str]
    optimal_timing: Dict[str, float]  # hour -> engagement probability
    risk_tolerance_score: float
    interaction_frequency: float
    feature_vector: List[float]

class MLSuggestionEnhancer:
    """
    ML-powered enhancement layer for your existing suggestion engine
    Integrates with existing SmartSuggestionEngine to add predictive capabilities
    """
    
    def __init__(self, model_path: str = "ml_models/"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        # ML models
        self.relevance_model = None
        self.engagement_model = None
        self.personalization_model = None
        
        # Feature processing
        self.text_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model metadata
        self.model_version = "1.0"
        self.last_training_date = None
        self.feature_columns = []
        
        # Load existing models if available
        self._load_models()
    
    def enhance_suggestions(
        self, 
        suggestions: List[SmartSuggestion],
        user_context: Dict[str, Any],
        portfolio_context: PortfolioContext,
        market_context: MarketContext,
        user_history: List[Dict[str, Any]]
    ) -> List[SmartSuggestion]:
        """
        Enhance existing suggestions with ML predictions
        This integrates with your existing suggestion_engine.py
        """
        
        if not suggestions:
            return suggestions
        
        try:
            # Analyze user behavior patterns
            behavior_pattern = self._analyze_user_behavior(user_history, user_context)
            
            # Generate feature vectors for each suggestion
            enhanced_suggestions = []
            
            for suggestion in suggestions:
                # Create feature vector for this suggestion
                features = self._create_suggestion_features(
                    suggestion, user_context, portfolio_context, 
                    market_context, behavior_pattern
                )
                
                # Get ML predictions
                ml_prediction = self._predict_suggestion_performance(features)
                
                # Apply ML enhancements
                enhanced_suggestion = self._apply_ml_enhancements(
                    suggestion, ml_prediction, behavior_pattern
                )
                
                enhanced_suggestions.append(enhanced_suggestion)
            
            # Re-rank using ML insights
            enhanced_suggestions = self._ml_rerank_suggestions(
                enhanced_suggestions, behavior_pattern
            )
            
            return enhanced_suggestions
            
        except Exception as e:
            print(f"ML enhancement error: {e}")
            # Fallback to original suggestions if ML fails
            return suggestions
    
    def _analyze_user_behavior(
        self, 
        user_history: List[Dict[str, Any]], 
        user_context: Dict[str, Any]
    ) -> UserBehaviorPattern:
        """Analyze user behavior patterns from historical data"""
        
        if not user_history:
            # Default pattern for new users
            return UserBehaviorPattern(
                preferred_categories=["Risk Management", "Performance"],
                optimal_timing={str(hour): 0.1 for hour in range(24)},
                risk_tolerance_score=0.5,
                interaction_frequency=0.1,
                feature_vector=[0.5] * 10
            )
        
        # Analyze category preferences
        category_counts = {}
        execution_counts = {}
        hourly_interactions = {str(hour): 0 for hour in range(24)}
        
        for interaction in user_history:
            category = interaction.get('category', 'General')
            executed = interaction.get('executed', False)
            timestamp = interaction.get('timestamp')
            
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if executed:
                execution_counts[category] = execution_counts.get(category, 0) + 1
            
            if timestamp:
                hour = str(timestamp.hour) if hasattr(timestamp, 'hour') else '12'
                hourly_interactions[hour] += 1
        
        # Calculate preferred categories (top 3)
        preferred_categories = sorted(
            category_counts.keys(), 
            key=lambda x: execution_counts.get(x, 0) / max(category_counts.get(x, 1), 1),
            reverse=True
        )[:3]
        
        # Calculate optimal timing (normalize hourly interactions)
        total_interactions = sum(hourly_interactions.values())
        if total_interactions > 0:
            optimal_timing = {
                hour: count / total_interactions 
                for hour, count in hourly_interactions.items()
            }
        else:
            optimal_timing = {str(hour): 1/24 for hour in range(24)}
        
        # Estimate risk tolerance from interaction patterns
        risk_related_interactions = sum(
            1 for interaction in user_history 
            if 'risk' in interaction.get('category', '').lower()
        )
        risk_tolerance_score = min(risk_related_interactions / max(len(user_history), 1), 1.0)
        
        # Calculate interaction frequency
        if user_history and len(user_history) > 1:
            first_interaction = min(
                interaction.get('timestamp', datetime.now()) 
                for interaction in user_history 
                if interaction.get('timestamp')
            )
            days_active = max((datetime.now() - first_interaction).days, 1)
            interaction_frequency = len(user_history) / days_active
        else:
            interaction_frequency = 0.1
        
        # Create feature vector for ML models
        feature_vector = [
            len(preferred_categories),
            risk_tolerance_score,
            interaction_frequency,
            len(user_history),
            sum(1 for i in user_history if i.get('executed', False)) / max(len(user_history), 1),
            max(optimal_timing.values()) if optimal_timing else 0.1,
            total_interactions,
            len(category_counts),
            max(execution_counts.values()) if execution_counts else 0,
            np.std(list(hourly_interactions.values())) if hourly_interactions else 0
        ]
        
        return UserBehaviorPattern(
            preferred_categories=preferred_categories or ["General"],
            optimal_timing=optimal_timing,
            risk_tolerance_score=risk_tolerance_score,
            interaction_frequency=interaction_frequency,
            feature_vector=feature_vector
        )
    
    def _create_suggestion_features(
        self,
        suggestion: SmartSuggestion,
        user_context: Dict[str, Any],
        portfolio_context: PortfolioContext,
        market_context: MarketContext,
        behavior_pattern: UserBehaviorPattern
    ) -> np.ndarray:
        """Create feature vector for ML prediction"""
        
        features = []
        
        # Suggestion features
        features.extend([
            suggestion.confidence,
            len(suggestion.description),
            len(suggestion.query),
            1.0 if suggestion.urgency.value == "high" else 0.5 if suggestion.urgency.value == "medium" else 0.0,
            1.0 if suggestion.category in behavior_pattern.preferred_categories else 0.0
        ])
        
        # Portfolio context features
        if portfolio_context:
            features.extend([
                portfolio_context.risk_score / 100.0,  # Normalize
                portfolio_context.volatility,
                abs(portfolio_context.risk_change_pct) / 100.0,
                portfolio_context.concentration_risk,
                len(portfolio_context.holdings) / 50.0  # Normalize assuming max 50 holdings
            ])
        else:
            features.extend([0.5, 0.15, 0.0, 0.1, 0.2])
        
        # Market context features
        features.extend([
            market_context.vix_level / 50.0,  # Normalize VIX
            1.0 if market_context.market_trend == "bullish" else -1.0 if market_context.market_trend == "bearish" else 0.0,
            len(market_context.sector_rotation) / 10.0  # Normalize
        ])
        
        # User behavior features
        features.extend(behavior_pattern.feature_vector[:5])  # Use first 5 behavior features
        
        # Time-based features
        current_hour = datetime.now().hour
        features.extend([
            behavior_pattern.optimal_timing.get(str(current_hour), 0.1),
            1.0 if 9 <= current_hour <= 16 else 0.0,  # Market hours
            1.0 if current_hour >= 18 else 0.0  # Evening
        ])
        
        return np.array(features)
    
    def _predict_suggestion_performance(self, features: np.ndarray) -> MLPrediction:
        """Predict suggestion performance using ML models"""
        
        if self.relevance_model is None or self.engagement_model is None:
            # Use heuristic scoring if models not available
            return MLPrediction(
                suggestion_relevance=0.7,
                user_engagement_probability=0.5,
                confidence_score=0.6,
                feature_importance={},
                model_version=self.model_version
            )
        
        try:
            # Reshape features for prediction
            features_reshaped = features.reshape(1, -1)
            
            # Ensure features match training dimensions
            if len(features) != len(self.feature_columns):
                # Pad or truncate features as needed
                if len(features) < len(self.feature_columns):
                    padded_features = np.pad(features, (0, len(self.feature_columns) - len(features)))
                    features_reshaped = padded_features.reshape(1, -1)
                else:
                    features_reshaped = features[:len(self.feature_columns)].reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features_reshaped)
            
            # Predict relevance
            relevance_prob = self.relevance_model.predict_proba(features_scaled)[0]
            suggestion_relevance = max(relevance_prob) if len(relevance_prob) > 1 else relevance_prob[0]
            
            # Predict engagement
            engagement_probability = self.engagement_model.predict(features_scaled)[0]
            
            # Calculate confidence based on model certainty
            confidence_score = min(suggestion_relevance * engagement_probability, 1.0)
            
            # Get feature importance
            if hasattr(self.relevance_model, 'feature_importances_'):
                importance_dict = dict(zip(
                    self.feature_columns[:len(self.relevance_model.feature_importances_)],
                    self.relevance_model.feature_importances_
                ))
            else:
                importance_dict = {}
            
            return MLPrediction(
                suggestion_relevance=float(suggestion_relevance),
                user_engagement_probability=float(max(0, min(1, engagement_probability))),
                confidence_score=float(confidence_score),
                feature_importance=importance_dict,
                model_version=self.model_version
            )
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return MLPrediction(
                suggestion_relevance=0.7,
                user_engagement_probability=0.5,
                confidence_score=0.6,
                feature_importance={},
                model_version=self.model_version
            )
    
    def _apply_ml_enhancements(
        self, 
        suggestion: SmartSuggestion, 
        ml_prediction: MLPrediction,
        behavior_pattern: UserBehaviorPattern
    ) -> SmartSuggestion:
        """Apply ML predictions to enhance the suggestion"""
        
        # Update confidence based on ML prediction
        enhanced_confidence = (suggestion.confidence + ml_prediction.confidence_score) / 2
        
        # Adjust urgency based on engagement probability
        if ml_prediction.user_engagement_probability > 0.8:
            # High engagement probability - maintain or increase urgency
            pass
        elif ml_prediction.user_engagement_probability < 0.3:
            # Low engagement probability - reduce urgency
            from smart_suggestions.suggestion_engine import Urgency
            if suggestion.urgency == Urgency.HIGH:
                suggestion.urgency = Urgency.MEDIUM
            elif suggestion.urgency == Urgency.MEDIUM:
                suggestion.urgency = Urgency.LOW
        
        # Personalize description based on user preferences
        if suggestion.category in behavior_pattern.preferred_categories:
            suggestion.description = f"⭐ {suggestion.description}"
        
        # Add ML metadata
        if suggestion.metadata is None:
            suggestion.metadata = {}
        
        suggestion.metadata.update({
            "ml_relevance": ml_prediction.suggestion_relevance,
            "ml_engagement_prob": ml_prediction.user_engagement_probability,
            "ml_confidence": ml_prediction.confidence_score,
            "ml_model_version": ml_prediction.model_version,
            "personalized": suggestion.category in behavior_pattern.preferred_categories
        })
        
        # Update overall confidence
        suggestion.confidence = enhanced_confidence
        
        return suggestion
    
    def _ml_rerank_suggestions(
        self, 
        suggestions: List[SmartSuggestion],
        behavior_pattern: UserBehaviorPattern
    ) -> List[SmartSuggestion]:
        """Re-rank suggestions using ML insights"""
        
        def ml_score(suggestion: SmartSuggestion) -> float:
            metadata = suggestion.metadata or {}
            
            base_score = suggestion.confidence
            ml_relevance = metadata.get("ml_relevance", 0.5)
            ml_engagement = metadata.get("ml_engagement_prob", 0.5)
            personalization_boost = 0.1 if metadata.get("personalized", False) else 0.0
            
            # Weight different factors
            score = (
                base_score * 0.4 +
                ml_relevance * 0.3 +
                ml_engagement * 0.2 +
                personalization_boost
            )
            
            # Boost preferred categories
            if suggestion.category in behavior_pattern.preferred_categories:
                score += 0.05
            
            return score
        
        return sorted(suggestions, key=ml_score, reverse=True)
    
    def train_models(self, training_data: List[Dict[str, Any]]):
        """Train ML models from user feedback and interaction data"""
        
        if len(training_data) < 50:  # Need minimum data for training
            print("Insufficient training data. Need at least 50 samples.")
            return False
        
        try:
            df = pd.DataFrame(training_data)
            
            # Prepare features
            feature_columns = [
                'suggestion_confidence', 'description_length', 'query_length',
                'urgency_score', 'category_match', 'portfolio_risk', 'portfolio_volatility',
                'risk_change', 'concentration_risk', 'holdings_count',
                'vix_normalized', 'market_trend_score', 'sector_rotation_count',
                'user_risk_tolerance', 'interaction_frequency', 'optimal_timing_score'
            ]
            
            # Create feature matrix
            X = df[feature_columns].fillna(0)
            
            # Create target variables
            y_relevance = df['user_rating'].apply(lambda x: 1 if x >= 4 else 0)  # Binary relevance
            y_engagement = df['executed'].astype(int)  # Binary engagement
            
            # Split data
            X_train, X_test, y_rel_train, y_rel_test, y_eng_train, y_eng_test = train_test_split(
                X, y_relevance, y_engagement, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train relevance model
            self.relevance_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.relevance_model.fit(X_train_scaled, y_rel_train)
            
            # Train engagement model
            self.engagement_model = GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
            self.engagement_model.fit(X_train_scaled, y_eng_train)
            
            # Evaluate models
            rel_pred = self.relevance_model.predict(X_test_scaled)
            eng_pred = self.engagement_model.predict(X_test_scaled)
            
            rel_accuracy = accuracy_score(y_rel_test, rel_pred)
            eng_mse = np.mean((y_eng_test - eng_pred) ** 2)
            
            print(f"Model training completed:")
            print(f"  Relevance accuracy: {rel_accuracy:.3f}")
            print(f"  Engagement MSE: {eng_mse:.3f}")
            
            # Update metadata
            self.feature_columns = feature_columns
            self.last_training_date = datetime.now()
            self.model_version = f"1.{len(training_data)}"
            
            # Save models
            self._save_models()
            
            return True
            
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            joblib.dump(self.relevance_model, self.model_path / "relevance_model.pkl")
            joblib.dump(self.engagement_model, self.model_path / "engagement_model.pkl")
            joblib.dump(self.scaler, self.model_path / "scaler.pkl")
            
            metadata = {
                "model_version": self.model_version,
                "last_training_date": self.last_training_date.isoformat() if self.last_training_date else None,
                "feature_columns": self.feature_columns
            }
            
            with open(self.model_path / "model_metadata.json", 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load existing models from disk"""
        try:
            if (self.model_path / "relevance_model.pkl").exists():
                self.relevance_model = joblib.load(self.model_path / "relevance_model.pkl")
                self.engagement_model = joblib.load(self.model_path / "engagement_model.pkl")
                self.scaler = joblib.load(self.model_path / "scaler.pkl")
                
                with open(self.model_path / "model_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    self.model_version = metadata["model_version"]
                    self.feature_columns = metadata["feature_columns"]
                    if metadata["last_training_date"]:
                        self.last_training_date = datetime.fromisoformat(metadata["last_training_date"])
                
                print(f"Loaded ML models version {self.model_version}")
        except Exception as e:
            print(f"No existing models found or error loading: {e}")

# Global ML enhancer instance
_ml_enhancer = None

def get_ml_enhancer() -> MLSuggestionEnhancer:
    """Get the global ML enhancer instance"""
    global _ml_enhancer
    if _ml_enhancer is None:
        _ml_enhancer = MLSuggestionEnhancer()
    return _ml_enhancer

# Integration function for existing suggestion engine
def enhance_suggestions_with_ml(
    suggestions: List[SmartSuggestion],
    user_context: Dict[str, Any],
    portfolio_context: PortfolioContext,
    market_context: MarketContext,
    user_history: List[Dict[str, Any]]
) -> List[SmartSuggestion]:
    """
    Main integration function to enhance suggestions with ML
    This function should be called from your existing suggestion_engine.py
    """
    enhancer = get_ml_enhancer()
    return enhancer.enhance_suggestions(
        suggestions, user_context, portfolio_context, market_context, user_history
    )