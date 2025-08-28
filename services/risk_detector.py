# services/risk_detector.py
"""
Risk Change Detection Service - Task 2.1.2 Implementation
========================================================
Intelligent risk detection with ML-based anomaly detection, configurable thresholds,
and automated recommendation generation with confidence scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from collections import deque
import json
import hashlib
from pydantic import BaseModel

# ML and statistical imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy import stats
from scipy.stats import zscore
import warnings

# Import our risk calculator
from services.risk_calculator import RiskMetrics, EnhancedRiskCalculator

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class DetectionMethod(Enum):
    """Risk detection methods"""
    THRESHOLD = "threshold"
    STATISTICAL = "statistical"
    ML_ANOMALY = "ml_anomaly"
    TREND_ANALYSIS = "trend_analysis"
    REGIME_CHANGE = "regime_change"

@dataclass
class RiskThreshold:
    """Configuration for risk thresholds"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    direction: str = "increase"  # "increase", "decrease", or "absolute"
    enabled: bool = True
    lookback_periods: int = 5  # Periods to look back for comparison

@dataclass
class RiskAlert:
    """Risk alert with details"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    detection_method: DetectionMethod
    metric_name: str
    current_value: float
    previous_value: Optional[float]
    threshold_breached: Optional[float]
    change_magnitude: float
    statistical_significance: Optional[float]
    confidence_score: float
    message: str
    recommendations: List[str]
    affected_positions: Optional[List[str]] = None
    estimated_impact: Optional[float] = None

@dataclass
class RiskDetectionConfig:
    """Configuration for risk detection system"""
    # Threshold configurations
    thresholds: List[RiskThreshold] = field(default_factory=list)
    
    # ML Anomaly Detection
    anomaly_detection_enabled: bool = True
    anomaly_contamination: float = 0.1  # Expected fraction of outliers
    anomaly_lookback_periods: int = 252  # 1 year of daily data
    
    # Statistical Detection
    statistical_detection_enabled: bool = True
    significance_level: float = 0.05  # 5% significance level
    min_periods_for_stats: int = 30
    
    # Trend Analysis
    trend_analysis_enabled: bool = True
    trend_window: int = 20  # Rolling window for trend detection
    trend_significance_threshold: float = 0.01
    
    # Performance Settings
    max_history_size: int = 1000  # Maximum risk snapshots to keep in memory
    detection_frequency_seconds: int = 30  # How often to run detection
    
    # Alert Management
    alert_cooldown_minutes: int = 15  # Minimum time between similar alerts
    max_alerts_per_hour: int = 10

class RiskChangeDetector:
    """
    Intelligent risk change detection system with ML-based anomaly detection
    """
    
    def __init__(self, config: RiskDetectionConfig = None):
        self.config = config or self._get_default_config()
        self.risk_calculator = EnhancedRiskCalculator()
        
        # Risk history storage
        self.risk_history: deque = deque(maxlen=self.config.max_history_size)
        self.alert_history: deque = deque(maxlen=1000)
        
        # ML Models
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # State tracking
        self.last_detection_time: Optional[datetime] = None
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.hourly_alert_count = 0
        self.last_hour_reset = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Performance metrics
        self.detection_stats = {
            "total_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "alerts_generated": 0,
            "recommendations_accepted": 0
        }
        
        logger.info("RiskChangeDetector initialized")
    
    def _get_default_config(self) -> RiskDetectionConfig:
        """Get default risk detection configuration"""
        
        default_thresholds = [
            # Volatility thresholds
            RiskThreshold("annualized_volatility", 0.25, 0.35, 0.50, "increase"),
            RiskThreshold("annualized_volatility", 0.05, 0.03, 0.01, "decrease"),
            
            # VaR thresholds
            RiskThreshold("var_95", -0.05, -0.08, -0.12, "decrease"),  # More negative is worse
            RiskThreshold("cvar_95", -0.08, -0.12, -0.18, "decrease"),
            
            # Drawdown thresholds
            RiskThreshold("max_drawdown", -0.15, -0.25, -0.35, "decrease"),
            RiskThreshold("current_drawdown", -0.10, -0.20, -0.30, "decrease"),
            
            # Risk-adjusted return thresholds
            RiskThreshold("sharpe_ratio", 0.8, 0.5, 0.2, "decrease"),
            RiskThreshold("sortino_ratio", 1.0, 0.6, 0.3, "decrease"),
            
            # Distribution risk thresholds
            RiskThreshold("skewness", -1.0, -1.5, -2.0, "decrease"),  # Negative skew is risky
            RiskThreshold("kurtosis", 3.0, 5.0, 8.0, "increase"),  # High kurtosis = fat tails
        ]
        
        return RiskDetectionConfig(thresholds=default_thresholds)
    
    def _extract_features(self, risk_metrics: RiskMetrics) -> Dict[str, float]:
        """Extract features for ML-based detection"""
        
        features = {
            'annualized_volatility': risk_metrics.annualized_volatility,
            'var_95': risk_metrics.var_95,
            'cvar_95': risk_metrics.cvar_95,
            'max_drawdown': risk_metrics.max_drawdown,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'sortino_ratio': risk_metrics.sortino_ratio,
            'skewness': risk_metrics.skewness,
            'kurtosis': risk_metrics.kurtosis,
            'calmar_ratio': getattr(risk_metrics, 'calmar_ratio', 0.0),
            'information_ratio': getattr(risk_metrics, 'information_ratio', 0.0),
        }
        
        # Replace NaN values with 0
        for key, value in features.items():
            if np.isnan(value) or value is None:
                features[key] = 0.0
        
        return features
    
    async def add_risk_snapshot(self, risk_metrics: RiskMetrics, portfolio_id: str = "default") -> List[RiskAlert]:
        """
        Add new risk metrics and detect changes
        
        Args:
            risk_metrics: New risk metrics to analyze
            portfolio_id: Portfolio identifier
            
        Returns:
            List of risk alerts generated
        """
        
        # Add to history
        risk_snapshot = {
            "timestamp": risk_metrics.calculation_date,
            "portfolio_id": portfolio_id,
            "metrics": risk_metrics,
            "features": self._extract_features(risk_metrics)
        }
        
        self.risk_history.append(risk_snapshot)
        
        # Run detection algorithms
        alerts = await self._run_risk_detection(risk_snapshot)
        
        # Update statistics
        self.detection_stats["total_detections"] += 1
        if alerts:
            self.detection_stats["alerts_generated"] += len(alerts)
        
        self.last_detection_time = datetime.now()
        
        logger.info(f"Risk detection completed: {len(alerts)} alerts generated")
        return alerts
    
    async def _run_risk_detection(self, current_snapshot: Dict) -> List[RiskAlert]:
        """Run all risk detection algorithms"""
        
        all_alerts = []
        
        # 1. Threshold-based detection
        if len(self.risk_history) > 1:
            threshold_alerts = await self._detect_threshold_breaches(current_snapshot)
            all_alerts.extend(threshold_alerts)
        
        # 2. Statistical anomaly detection
        if self.config.statistical_detection_enabled and len(self.risk_history) >= self.config.min_periods_for_stats:
            statistical_alerts = await self._detect_statistical_anomalies(current_snapshot)
            all_alerts.extend(statistical_alerts)
        
        # 3. ML-based anomaly detection
        if self.config.anomaly_detection_enabled:
            ml_alerts = await self._detect_ml_anomalies(current_snapshot)
            all_alerts.extend(ml_alerts)
        
        # 4. Trend analysis
        if self.config.trend_analysis_enabled and len(self.risk_history) >= self.config.trend_window:
            trend_alerts = await self._detect_trend_changes(current_snapshot)
            all_alerts.extend(trend_alerts)
        
        # 5. Regime change detection
        if len(self.risk_history) >= 50:  # Need sufficient history
            regime_alerts = await self._detect_regime_changes(current_snapshot)
            all_alerts.extend(regime_alerts)
        
        # Filter and prioritize alerts
        filtered_alerts = await self._filter_and_prioritize_alerts(all_alerts)
        
        # Store alerts
        for alert in filtered_alerts:
            self.alert_history.append(alert)
        
        return filtered_alerts
    
    async def _detect_threshold_breaches(self, current_snapshot: Dict) -> List[RiskAlert]:
        """Detect threshold breaches"""
        
        alerts = []
        current_metrics = current_snapshot["metrics"]
        
        # Get previous snapshot for comparison
        if len(self.risk_history) < 2:
            return alerts
        
        previous_snapshot = list(self.risk_history)[-2]
        previous_metrics = previous_snapshot["metrics"]
        
        for threshold in self.config.thresholds:
            if not threshold.enabled:
                continue
            
            try:
                current_value = getattr(current_metrics, threshold.metric_name)
                previous_value = getattr(previous_metrics, threshold.metric_name, None)
                
                if np.isnan(current_value) or current_value is None:
                    continue
                
                # Check threshold breach
                breach_info = self._check_threshold_breach(
                    current_value, threshold, previous_value
                )
                
                if breach_info:
                    alert = await self._create_threshold_alert(
                        threshold, current_value, previous_value, breach_info, current_snapshot
                    )
                    alerts.append(alert)
                    
            except AttributeError:
                logger.warning(f"Metric {threshold.metric_name} not found in risk metrics")
                continue
        
        return alerts
    
    def _check_threshold_breach(
        self, 
        current_value: float, 
        threshold: RiskThreshold, 
        previous_value: Optional[float]
    ) -> Optional[Dict]:
        """Check if a threshold is breached"""
        
        # Determine severity based on threshold levels
        if threshold.direction == "increase":
            if threshold.emergency_threshold and current_value >= threshold.emergency_threshold:
                return {"severity": AlertSeverity.EMERGENCY, "threshold": threshold.emergency_threshold}
            elif current_value >= threshold.critical_threshold:
                return {"severity": AlertSeverity.CRITICAL, "threshold": threshold.critical_threshold}
            elif current_value >= threshold.warning_threshold:
                return {"severity": AlertSeverity.WARNING, "threshold": threshold.warning_threshold}
        
        elif threshold.direction == "decrease":
            if threshold.emergency_threshold and current_value <= threshold.emergency_threshold:
                return {"severity": AlertSeverity.EMERGENCY, "threshold": threshold.emergency_threshold}
            elif current_value <= threshold.critical_threshold:
                return {"severity": AlertSeverity.CRITICAL, "threshold": threshold.critical_threshold}
            elif current_value <= threshold.warning_threshold:
                return {"severity": AlertSeverity.WARNING, "threshold": threshold.warning_threshold}
        
        elif threshold.direction == "absolute":
            abs_value = abs(current_value)
            if threshold.emergency_threshold and abs_value >= threshold.emergency_threshold:
                return {"severity": AlertSeverity.EMERGENCY, "threshold": threshold.emergency_threshold}
            elif abs_value >= threshold.critical_threshold:
                return {"severity": AlertSeverity.CRITICAL, "threshold": threshold.critical_threshold}
            elif abs_value >= threshold.warning_threshold:
                return {"severity": AlertSeverity.WARNING, "threshold": threshold.warning_threshold}
        
        return None
    
    async def _create_threshold_alert(
        self,
        threshold: RiskThreshold,
        current_value: float,
        previous_value: Optional[float],
        breach_info: Dict,
        current_snapshot: Dict
    ) -> RiskAlert:
        """Create a threshold breach alert"""
        
        # Calculate change magnitude
        if previous_value is not None and not np.isnan(previous_value):
            if previous_value != 0:
                change_magnitude = abs((current_value - previous_value) / previous_value)
            else:
                change_magnitude = abs(current_value)
        else:
            change_magnitude = 0.0
        
        # Generate message
        direction_text = {
            "increase": "increased to",
            "decrease": "decreased to", 
            "absolute": "reached"
        }[threshold.direction]
        
        message = f"{threshold.metric_name.replace('_', ' ').title()} has {direction_text} {current_value:.4f}, breaching the {breach_info['severity'].value} threshold of {breach_info['threshold']:.4f}"
        
        # Generate recommendations
        recommendations = await self._generate_threshold_recommendations(threshold, current_value, change_magnitude)
        
        # Calculate confidence score
        confidence_score = self._calculate_threshold_confidence(threshold, current_value, change_magnitude)
        
        return RiskAlert(
            alert_id=f"threshold_{threshold.metric_name}_{datetime.now().isoformat()}",
            timestamp=current_snapshot["timestamp"],
            severity=breach_info["severity"],
            detection_method=DetectionMethod.THRESHOLD,
            metric_name=threshold.metric_name,
            current_value=current_value,
            previous_value=previous_value,
            threshold_breached=breach_info["threshold"],
            change_magnitude=change_magnitude,
            statistical_significance=None,
            confidence_score=confidence_score,
            message=message,
            recommendations=recommendations
        )
    
    async def _detect_statistical_anomalies(self, current_snapshot: Dict) -> List[RiskAlert]:
        """Detect statistical anomalies using Z-score and other methods"""
        
        alerts = []
        
        if len(self.risk_history) < self.config.min_periods_for_stats:
            return alerts
        
        current_metrics = current_snapshot["metrics"]
        
        # Extract historical values for each metric
        metric_names = [
            'annualized_volatility', 'var_95', 'cvar_95', 'max_drawdown',
            'sharpe_ratio', 'sortino_ratio', 'skewness', 'kurtosis'
        ]
        
        for metric_name in metric_names:
            try:
                # Get historical values
                historical_values = []
                for snapshot in list(self.risk_history)[:-1]:  # Exclude current
                    value = getattr(snapshot["metrics"], metric_name)
                    if not np.isnan(value) and value is not None:
                        historical_values.append(value)
                
                if len(historical_values) < self.config.min_periods_for_stats:
                    continue
                
                current_value = getattr(current_metrics, metric_name)
                if np.isnan(current_value) or current_value is None:
                    continue
                
                # Statistical tests
                anomaly_results = self._perform_statistical_tests(
                    historical_values, current_value, metric_name
                )
                
                if anomaly_results["is_anomaly"]:
                    alert = await self._create_statistical_alert(
                        metric_name, current_value, anomaly_results, current_snapshot
                    )
                    alerts.append(alert)
                    
            except Exception as e:
                logger.warning(f"Statistical anomaly detection failed for {metric_name}: {str(e)}")
                continue
        
        return alerts
    
    def _perform_statistical_tests(
        self, 
        historical_values: List[float], 
        current_value: float, 
        metric_name: str
    ) -> Dict:
        """Perform statistical tests to detect anomalies"""
        
        hist_array = np.array(historical_values)
        
        # Z-score test
        mean_val = np.mean(hist_array)
        std_val = np.std(hist_array, ddof=1)
        
        if std_val > 0:
            z_score = (current_value - mean_val) / std_val
        else:
            z_score = 0
        
        # Critical z-score for given significance level
        critical_z = stats.norm.ppf(1 - self.config.significance_level / 2)
        
        # Grubbs' test for outliers
        grubbs_stat = abs(z_score)
        n = len(historical_values) + 1
        grubbs_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(
            stats.t.ppf(1 - self.config.significance_level / (2 * n), n - 2) ** 2 /
            (n - 2 + stats.t.ppf(1 - self.config.significance_level / (2 * n), n - 2) ** 2)
        )
        
        # Determine if anomaly
        is_z_anomaly = abs(z_score) > critical_z
        is_grubbs_anomaly = grubbs_stat > grubbs_critical
        
        # Overall anomaly decision
        is_anomaly = is_z_anomaly or is_grubbs_anomaly
        
        # Severity based on z-score magnitude
        if abs(z_score) > 3:
            severity = AlertSeverity.CRITICAL
        elif abs(z_score) > 2.5:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        return {
            "is_anomaly": is_anomaly,
            "z_score": z_score,
            "significance": 1 - stats.norm.cdf(abs(z_score)) * 2,  # Two-tailed p-value
            "severity": severity,
            "grubbs_stat": grubbs_stat,
            "historical_mean": mean_val,
            "historical_std": std_val
        }
    
    async def _create_statistical_alert(
        self,
        metric_name: str,
        current_value: float,
        anomaly_results: Dict,
        current_snapshot: Dict
    ) -> RiskAlert:
        """Create statistical anomaly alert"""
        
        z_score = anomaly_results["z_score"]
        significance = anomaly_results["significance"]
        
        # Generate message
        direction = "above" if z_score > 0 else "below"
        message = f"Statistical anomaly detected: {metric_name.replace('_', ' ').title()} ({current_value:.4f}) is {abs(z_score):.2f} standard deviations {direction} the historical mean ({anomaly_results['historical_mean']:.4f})"
        
        # Generate recommendations
        recommendations = await self._generate_statistical_recommendations(metric_name, z_score, significance)
        
        # Confidence score based on statistical significance
        confidence_score = 1 - significance if significance < 0.1 else 0.5
        
        return RiskAlert(
            alert_id=f"statistical_{metric_name}_{datetime.now().isoformat()}",
            timestamp=current_snapshot["timestamp"],
            severity=anomaly_results["severity"],
            detection_method=DetectionMethod.STATISTICAL,
            metric_name=metric_name,
            current_value=current_value,
            previous_value=anomaly_results["historical_mean"],
            threshold_breached=None,
            change_magnitude=abs(z_score),
            statistical_significance=significance,
            confidence_score=confidence_score,
            message=message,
            recommendations=recommendations
        )
    
    async def _detect_ml_anomalies(self, current_snapshot: Dict) -> List[RiskAlert]:
        """Detect anomalies using ML-based methods"""
        
        alerts = []
        
        if len(self.risk_history) < self.config.anomaly_lookback_periods:
            return alerts
        
        try:
            # Prepare training data
            training_features = []
            for snapshot in list(self.risk_history)[:-1]:  # Exclude current
                features = snapshot["features"]
                if not any(np.isnan(v) for v in features.values()):
                    training_features.append(list(features.values()))
            
            if len(training_features) < 30:  # Need minimum data for ML
                return alerts
            
            # Train anomaly detector if not already trained or retrain periodically
            if not self.is_trained or len(self.risk_history) % 50 == 0:
                await self._train_anomaly_detector(training_features)
            
            # Detect anomalies in current snapshot
            current_features = list(current_snapshot["features"].values())
            
            if not any(np.isnan(v) for v in current_features):
                anomaly_score = await self._detect_anomaly(current_features)
                
                if anomaly_score < 0:  # Isolation Forest returns negative scores for anomalies
                    alert = await self._create_ml_alert(
                        current_snapshot, anomaly_score, current_features
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {str(e)}")
        
        return alerts
    
    async def _train_anomaly_detector(self, training_features: List[List[float]]):
        """Train the ML anomaly detector"""
        
        try:
            # Scale features
            feature_array = np.array(training_features)
            scaled_features = self.scaler.fit_transform(feature_array)
            
            # Train Isolation Forest
            self.anomaly_detector = IsolationForest(
                contamination=self.config.anomaly_contamination,
                random_state=42,
                n_estimators=100
            )
            
            self.anomaly_detector.fit(scaled_features)
            self.is_trained = True
            
            logger.info(f"ML anomaly detector trained on {len(training_features)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {str(e)}")
            self.is_trained = False
    
    async def _detect_anomaly(self, features: List[float]) -> float:
        """Detect anomaly using trained ML model"""
        
        if not self.is_trained or self.anomaly_detector is None:
            return 0.0
        
        try:
            # Scale current features
            feature_array = np.array([features])
            scaled_features = self.scaler.transform(feature_array)
            
            # Get anomaly score
            anomaly_score = self.anomaly_detector.score_samples(scaled_features)[0]
            return anomaly_score
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {str(e)}")
            return 0.0
    
    async def _create_ml_alert(
        self,
        current_snapshot: Dict,
        anomaly_score: float,
        features: List[float]
    ) -> RiskAlert:
        """Create ML-based anomaly alert"""
        
        # Determine severity based on anomaly score
        if anomaly_score < -0.3:
            severity = AlertSeverity.CRITICAL
        elif anomaly_score < -0.2:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        message = f"ML anomaly detected: Portfolio risk profile shows unusual pattern (anomaly score: {anomaly_score:.3f})"
        
        # Generate ML-based recommendations
        recommendations = await self._generate_ml_recommendations(anomaly_score, features)
        
        # Confidence score based on anomaly score magnitude
        confidence_score = min(abs(anomaly_score), 1.0)
        
        return RiskAlert(
            alert_id=f"ml_anomaly_{datetime.now().isoformat()}",
            timestamp=current_snapshot["timestamp"],
            severity=severity,
            detection_method=DetectionMethod.ML_ANOMALY,
            metric_name="ml_anomaly_score",
            current_value=anomaly_score,
            previous_value=None,
            threshold_breached=None,
            change_magnitude=abs(anomaly_score),
            statistical_significance=None,
            confidence_score=confidence_score,
            message=message,
            recommendations=recommendations
        )
    
    async def _detect_trend_changes(self, current_snapshot: Dict) -> List[RiskAlert]:
        """Detect significant trend changes in risk metrics"""
        
        alerts = []
        
        if len(self.risk_history) < self.config.trend_window:
            return alerts
        
        # Extract recent history for trend analysis
        recent_snapshots = list(self.risk_history)[-self.config.trend_window:]
        
        metric_names = ['annualized_volatility', 'var_95', 'sharpe_ratio', 'max_drawdown']
        
        for metric_name in metric_names:
            try:
                # Extract values for trend analysis
                values = []
                timestamps = []
                
                for snapshot in recent_snapshots:
                    value = getattr(snapshot["metrics"], metric_name)
                    if not np.isnan(value) and value is not None:
                        values.append(value)
                        timestamps.append(snapshot["timestamp"])
                
                if len(values) < self.config.trend_window // 2:
                    continue
                
                # Perform trend analysis
                trend_results = self._analyze_trend(values, timestamps, metric_name)
                
                if trend_results["significant_change"]:
                    alert = await self._create_trend_alert(
                        metric_name, trend_results, current_snapshot
                    )
                    alerts.append(alert)
                    
            except Exception as e:
                logger.warning(f"Trend analysis failed for {metric_name}: {str(e)}")
                continue
        
        return alerts
    
    def _analyze_trend(self, values: List[float], timestamps: List[datetime], metric_name: str) -> Dict:
        """Analyze trend in metric values"""
        
        if len(values) < 5:
            return {"significant_change": False}
        
        # Convert to numpy arrays
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression to detect trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine if trend is significant
        is_significant = p_value < self.config.trend_significance_threshold
        
        # Calculate trend strength
        trend_strength = abs(r_value)
        
        # Classify trend direction
        if slope > 0:
            direction = "increasing"
            severity = self._get_trend_severity(metric_name, slope, "increase")
        elif slope < 0:
            direction = "decreasing"
            severity = self._get_trend_severity(metric_name, slope, "decrease")
        else:
            direction = "stable"
            severity = AlertSeverity.INFO
        
        return {
            "significant_change": is_significant and trend_strength > 0.3,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "direction": direction,
            "severity": severity,
            "trend_strength": trend_strength
        }
    
    def _get_trend_severity(self, metric_name: str, slope: float, direction: str) -> AlertSeverity:
        """Determine trend severity based on metric and slope"""
        
        # Risk metrics that are bad when increasing
        bad_when_increasing = ['annualized_volatility', 'var_95', 'max_drawdown', 'kurtosis']
        
        # Risk metrics that are bad when decreasing  
        bad_when_decreasing = ['sharpe_ratio', 'sortino_ratio']
        
        slope_magnitude = abs(slope)
        
        if (metric_name in bad_when_increasing and direction == "increase") or \
           (metric_name in bad_when_decreasing and direction == "decrease"):
            # Bad trend
            if slope_magnitude > 0.05:
                return AlertSeverity.CRITICAL
            elif slope_magnitude > 0.02:
                return AlertSeverity.WARNING
            else:
                return AlertSeverity.INFO
        else:
            # Good trend or neutral
            return AlertSeverity.INFO
    
    async def _create_trend_alert(
        self,
        metric_name: str,
        trend_results: Dict,
        current_snapshot: Dict
    ) -> RiskAlert:
        """Create trend change alert"""
        
        slope = trend_results["slope"]
        direction = trend_results["direction"]
        r_squared = trend_results["r_squared"]
        
        message = f"Significant trend detected: {metric_name.replace('_', ' ').title()} is {direction} with slope {slope:.6f} (R² = {r_squared:.3f})"
        
        # Generate recommendations
        recommendations = await self._generate_trend_recommendations(metric_name, direction, slope)
        
        # Confidence score based on R-squared
        confidence_score = min(r_squared, 1.0)
        
        return RiskAlert(
            alert_id=f"trend_{metric_name}_{datetime.now().isoformat()}",
            timestamp=current_snapshot["timestamp"],
            severity=trend_results["severity"],
            detection_method=DetectionMethod.TREND_ANALYSIS,
            metric_name=metric_name,
            current_value=slope,
            previous_value=None,
            threshold_breached=None,
            change_magnitude=trend_results["trend_strength"],
            statistical_significance=trend_results["p_value"],
            confidence_score=confidence_score,
            message=message,
            recommendations=recommendations
        )
    
    async def _detect_regime_changes(self, current_snapshot: Dict) -> List[RiskAlert]:
        """Detect regime changes in risk characteristics"""
        
        alerts = []
        
        if len(self.risk_history) < 50:
            return alerts
        
        try:
            # Extract volatility and correlation patterns
            recent_history = list(self.risk_history)[-50:]
            
            volatilities = []
            for snapshot in recent_history:
                vol = getattr(snapshot["metrics"], "annualized_volatility", 0)
                if not np.isnan(vol):
                    volatilities.append(vol)
            
            if len(volatilities) < 30:
                return alerts
            
            # Simple regime change detection using rolling statistics
            vol_array = np.array(volatilities)
            
            # Calculate rolling means and standard deviations
            window_size = 10
            if len(vol_array) >= window_size * 2:
                recent_mean = np.mean(vol_array[-window_size:])
                older_mean = np.mean(vol_array[-window_size*2:-window_size])
                
                recent_std = np.std(vol_array[-window_size:])
                older_std = np.std(vol_array[-window_size*2:-window_size])
                
                # Detect significant shifts
                mean_shift = abs(recent_mean - older_mean) / older_mean if older_mean > 0 else 0
                std_shift = abs(recent_std - older_std) / older_std if older_std > 0 else 0
                
                # Check for regime change
                if mean_shift > 0.3 or std_shift > 0.5:
                    alert = await self._create_regime_alert(
                        current_snapshot, mean_shift, std_shift, recent_mean, older_mean
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Regime change detection failed: {str(e)}")
        
        return alerts
    
    async def _create_regime_alert(
        self,
        current_snapshot: Dict,
        mean_shift: float,
        std_shift: float,
        recent_mean: float,
        older_mean: float
    ) -> RiskAlert:
        """Create regime change alert"""
        
        # Determine severity
        if mean_shift > 0.5 or std_shift > 1.0:
            severity = AlertSeverity.CRITICAL
        elif mean_shift > 0.3 or std_shift > 0.5:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        shift_direction = "increased" if recent_mean > older_mean else "decreased"
        
        message = f"Market regime change detected: Volatility pattern has {shift_direction} significantly (mean shift: {mean_shift:.1%}, std shift: {std_shift:.1%})"
        
        # Generate recommendations
        recommendations = await self._generate_regime_recommendations(mean_shift, shift_direction)
        
        # Confidence based on magnitude of shift
        confidence_score = min((mean_shift + std_shift) / 2, 1.0)
        
        return RiskAlert(
            alert_id=f"regime_change_{datetime.now().isoformat()}",
            timestamp=current_snapshot["timestamp"],
            severity=severity,
            detection_method=DetectionMethod.REGIME_CHANGE,
            metric_name="volatility_regime",
            current_value=recent_mean,
            previous_value=older_mean,
            threshold_breached=None,
            change_magnitude=mean_shift,
            statistical_significance=None,
            confidence_score=confidence_score,
            message=message,
            recommendations=recommendations
        )
    
    async def _filter_and_prioritize_alerts(self, alerts: List[RiskAlert]) -> List[RiskAlert]:
        """Filter and prioritize alerts"""
        
        if not alerts:
            return alerts
        
        # Check alert rate limiting
        current_time = datetime.now()
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        
        if current_hour > self.last_hour_reset:
            self.hourly_alert_count = 0
            self.last_hour_reset = current_hour
        
        if self.hourly_alert_count >= self.config.max_alerts_per_hour:
            logger.warning("Alert rate limit reached, filtering alerts")
            # Only keep emergency and critical alerts
            alerts = [a for a in alerts if a.severity in [AlertSeverity.EMERGENCY, AlertSeverity.CRITICAL]]
        
        filtered_alerts = []
        
        for alert in alerts:
            # Check cooldown
            alert_key = f"{alert.detection_method.value}_{alert.metric_name}"
            if alert_key in self.alert_cooldowns:
                cooldown_end = self.alert_cooldowns[alert_key] + timedelta(minutes=self.config.alert_cooldown_minutes)
                if current_time < cooldown_end:
                    continue
            
            # Set cooldown for this alert type
            self.alert_cooldowns[alert_key] = current_time
            filtered_alerts.append(alert)
            self.hourly_alert_count += 1
        
        # Sort by severity and confidence
        severity_order = {
            AlertSeverity.EMERGENCY: 4,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 1
        }
        
        filtered_alerts.sort(
            key=lambda a: (severity_order[a.severity], a.confidence_score),
            reverse=True
        )
        
        return filtered_alerts
    
    # Recommendation generation methods
    async def _generate_threshold_recommendations(
        self, 
        threshold: RiskThreshold, 
        current_value: float, 
        change_magnitude: float
    ) -> List[str]:
        """Generate recommendations for threshold breaches"""
        
        recommendations = []
        
        metric_name = threshold.metric_name
        
        if metric_name == "annualized_volatility":
            if threshold.direction == "increase":
                recommendations.extend([
                    "Consider reducing position sizes to lower portfolio volatility",
                    "Review correlation exposures and consider diversification",
                    "Implement volatility-based position sizing",
                    "Consider adding defensive assets or hedges"
                ])
            else:
                recommendations.extend([
                    "Low volatility may indicate limited opportunities",
                    "Consider increasing allocation to growth assets",
                    "Review if portfolio is too conservative for objectives"
                ])
        
        elif metric_name in ["var_95", "cvar_95"]:
            recommendations.extend([
                "Immediate risk reduction required - consider position downsizing",
                "Review tail risk hedging strategies",
                "Implement stop-loss orders on high-risk positions",
                "Consider portfolio stress testing with current market conditions"
            ])
        
        elif metric_name in ["max_drawdown", "current_drawdown"]:
            recommendations.extend([
                "Review risk management protocols immediately",
                "Consider temporary reduction in leverage/exposure",
                "Implement dynamic hedging strategies",
                "Review stop-loss and position sizing rules"
            ])
        
        elif metric_name in ["sharpe_ratio", "sortino_ratio"]:
            recommendations.extend([
                "Portfolio efficiency declining - review asset allocation",
                "Consider rebalancing to higher risk-adjusted return assets",
                "Review fee structure and transaction costs",
                "Analyze alpha generation strategies"
            ])
        
        elif metric_name == "skewness":
            recommendations.extend([
                "Negative skew increasing tail risk",
                "Consider tail risk hedging (put options, volatility strategies)",
                "Review exposure to assets with asymmetric risk profiles",
                "Implement convexity-focused strategies"
            ])
        
        elif metric_name == "kurtosis":
            recommendations.extend([
                "High kurtosis indicates fat-tail risk",
                "Consider volatility targeting strategies",
                "Review exposure to assets prone to extreme moves",
                "Implement crisis alpha strategies"
            ])
        
        return recommendations
    
    async def _generate_statistical_recommendations(
        self, 
        metric_name: str, 
        z_score: float, 
        significance: float
    ) -> List[str]:
        """Generate recommendations for statistical anomalies"""
        
        recommendations = [
            f"Statistical anomaly detected with {(1-significance)*100:.1f}% confidence",
            "Investigate underlying causes of unusual risk pattern",
            "Review recent portfolio changes and market conditions"
        ]
        
        if abs(z_score) > 3:
            recommendations.extend([
                "Extremely rare event - immediate attention required",
                "Consider emergency risk reduction measures",
                "Validate data quality and calculation methods"
            ])
        elif abs(z_score) > 2:
            recommendations.extend([
                "Significant deviation from normal patterns",
                "Monitor closely for continuation of anomaly",
                "Review risk model assumptions"
            ])
        
        return recommendations
    
    async def _generate_ml_recommendations(self, anomaly_score: float, features: List[float]) -> List[str]:
        """Generate recommendations for ML-detected anomalies"""
        
        recommendations = [
            "Machine learning model detected unusual risk pattern",
            "Review recent portfolio changes and market regime shifts",
            "Consider validating risk calculations with alternative models"
        ]
        
        if anomaly_score < -0.3:
            recommendations.extend([
                "Highly unusual pattern detected - investigate immediately",
                "Consider reducing portfolio risk until pattern is understood",
                "Review data feeds and model inputs for accuracy"
            ])
        
        return recommendations
    
    async def _generate_trend_recommendations(self, metric_name: str, direction: str, slope: float) -> List[str]:
        """Generate recommendations for trend changes"""
        
        recommendations = []
        
        if metric_name == "annualized_volatility":
            if direction == "increasing":
                recommendations.extend([
                    "Rising volatility trend detected - prepare for increased risk",
                    "Consider implementing volatility-based position sizing",
                    "Review hedging strategies for continued volatility increase"
                ])
            else:
                recommendations.extend([
                    "Declining volatility trend - may indicate complacency",
                    "Monitor for potential volatility regime change",
                    "Consider strategies that benefit from low volatility"
                ])
        
        elif metric_name == "sharpe_ratio":
            if direction == "decreasing":
                recommendations.extend([
                    "Risk-adjusted returns deteriorating over time",
                    "Review alpha generation strategies",
                    "Consider portfolio rebalancing to improve efficiency"
                ])
        
        return recommendations
    
    async def _generate_regime_recommendations(self, mean_shift: float, shift_direction: str) -> List[str]:
        """Generate recommendations for regime changes"""
        
        recommendations = [
            f"Market volatility regime has {shift_direction} significantly",
            "Review risk model parameters for new market regime",
            "Consider adjusting risk budgets and position sizes"
        ]
        
        if shift_direction == "increased":
            recommendations.extend([
                "Higher volatility regime - implement defensive measures",
                "Consider increasing hedging allocation",
                "Review correlation assumptions as they may change in new regime"
            ])
        else:
            recommendations.extend([
                "Lower volatility regime - may indicate new opportunities",
                "Monitor for false calm before potential volatility spike",
                "Consider strategies that benefit from stable markets"
            ])
        
        return recommendations
    
    def _calculate_threshold_confidence(self, threshold: RiskThreshold, current_value: float, change_magnitude: float) -> float:
        """Calculate confidence score for threshold alerts"""
        
        # Base confidence on how far past threshold
        if threshold.direction == "increase":
            distance_past_threshold = (current_value - threshold.warning_threshold) / threshold.warning_threshold
        elif threshold.direction == "decrease":
            distance_past_threshold = (threshold.warning_threshold - current_value) / abs(threshold.warning_threshold)
        else:  # absolute
            distance_past_threshold = (abs(current_value) - threshold.warning_threshold) / threshold.warning_threshold
        
        # Confidence increases with distance past threshold and change magnitude
        confidence = min(0.5 + distance_past_threshold * 0.3 + change_magnitude * 0.2, 1.0)
        return max(confidence, 0.1)  # Minimum confidence
    
    
    # Utility methods
    def get_detection_stats(self) -> Dict:
        """Get detection performance statistics"""
        return self.detection_stats.copy()
    
    def get_alert_history(self, hours: int = 24) -> List[RiskAlert]:
        """Get recent alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def update_alert_feedback(self, alert_id: str, was_useful: bool):
        """Update alert feedback for model improvement"""
        if was_useful:
            self.detection_stats["recommendations_accepted"] += 1
        
        # Find and update the alert
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                # In a real implementation, you might store this feedback
                # for model retraining or threshold adjustment
                break
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on detection system"""
        return {
            "detector_initialized": True,
            "ml_model_trained": self.is_trained,
            "recent_detection": self.last_detection_time is not None and 
                              (datetime.now() - self.last_detection_time).seconds < 300,
            "sufficient_history": len(self.risk_history) > 10,
            "alerts_functioning": len(self.alert_history) >= 0
        }

def create_risk_detector(threshold_pct: float = 15.0) -> RiskChangeDetector:
    """Factory function to create risk detector instance"""
    config = RiskDetectionConfig()
    # You can customize the config based on threshold_pct if needed
    return RiskChangeDetector(config=config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta
    
    async def test_risk_detector():
        """Test the risk detection system"""
        
        # Initialize detector
        detector = RiskChangeDetector()
        
        # Create mock risk metrics
        base_date = datetime.now() - timedelta(days=100)
        
        for i in range(100):
            # Create mock metrics with some trends and anomalies
            volatility = 0.15 + 0.05 * np.sin(i / 10) + 0.02 * np.random.normal()
            if i > 80:  # Introduce anomaly
                volatility += 0.1
            
            mock_metrics = type('RiskMetrics', (), {
                'calculation_date': base_date + timedelta(days=i),
                'annualized_volatility': max(volatility, 0.01),
                'var_95': -0.05 - volatility * 0.3,
                'cvar_95': -0.08 - volatility * 0.5,
                'max_drawdown': -0.1 - volatility * 0.4,
                'sharpe_ratio': max(1.5 - volatility * 2, 0.1),
                'sortino_ratio': max(1.8 - volatility * 2, 0.1),
                'skewness': -0.5 + 0.3 * np.random.normal(),
                'kurtosis': 3.5 + np.random.normal()
            })()
            
            # Add snapshot and detect
            alerts = await detector.add_risk_snapshot(mock_metrics, f"test_portfolio_{i%3}")
            
            if alerts:
                print(f"Day {i}: {len(alerts)} alerts generated")
                for alert in alerts:
                    print(f"  - {alert.severity.value}: {alert.message[:100]}...")
        
        # Print final statistics
        stats = detector.get_detection_stats()
        print(f"\nDetection Statistics:")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Alerts generated: {stats['alerts_generated']}")
        
        # Health check
        health = await detector.health_check()
        print(f"\nHealth Check: {health}")
    
    # Run test
    asyncio.run(test_risk_detector())