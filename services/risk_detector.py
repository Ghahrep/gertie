"""
Risk Detector Service - Task 4.1.1.3
====================================

Integrates with existing proactive_monitor.py to add comprehensive risk change detection.
Compares current vs historical risk metrics and triggers workflows when thresholds are breached.

Author: Quant Platform Development
Created: August 19, 2025
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from sqlalchemy.orm import Session
from db.session import get_db
from db.models import PortfolioRiskSnapshot, RiskChangeEvent, ProactiveAlert, Portfolio
from services.risk_calculator import RiskCalculatorService, RiskMetrics
from services.proactive_monitor import ProactiveRiskMonitor, AlertType, AlertPriority

logger = logging.getLogger(__name__)

@dataclass
class RiskChangeAnalysis:
    """Results of risk change detection analysis"""
    portfolio_id: str
    current_metrics: RiskMetrics
    previous_metrics: Optional[RiskMetrics]
    risk_direction: str
    risk_magnitude_pct: float
    threshold_breached: bool
    significant_changes: Dict[str, float]
    all_changes: Dict[str, float]
    recommendation: str
    should_trigger_workflow: bool

class RiskDetectorService:
    """
    Enhanced risk detection service that integrates with existing proactive monitoring.
    """
    
    def __init__(self, threshold_pct: float = 15.0):
        self.threshold_pct = threshold_pct
        self.risk_calculator = RiskCalculatorService()
        
        # Risk thresholds for different metrics
        self.metric_thresholds = {
            'volatility': 15.0,      # 15% change in volatility
            'beta': 20.0,            # 20% change in beta
            'max_drawdown': 10.0,    # 10% change in max drawdown
            'var_99': 15.0,          # 15% change in VaR
            'cvar_99': 15.0,         # 15% change in CVaR
            'risk_score': 15.0,      # 15% change in composite risk score
            'sentiment_index': 20.0, # 20% change in sentiment
            'sharpe_ratio': 25.0     # 25% change in Sharpe ratio
        }
        
    async def detect_portfolio_risk_changes(
        self,
        portfolio_id: int,
        user_id: int,
        db: Session
    ) -> Optional[RiskChangeAnalysis]:
        """
        Detect significant risk changes for a portfolio.
        
        Parameters:
        -----------
        portfolio_id : int
            Portfolio ID to analyze
        user_id : int
            User ID for the portfolio
        db : Session
            Database session
            
        Returns:
        --------
        RiskChangeAnalysis or None
            Analysis results if successful
        """
        try:
            # 1. Get portfolio holdings and calculate current risk metrics
            current_metrics = await self._calculate_current_risk_metrics(
                portfolio_id, user_id, db
            )
            
            if not current_metrics:
                logger.warning(f"Could not calculate current risk metrics for portfolio {portfolio_id}")
                return None
            
            # 2. Get previous risk snapshot for comparison
            previous_snapshot = self._get_latest_risk_snapshot(portfolio_id, db)
            previous_metrics = self._snapshot_to_risk_metrics(previous_snapshot) if previous_snapshot else None
            
            # 3. Store current risk snapshot
            current_snapshot_id = await self._store_risk_snapshot(
                current_metrics, portfolio_id, user_id, db
            )
            
            # 4. Compare risks if we have previous data
            if previous_metrics:
                risk_analysis = self._analyze_risk_changes(
                    current_metrics, previous_metrics, portfolio_id
                )
                
                # 5. Store risk change event if significant
                if risk_analysis.threshold_breached:
                    await self._store_risk_change_event(
                        risk_analysis, current_snapshot_id, previous_snapshot.id, user_id, db
                    )
                
                return risk_analysis
            else:
                logger.info(f"No previous risk data for portfolio {portfolio_id}, storing baseline")
                return RiskChangeAnalysis(
                    portfolio_id=str(portfolio_id),
                    current_metrics=current_metrics,
                    previous_metrics=None,
                    risk_direction="BASELINE",
                    risk_magnitude_pct=0.0,
                    threshold_breached=False,
                    significant_changes={},
                    all_changes={},
                    recommendation="Baseline risk metrics established",
                    should_trigger_workflow=False
                )
                
        except Exception as e:
            logger.error(f"Error detecting risk changes for portfolio {portfolio_id}: {e}")
            return None
    
    async def _calculate_current_risk_metrics(
        self, 
        portfolio_id: int, 
        user_id: int, 
        db: Session
    ) -> Optional[RiskMetrics]:
        """Calculate current risk metrics for a portfolio"""
        try:
            # Get portfolio holdings from database
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            if not portfolio:
                logger.error(f"Portfolio {portfolio_id} not found")
                return None
            
            # Extract tickers and weights from holdings
            tickers = []
            weights = []
            total_value = sum(holding.shares * (holding.purchase_price or 100) for holding in portfolio.holdings)
            
            if total_value == 0:
                logger.warning(f"Portfolio {portfolio_id} has no holdings or zero value")
                return None
            
            for holding in portfolio.holdings:
                if holding.asset and holding.asset.ticker:
                    tickers.append(holding.asset.ticker)
                    holding_value = holding.shares * (holding.purchase_price or 100)
                    weights.append(holding_value / total_value)
            
            if not tickers:
                logger.warning(f"Portfolio {portfolio_id} has no valid tickers")
                return None
            
            # Calculate risk metrics using risk calculator
            risk_metrics = self.risk_calculator.calculate_portfolio_risk_from_tickers(
                tickers=tickers,
                weights=weights,
                lookback_days=252
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating current risk metrics: {e}")
            return None
    
    def _get_latest_risk_snapshot(self, portfolio_id: int, db: Session) -> Optional[PortfolioRiskSnapshot]:
        """Get the most recent risk snapshot for a portfolio"""
        try:
            return db.query(PortfolioRiskSnapshot)\
                    .filter(PortfolioRiskSnapshot.portfolio_id == portfolio_id)\
                    .order_by(PortfolioRiskSnapshot.snapshot_date.desc())\
                    .first()
        except Exception as e:
            logger.error(f"Error getting latest risk snapshot: {e}")
            return None
    
    def _snapshot_to_risk_metrics(self, snapshot: PortfolioRiskSnapshot) -> RiskMetrics:
        """Convert database snapshot to RiskMetrics object"""
        return RiskMetrics(
            volatility=snapshot.volatility,
            beta=snapshot.beta,
            max_drawdown=snapshot.max_drawdown,
            var_95=snapshot.var_95,
            var_99=snapshot.var_99,
            cvar_95=snapshot.cvar_95,
            cvar_99=snapshot.cvar_99,
            sharpe_ratio=snapshot.sharpe_ratio,
            sortino_ratio=snapshot.sortino_ratio,
            calmar_ratio=snapshot.calmar_ratio,
            hurst_exponent=snapshot.hurst_exponent or 0.5,
            dfa_alpha=snapshot.dfa_alpha or 0.5,
            risk_score=snapshot.risk_score,
            sentiment_index=snapshot.sentiment_index,
            regime_volatility=snapshot.regime_volatility or snapshot.volatility,
            timestamp=snapshot.snapshot_date
        )
    
    def _analyze_risk_changes(
        self,
        current: RiskMetrics,
        previous: RiskMetrics,
        portfolio_id: int
    ) -> RiskChangeAnalysis:
        """Analyze changes between current and previous risk metrics"""
        try:
            def calculate_pct_change(current_val: float, previous_val: float) -> float:
                if previous_val == 0:
                    return 0.0
                return ((current_val - previous_val) / abs(previous_val)) * 100
            
            # Calculate percentage changes for all metrics
            all_changes = {
                'volatility': calculate_pct_change(current.volatility, previous.volatility),
                'beta': calculate_pct_change(current.beta, previous.beta),
                'max_drawdown': calculate_pct_change(current.max_drawdown, previous.max_drawdown),
                'var_99': calculate_pct_change(current.var_99, previous.var_99),
                'cvar_99': calculate_pct_change(current.cvar_99, previous.cvar_99),
                'risk_score': calculate_pct_change(current.risk_score, previous.risk_score),
                'sentiment_index': calculate_pct_change(current.sentiment_index, previous.sentiment_index),
                'sharpe_ratio': calculate_pct_change(current.sharpe_ratio, previous.sharpe_ratio)
            }
            
            # Identify significant changes based on metric-specific thresholds
            significant_changes = {}
            for metric, change in all_changes.items():
                threshold = self.metric_thresholds.get(metric, self.threshold_pct)
                if abs(change) >= threshold:
                    significant_changes[metric] = change
            
            # Determine overall risk direction and magnitude
            risk_direction = "INCREASED" if current.risk_score > previous.risk_score else "DECREASED"
            risk_magnitude = abs(all_changes['risk_score'])
            
            # Check if any threshold was breached
            threshold_breached = len(significant_changes) > 0
            
            # Generate recommendation
            recommendation = self._generate_risk_recommendation(
                risk_direction, risk_magnitude, significant_changes
            )
            
            # Determine if workflow should be triggered
            should_trigger_workflow = (
                threshold_breached and 
                risk_magnitude >= self.threshold_pct and
                risk_direction == "INCREASED"
            )
            
            return RiskChangeAnalysis(
                portfolio_id=str(portfolio_id),
                current_metrics=current,
                previous_metrics=previous,
                risk_direction=risk_direction,
                risk_magnitude_pct=risk_magnitude,
                threshold_breached=threshold_breached,
                significant_changes=significant_changes,
                all_changes=all_changes,
                recommendation=recommendation,
                should_trigger_workflow=should_trigger_workflow
            )
            
        except Exception as e:
            logger.error(f"Error analyzing risk changes: {e}")
            return RiskChangeAnalysis(
                portfolio_id=str(portfolio_id),
                current_metrics=current,
                previous_metrics=previous,
                risk_direction="ERROR",
                risk_magnitude_pct=0.0,
                threshold_breached=False,
                significant_changes={},
                all_changes={},
                recommendation="Error analyzing risk changes",
                should_trigger_workflow=False
            )
    
    def _generate_risk_recommendation(
        self,
        risk_direction: str,
        risk_magnitude: float,
        significant_changes: Dict[str, float]
    ) -> str:
        """Generate human-readable risk recommendation"""
        if not significant_changes:
            return "Portfolio risk levels remain stable within normal parameters."
        
        if risk_direction == "INCREASED":
            if risk_magnitude > 25:
                severity = "significantly"
            elif risk_magnitude > 15:
                severity = "moderately"
            else:
                severity = "slightly"
            
            key_changes = []
            if 'volatility' in significant_changes:
                key_changes.append(f"volatility {significant_changes['volatility']:+.1f}%")
            if 'max_drawdown' in significant_changes:
                key_changes.append(f"max drawdown {significant_changes['max_drawdown']:+.1f}%")
            if 'var_99' in significant_changes:
                key_changes.append(f"VaR {significant_changes['var_99']:+.1f}%")
            
            changes_text = ", ".join(key_changes[:2])  # Limit to most important
            
            return f"Portfolio risk has {severity} increased ({risk_magnitude:.1f}%). Key changes: {changes_text}. Consider reviewing position sizes and diversification."
        
        else:  # DECREASED
            return f"Portfolio risk has decreased by {risk_magnitude:.1f}%. Risk profile has improved across key metrics."
    
    async def _store_risk_snapshot(
        self,
        risk_metrics: RiskMetrics,
        portfolio_id: int,
        user_id: int,
        db: Session
    ) -> int:
        """Store risk metrics snapshot in database"""
        try:
            snapshot = PortfolioRiskSnapshot(
                portfolio_id=portfolio_id,
                user_id=user_id,
                volatility=risk_metrics.volatility,
                beta=risk_metrics.beta,
                max_drawdown=risk_metrics.max_drawdown,
                var_95=risk_metrics.var_95,
                var_99=risk_metrics.var_99,
                cvar_95=risk_metrics.cvar_95,
                cvar_99=risk_metrics.cvar_99,
                sharpe_ratio=risk_metrics.sharpe_ratio,
                sortino_ratio=risk_metrics.sortino_ratio,
                calmar_ratio=risk_metrics.calmar_ratio,
                hurst_exponent=risk_metrics.hurst_exponent,
                dfa_alpha=risk_metrics.dfa_alpha,
                risk_score=risk_metrics.risk_score,
                sentiment_index=risk_metrics.sentiment_index,
                regime_volatility=risk_metrics.regime_volatility,
                snapshot_date=risk_metrics.timestamp,
                calculation_method="comprehensive",
                data_window_days=252
            )
            
            db.add(snapshot)
            db.commit()
            db.refresh(snapshot)
            
            logger.info(f"Stored risk snapshot {snapshot.id} for portfolio {portfolio_id}")
            return snapshot.id
            
        except Exception as e:
            logger.error(f"Error storing risk snapshot: {e}")
            db.rollback()
            raise
    
    async def _store_risk_change_event(
        self,
        analysis: RiskChangeAnalysis,
        current_snapshot_id: int,
        previous_snapshot_id: int,
        user_id: int,
        db: Session
    ) -> int:
        """Store risk change event in database"""
        try:
            risk_event = RiskChangeEvent(
                portfolio_id=int(analysis.portfolio_id),
                user_id=user_id,
                current_snapshot_id=current_snapshot_id,
                previous_snapshot_id=previous_snapshot_id,
                risk_direction=analysis.risk_direction,
                risk_magnitude_pct=analysis.risk_magnitude_pct,
                threshold_breached=analysis.threshold_breached,
                risk_changes=analysis.all_changes,
                significant_changes=analysis.significant_changes,
                workflow_triggered=False,  # Will be updated when workflow is triggered
                threshold_pct=self.threshold_pct
            )
            
            db.add(risk_event)
            db.commit()
            db.refresh(risk_event)
            
            logger.info(f"Stored risk change event {risk_event.id} for portfolio {analysis.portfolio_id}")
            return risk_event.id
            
        except Exception as e:
            logger.error(f"Error storing risk change event: {e}")
            db.rollback()
            raise
    
    async def integrate_with_proactive_monitor(
        self,
        analysis: RiskChangeAnalysis,
        proactive_monitor: ProactiveRiskMonitor
    ) -> List[Dict]:
        """
        Convert risk analysis to alerts compatible with existing proactive monitor.
        """
        alerts = []
        
        if not analysis.threshold_breached:
            return alerts
        
        try:
            # Generate alerts based on significant changes
            for metric, change_pct in analysis.significant_changes.items():
                alert_type, priority, message = self._metric_change_to_alert(
                    metric, change_pct, analysis.risk_direction
                )
                
                if alert_type:
                    alerts.append({
                        'portfolio_id': analysis.portfolio_id,
                        'alert_type': alert_type,
                        'priority': priority,
                        'message': message,
                        'details': {
                            'metric': metric,
                            'change_pct': change_pct,
                            'risk_direction': analysis.risk_direction,
                            'current_risk_score': analysis.current_metrics.risk_score,
                            'recommendation': analysis.recommendation
                        }
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error integrating with proactive monitor: {e}")
            return []
    
    def _metric_change_to_alert(
        self, 
        metric: str, 
        change_pct: float, 
        direction: str
    ) -> Tuple[Optional[AlertType], AlertPriority, str]:
        """Convert metric change to appropriate alert type and message"""
        
        if direction != "INCREASED":
            return None, AlertPriority.LOW, ""
        
        # Map metrics to alert types
        metric_mapping = {
            'volatility': (AlertType.VOLATILITY_SPIKE, "Portfolio volatility spike detected"),
            'var_99': (AlertType.VAR_BREACH, "Value-at-Risk threshold breached"),
            'max_drawdown': (AlertType.VAR_BREACH, "Maximum drawdown increased significantly"),
            'risk_score': (AlertType.VOLATILITY_SPIKE, "Overall portfolio risk increased"),
            'sentiment_index': (AlertType.REGIME_CHANGE, "Risk sentiment deteriorated")
        }
        
        if metric not in metric_mapping:
            return None, AlertPriority.LOW, ""
        
        alert_type, base_message = metric_mapping[metric]
        
        # Determine priority based on magnitude
        if abs(change_pct) > 30:
            priority = AlertPriority.CRITICAL
        elif abs(change_pct) > 20:
            priority = AlertPriority.HIGH
        else:
            priority = AlertPriority.MEDIUM
        
        # Create detailed message
        message = f"{base_message}: {metric} changed by {change_pct:+.1f}%"
        
        return alert_type, priority, message

# Factory function
def create_risk_detector(threshold_pct: float = 15.0) -> RiskDetectorService:
    """Create a risk detector service instance"""
    return RiskDetectorService(threshold_pct=threshold_pct)

# Example usage
if __name__ == "__main__":
    # Test the risk detector
    detector = create_risk_detector(threshold_pct=15.0)
    print("Risk Detector Service created successfully!")