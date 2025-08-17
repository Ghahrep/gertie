from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging

from db import models
from core.data_handler import get_market_data_for_portfolio

logger = logging.getLogger(__name__)

class AlertService:
    """Service for monitoring and triggering portfolio alerts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_user_alerts(self, db: Session, user_id: int, portfolio_id: Optional[int] = None) -> List[models.Alert]:
        """Check all active alerts for a user and trigger if conditions are met"""
        try:
            # Get active alerts for user
            query = db.query(models.Alert).filter(
                models.Alert.user_id == user_id,
                models.Alert.is_active == True,
                models.Alert.status == models.AlertStatus.ACTIVE
            )
            
            if portfolio_id:
                query = query.filter(models.Alert.portfolio_id == portfolio_id)
            
            alerts = query.all()
            triggered_alerts = []
            
            for alert in alerts:
                if self._check_alert_condition(db, alert):
                    self._trigger_alert(db, alert)
                    triggered_alerts.append(alert)
                else:
                    # Update last checked time
                    alert.last_checked = datetime.utcnow()
            
            db.commit()
            return triggered_alerts
            
        except Exception as e:
            self.logger.error(f"Error checking alerts for user {user_id}: {e}")
            db.rollback()
            return []
    
    def _check_alert_condition(self, db: Session, alert: models.Alert) -> bool:
        """Check if alert condition is met"""
        try:
            if alert.alert_type == models.AlertType.PRICE_CHANGE and alert.asset_ticker:
                return self._check_price_alert(alert)
            
            elif alert.alert_type == models.AlertType.PORTFOLIO_VALUE and alert.portfolio_id:
                return self._check_portfolio_value_alert(db, alert)
            
            elif alert.alert_type == models.AlertType.RISK_THRESHOLD and alert.portfolio_id:
                return self._check_risk_alert(db, alert)
            
            # Add more alert type checks as needed
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking alert {alert.id}: {e}")
            return False
    
    def _check_price_alert(self, alert: models.Alert) -> bool:
        """Check price change alert for specific asset"""
        try:
            import yfinance as yf
            
            # Get current price
            ticker = yf.Ticker(alert.asset_ticker)
            hist = ticker.history(period="2d")
            
            if hist.empty:
                return False
            
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            if alert.threshold_direction == "above":
                return current_price > alert.threshold_value
            elif alert.threshold_direction == "below":
                return current_price < alert.threshold_value
            elif alert.threshold_direction == "change":
                price_change_pct = ((current_price - previous_price) / previous_price) * 100
                return abs(price_change_pct) > alert.threshold_value
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking price alert for {alert.asset_ticker}: {e}")
            return False
    
    def _check_portfolio_value_alert(self, db: Session, alert: models.Alert) -> bool:
        """Check portfolio value alert"""
        try:
            portfolio = db.query(models.Portfolio).filter(
                models.Portfolio.id == alert.portfolio_id
            ).first()
            
            if not portfolio:
                return False
            
            portfolio_context = get_market_data_for_portfolio(portfolio.holdings)
            current_value = portfolio_context.get('total_value', 0)
            
            if alert.threshold_direction == "above":
                return current_value > alert.threshold_value
            elif alert.threshold_direction == "below":
                return current_value < alert.threshold_value
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio value alert: {e}")
            return False
    
    def _check_risk_alert(self, db: Session, alert: models.Alert) -> bool:
        """Check risk threshold alert using existing risk tools"""
        try:
            from agents.orchestrator import FinancialOrchestrator
            
            portfolio = db.query(models.Portfolio).filter(
                models.Portfolio.id == alert.portfolio_id
            ).first()
            
            if not portfolio:
                return False
            
            # Get portfolio context and run risk analysis
            portfolio_context = get_market_data_for_portfolio(portfolio.holdings)
            
            orchestrator = FinancialOrchestrator()
            risk_result = orchestrator.roster["QuantitativeAnalystAgent"].run(
                "calculate risk metrics", context=portfolio_context
            )
            
            if not risk_result.get("success"):
                return False
            
            # Check specific risk metric based on alert configuration
            risk_data = risk_result.get("data", {})
            
            # Example: Check volatility
            volatility = risk_data.get("performance_stats", {}).get("annualized_volatility_pct", 0)
            
            if alert.threshold_direction == "above":
                return volatility > alert.threshold_value
            elif alert.threshold_direction == "below":
                return volatility < alert.threshold_value
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking risk alert: {e}")
            return False
    
    def _trigger_alert(self, db: Session, alert: models.Alert):
        """Trigger an alert and update its status"""
        try:
            alert.status = models.AlertStatus.TRIGGERED
            alert.last_triggered = datetime.utcnow()
            alert.trigger_count += 1
            
            # Here you could add notification logic (email, push, etc.)
            self.logger.info(f"Alert {alert.id} triggered: {alert.name}")
            
            # Optional: Auto-pause alerts that trigger frequently
            if alert.trigger_count > 10:
                alert.status = models.AlertStatus.PAUSED
                alert.is_active = False
                self.logger.info(f"Alert {alert.id} auto-paused due to frequent triggers")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert {alert.id}: {e}")