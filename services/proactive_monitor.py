# services/proactive_monitor.py
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from mcp.schemas import JobRequest
from services.mcp_client import get_mcp_client

logger = logging.getLogger(__name__)

class AlertPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(str, Enum):
    VAR_BREACH = "var_breach"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    CONCENTRATION_RISK = "concentration_risk"
    NEWS_IMPACT = "news_impact"
    REGIME_CHANGE = "regime_change"
    TIMING_SIGNAL = "timing_signal"

@dataclass
class MonitoringAlert:
    alert_id: str
    portfolio_id: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    details: Dict
    timestamp: datetime
    resolved: bool = False
    user_id: Optional[str] = None

class ProactiveRiskMonitor:
    """Proactive portfolio monitoring service"""
    
    def __init__(self):
        self.active_monitors: Dict[str, asyncio.Task] = {}
        self.alert_history: List[MonitoringAlert] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            "var_breach": 0.025,  # 2.5% daily VaR threshold
            "correlation_spike": 0.8,  # High correlation warning
            "volatility_spike": 2.0,  # 2x normal volatility
            "concentration_risk": 0.25,  # 25% in single position
            "news_sentiment_drop": -0.3,  # Sentiment drop threshold
            "regime_confidence": 0.8  # Regime change confidence threshold
        }
        
        # Monitoring intervals (seconds)
        self.monitoring_intervals = {
            "risk_check": 300,      # 5 minutes
            "market_regime": 900,   # 15 minutes  
            "news_scan": 180,       # 3 minutes
            "timing_signals": 600   # 10 minutes
        }
        
        # Rate limiting to prevent spam
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=30)  # 30 min cooldown per alert type
        
    async def start_portfolio_monitoring(self, portfolio_id: str, user_id: str):
        """Start continuous monitoring for a portfolio"""
        if portfolio_id in self.active_monitors:
            logger.info(f"Portfolio {portfolio_id} already being monitored")
            return {"status": "already_active", "portfolio_id": portfolio_id}
        
        # Create monitoring task
        monitor_task = asyncio.create_task(
            self._monitor_portfolio_continuously(portfolio_id, user_id)
        )
        self.active_monitors[portfolio_id] = monitor_task
        
        logger.info(f"Started proactive monitoring for portfolio {portfolio_id}")
        return {
            "status": "started",
            "portfolio_id": portfolio_id,
            "monitoring_intervals": self.monitoring_intervals,
            "alert_thresholds": self.alert_thresholds
        }
    
    async def stop_portfolio_monitoring(self, portfolio_id: str):
        """Stop monitoring for a portfolio"""
        if portfolio_id in self.active_monitors:
            self.active_monitors[portfolio_id].cancel()
            del self.active_monitors[portfolio_id]
            logger.info(f"Stopped monitoring for portfolio {portfolio_id}")
            return {"status": "stopped", "portfolio_id": portfolio_id}
        else:
            return {"status": "not_active", "portfolio_id": portfolio_id}
    
    async def _monitor_portfolio_continuously(self, portfolio_id: str, user_id: str):
        """Main monitoring loop for a portfolio"""
        logger.info(f"Starting continuous monitoring loop for portfolio {portfolio_id}")
        
        monitoring_cycle = 0
        
        try:
            while True:
                monitoring_cycle += 1
                logger.debug(f"Monitoring cycle {monitoring_cycle} for portfolio {portfolio_id}")
                
                # Stagger different types of monitoring
                tasks = []
                
                # Risk monitoring (every cycle)
                tasks.append(self._check_risk_thresholds(portfolio_id, user_id))
                
                # Market regime monitoring (every 3rd cycle = ~15 min)
                if monitoring_cycle % 3 == 0:
                    tasks.append(self._check_market_regime_changes(portfolio_id, user_id))
                
                # News monitoring (every cycle for high-impact news)
                tasks.append(self._scan_relevant_news(portfolio_id, user_id))
                
                # Timing signals (every 2nd cycle = ~10 min)
                if monitoring_cycle % 2 == 0:
                    tasks.append(self._check_timing_signals(portfolio_id, user_id))
                
                # Execute monitoring tasks
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait before next cycle
                await asyncio.sleep(self.monitoring_intervals["risk_check"])
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for portfolio {portfolio_id}")
        except Exception as e:
            logger.error(f"Error in monitoring loop for {portfolio_id}: {str(e)}")
            # Continue monitoring after error
            await asyncio.sleep(60)

    
    async def _check_risk_thresholds(self, portfolio_id: str, user_id: str):
        """Check if portfolio risk metrics breach thresholds"""
        try:
            # Submit risk analysis job to MCP
            job_request = JobRequest(
                query="Check portfolio risk thresholds for monitoring alerts",
                context={
                    "portfolio_id": portfolio_id,
                    "user_id": user_id,
                    "monitoring_mode": True,
                    "thresholds": self.alert_thresholds,
                    "alert_check": True
                },
                priority=5,
                timeout_seconds=60,
                required_capabilities=["risk_analysis"]
            )
            
            mcp_client = await get_mcp_client()
            job_response = await mcp_client.submit_job(job_request)
            
            # For demo purposes, simulate risk threshold checks
            # In production, would analyze actual job results
            await self._simulate_risk_threshold_check(portfolio_id, user_id)
            
        except Exception as e:
            logger.error(f"Error checking risk thresholds for {portfolio_id}: {str(e)}")
    
    async def _check_market_regime_changes(self, portfolio_id: str, user_id: str):
        """Monitor for significant market regime changes"""
        try:
            job_request = JobRequest(
                query="Detect market regime changes for monitoring",
                context={
                    "portfolio_id": portfolio_id,
                    "user_id": user_id,
                    "monitoring_mode": True,
                    "regime_change_threshold": self.alert_thresholds["regime_confidence"]
                },
                priority=4,
                required_capabilities=["regime_detection"]
            )
            
            mcp_client = await get_mcp_client()
            job_response = await mcp_client.submit_job(job_request)
            
            # Simulate regime change detection
            await self._simulate_regime_change_check(portfolio_id, user_id)
            
        except Exception as e:
            logger.error(f"Error checking regime changes for {portfolio_id}: {str(e)}")
    
    async def _scan_relevant_news(self, portfolio_id: str, user_id: str):
        """Scan for news that significantly impacts portfolio holdings"""
        try:
            job_request = JobRequest(
                query="Scan news for significant portfolio impact",
                context={
                    "portfolio_id": portfolio_id,
                    "user_id": user_id,
                    "monitoring_mode": True,
                    "news_impact_threshold": 0.7,
                    "sentiment_threshold": self.alert_thresholds["news_sentiment_drop"]
                },
                priority=6,
                required_capabilities=["news_correlation"]
            )
            
            mcp_client = await get_mcp_client()
            job_response = await mcp_client.submit_job(job_request)
            
            # Simulate news impact monitoring
            await self._simulate_news_impact_check(portfolio_id, user_id)
            
        except Exception as e:
            logger.error(f"Error scanning news for {portfolio_id}: {str(e)}")
    
    async def _check_timing_signals(self, portfolio_id: str, user_id: str):
        """Monitor for strong market timing signal changes"""
        try:
            job_request = JobRequest(
                query="Monitor market timing signals for significant changes",
                context={
                    "portfolio_id": portfolio_id,
                    "user_id": user_id,
                    "monitoring_mode": True,
                    "signal_strength_threshold": 0.7
                },
                priority=5,
                required_capabilities=["market_timing"]
            )
            
            mcp_client = await get_mcp_client()
            job_response = await mcp_client.submit_job(job_request)
            
            # Simulate timing signal monitoring
            await self._simulate_timing_signal_check(portfolio_id, user_id)
            
        except Exception as e:
            logger.error(f"Error checking timing signals for {portfolio_id}: {str(e)}")
    
    # Simulation methods for demonstration (replace with real analysis in production)
    async def _simulate_risk_threshold_check(self, portfolio_id: str, user_id: str):
        """Simulate risk threshold checking"""
        import random
        
        # Randomly trigger alerts for demonstration
        if random.random() < 0.05:  # 5% chance per check
            alert_type = random.choice([AlertType.VAR_BREACH, AlertType.CONCENTRATION_RISK, AlertType.VOLATILITY_SPIKE])
            
            if self._should_send_alert(f"{portfolio_id}_{alert_type.value}"):
                alert = await self._create_alert(
                    portfolio_id, user_id, alert_type, AlertPriority.HIGH,
                    f"Risk threshold breached: {alert_type.value}",
                    {"simulated": True, "threshold_value": 0.025}
                )
                await self._send_alert(alert)
    
    async def _simulate_regime_change_check(self, portfolio_id: str, user_id: str):
        """Simulate regime change detection"""
        import random
        
        if random.random() < 0.02:  # 2% chance per check
            if self._should_send_alert(f"{portfolio_id}_regime_change"):
                alert = await self._create_alert(
                    portfolio_id, user_id, AlertType.REGIME_CHANGE, AlertPriority.MEDIUM,
                    "Market regime change detected - from bull_market to late_cycle",
                    {"old_regime": "bull_market", "new_regime": "late_cycle", "confidence": 0.85}
                )
                await self._send_alert(alert)
    
    async def _simulate_news_impact_check(self, portfolio_id: str, user_id: str):
        """Simulate news impact monitoring"""
        import random
        
        if random.random() < 0.03:  # 3% chance per check
            if self._should_send_alert(f"{portfolio_id}_news_impact"):
                alert = await self._create_alert(
                    portfolio_id, user_id, AlertType.NEWS_IMPACT, AlertPriority.MEDIUM,
                    "Significant negative news impact detected on portfolio holdings",
                    {"sentiment_drop": -0.35, "affected_holdings": ["AAPL", "GOOGL"]}
                )
                await self._send_alert(alert)
    
    async def _simulate_timing_signal_check(self, portfolio_id: str, user_id: str):
        """Simulate timing signal monitoring"""
        import random
        
        if random.random() < 0.04:  # 4% chance per check
            signal_type = random.choice(["strong_buy", "strong_sell"])
            priority = AlertPriority.HIGH if signal_type == "strong_sell" else AlertPriority.MEDIUM
            
            if self._should_send_alert(f"{portfolio_id}_timing_signal"):
                alert = await self._create_alert(
                    portfolio_id, user_id, AlertType.TIMING_SIGNAL, priority,
                    f"Strong {signal_type.replace('_', ' ')} signal detected",
                    {"signal": signal_type, "confidence": 0.82, "timeframe": "short_term"}
                )
                await self._send_alert(alert)
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """Check if we should send alert (rate limiting)"""
        now = datetime.utcnow()
        last_alert = self.last_alert_times.get(alert_key)
        
        if last_alert is None or (now - last_alert) > self.alert_cooldown:
            self.last_alert_times[alert_key] = now
            return True
        return False
    
    async def _create_alert(self, portfolio_id: str, user_id: str, alert_type: AlertType, 
                          priority: AlertPriority, message: str, details: Dict) -> MonitoringAlert:
        """Create a monitoring alert"""
        alert_id = f"alert_{portfolio_id}_{alert_type.value}_{datetime.utcnow().timestamp()}"
        
        return MonitoringAlert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            user_id=user_id,
            alert_type=alert_type,
            priority=priority,
            message=message,
            details=details,
            timestamp=datetime.utcnow()
        )
    
    async def _send_alert(self, alert: MonitoringAlert):
        """Send/store alert"""
        # Add to alert history
        self.alert_history.append(alert)
        
        # Log the alert
        logger.warning(f"ALERT GENERATED: {alert.priority.value.upper()} - {alert.message} (Portfolio: {alert.portfolio_id})")
        
        # In production, would send notifications via email, webhooks, etc.
        # For now, just store in memory
        
        # Keep only last 1000 alerts in memory
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    async def generate_proactive_alerts(self, portfolio_id: str, risk_changes: Dict) -> List[MonitoringAlert]:
        """Generate alerts based on risk changes"""
        alerts = []
        user_id = risk_changes.get("user_id", "unknown")
        
        # VaR breach alert
        if risk_changes.get("var_breach"):
            alerts.append(await self._create_alert(
                portfolio_id, user_id, AlertType.VAR_BREACH, AlertPriority.HIGH,
                "Portfolio VaR threshold breached - risk exposure elevated",
                {
                    "current_var": risk_changes.get("current_var", 0),
                    "threshold": self.alert_thresholds["var_breach"],
                    "recommended_action": "Consider reducing position sizes"
                }
            ))
        
        # Correlation spike alert
        if risk_changes.get("correlation_spike"):
            alerts.append(await self._create_alert(
                portfolio_id, user_id, AlertType.CORRELATION_SPIKE, AlertPriority.MEDIUM,
                "Portfolio correlation spike detected - diversification reduced",
                {
                    "avg_correlation": risk_changes.get("avg_correlation", 0),
                    "threshold": self.alert_thresholds["correlation_spike"],
                    "affected_pairs": risk_changes.get("high_corr_pairs", [])
                }
            ))
        
        # Concentration risk alert
        if risk_changes.get("concentration_risk"):
            alerts.append(await self._create_alert(
                portfolio_id, user_id, AlertType.CONCENTRATION_RISK, AlertPriority.HIGH,
                "High concentration risk detected in portfolio",
                {
                    "largest_position_pct": risk_changes.get("largest_position_pct", 0),
                    "threshold": self.alert_thresholds["concentration_risk"] * 100,
                    "concentrated_holdings": risk_changes.get("concentrated_holdings", [])
                }
            ))
        
        # Volatility spike alert  
        if risk_changes.get("volatility_spike"):
            alerts.append(await self._create_alert(
                portfolio_id, user_id, AlertType.VOLATILITY_SPIKE, AlertPriority.MEDIUM,
                "Portfolio volatility spike detected",
                {
                    "current_volatility": risk_changes.get("current_volatility", 0),
                    "normal_volatility": risk_changes.get("normal_volatility", 0),
                    "spike_magnitude": risk_changes.get("volatility_multiplier", 1)
                }
            ))
        
        # Store alerts
        for alert in alerts:
            await self._store_alert(alert)
        
        return alerts
    
    async def _store_alert(self, alert: MonitoringAlert):
        """Store alert in system"""
        # Add to in-memory storage
        self.alert_history.append(alert)
        
        # Log alert with structured data
        logger.info(f"Alert stored: {alert.alert_type.value} for portfolio {alert.portfolio_id}", 
                   extra={
                       "alert_id": alert.alert_id,
                       "portfolio_id": alert.portfolio_id,
                       "alert_type": alert.alert_type.value,
                       "priority": alert.priority.value,
                       "timestamp": alert.timestamp.isoformat()
                   })
        
        # In production, would save to database:
        # await database.insert("monitoring_alerts", alert.__dict__)
    
    def get_active_monitors(self) -> List[str]:
        """Get list of actively monitored portfolios"""
        return list(self.active_monitors.keys())
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring system statistics"""
        total_alerts = len(self.alert_history)
        recent_alerts = len([a for a in self.alert_history if 
                           (datetime.utcnow() - a.timestamp).total_seconds() < 3600])  # Last hour
        
        alert_by_priority = {}
        for priority in AlertPriority:
            alert_by_priority[priority.value] = len([a for a in self.alert_history if a.priority == priority])
        
        return {
            "active_monitors": len(self.active_monitors),
            "total_alerts": total_alerts,
            "recent_alerts_1h": recent_alerts,
            "alerts_by_priority": alert_by_priority,
            "monitoring_intervals": self.monitoring_intervals,
            "alert_thresholds": self.alert_thresholds
        }
    
    def get_portfolio_alerts(self, portfolio_id: str, limit: int = 50) -> List[Dict]:
        """Get alerts for a specific portfolio"""
        portfolio_alerts = [
            alert for alert in self.alert_history 
            if alert.portfolio_id == portfolio_id
        ]
        
        # Sort by timestamp (most recent first)
        portfolio_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Convert to dict format and limit results
        return [
            {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "priority": alert.priority.value,
                "message": alert.message,
                "details": alert.details,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            for alert in portfolio_alerts[:limit]
        ]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Alert {alert_id} marked as resolved")
                return True
        return False
    
    async def update_thresholds(self, new_thresholds: Dict) -> Dict:
        """Update monitoring thresholds"""
        valid_thresholds = {}
        
        for key, value in new_thresholds.items():
            if key in self.alert_thresholds and isinstance(value, (int, float)):
                valid_thresholds[key] = value
                self.alert_thresholds[key] = value
        
        logger.info(f"Updated monitoring thresholds: {valid_thresholds}")
        return {
            "updated_thresholds": valid_thresholds,
            "current_thresholds": self.alert_thresholds
        }
    
    async def get_alert_summary(self, portfolio_id: str, hours: int = 24) -> Dict:
        """Get alert summary for portfolio over specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.portfolio_id == portfolio_id and alert.timestamp >= cutoff_time
        ]
        
        if not recent_alerts:
            return {
                "portfolio_id": portfolio_id,
                "time_period_hours": hours,
                "total_alerts": 0,
                "alert_summary": "No alerts in specified time period",
                "risk_level": "normal"
            }
        
        # Count by priority
        priority_counts = {}
        for priority in AlertPriority:
            priority_counts[priority.value] = len([a for a in recent_alerts if a.priority == priority])
        
        # Determine risk level
        if priority_counts.get("critical", 0) > 0:
            risk_level = "critical"
        elif priority_counts.get("high", 0) > 2:
            risk_level = "high"
        elif priority_counts.get("high", 0) > 0 or priority_counts.get("medium", 0) > 3:
            risk_level = "elevated"
        else:
            risk_level = "normal"
        
        return {
            "portfolio_id": portfolio_id,
            "time_period_hours": hours,
            "total_alerts": len(recent_alerts),
            "alerts_by_priority": priority_counts,
            "risk_level": risk_level,
            "alert_summary": f"{len(recent_alerts)} alerts in last {hours} hours",
            "most_recent_alert": recent_alerts[-1].timestamp.isoformat() if recent_alerts else None
        }

# Global monitor instance
_proactive_monitor = ProactiveRiskMonitor()

async def get_proactive_monitor() -> ProactiveRiskMonitor:
    """Get the global proactive monitor instance"""
    return _proactive_monitor