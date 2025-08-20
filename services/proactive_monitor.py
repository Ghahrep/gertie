# services/proactive_monitor.py - CLEAN VERSION (fixes circular import)
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# MCP imports (your existing)
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
    """Enhanced Proactive portfolio monitoring service with real risk detection"""
    
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
        
        # 🆕 Risk detection service - LAZY IMPORT to avoid circular dependency
        self._risk_detector = None
        
    def _get_risk_detector(self):
        """Lazy initialization of risk detector to avoid circular imports"""
        if self._risk_detector is None:
            try:
                from services.risk_detector import create_risk_detector
                self._risk_detector = create_risk_detector(threshold_pct=15.0)
            except Exception as e:
                logger.warning(f"Could not initialize risk detector: {e}")
                self._risk_detector = False  # Mark as failed
        return self._risk_detector if self._risk_detector is not False else None
        
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
                
                # Risk monitoring (every cycle) - 🔄 ENHANCED
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

    # 🔄 ENHANCED: Replace your existing _check_risk_thresholds method
    async def _check_risk_thresholds(self, portfolio_id: str, user_id: str):
        """Enhanced risk threshold checking with real calculations and MCP integration"""
        try:
            # 🆕 NEW: Real risk detection alongside MCP (if available)
            risk_detector = self._get_risk_detector()
            if risk_detector:
                await self._perform_real_risk_detection(portfolio_id, user_id, risk_detector)
            
            # 🔄 ENHANCED: Still submit to MCP but with enriched context
            job_request = JobRequest(
                query="Check portfolio risk thresholds for monitoring alerts with comprehensive analysis",
                context={
                    "portfolio_id": portfolio_id,
                    "user_id": user_id,
                    "monitoring_mode": True,
                    "thresholds": self.alert_thresholds,
                    "alert_check": True,
                    "enhanced_risk_detection": risk_detector is not None,  # Flag for enhanced analysis
                    "risk_metrics_available": risk_detector is not None
                },
                priority=5,
                timeout_seconds=60,
                required_capabilities=["risk_analysis", "portfolio_monitoring"]
            )
            
            mcp_client = await get_mcp_client()
            job_response = await mcp_client.submit_job(job_request)
            
            logger.debug(f"MCP risk analysis job submitted for portfolio {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Error in enhanced risk threshold check for {portfolio_id}: {str(e)}")
            # Fallback to basic monitoring if enhanced detection fails
            await self._simulate_risk_threshold_check(portfolio_id, user_id)

    # 🆕 NEW: Real risk detection method
    async def _perform_real_risk_detection(self, portfolio_id: str, user_id: str, risk_detector):
        """Perform real risk detection using the RiskDetectorService"""
        db = None
        try:
            # LAZY IMPORT to avoid circular dependency
            from db.session import get_db
            db = next(get_db())
            
            # Use the risk detector to analyze portfolio risk changes
            risk_analysis = await risk_detector.detect_portfolio_risk_changes(
                portfolio_id=int(portfolio_id),
                user_id=int(user_id),
                db=db
            )
            
            if risk_analysis and risk_analysis.threshold_breached:
                logger.warning(f"Risk threshold breached for portfolio {portfolio_id}: {risk_analysis.risk_direction} by {risk_analysis.risk_magnitude_pct:.1f}%")
                
                # Convert risk analysis to alerts compatible with existing system
                alerts = await risk_detector.integrate_with_proactive_monitor(
                    risk_analysis, self
                )
                
                # Generate and send alerts using existing alert system
                for alert_data in alerts:
                    alert = await self._create_alert(
                        portfolio_id=alert_data['portfolio_id'],
                        user_id=user_id,
                        alert_type=alert_data['alert_type'],
                        priority=alert_data['priority'],
                        message=alert_data['message'],
                        details=alert_data['details']
                    )
                    await self._send_alert(alert)
                
                # 🚀 Trigger workflow if significant risk increase
                if risk_analysis.should_trigger_workflow:
                    await self._trigger_risk_workflow(portfolio_id, user_id, risk_analysis)
                    
            elif risk_analysis:
                logger.debug(f"Portfolio {portfolio_id} risk levels normal - no threshold breach")
            else:
                logger.warning(f"Could not analyze risk for portfolio {portfolio_id}")
                
        except Exception as e:
            logger.error(f"Error in real risk detection for portfolio {portfolio_id}: {e}")
        finally:
            if db:
                db.close()

    # 🆕 NEW: Risk workflow trigger method
    async def _trigger_risk_workflow(self, portfolio_id: str, user_id: str, risk_analysis):
        """Trigger AI workflow when significant risk increase detected"""
        try:
            # Generate system prompt based on risk analysis
            workflow_prompt = self._generate_risk_workflow_prompt(portfolio_id, risk_analysis)
            
            # Submit workflow job to MCP
            workflow_job = JobRequest(
                query=workflow_prompt,
                context={
                    "portfolio_id": portfolio_id,
                    "user_id": user_id,
                    "trigger_type": "PROACTIVE_RISK_ALERT",
                    "risk_analysis": {
                        "risk_direction": risk_analysis.risk_direction,
                        "risk_magnitude_pct": risk_analysis.risk_magnitude_pct,
                        "significant_changes": risk_analysis.significant_changes,
                        "recommendation": risk_analysis.recommendation
                    },
                    "workflow_mode": True,
                    "priority": "high"
                },
                priority=3,  # High priority for risk-triggered workflows
                timeout_seconds=300,  # 5 minutes for comprehensive analysis
                required_capabilities=["multi_agent_workflow", "risk_analysis", "portfolio_strategy"]
            )
            
            mcp_client = await get_mcp_client()
            workflow_response = await mcp_client.submit_job(workflow_job)
            
            logger.info(f"Risk-triggered workflow initiated for portfolio {portfolio_id}")
            
            # Create alert about workflow initiation
            workflow_alert = await self._create_alert(
                portfolio_id=portfolio_id,
                user_id=user_id,
                alert_type=AlertType.REGIME_CHANGE,  # Using closest existing type
                priority=AlertPriority.HIGH,
                message=f"AI Team Analysis Initiated: Portfolio risk increased by {risk_analysis.risk_magnitude_pct:.1f}%",
                details={
                    "workflow_triggered": True,
                    "risk_analysis_summary": risk_analysis.recommendation,
                    "analysis_type": "multi_agent_collaborative"
                }
            )
            await self._send_alert(workflow_alert)
            
        except Exception as e:
            logger.error(f"Error triggering risk workflow for portfolio {portfolio_id}: {e}")

    # 🆕 NEW: Generate workflow prompt based on risk analysis
    def _generate_risk_workflow_prompt(self, portfolio_id: str, risk_analysis) -> str:
        """Generate contextual prompt for risk-triggered workflow"""
        
        # Format significant changes for readability
        changes_text = []
        for metric, change in risk_analysis.significant_changes.items():
            changes_text.append(f"  • {metric.replace('_', ' ').title()}: {change:+.1f}%")
        changes_summary = "\n".join(changes_text) if changes_text else "  • Overall risk profile changes detected"
        
        prompt = f"""
System Alert: Significant Portfolio Risk Increase Detected

Portfolio ID: {portfolio_id}
Risk Assessment:
  • Direction: {risk_analysis.risk_direction}
  • Magnitude: {risk_analysis.risk_magnitude_pct:.1f}%
  • Current Risk Score: {risk_analysis.current_metrics.risk_score}/100

Key Risk Changes:
{changes_summary}

Risk Analysis Summary:
{risk_analysis.recommendation}

Team, please conduct a comprehensive 4-step analysis:

1. STRATEGY ANALYSIS: Analyze the root causes of this risk increase
2. SECURITY SCREENING: Review current holdings for risk contributors  
3. RISK ASSESSMENT: Quantify the impact and provide stress test scenarios
4. RECOMMENDATIONS: Provide specific, actionable steps to mitigate the increased risk

Please focus on:
- Immediate actions to reduce risk exposure
- Portfolio rebalancing recommendations
- Hedging strategies if appropriate
- Timeline for implementing changes
- Monitoring adjustments needed

This is a proactive risk alert requiring urgent attention.
        """
        
        return prompt.strip()
    
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
        """Simulate risk threshold checking - FALLBACK ONLY"""
        import random
        
        # Only used as fallback when real risk detection fails
        logger.info(f"Using fallback simulated risk check for portfolio {portfolio_id}")
        
        # Randomly trigger alerts for demonstration
        if random.random() < 0.05:  # 5% chance per check
            alert_type = random.choice([AlertType.VAR_BREACH, AlertType.CONCENTRATION_RISK, AlertType.VOLATILITY_SPIKE])
            
            if self._should_send_alert(f"{portfolio_id}_{alert_type.value}"):
                alert = await self._create_alert(
                    portfolio_id, user_id, alert_type, AlertPriority.HIGH,
                    f"Risk threshold breached: {alert_type.value} (simulated fallback)",
                    {"simulated": True, "fallback_mode": True, "threshold_value": 0.025}
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

    # 🔄 ENHANCED: Add risk context to existing monitoring stats
    def get_monitoring_stats(self) -> Dict:
        """Enhanced monitoring system statistics with risk metrics"""
        # Get your existing stats
        total_alerts = len(self.alert_history)
        recent_alerts = len([a for a in self.alert_history if 
                           (datetime.utcnow() - a.timestamp).total_seconds() < 3600])
        
        alert_by_priority = {}
        for priority in AlertPriority:
            alert_by_priority[priority.value] = len([a for a in self.alert_history if a.priority == priority])
        
        # 🆕 NEW: Add risk-specific statistics
        risk_alerts = len([a for a in self.alert_history if 
                          a.alert_type in [AlertType.VAR_BREACH, AlertType.VOLATILITY_SPIKE]])
        
        workflow_triggered_alerts = len([a for a in self.alert_history if 
                                       a.details.get("workflow_triggered", False)])
        
        risk_detector = self._get_risk_detector()
        
        return {
            "active_monitors": len(self.active_monitors),
            "total_alerts": total_alerts,
            "recent_alerts_1h": recent_alerts,
            "alerts_by_priority": alert_by_priority,
            "risk_specific_alerts": risk_alerts,
            "workflow_triggered_count": workflow_triggered_alerts,
            "monitoring_intervals": self.monitoring_intervals,
            "alert_thresholds": self.alert_thresholds,
            "enhanced_risk_detection": risk_detector is not None  # Flag indicating enhanced capabilities
        }
    
    # 🆕 NEW: Method to manually trigger risk analysis for testing
    async def manual_risk_check(self, portfolio_id: str, user_id: str) -> Dict:
        """Manually trigger risk analysis for testing/debugging"""
        try:
            logger.info(f"Manual risk check initiated for portfolio {portfolio_id}")
            
            risk_detector = self._get_risk_detector()
            if risk_detector:
                await self._perform_real_risk_detection(portfolio_id, user_id, risk_detector)
                return {
                    "status": "completed",
                    "portfolio_id": portfolio_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Manual risk analysis completed successfully"
                }
            else:
                return {
                    "status": "completed",
                    "portfolio_id": portfolio_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Manual risk check completed (risk detector not available)"
                }
        except Exception as e:
            logger.error(f"Manual risk check failed for portfolio {portfolio_id}: {e}")
            return {
                "status": "error",
                "portfolio_id": portfolio_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    # Add remaining methods from your original file...
    def get_active_monitors(self) -> List[str]:
        """Get list of actively monitored portfolios"""
        return list(self.active_monitors.keys())
    
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

# Global monitor instance
_proactive_monitor = ProactiveRiskMonitor()

async def get_proactive_monitor() -> ProactiveRiskMonitor:
    """Get the global proactive monitor instance"""
    return _proactive_monitor