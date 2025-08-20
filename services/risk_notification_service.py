# services/risk_notification_service.py
"""
Risk Notification Service
Integrates risk attribution with WebSocket notifications
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timezone

from websocket.connection_manager import get_connection_manager

logger = logging.getLogger(__name__)

class RiskNotificationService:
    """
    Service to handle risk-related notifications via WebSocket
    Integrates with your existing risk attribution system
    """
    
    def __init__(self):
        self.connection_manager = get_connection_manager()
        self.notification_history = {}  # Track recent notifications to prevent spam
    
    async def send_threshold_breach_alert(
        self, 
        user_id: str, 
        portfolio_data: Dict, 
        risk_snapshot: Dict,
        workflow_id: Optional[str] = None
    ) -> bool:
        """
        Send alert when risk thresholds are breached
        
        Args:
            user_id: User to notify
            portfolio_data: Portfolio information
            risk_snapshot: Risk calculation results
            workflow_id: Optional workflow session ID
            
        Returns:
            bool: True if notification sent successfully
        """
        
        try:
            # Check if we recently sent an alert for this portfolio to prevent spam
            alert_key = f"{user_id}_{portfolio_data.get('portfolio_id')}"
            recent_alert = self.notification_history.get(alert_key)
            
            if recent_alert:
                time_since_last = (datetime.now(timezone.utc) - recent_alert).total_seconds()
                if time_since_last < 300:  # 5 minutes cooldown
                    logger.info(f"⏰ Skipping duplicate alert for {alert_key} (cooldown)")
                    return False
            
            # Prepare alert data
            alert_data = {
                "portfolio_id": portfolio_data.get('portfolio_id'),
                "portfolio_name": portfolio_data.get('portfolio_name', portfolio_data.get('name', 'Portfolio')),
                "risk_score": risk_snapshot.get('risk_score'),
                "risk_change_pct": risk_snapshot.get('risk_score_change_pct'),
                "volatility": risk_snapshot.get('volatility'),
                "volatility_change_pct": risk_snapshot.get('volatility_change_pct'),
                "threshold_breached": risk_snapshot.get('is_threshold_breach', True),
                "severity": self._determine_severity(risk_snapshot),
                "workflow_id": workflow_id,
                "alert_id": risk_snapshot.get('snapshot_id'),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Send the alert
            success = await self.connection_manager.send_risk_alert(user_id, alert_data)
            
            if success:
                # Track this notification
                self.notification_history[alert_key] = datetime.now(timezone.utc)
                
                logger.info(f"🚨 Threshold breach alert sent: {portfolio_data.get('portfolio_name')} ({risk_snapshot.get('risk_score_change_pct', 'N/A')}% change)")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to send threshold breach alert: {e}")
            return False
    
    async def send_workflow_started_notification(
        self, 
        user_id: str, 
        workflow_data: Dict
    ) -> bool:
        """
        Notify user when AI workflow analysis is started
        
        Args:
            user_id: User to notify
            workflow_data: Workflow information
            
        Returns:
            bool: True if notification sent successfully
        """
        
        try:
            notification_data = {
                "workflow_id": workflow_data.get('workflow_id'),
                "message": f"AI team analyzing {workflow_data.get('portfolio_name', 'your portfolio')} risk changes",
                "status": "started",
                "progress": 0,
                "current_agent": workflow_data.get('current_agent', 'Dr. Sarah Chen'),
                "step": "Risk Analysis Initialization",
                "estimated_duration": "2-3 minutes"
            }
            
            success = await self.connection_manager.send_workflow_update(user_id, notification_data)
            
            if success:
                logger.info(f"🤖 Workflow started notification sent: {workflow_data.get('workflow_id')}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to send workflow started notification: {e}")
            return False
    
    async def send_workflow_progress_update(
        self, 
        user_id: str, 
        workflow_id: str,
        progress: int,
        current_agent: str,
        step: str,
        message: str = None
    ) -> bool:
        """
        Send workflow progress updates
        
        Args:
            user_id: User to notify
            workflow_id: Workflow session ID
            progress: Progress percentage (0-100)
            current_agent: Current agent working
            step: Current analysis step
            message: Optional custom message
            
        Returns:
            bool: True if notification sent successfully
        """
        
        try:
            notification_data = {
                "workflow_id": workflow_id,
                "message": message or f"{current_agent} is {step.lower()}",
                "status": "in_progress",
                "progress": progress,
                "current_agent": current_agent,
                "step": step
            }
            
            success = await self.connection_manager.send_workflow_update(user_id, notification_data)
            
            if success:
                logger.debug(f"🔄 Workflow progress update sent: {workflow_id} ({progress}%)")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to send workflow progress update: {e}")
            return False
    
    async def send_workflow_completed_notification(
        self, 
        user_id: str, 
        workflow_data: Dict,
        analysis_summary: Dict = None
    ) -> bool:
        """
        Notify user when AI workflow analysis is completed
        
        Args:
            user_id: User to notify
            workflow_data: Workflow information
            analysis_summary: Optional summary of analysis results
            
        Returns:
            bool: True if notification sent successfully
        """
        
        try:
            summary_text = "Analysis complete"
            if analysis_summary:
                summary_text = f"Risk assessment complete: {analysis_summary.get('recommendation', 'View full analysis')}"
            
            notification_data = {
                "workflow_id": workflow_data.get('workflow_id'),
                "message": summary_text,
                "status": "completed",
                "progress": 100,
                "current_agent": "AI Team",
                "step": "Analysis Complete",
                "analysis_summary": analysis_summary
            }
            
            success = await self.connection_manager.send_workflow_update(user_id, notification_data)
            
            if success:
                logger.info(f"✅ Workflow completed notification sent: {workflow_data.get('workflow_id')}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to send workflow completed notification: {e}")
            return False
    
    async def send_portfolio_monitoring_status(
        self, 
        user_id: str, 
        portfolios_monitored: int,
        alerts_triggered: int = 0
    ) -> bool:
        """
        Send daily/periodic monitoring status update
        
        Args:
            user_id: User to notify
            portfolios_monitored: Number of portfolios monitored
            alerts_triggered: Number of alerts triggered
            
        Returns:
            bool: True if notification sent successfully
        """
        
        try:
            if alerts_triggered > 0:
                message = f"📊 Daily monitoring: {portfolios_monitored} portfolios checked, {alerts_triggered} alerts triggered"
                severity = "medium"
            else:
                message = f"📊 Daily monitoring: {portfolios_monitored} portfolios checked, all within normal risk levels"
                severity = "low"
            
            notification_data = {
                "type": "monitoring_status",
                "title": "Portfolio Monitoring Report",
                "message": message,
                "data": {
                    "portfolios_monitored": portfolios_monitored,
                    "alerts_triggered": alerts_triggered,
                    "monitoring_date": datetime.now(timezone.utc).date().isoformat()
                },
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            success = await self.connection_manager.send_personal_message(user_id, notification_data)
            
            if success:
                logger.info(f"📈 Monitoring status sent to {user_id}: {portfolios_monitored} portfolios, {alerts_triggered} alerts")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to send monitoring status: {e}")
            return False
    
    def _determine_severity(self, risk_snapshot: Dict) -> str:
        """
        Determine alert severity based on risk metrics
        
        Args:
            risk_snapshot: Risk calculation results
            
        Returns:
            str: Severity level ('low', 'medium', 'high', 'critical')
        """
        
        risk_score = risk_snapshot.get('risk_score', 0)
        risk_change = abs(risk_snapshot.get('risk_score_change_pct', 0))
        volatility = risk_snapshot.get('volatility', 0)
        
        # Critical: Very high risk or massive change
        if risk_score > 90 or risk_change > 50 or volatility > 0.6:
            return 'critical'
        
        # High: High risk or significant change
        if risk_score > 80 or risk_change > 30 or volatility > 0.4:
            return 'high'
        
        # Medium: Moderate risk or noticeable change
        if risk_score > 60 or risk_change > 15 or volatility > 0.25:
            return 'medium'
        
        # Low: Minor changes
        return 'low'
    
    def get_notification_stats(self, user_id: str = None) -> Dict:
        """
        Get notification statistics
        
        Args:
            user_id: Optional user filter
            
        Returns:
            Dict: Notification statistics
        """
        
        connection_stats = self.connection_manager.get_connection_stats()
        
        # Filter notification history if user_id provided
        if user_id:
            user_notifications = {
                k: v for k, v in self.notification_history.items() 
                if k.startswith(f"{user_id}_")
            }
        else:
            user_notifications = self.notification_history
        
        return {
            "connection_stats": connection_stats,
            "recent_notifications": len(user_notifications),
            "notification_history": user_notifications if user_id else {}
        }


# Global service instance
_risk_notification_service = None

def get_risk_notification_service() -> RiskNotificationService:
    """Get global risk notification service instance"""
    global _risk_notification_service
    if _risk_notification_service is None:
        _risk_notification_service = RiskNotificationService()
    return _risk_notification_service


# Helper functions for easy integration with existing services

async def notify_threshold_breach(
    user_id: str, 
    portfolio_data: Dict, 
    risk_snapshot: Dict,
    workflow_id: str = None
) -> bool:
    """
    Helper function to send threshold breach notification
    Call this from your risk attribution service
    """
    service = get_risk_notification_service()
    return await service.send_threshold_breach_alert(user_id, portfolio_data, risk_snapshot, workflow_id)

async def notify_workflow_started(user_id: str, workflow_data: Dict) -> bool:
    """
    Helper function to send workflow started notification
    Call this from your workflow orchestrator
    """
    service = get_risk_notification_service()
    return await service.send_workflow_started_notification(user_id, workflow_data)

async def notify_workflow_progress(
    user_id: str, 
    workflow_id: str, 
    progress: int, 
    agent: str, 
    step: str
) -> bool:
    """
    Helper function to send workflow progress notification
    Call this during workflow execution
    """
    service = get_risk_notification_service()
    return await service.send_workflow_progress_update(user_id, workflow_id, progress, agent, step)

async def notify_workflow_completed(user_id: str, workflow_data: Dict, summary: Dict = None) -> bool:
    """
    Helper function to send workflow completed notification
    Call this when workflow finishes
    """
    service = get_risk_notification_service()
    return await service.send_workflow_completed_notification(user_id, workflow_data, summary)