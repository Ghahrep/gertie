# services/alert_management_system.py
"""
Task 2.2.2: Complete Alert Generation and Management System
Builds on your existing alert service with enhanced features
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from sqlalchemy.orm import Session
from db.session import get_db
from db.models import ProactiveAlert, User, Portfolio, RiskChangeEvent
from db.crud import (
    create_risk_change_event,
    get_user_notification_preferences,
    update_user_notification_preferences
)

logger = logging.getLogger(__name__)

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel(str, Enum):
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"

@dataclass
class AlertRule:
    """Defines when and how alerts should be generated"""
    rule_id: str
    portfolio_id: Optional[int]
    user_id: int
    metric_name: str
    condition: str  # "greater_than", "less_than", "change_percent"
    threshold_value: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 30
    escalation_rules: List[Dict] = None

class AlertManagementSystem:
    """
    Complete alert generation and management system
    Extends your existing alert service with advanced features
    """
    
    def __init__(self):
        self.active_rules: Dict[str, AlertRule] = {}
        self.alert_cache: Dict[str, datetime] = {}  # For cooldown tracking
        self.notification_handlers = {
            NotificationChannel.WEBSOCKET: self._send_websocket_notification,
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.SMS: self._send_sms_notification,
            NotificationChannel.PUSH: self._send_push_notification,
            NotificationChannel.SLACK: self._send_slack_notification,
        }
        
        logger.info("Alert Management System initialized")
    
    async def generate_risk_alert(
        self,
        portfolio_id: int,
        user_id: int,
        risk_change_event: RiskChangeEvent,
        severity: AlertSeverity = AlertSeverity.MEDIUM
    ) -> Optional[ProactiveAlert]:
        """
        Generate alert from risk change event
        Core integration with your existing risk detection
        """
        try:
            db = next(get_db())
            try:
                # Create alert based on risk change
                alert_data = self._create_alert_from_risk_event(
                    risk_change_event, severity, user_id, portfolio_id
                )
                
                # Check cooldown to prevent spam
                cooldown_key = f"{portfolio_id}_{alert_data['alert_type']}"
                if not self._check_alert_cooldown(cooldown_key):
                    logger.debug(f"Alert {cooldown_key} still in cooldown period")
                    return None
                
                # Create alert in database
                alert = ProactiveAlert(
                    portfolio_id=portfolio_id,
                    user_id=user_id,
                    risk_change_event_id=risk_change_event.id,
                    **alert_data
                )
                
                db.add(alert)
                db.commit()
                db.refresh(alert)
                
                # Send notifications
                await self._process_alert_notifications(alert, db)
                
                # Update cooldown
                self.alert_cache[cooldown_key] = datetime.utcnow()
                
                logger.info(f"Risk alert generated: {alert.id} for portfolio {portfolio_id}")
                return alert
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error generating risk alert: {e}")
            return None
    
    def _create_alert_from_risk_event(
        self, 
        risk_event: RiskChangeEvent, 
        severity: AlertSeverity,
        user_id: int,
        portfolio_id: int
    ) -> Dict[str, Any]:
        """Create alert data from risk change event"""
        
        # Determine alert type based on risk changes
        alert_type = "risk_increase"
        if risk_event.risk_direction == "decrease":
            alert_type = "risk_improvement"
        
        # Create title and message
        direction_text = "increased" if risk_event.risk_direction == "increase" else "decreased"
        title = f"Portfolio Risk {direction_text.title()}"
        
        message = f"Portfolio risk has {direction_text} by {risk_event.risk_magnitude_pct:.1f}%"
        
        if risk_event.threshold_breached:
            message += " (Threshold Breach)"
        
        # Add specific changes
        if risk_event.significant_changes:
            key_changes = []
            for metric, change in list(risk_event.significant_changes.items())[:3]:  # Top 3
                key_changes.append(f"{metric.replace('_', ' ').title()}: {change:+.1f}%")
            
            if key_changes:
                message += f"\n\nKey changes: {', '.join(key_changes)}"
        
        # Determine priority based on severity and magnitude
        if risk_event.risk_magnitude_pct > 25 or severity == AlertSeverity.CRITICAL:
            priority = "critical"
        elif risk_event.risk_magnitude_pct > 15 or severity == AlertSeverity.HIGH:
            priority = "high"
        else:
            priority = "medium"
        
        return {
            "alert_type": alert_type,
            "priority": priority,
            "title": title,
            "message": message,
            "details": {
                "risk_direction": risk_event.risk_direction,
                "risk_magnitude_pct": risk_event.risk_magnitude_pct,
                "threshold_breached": risk_event.threshold_breached,
                "significant_changes": risk_event.significant_changes,
                "workflow_triggered": risk_event.workflow_triggered,
                "detected_at": risk_event.detected_at.isoformat(),
                "analysis_summary": "Comprehensive risk analysis completed"
            },
            "triggered_risk_score": getattr(risk_event.current_snapshot, 'risk_score', None)
        }
    
    async def _process_alert_notifications(self, alert: ProactiveAlert, db: Session):
        """Send notifications through configured channels"""
        try:
            # Get user notification preferences
            preferences = get_user_notification_preferences(db, alert.user_id)
            
            # Determine which channels to use based on severity
            channels_to_use = self._get_notification_channels_for_alert(
                alert, preferences
            )
            
            # Send notifications
            delivery_status = {}
            for channel in channels_to_use:
                try:
                    success = await self._send_notification(alert, channel, db)
                    delivery_status[channel.value] = "sent" if success else "failed"
                except Exception as e:
                    logger.error(f"Failed to send {channel.value} notification: {e}")
                    delivery_status[channel.value] = "error"
            
            # Update alert with delivery status
            alert.delivery_channels = [ch.value for ch in channels_to_use]
            alert.delivery_status = delivery_status
            alert.sent_at = datetime.utcnow()
            alert.is_sent = True
            
            db.commit()
            
            logger.info(f"Alert {alert.id} notifications sent: {delivery_status}")
            
        except Exception as e:
            logger.error(f"Error processing alert notifications: {e}")
    
    def _get_notification_channels_for_alert(
        self, 
        alert: ProactiveAlert, 
        preferences: Dict
    ) -> List[NotificationChannel]:
        """Determine notification channels based on alert priority and user preferences"""
        
        channels = []
        risk_prefs = preferences.get("risk_alerts", {})
        
        if not risk_prefs.get("enabled", True):
            return channels
        
        configured_channels = risk_prefs.get("channels", ["websocket"])
        
        # Always include websocket for real-time updates
        if "websocket" in configured_channels:
            channels.append(NotificationChannel.WEBSOCKET)
        
        # Add email for medium+ priority
        if alert.priority in ["medium", "high", "critical"] and "email" in configured_channels:
            channels.append(NotificationChannel.EMAIL)
        
        # Add SMS for high+ priority
        if alert.priority in ["high", "critical"] and "sms" in configured_channels:
            channels.append(NotificationChannel.SMS)
        
        # Add push for all priorities if configured
        if "push" in configured_channels:
            channels.append(NotificationChannel.PUSH)
        
        return channels
    
    async def _send_notification(
        self, 
        alert: ProactiveAlert, 
        channel: NotificationChannel,
        db: Session
    ) -> bool:
        """Send notification through specific channel"""
        try:
            handler = self.notification_handlers.get(channel)
            if handler:
                return await handler(alert, db)
            else:
                logger.warning(f"No handler for notification channel: {channel}")
                return False
        except Exception as e:
            logger.error(f"Error sending {channel.value} notification: {e}")
            return False
    
    # Notification handlers (implement based on your infrastructure)
    async def _send_websocket_notification(self, alert: ProactiveAlert, db: Session) -> bool:
        """Send WebSocket notification"""
        try:
            # Integration with your WebSocket system
            # This would connect to your existing WebSocket infrastructure
            
            websocket_message = {
                "type": "risk_alert",
                "alert_id": alert.id,
                "portfolio_id": alert.portfolio_id,
                "priority": alert.priority,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.created_at.isoformat(),
                "details": alert.details
            }
            
            # Mock WebSocket send (replace with your implementation)
            logger.info(f"WebSocket alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket notification failed: {e}")
            return False
    
    async def _send_email_notification(self, alert: ProactiveAlert, db: Session) -> bool:
        """Send email notification"""
        try:
            # Get user email
            user = db.query(User).filter(User.id == alert.user_id).first()
            if not user:
                return False
            
            # Create email content
            email_subject = f"[Portfolio Alert] {alert.title}"
            email_body = self._create_email_body(alert)
            
            # Mock email send (replace with your email service)
            logger.info(f"Email alert sent to {user.email}: {alert.title}")
            
            # Here you would integrate with your email service
            # await email_service.send(user.email, email_subject, email_body)
            
            return True
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    async def _send_sms_notification(self, alert: ProactiveAlert, db: Session) -> bool:
        """Send SMS notification"""
        try:
            # Mock SMS implementation
            sms_message = f"{alert.title}: {alert.message[:100]}..."
            logger.info(f"SMS alert sent: {alert.title}")
            
            # Here you would integrate with SMS service
            # await sms_service.send(user.phone, sms_message)
            
            return True
            
        except Exception as e:
            logger.error(f"SMS notification failed: {e}")
            return False
    
    async def _send_push_notification(self, alert: ProactiveAlert, db: Session) -> bool:
        """Send push notification"""
        try:
            # Mock push notification
            push_message = {
                "title": alert.title,
                "body": alert.message,
                "data": {"alert_id": alert.id, "portfolio_id": alert.portfolio_id}
            }
            
            logger.info(f"Push notification sent: {alert.title}")
            
            # Here you would integrate with push notification service
            # await push_service.send(user_id, push_message)
            
            return True
            
        except Exception as e:
            logger.error(f"Push notification failed: {e}")
            return False
    
    async def _send_slack_notification(self, alert: ProactiveAlert, db: Session) -> bool:
        """Send Slack notification"""
        try:
            # Mock Slack integration
            slack_message = {
                "text": alert.title,
                "attachments": [{
                    "color": self._get_slack_color_for_priority(alert.priority),
                    "fields": [
                        {"title": "Portfolio", "value": str(alert.portfolio_id), "short": True},
                        {"title": "Priority", "value": alert.priority.upper(), "short": True}
                    ]
                }]
            }
            
            logger.info(f"Slack notification sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False
    
    def _create_email_body(self, alert: ProactiveAlert) -> str:
        """Create formatted email body"""
        return f"""
        {alert.title}
        
        {alert.message}
        
        Portfolio ID: {alert.portfolio_id}
        Priority: {alert.priority.upper()}
        Detected: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
        
        {self._format_alert_details(alert.details)}
        
        This is an automated alert from your portfolio monitoring system.
        """
    
    def _format_alert_details(self, details: Dict) -> str:
        """Format alert details for display"""
        if not details:
            return ""
        
        formatted = "Alert Details:\n"
        
        if "significant_changes" in details and details["significant_changes"]:
            formatted += "Key Risk Changes:\n"
            for metric, change in details["significant_changes"].items():
                formatted += f"  • {metric.replace('_', ' ').title()}: {change:+.1f}%\n"
        
        if "workflow_triggered" in details and details["workflow_triggered"]:
            formatted += "\nAI Team Analysis: Initiated\n"
        
        return formatted
    
    def _get_slack_color_for_priority(self, priority: str) -> str:
        """Get Slack color based on alert priority"""
        colors = {
            "low": "#36a64f",      # Green
            "medium": "#ff9500",   # Orange  
            "high": "#ff0000",     # Red
            "critical": "#8b0000"  # Dark red
        }
        return colors.get(priority, "#36a64f")
    
    def _check_alert_cooldown(self, cooldown_key: str, cooldown_minutes: int = 30) -> bool:
        """Check if alert is still in cooldown period"""
        if cooldown_key not in self.alert_cache:
            return True
        
        last_sent = self.alert_cache[cooldown_key]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.utcnow() - last_sent > cooldown_period
    
    # Alert management methods
    async def acknowledge_alert(self, alert_id: int, user_id: int) -> bool:
        """Mark alert as acknowledged by user"""
        try:
            db = next(get_db())
            try:
                alert = db.query(ProactiveAlert).filter(
                    ProactiveAlert.id == alert_id,
                    ProactiveAlert.user_id == user_id
                ).first()
                
                if alert:
                    alert.is_acknowledged = True
                    alert.acknowledged_at = datetime.utcnow()
                    db.commit()
                    
                    logger.info(f"Alert {alert_id} acknowledged by user {user_id}")
                    return True
                
                return False
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: int, user_id: int) -> bool:
        """Mark alert as resolved"""
        try:
            db = next(get_db())
            try:
                alert = db.query(ProactiveAlert).filter(
                    ProactiveAlert.id == alert_id,
                    ProactiveAlert.user_id == user_id
                ).first()
                
                if alert:
                    alert.is_resolved = True
                    alert.resolved_at = datetime.utcnow()
                    alert.is_active = False
                    db.commit()
                    
                    logger.info(f"Alert {alert_id} resolved by user {user_id}")
                    return True
                
                return False
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def get_user_active_alerts(
        self, 
        user_id: int, 
        portfolio_id: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get active alerts for user"""
        try:
            db = next(get_db())
            try:
                query = db.query(ProactiveAlert).filter(
                    ProactiveAlert.user_id == user_id,
                    ProactiveAlert.is_active == True
                )
                
                if portfolio_id:
                    query = query.filter(ProactiveAlert.portfolio_id == portfolio_id)
                
                alerts = query.order_by(
                    ProactiveAlert.created_at.desc()
                ).limit(limit).all()
                
                return [self._alert_to_dict(alert) for alert in alerts]
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting user alerts: {e}")
            return []
    
    async def get_alert_statistics(self, user_id: int, days: int = 7) -> Dict:
        """Get alert statistics for user"""
        try:
            db = next(get_db())
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                alerts = db.query(ProactiveAlert).filter(
                    ProactiveAlert.user_id == user_id,
                    ProactiveAlert.created_at >= cutoff_date
                ).all()
                
                stats = {
                    "total_alerts": len(alerts),
                    "active_alerts": len([a for a in alerts if a.is_active]),
                    "acknowledged_alerts": len([a for a in alerts if a.is_acknowledged]),
                    "resolved_alerts": len([a for a in alerts if a.is_resolved]),
                    "priority_breakdown": {
                        "critical": len([a for a in alerts if a.priority == "critical"]),
                        "high": len([a for a in alerts if a.priority == "high"]),
                        "medium": len([a for a in alerts if a.priority == "medium"]),
                        "low": len([a for a in alerts if a.priority == "low"])
                    },
                    "delivery_success_rate": self._calculate_delivery_success_rate(alerts),
                    "average_acknowledgment_time_minutes": self._calculate_avg_ack_time(alerts)
                }
                
                return stats
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {}
    
    def _alert_to_dict(self, alert: ProactiveAlert) -> Dict:
        """Convert alert to dictionary"""
        return {
            "alert_id": alert.id,
            "portfolio_id": alert.portfolio_id,
            "alert_type": alert.alert_type,
            "priority": alert.priority,
            "title": alert.title,
            "message": alert.message,
            "details": alert.details,
            "created_at": alert.created_at.isoformat(),
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "is_active": alert.is_active,
            "is_acknowledged": alert.is_acknowledged,
            "is_resolved": alert.is_resolved,
            "delivery_channels": alert.delivery_channels,
            "delivery_status": alert.delivery_status,
            "triggered_risk_score": alert.triggered_risk_score
        }
    
    def _calculate_delivery_success_rate(self, alerts: List[ProactiveAlert]) -> float:
        """Calculate notification delivery success rate"""
        if not alerts:
            return 0.0
        
        sent_alerts = [a for a in alerts if a.delivery_status]
        if not sent_alerts:
            return 0.0
        
        total_deliveries = 0
        successful_deliveries = 0
        
        for alert in sent_alerts:
            for channel, status in alert.delivery_status.items():
                total_deliveries += 1
                if status == "sent":
                    successful_deliveries += 1
        
        return (successful_deliveries / total_deliveries) * 100 if total_deliveries > 0 else 0.0
    
    def _calculate_avg_ack_time(self, alerts: List[ProactiveAlert]) -> Optional[float]:
        """Calculate average acknowledgment time in minutes"""
        ack_times = []
        
        for alert in alerts:
            if alert.acknowledged_at and alert.created_at:
                ack_time = (alert.acknowledged_at - alert.created_at).total_seconds() / 60
                ack_times.append(ack_time)
        
        return sum(ack_times) / len(ack_times) if ack_times else None
    
    async def update_notification_preferences(
        self, 
        user_id: int, 
        preferences: Dict
    ) -> bool:
        """Update user notification preferences"""
        try:
            db = next(get_db())
            try:
                success = update_user_notification_preferences(db, user_id, preferences)
                
                if success:
                    logger.info(f"Updated notification preferences for user {user_id}")
                
                return success
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating notification preferences: {e}")
            return False


# Global alert system instance
_alert_management_system = None

async def get_alert_management_system() -> AlertManagementSystem:
    """Get global alert management system instance"""
    global _alert_management_system
    if _alert_management_system is None:
        _alert_management_system = AlertManagementSystem()
    return _alert_management_system

# Convenience functions for integration
async def generate_risk_alert(portfolio_id: int, user_id: int, risk_event: RiskChangeEvent):
    """Generate alert from risk event"""
    alert_system = await get_alert_management_system()
    return await alert_system.generate_risk_alert(portfolio_id, user_id, risk_event)

async def acknowledge_alert(alert_id: int, user_id: int):
    """Acknowledge an alert"""
    alert_system = await get_alert_management_system()
    return await alert_system.acknowledge_alert(alert_id, user_id)

async def get_user_alerts(user_id: int, portfolio_id: Optional[int] = None):
    """Get active alerts for user"""
    alert_system = await get_alert_management_system()
    return await alert_system.get_user_active_alerts(user_id, portfolio_id)


if __name__ == "__main__":
    async def test_alert_system():
        print("🚨 Testing Alert Management System")
        print("=" * 50)
        
        alert_system = AlertManagementSystem()
        
        # Mock user and portfolio
        test_user_id = 1
        test_portfolio_id = 1
        
        # Test alert generation (would normally come from risk detection)
        print("📧 Testing notification preferences...")
        
        # Test notification preferences
        test_prefs = {
            "risk_alerts": {
                "enabled": True,
                "channels": ["websocket", "email"],
                "threshold": "medium"
            }
        }
        
        success = await alert_system.update_notification_preferences(
            test_user_id, test_prefs
        )
        print(f"✅ Notification preferences updated: {success}")
        
        # Get alert statistics
        stats = await alert_system.get_alert_statistics(test_user_id)
        print(f"📊 Alert statistics: {stats}")
        
        print("\n🎉 Alert Management System testing complete!")
    
    asyncio.run(test_alert_system())