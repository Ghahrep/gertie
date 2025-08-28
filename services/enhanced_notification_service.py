# services/enhanced_notification_service.py
"""
Enhanced multi-channel notification service with rate limiting,
history tracking, and acknowledgment support
"""

import asyncio
import logging
import json
import smtplib
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid
from sqlalchemy.orm import Session

from db.session import get_db
from db import crud, models
from websocket.enhanced_connection_manager import get_enhanced_connection_manager

logger = logging.getLogger(__name__)

@dataclass 
class NotificationConfig:
    """Configuration for notification channels"""
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = False
    slack_enabled: bool = False
    webhook_enabled: bool = False
    
    # Rate limiting
    max_emails_per_hour: int = 20
    max_sms_per_hour: int = 10
    max_push_per_minute: int = 30
    
    # Delivery settings
    retry_attempts: int = 3
    retry_delay_seconds: int = 30
    batch_size: int = 50
    
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    
    # SMS settings (Twilio example)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    
    # Push notification settings (FCM example) 
    fcm_server_key: str = ""
    fcm_project_id: str = ""
    
    # Slack settings
    slack_webhook_url: str = ""
    slack_bot_token: str = ""

@dataclass
class NotificationHistory:
    """Track notification delivery history"""
    notification_id: str
    user_id: str
    channel: str
    message_type: str
    sent_at: datetime
    delivered: bool
    acknowledged_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

class RateLimiter:
    """Token bucket rate limiter for different channels"""
    
    def __init__(self):
        self.buckets: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
    
    def check_rate_limit(self, user_id: str, channel: str, max_per_hour: int) -> bool:
        """Check if user is within rate limit for channel"""
        now = datetime.now()
        user_bucket = self.buckets[user_id][channel]
        
        # Remove old timestamps
        cutoff_time = now - timedelta(hours=1)
        while user_bucket and user_bucket[0] < cutoff_time:
            user_bucket.popleft()
        
        # Check if under limit
        if len(user_bucket) >= max_per_hour:
            return False
        
        user_bucket.append(now)
        return True

class EmailProvider:
    """Email notification provider"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_email(self, to_email: str, subject: str, html_body: str, 
                        text_body: str = None) -> Tuple[bool, Optional[str]]:
        """Send email notification"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config.email_from
            msg['To'] = to_email
            
            # Add text version
            if text_body:
                text_part = MIMEText(text_body, 'plain', 'utf-8')
                msg.attach(text_part)
            
            # Add HTML version
            html_part = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(html_part)
            
            # Send via SMTP
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False, str(e)

class SMSProvider:
    """SMS notification provider (Twilio example)"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_sms(self, to_number: str, message: str) -> Tuple[bool, Optional[str]]:
        """Send SMS notification"""
        try:
            # Mock SMS sending - replace with actual Twilio/provider code
            logger.info(f"SMS sent to {to_number}: {message[:50]}...")
            
            # Here you would integrate with actual SMS provider:
            # from twilio.rest import Client
            # client = Client(self.config.twilio_account_sid, self.config.twilio_auth_token)
            # message = client.messages.create(
            #     body=message,
            #     from_=self.config.twilio_from_number,
            #     to=to_number
            # )
            
            return True, None
            
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
            return False, str(e)

class PushProvider:
    """Push notification provider (FCM example)"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_push(self, device_tokens: List[str], title: str, body: str, 
                       data: Dict = None) -> Tuple[bool, Optional[str]]:
        """Send push notification"""
        try:
            # Mock push sending - replace with actual FCM code
            logger.info(f"Push sent to {len(device_tokens)} devices: {title}")
            
            # Here you would integrate with FCM:
            # from pyfcm import FCMNotification
            # push_service = FCMNotification(api_key=self.config.fcm_server_key)
            # result = push_service.notify_multiple_devices(
            #     registration_ids=device_tokens,
            #     message_title=title,
            #     message_body=body,
            #     data_message=data
            # )
            
            return True, None
            
        except Exception as e:
            logger.error(f"Push send failed: {e}")
            return False, str(e)

class SlackProvider:
    """Slack notification provider"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_slack_message(self, channel: str, text: str, 
                                attachments: List[Dict] = None) -> Tuple[bool, Optional[str]]:
        """Send Slack notification"""
        try:
            payload = {
                "text": text,
                "channel": channel
            }
            
            if attachments:
                payload["attachments"] = attachments
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        return True, None
                    else:
                        error_msg = f"Slack API error: {response.status}"
                        return False, error_msg
                        
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False, str(e)

class EnhancedNotificationService:
    """
    Production-ready multi-channel notification service with:
    - Multi-channel delivery (WebSocket, Email, SMS, Push, Slack)
    - Rate limiting and throttling
    - Delivery confirmation and retry logic
    - Notification history and analytics
    - Intelligent priority and routing
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self.rate_limiter = RateLimiter()
        self.notification_history: Dict[str, NotificationHistory] = {}
        self.retry_queue: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize providers
        self.email_provider = EmailProvider(self.config) if self.config.email_enabled else None
        self.sms_provider = SMSProvider(self.config) if self.config.sms_enabled else None
        self.push_provider = PushProvider(self.config) if self.config.push_enabled else None
        self.slack_provider = SlackProvider(self.config) if self.config.slack_enabled else None
        
        # Get WebSocket manager
        self.websocket_manager = get_enhanced_connection_manager()
        
        logger.info("Enhanced Notification Service initialized")
        
        # Start background tasks
        asyncio.create_task(self._retry_failed_notifications())
        asyncio.create_task(self._cleanup_old_notifications())
    
    async def send_risk_alert(self, user_id: str, alert_data: Dict[str, Any], 
                             channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """Send risk alert across multiple channels with intelligent routing"""
        
        # Get user preferences
        db = next(get_db())
        try:
            user_prefs = crud.get_user_notification_preferences(db, int(user_id))
            risk_prefs = user_prefs.get("risk_alerts", {})
            
            # Determine channels to use
            if not channels:
                channels = risk_prefs.get("channels", ["websocket", "email"])
            
            # Filter based on alert severity and user preferences
            severity = alert_data.get("severity", "medium")
            if severity in ["critical", "emergency"]:
                # Force critical alerts through all available channels
                if self.config.sms_enabled and "sms" not in channels:
                    channels.append("sms")
                if self.config.push_enabled and "push" not in channels:
                    channels.append("push")
            
            # Send through each channel
            results = {}
            notification_id = str(uuid.uuid4())
            
            for channel in channels:
                try:
                    success = await self._send_via_channel(
                        channel, user_id, "risk_alert", alert_data, notification_id
                    )
                    results[channel] = success
                    
                    # Log the attempt
                    self.notification_history[f"{notification_id}_{channel}"] = NotificationHistory(
                        notification_id=notification_id,
                        user_id=user_id,
                        channel=channel,
                        message_type="risk_alert",
                        sent_at=datetime.now(timezone.utc),
                        delivered=success
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to send risk alert via {channel}: {e}")
                    results[channel] = False
            
            # Log to database
            try:
                crud.log_risk_alert(
                    db=db,
                    user_id=user_id,
                    portfolio_id=alert_data.get("portfolio_id"),
                    alert_type="threshold_breach",
                    alert_message=alert_data.get("message", "Risk threshold exceeded"),
                    severity=severity,
                    triggered_metrics=alert_data,
                    workflow_id=alert_data.get("workflow_id")
                )
            except Exception as e:
                logger.error(f"Failed to log risk alert to database: {e}")
            
            return results
            
        finally:
            db.close()
    
    async def send_workflow_update(self, user_id: str, workflow_data: Dict[str, Any]) -> bool:
        """Send workflow progress update"""
        
        notification_id = str(uuid.uuid4())
        
        # Workflow updates typically go through WebSocket only
        success = await self._send_via_channel(
            "websocket", user_id, "workflow_update", workflow_data, notification_id
        )
        
        # Log the notification
        self.notification_history[f"{notification_id}_websocket"] = NotificationHistory(
            notification_id=notification_id,
            user_id=user_id,
            channel="websocket",
            message_type="workflow_update",
            sent_at=datetime.now(timezone.utc),
            delivered=success
        )
        
        return success
    
    async def send_portfolio_report(self, user_id: str, report_data: Dict[str, Any],
                                  preferred_channel: str = "email") -> bool:
        """Send portfolio report via preferred channel"""
        
        notification_id = str(uuid.uuid4())
        
        success = await self._send_via_channel(
            preferred_channel, user_id, "portfolio_report", report_data, notification_id
        )
        
        self.notification_history[f"{notification_id}_{preferred_channel}"] = NotificationHistory(
            notification_id=notification_id,
            user_id=user_id,
            channel=preferred_channel,
            message_type="portfolio_report",
            sent_at=datetime.now(timezone.utc),
            delivered=success
        )
        
        return success
    
    async def _send_via_channel(self, channel: str, user_id: str, message_type: str,
                               data: Dict[str, Any], notification_id: str) -> bool:
        """Send notification via specific channel"""
        
        # Check rate limits
        rate_limits = {
            "email": self.config.max_emails_per_hour,
            "sms": self.config.max_sms_per_hour,
            "push": self.config.max_push_per_minute * 60  # Convert to per hour
        }
        
        if channel in rate_limits:
            if not self.rate_limiter.check_rate_limit(user_id, channel, rate_limits[channel]):
                logger.warning(f"Rate limit exceeded for user {user_id} on channel {channel}")
                return False
        
        # Route to appropriate provider
        try:
            if channel == "websocket":
                return await self._send_websocket(user_id, message_type, data)
            elif channel == "email" and self.email_provider:
                return await self._send_email(user_id, message_type, data)
            elif channel == "sms" and self.sms_provider:
                return await self._send_sms(user_id, message_type, data)
            elif channel == "push" and self.push_provider:
                return await self._send_push(user_id, message_type, data)
            elif channel == "slack" and self.slack_provider:
                return await self._send_slack(user_id, message_type, data)
            else:
                logger.warning(f"Channel {channel} not available or configured")
                return False
                
        except Exception as e:
            logger.error(f"Error sending via {channel}: {e}")
            # Add to retry queue
            self.retry_queue[user_id].append(notification_id)
            return False
    
    async def _send_websocket(self, user_id: str, message_type: str, data: Dict[str, Any]) -> bool:
        """Send via WebSocket"""
        if message_type == "risk_alert":
            return await self.websocket_manager.send_risk_alert(user_id, data)
        elif message_type == "workflow_update":
            return await self.websocket_manager.send_workflow_update(user_id, data)
        else:
            # Generic message
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            return await self.websocket_manager.send_to_user(user_id, message) > 0
    
    async def _send_email(self, user_id: str, message_type: str, data: Dict[str, Any]) -> bool:
        """Send via Email"""
        try:
            # Get user email from database
            db = next(get_db())
            try:
                user = db.query(models.User).filter(models.User.id == int(user_id)).first()
                if not user:
                    return False
                
                # Create email content based on message type
                if message_type == "risk_alert":
                    subject = f"Portfolio Risk Alert - {data.get('portfolio_name', 'Your Portfolio')}"
                    html_body = self._create_risk_alert_email_html(data)
                    text_body = self._create_risk_alert_email_text(data)
                elif message_type == "portfolio_report":
                    subject = f"Portfolio Report - {data.get('report_date', 'Recent')}"
                    html_body = self._create_portfolio_report_email_html(data)
                    text_body = self._create_portfolio_report_email_text(data)
                else:
                    subject = "Portfolio Notification"
                    html_body = f"<p>{json.dumps(data, indent=2)}</p>"
                    text_body = json.dumps(data, indent=2)
                
                success, error = await self.email_provider.send_email(
                    user.email, subject, html_body, text_body
                )
                
                if not success:
                    logger.error(f"Email send failed for user {user_id}: {error}")
                
                return success
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in email sending: {e}")
            return False
    
    async def _send_sms(self, user_id: str, message_type: str, data: Dict[str, Any]) -> bool:
        """Send via SMS"""
        try:
            # Get user phone from database (you'd need to add phone field to User model)
            db = next(get_db())
            try:
                user = db.query(models.User).filter(models.User.id == int(user_id)).first()
                if not user:
                    return False
                
                # Get phone number from preferences
                phone = user.preferences.get("phone_number") if user.preferences else None
                if not phone:
                    return False
                
                # Create SMS content
                if message_type == "risk_alert":
                    message = f"RISK ALERT: {data.get('portfolio_name', 'Portfolio')} risk increased by {data.get('risk_change_pct', 'N/A')}%. View details in app."
                else:
                    message = f"Portfolio update: {data.get('message', 'Check your portfolio')}"
                
                # Truncate to SMS limit
                message = message[:160]
                
                success, error = await self.sms_provider.send_sms(phone, message)
                
                if not success:
                    logger.error(f"SMS send failed for user {user_id}: {error}")
                
                return success
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in SMS sending: {e}")
            return False
    
    async def _send_push(self, user_id: str, message_type: str, data: Dict[str, Any]) -> bool:
        """Send via Push Notification"""
        try:
            # Get user device tokens from database (you'd need to add device_tokens to User)
            db = next(get_db())
            try:
                user = db.query(models.User).filter(models.User.id == int(user_id)).first()
                if not user:
                    return False
                
                device_tokens = user.preferences.get("device_tokens", []) if user.preferences else []
                if not device_tokens:
                    return False
                
                # Create push notification content
                if message_type == "risk_alert":
                    title = "Portfolio Risk Alert"
                    body = f"{data.get('portfolio_name', 'Portfolio')} risk increased"
                    push_data = {"type": "risk_alert", "portfolio_id": str(data.get("portfolio_id", ""))}
                else:
                    title = "Portfolio Update"
                    body = data.get("message", "Portfolio notification")
                    push_data = {"type": message_type}
                
                success, error = await self.push_provider.send_push(device_tokens, title, body, push_data)
                
                if not success:
                    logger.error(f"Push send failed for user {user_id}: {error}")
                
                return success
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in push sending: {e}")
            return False
    
    async def _send_slack(self, user_id: str, message_type: str, data: Dict[str, Any]) -> bool:
        """Send via Slack"""
        try:
            # Get user Slack channel/username from database
            db = next(get_db())
            try:
                user = db.query(models.User).filter(models.User.id == int(user_id)).first()
                if not user:
                    return False
                
                slack_channel = user.preferences.get("slack_channel") if user.preferences else None
                if not slack_channel:
                    slack_channel = f"@{user.email.split('@')[0]}"  # Default to username
                
                # Create Slack message
                if message_type == "risk_alert":
                    text = f":warning: *Portfolio Risk Alert*"
                    attachments = [{
                        "color": "danger" if data.get("severity") == "high" else "warning",
                        "fields": [
                            {"title": "Portfolio", "value": data.get("portfolio_name", "Unknown"), "short": True},
                            {"title": "Risk Change", "value": f"{data.get('risk_change_pct', 'N/A')}%", "short": True},
                            {"title": "Current Risk Score", "value": str(data.get("risk_score", "N/A")), "short": True}
                        ]
                    }]
                else:
                    text = f"Portfolio Update: {data.get('message', 'Notification')}"
                    attachments = None
                
                success, error = await self.slack_provider.send_slack_message(
                    slack_channel, text, attachments
                )
                
                if not success:
                    logger.error(f"Slack send failed for user {user_id}: {error}")
                
                return success
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in Slack sending: {e}")
            return False
    
    def _create_risk_alert_email_html(self, data: Dict[str, Any]) -> str:
        """Create HTML email for risk alert"""
        severity_color = "#dc3545" if data.get("severity") == "high" else "#fd7e14"
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px; border-left: 4px solid {severity_color};">
                <h2 style="color: {severity_color}; margin-top: 0;">🚨 Portfolio Risk Alert</h2>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0;">
                    <p style="margin: 0; font-size: 16px;"><strong>{data.get('portfolio_name', 'Your Portfolio')}</strong></p>
                    <p style="margin: 5px 0 0 0; color: #666;">Risk increased by {data.get('risk_change_pct', 'N/A')}%</p>
                </div>
                
                <div style="margin: 20px 0;">
                    <h4 style="margin-bottom: 10px;">Key Metrics:</h4>
                    <ul style="list-style: none; padding: 0;">
                        <li style="padding: 5px 0;"><strong>Current Risk Score:</strong> {data.get('risk_score', 'N/A')}</li>
                        <li style="padding: 5px 0;"><strong>Volatility:</strong> {data.get('volatility', 'N/A')}</li>
                        <li style="padding: 5px 0;"><strong>Alert Time:</strong> {data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}</li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin: 25px 0;">
                    <a href="#" style="background: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block;">
                        View Portfolio Details
                    </a>
                </div>
                
                <p style="font-size: 12px; color: #666; margin-top: 30px;">
                    This is an automated alert from your portfolio monitoring system. 
                    If you need to adjust alert settings, please contact support.
                </p>
            </div>
        </body>
        </html>
        """
    
    def _create_risk_alert_email_text(self, data: Dict[str, Any]) -> str:
        """Create text email for risk alert"""
        return f"""
PORTFOLIO RISK ALERT

Portfolio: {data.get('portfolio_name', 'Your Portfolio')}
Risk Change: +{data.get('risk_change_pct', 'N/A')}%

Key Metrics:
- Current Risk Score: {data.get('risk_score', 'N/A')}
- Volatility: {data.get('volatility', 'N/A')}
- Alert Time: {data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}

Please review your portfolio and consider taking appropriate action.

This is an automated alert. To adjust settings, contact support.
        """
    
    def _create_portfolio_report_email_html(self, data: Dict[str, Any]) -> str:
        """Create HTML email for portfolio report"""
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px;">
                <h2 style="color: #28a745; margin-top: 0;">📊 Portfolio Report</h2>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0;">
                    <p style="margin: 0; font-size: 16px;"><strong>Report Date:</strong> {data.get('report_date', 'Recent')}</p>
                    <p style="margin: 5px 0 0 0; color: #666;">Portfolio Performance Summary</p>
                </div>
                
                <div style="margin: 20px 0;">
                    <h4 style="margin-bottom: 10px;">Performance Highlights:</h4>
                    <ul>
                        <li>Total Return: {data.get('total_return', 'N/A')}</li>
                        <li>Risk-Adjusted Return: {data.get('risk_adjusted_return', 'N/A')}</li>
                        <li>Sharpe Ratio: {data.get('sharpe_ratio', 'N/A')}</li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin: 25px 0;">
                    <a href="#" style="background: #28a745; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block;">
                        View Full Report
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_portfolio_report_email_text(self, data: Dict[str, Any]) -> str:
        """Create text email for portfolio report"""
        return f"""
PORTFOLIO REPORT - {data.get('report_date', 'Recent')}

Performance Highlights:
- Total Return: {data.get('total_return', 'N/A')}
- Risk-Adjusted Return: {data.get('risk_adjusted_return', 'N/A')}
- Sharpe Ratio: {data.get('sharpe_ratio', 'N/A')}

For detailed analysis, please visit your dashboard.
        """
    
    async def acknowledge_notification(self, notification_id: str, user_id: str) -> bool:
        """Mark notification as acknowledged"""
        for key, history in self.notification_history.items():
            if key.startswith(notification_id) and history.user_id == user_id:
                history.acknowledged_at = datetime.now(timezone.utc)
                
                # Also update database if it's a risk alert
                if history.message_type == "risk_alert":
                    db = next(get_db())
                    try:
                        crud.acknowledge_alert(db, notification_id, user_id)
                    except Exception as e:
                        logger.error(f"Failed to acknowledge alert in database: {e}")
                    finally:
                        db.close()
                
                return True
        return False
    
    async def _retry_failed_notifications(self):
        """Background task to retry failed notifications"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for user_id, notification_ids in list(self.retry_queue.items()):
                    for notification_id in notification_ids[:]:  # Copy list to avoid modification during iteration
                        # Find the failed notification
                        failed_notifications = [
                            (key, history) for key, history in self.notification_history.items()
                            if key.startswith(notification_id) and not history.delivered
                        ]
                        
                        for key, history in failed_notifications:
                            if history.retry_count < self.config.retry_attempts:
                                history.retry_count += 1
                                
                                # Retry the notification
                                success = await self._send_via_channel(
                                    history.channel, user_id, history.message_type, 
                                    {}, notification_id  # Would need to store original data
                                )
                                
                                if success:
                                    history.delivered = True
                                    history.sent_at = datetime.now(timezone.utc)
                                    notification_ids.remove(notification_id)
                                    logger.info(f"Retry successful for notification {notification_id}")
                                else:
                                    logger.warning(f"Retry {history.retry_count} failed for notification {notification_id}")
                            else:
                                # Max retries reached
                                notification_ids.remove(notification_id)
                                logger.error(f"Max retries reached for notification {notification_id}")
                
            except Exception as e:
                logger.error(f"Error in retry task: {e}")
    
    async def _cleanup_old_notifications(self):
        """Background task to clean up old notification history"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
                
                # Remove old notification history
                old_notifications = [
                    key for key, history in self.notification_history.items()
                    if history.sent_at < cutoff_time
                ]
                
                for key in old_notifications:
                    del self.notification_history[key]
                
                if old_notifications:
                    logger.info(f"Cleaned up {len(old_notifications)} old notifications")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def get_notification_stats(self, user_id: Optional[str] = None, 
                              hours: int = 24) -> Dict[str, Any]:
        """Get notification delivery statistics"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Filter notifications
        notifications = [
            history for history in self.notification_history.values()
            if (not user_id or history.user_id == user_id) and history.sent_at >= cutoff_time
        ]
        
        if not notifications:
            return {
                "total_notifications": 0,
                "delivery_rate": 0.0,
                "channel_breakdown": {},
                "message_type_breakdown": {}
            }
        
        # Calculate statistics
        total_notifications = len(notifications)
        delivered_notifications = len([n for n in notifications if n.delivered])
        delivery_rate = (delivered_notifications / total_notifications) * 100
        
        # Channel breakdown
        channel_stats = defaultdict(lambda: {"total": 0, "delivered": 0, "rate": 0.0})
        for notification in notifications:
            channel_stats[notification.channel]["total"] += 1
            if notification.delivered:
                channel_stats[notification.channel]["delivered"] += 1
        
        for channel, stats in channel_stats.items():
            stats["rate"] = (stats["delivered"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        
        # Message type breakdown
        type_stats = defaultdict(lambda: {"total": 0, "delivered": 0, "rate": 0.0})
        for notification in notifications:
            type_stats[notification.message_type]["total"] += 1
            if notification.delivered:
                type_stats[notification.message_type]["delivered"] += 1
        
        for msg_type, stats in type_stats.items():
            stats["rate"] = (stats["delivered"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        
        return {
            "total_notifications": total_notifications,
            "delivered_notifications": delivered_notifications,
            "delivery_rate": round(delivery_rate, 2),
            "channel_breakdown": dict(channel_stats),
            "message_type_breakdown": dict(type_stats),
            "acknowledgment_rate": round(
                (len([n for n in notifications if n.acknowledged_at]) / total_notifications) * 100, 2
            )
        }

# Global notification service instance
_notification_service = None

def get_notification_service(config: Optional[NotificationConfig] = None) -> EnhancedNotificationService:
    """Get the global notification service instance"""
    global _notification_service
    if _notification_service is None:
        _notification_service = EnhancedNotificationService(config)
    return _notification_service

# Convenience functions for easy integration
async def notify_risk_alert(user_id: str, alert_data: Dict[str, Any], 
                          channels: Optional[List[str]] = None) -> Dict[str, bool]:
    """Send risk alert notification across multiple channels"""
    service = get_notification_service()
    return await service.send_risk_alert(user_id, alert_data, channels)

async def notify_workflow_update(user_id: str, workflow_data: Dict[str, Any]) -> bool:
    """Send workflow progress notification"""
    service = get_notification_service()
    return await service.send_workflow_update(user_id, workflow_data)

async def acknowledge_notification(notification_id: str, user_id: str) -> bool:
    """Acknowledge a notification"""
    service = get_notification_service()
    return await service.acknowledge_notification(notification_id, user_id)