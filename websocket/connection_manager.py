# websocket/connection_manager.py
"""
WebSocket Connection Manager for Real-Time Risk Alerts
Manages user connections and delivers risk notifications
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, List
import json
import logging
from datetime import datetime, timezone
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connections for real-time notifications
    Supports user-specific messaging and connection lifecycle
    """
    
    def __init__(self):
        # Store active connections by user_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Track connection metadata
        self.connection_info: Dict[WebSocket, dict] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str, token: str = None):
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Initialize user connection set
            if user_id not in self.active_connections:
                self.active_connections[user_id] = set()
            
            # Add connection
            self.active_connections[user_id].add(websocket)
            
            # Store connection metadata
            self.connection_info[websocket] = {
                'user_id': user_id,
                'connected_at': datetime.now(timezone.utc),
                'token': token
            }
            
            logger.info(f"🔌 WebSocket connected: user={user_id}, total_connections={len(self.active_connections[user_id])}")
            
            # Send welcome message
            await self.send_personal_message(user_id, {
                "type": "connection_established",
                "message": "Risk monitoring connected",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            raise
    
    async def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection"""
        try:
            # Remove from active connections
            if user_id in self.active_connections:
                self.active_connections[user_id].discard(websocket)
                
                # Clean up empty user sets
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            
            # Remove connection metadata
            if websocket in self.connection_info:
                del self.connection_info[websocket]
            
            logger.info(f"🔌 WebSocket disconnected: user={user_id}")
            
        except Exception as e:
            logger.error(f"❌ WebSocket disconnect error: {e}")
    
    async def send_personal_message(self, user_id: str, message: dict):
        """Send message to all connections for a specific user"""
        if user_id not in self.active_connections:
            logger.warning(f"⚠️ No active connections for user: {user_id}")
            return False
        
        message_json = json.dumps(message)
        disconnected_connections = []
        successful_sends = 0
        
        for connection in self.active_connections[user_id].copy():
            try:
                await connection.send_text(message_json)
                successful_sends += 1
            except WebSocketDisconnect:
                disconnected_connections.append(connection)
            except Exception as e:
                logger.error(f"❌ Error sending message to user {user_id}: {e}")
                disconnected_connections.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected_connections:
            await self.disconnect(connection, user_id)
        
        logger.info(f"📤 Message sent to user {user_id}: {successful_sends} connections")
        return successful_sends > 0
    
    async def send_risk_alert(self, user_id: str, alert_data: dict):
        """Send risk alert notification to user"""
        
        # Format risk alert message
        risk_change = alert_data.get('risk_change_pct', 'N/A')
        portfolio_name = alert_data.get('portfolio_name', 'Portfolio')
        
        message = {
            "type": "risk_alert",
            "title": "🚨 Portfolio Risk Alert",
            "message": f"{portfolio_name} risk increased by {risk_change}%",
            "alert_data": {
                "portfolio_id": alert_data.get('portfolio_id'),
                "portfolio_name": portfolio_name,
                "risk_score": alert_data.get('risk_score'),
                "risk_change_pct": risk_change,
                "volatility": alert_data.get('volatility'),
                "threshold_breached": alert_data.get('threshold_breached', False)
            },
            "actions": {
                "view_analysis": {
                    "text": "View Analysis",
                    "url": f"/dashboard/workflow/{alert_data.get('workflow_id')}" if alert_data.get('workflow_id') else None
                },
                "view_portfolio": {
                    "text": "View Portfolio", 
                    "url": f"/dashboard/portfolio/{alert_data.get('portfolio_id')}"
                }
            },
            "severity": alert_data.get('severity', 'medium'),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_id": alert_data.get('alert_id')
        }
        
        success = await self.send_personal_message(user_id, message)
        
        if success:
            logger.info(f"🚨 Risk alert sent to user {user_id}: {portfolio_name} ({risk_change}% change)")
        else:
            logger.warning(f"⚠️ Failed to send risk alert to user {user_id}")
        
        return success
    
    async def send_workflow_update(self, user_id: str, workflow_data: dict):
        """Send workflow status update to user"""
        
        message = {
            "type": "workflow_update",
            "title": "🤖 AI Analysis Update",
            "message": workflow_data.get('message', 'AI team is analyzing your portfolio'),
            "workflow_data": {
                "workflow_id": workflow_data.get('workflow_id'),
                "status": workflow_data.get('status'),
                "progress": workflow_data.get('progress', 0),
                "current_agent": workflow_data.get('current_agent'),
                "step": workflow_data.get('step')
            },
            "actions": {
                "view_progress": {
                    "text": "View Progress",
                    "url": f"/dashboard/workflow/{workflow_data.get('workflow_id')}"
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return await self.send_personal_message(user_id, message)
    
    async def broadcast_system_message(self, message: dict, user_ids: List[str] = None):
        """Broadcast system message to specific users or all users"""
        
        target_users = user_ids if user_ids else list(self.active_connections.keys())
        successful_broadcasts = 0
        
        for user_id in target_users:
            success = await self.send_personal_message(user_id, {
                "type": "system_message",
                "title": "📢 System Update",
                "message": message.get('message', 'System notification'),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **message
            })
            
            if success:
                successful_broadcasts += 1
        
        logger.info(f"📢 System message broadcast to {successful_broadcasts}/{len(target_users)} users")
        return successful_broadcasts
    
    def get_connection_stats(self) -> dict:
        """Get current connection statistics"""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        
        return {
            "total_users": len(self.active_connections),
            "total_connections": total_connections,
            "users_online": list(self.active_connections.keys()),
            "connections_per_user": {
                user_id: len(connections) 
                for user_id, connections in self.active_connections.items()
            }
        }
    
    async def ping_all_connections(self):
        """Ping all connections to check if they're still alive"""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        disconnected_users = []
        
        for user_id in list(self.active_connections.keys()):
            try:
                await self.send_personal_message(user_id, ping_message)
            except Exception as e:
                logger.warning(f"⚠️ Ping failed for user {user_id}: {e}")
                disconnected_users.append(user_id)
        
        # Clean up failed connections
        for user_id in disconnected_users:
            if user_id in self.active_connections:
                del self.active_connections[user_id]
        
        return len(disconnected_users)


# Global connection manager instance
connection_manager = ConnectionManager()

def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance"""
    return connection_manager


# Background task to maintain connections
async def connection_heartbeat_task():
    """Background task to ping connections and clean up dead ones"""
    while True:
        try:
            await asyncio.sleep(30)  # Ping every 30 seconds
            disconnected = await connection_manager.ping_all_connections()
            
            if disconnected > 0:
                logger.info(f"🧹 Cleaned up {disconnected} dead connections")
                
        except Exception as e:
            logger.error(f"❌ Heartbeat task error: {e}")
            await asyncio.sleep(60)  # Wait longer on error


# Helper function to start background task
def start_websocket_heartbeat():
    """Start the WebSocket heartbeat background task"""
    asyncio.create_task(connection_heartbeat_task())
    logger.info("💓 WebSocket heartbeat task started")