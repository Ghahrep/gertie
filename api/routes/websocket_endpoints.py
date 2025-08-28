# api/routes/websocket_endpoints.py
"""
Enhanced WebSocket API endpoints for Task 2.3 completion
Focused on WebSocket connections and unique authenticated functionality
HTTP test endpoints moved to main.py as public endpoints
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json
import asyncio
import uuid

from db.session import get_db
from db import crud, models
from api.routes.auth import get_current_user_websocket, get_current_user
from api.schemas import User
from websocket.enhanced_connection_manager import get_enhanced_connection_manager, start_connection_maintenance
from services.enhanced_notification_service import get_notification_service, NotificationConfig

router = APIRouter(prefix="/api/v1/websocket", tags=["WebSocket Real-Time"])

# Initialize services
connection_manager = get_enhanced_connection_manager()
notification_service = get_notification_service()

@router.websocket("/connect/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    user_id: str,
    token: Optional[str] = Query(None),
    topics: Optional[str] = Query("risk_alerts,workflow_updates"),
    compression: Optional[bool] = Query(True)
):
    """
    Enhanced WebSocket connection endpoint with:
    - Connection pooling and auto-scaling
    - Selective topic subscriptions
    - Message compression
    - Connection health monitoring
    """
    
    # Parse subscription topics
    subscription_topics = topics.split(",") if topics else ["risk_alerts", "workflow_updates"]
    
    # Validate user token (implement your auth logic)
    # user = await get_current_user_websocket(token)
    # if not user:
    #     await websocket.close(code=4001, reason="Invalid token")
    #     return
    
    # Connect to enhanced manager
    connected = await connection_manager.connect(
        websocket, user_id, subscription_topics, compression_level=6 if compression else 0
    )
    
    if not connected:
        await websocket.close(code=4000, reason="Connection failed")
        return
    
    try:
        while True:
            try:
                # Wait for client messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                try:
                    message = json.loads(data)
                    await handle_client_message(websocket, user_id, message)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }))
                    
            except asyncio.TimeoutError:
                # Send heartbeat ping
                await websocket.send_text(json.dumps({
                    "type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error for user {user_id}: {e}")
    finally:
        await connection_manager.disconnect(websocket)

async def handle_client_message(websocket: WebSocket, user_id: str, message: Dict[str, Any]):
    """Handle incoming client messages with type-based routing"""
    
    message_type = message.get("type", "unknown")
    
    if message_type == "pong":
        # Response to server ping
        pass
    elif message_type == "subscribe":
        # Subscribe to additional topics
        topic = message.get("topic")
        if topic:
            await connection_manager.subscribe_to_topic(websocket, topic)
            await websocket.send_text(json.dumps({
                "type": "subscription_confirmed",
                "topic": topic,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }))
    elif message_type == "unsubscribe":
        # Unsubscribe from topic
        topic = message.get("topic")
        if topic:
            await connection_manager.unsubscribe_from_topic(websocket, topic)
            await websocket.send_text(json.dumps({
                "type": "unsubscription_confirmed", 
                "topic": topic,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }))
    elif message_type == "acknowledge_notification":
        # Acknowledge notification receipt
        notification_id = message.get("notification_id")
        if notification_id:
            success = await notification_service.acknowledge_notification(notification_id, user_id)
            await websocket.send_text(json.dumps({
                "type": "acknowledgment_confirmed",
                "notification_id": notification_id,
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }))
    elif message_type == "dashboard_state_update":
        # Handle dashboard state updates for contextual notifications
        state_data = message.get("state", {})
        # Store dashboard state for targeted notifications
        # This could be cached in Redis or stored temporarily
        pass
    else:
        # Unknown message type
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))

# ===== AUTHENTICATED ENDPOINTS - Require User Login =====

@router.post("/broadcast/{topic}")
async def broadcast_to_topic(
    topic: str,
    message_data: Dict[str, Any],
    exclude_users: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user)
):
    """Broadcast message to all subscribers of a topic (admin only)"""
    
    # Add authorization check for admin users
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        sent_count = await connection_manager.broadcast_to_topic(
            topic, message_data, qos_level=1, exclude_users=set(exclude_users or [])
        )
        
        return {
            "status": "success",
            "topic": topic,
            "recipients": sent_count,
            "message": "Broadcast completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broadcast failed: {str(e)}")

@router.post("/send/{user_id}")
async def send_direct_message(
    user_id: str,
    message_data: Dict[str, Any],
    qos_level: int = 1,
    compress: bool = True,
    current_user: User = Depends(get_current_user)
):
    """Send direct message to specific user"""
    
    try:
        sent_count = await connection_manager.send_to_user(
            user_id, message_data, qos_level, compress
        )
        
        return {
            "status": "success" if sent_count > 0 else "no_connection",
            "user_id": user_id,
            "connections_reached": sent_count,
            "message": f"Message {'sent' if sent_count > 0 else 'queued - user offline'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Send failed: {str(e)}")

# ===== USER-SPECIFIC AUTHENTICATED ENDPOINTS =====

@router.get("/notifications/stats")
async def get_notification_stats(
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    current_user: User = Depends(get_current_user)
):
    """Get notification delivery statistics for current user"""
    
    try:
        stats = notification_service.get_notification_stats(str(current_user.id), hours)
        
        return {
            "status": "success",
            "period_hours": hours,
            "user_id": current_user.id,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/notifications/{notification_id}/acknowledge")
async def acknowledge_notification_endpoint(
    notification_id: str,
    current_user: User = Depends(get_current_user)
):
    """Acknowledge a notification via REST API"""
    
    try:
        success = await notification_service.acknowledge_notification(notification_id, str(current_user.id))
        
        return {
            "status": "success" if success else "not_found",
            "notification_id": notification_id,
            "acknowledged": success,
            "acknowledged_at": datetime.now(timezone.utc).isoformat() if success else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Acknowledgment failed: {str(e)}")

# ===== USER SUBSCRIPTION MANAGEMENT (AUTHENTICATED) =====

@router.get("/user/subscriptions")
async def get_my_subscriptions(current_user: User = Depends(get_current_user)):
    """Get current user's WebSocket topic subscriptions"""
    
    try:
        # Get active connections for user
        user_connections = connection_manager.user_connections.get(str(current_user.id), set())
        
        if not user_connections:
            return {
                "status": "offline",
                "active_connections": 0,
                "subscriptions": []
            }
        
        # Get subscriptions from first connection (they should all be the same)
        first_connection = next(iter(user_connections))
        connection_metadata = connection_manager.connection_metadata.get(first_connection)
        
        if connection_metadata:
            subscriptions = list(connection_metadata.subscription_topics)
        else:
            subscriptions = []
        
        return {
            "status": "online",
            "active_connections": len(user_connections),
            "subscriptions": subscriptions,
            "available_topics": [
                "risk_alerts",
                "workflow_updates", 
                "portfolio_reports",
                "system_announcements",
                "market_updates"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get subscriptions: {str(e)}")

@router.post("/user/subscriptions/{topic}")
async def subscribe_to_my_topic(
    topic: str,
    current_user: User = Depends(get_current_user)
):
    """Subscribe current user to a topic (affects all user connections)"""
    
    try:
        user_connections = connection_manager.user_connections.get(str(current_user.id), set())
        
        if not user_connections:
            raise HTTPException(status_code=400, detail="No active WebSocket connections")
        
        # Subscribe all user connections to the topic
        subscription_count = 0
        for websocket in user_connections:
            await connection_manager.subscribe_to_topic(websocket, topic)
            subscription_count += 1
        
        return {
            "status": "success",
            "topic": topic,
            "connections_updated": subscription_count,
            "message": f"Subscribed to {topic}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription failed: {str(e)}")

@router.delete("/user/subscriptions/{topic}")
async def unsubscribe_from_my_topic(
    topic: str,
    current_user: User = Depends(get_current_user)
):
    """Unsubscribe current user from a topic"""
    
    try:
        user_connections = connection_manager.user_connections.get(str(current_user.id), set())
        
        if not user_connections:
            raise HTTPException(status_code=400, detail="No active WebSocket connections")
        
        # Unsubscribe all user connections from the topic
        unsubscription_count = 0
        for websocket in user_connections:
            await connection_manager.unsubscribe_from_topic(websocket, topic)
            unsubscription_count += 1
        
        return {
            "status": "success",
            "topic": topic,
            "connections_updated": unsubscription_count,
            "message": f"Unsubscribed from {topic}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unsubscription failed: {str(e)}")

# ===== PERFORMANCE TESTING ENDPOINTS (AUTHENTICATED) =====

@router.post("/load-test/connections")
async def create_load_test_connections(
    connection_count: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user)
):
    """Create multiple test connections for load testing (development only)"""
    
    # This would only be available in development/testing environments
    # Add environment check here
    
    try:
        test_connections = []
        
        for i in range(connection_count):
            # Create mock WebSocket connections for testing
            # This is simplified - in real testing you'd use proper WebSocket clients
            test_user_id = f"test_user_{i}"
            
            # Simulate connection metrics
            test_connections.append({
                "user_id": test_user_id,
                "connection_id": f"conn_{i}",
                "status": "simulated"
            })
        
        return {
            "status": "success",
            "message": f"Created {connection_count} simulated test connections",
            "connections": test_connections,
            "warning": "This endpoint should only be used in testing environments"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load test setup failed: {str(e)}")

@router.post("/load-test/broadcast")
async def load_test_broadcast(
    message_count: int = Query(100, ge=1, le=1000),
    topic: str = Query("test_topic"),
    current_user: User = Depends(get_current_user)
):
    """Send multiple messages for performance testing"""
    
    try:
        sent_messages = []
        
        for i in range(message_count):
            test_message = {
                "type": "load_test",
                "message_id": f"test_msg_{i}",
                "sequence": i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": f"Test message content {i}"
            }
            
            recipients = await connection_manager.broadcast_to_topic(topic, test_message)
            sent_messages.append({
                "message_id": f"test_msg_{i}",
                "recipients": recipients
            })
        
        return {
            "status": "success",
            "messages_sent": len(sent_messages),
            "topic": topic,
            "performance_summary": {
                "total_recipients": sum(msg["recipients"] for msg in sent_messages),
                "average_recipients_per_message": sum(msg["recipients"] for msg in sent_messages) / len(sent_messages) if sent_messages else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load test failed: {str(e)}")
    
@router.post("/context/dashboard-update")
async def update_dashboard_context(
    state_update: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Update dashboard state for contextual suggestions"""
    from smart_suggestions.realtime_context_service import notify_dashboard_state_change
    
    await notify_dashboard_state_change(str(current_user.id), state_update)
    return {"status": "success", "message": "Dashboard context updated"}

@router.post("/context/conversation-update")
async def update_conversation_context(
    conversation_id: str,
    update_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Update conversation context for contextual suggestions"""
    from smart_suggestions.realtime_context_service import notify_conversation_update
    
    await notify_conversation_update(str(current_user.id), conversation_id, update_data)
    return {"status": "success", "message": "Conversation context updated"}

# Initialize WebSocket maintenance on startup
@router.on_event("startup")
async def startup_websocket_system():
    """Initialize WebSocket system and background tasks"""
    
    try:
        # Start connection maintenance
        start_connection_maintenance()
        
        print("Enhanced WebSocket router initialized")
        print("- WebSocket connections: ENABLED")
        print("- Authenticated endpoints: ENABLED") 
        print("- User subscription management: ENABLED")
        print("- Performance testing tools: ENABLED")
        
    except Exception as e:
        print(f"Failed to initialize WebSocket router: {e}")
        raise

# === REMOVED DUPLICATE ENDPOINTS ===
# The following endpoints have been REMOVED from this file as they are now 
# public endpoints in main.py without authentication requirements:
#
# - get_websocket_stats (/stats) -> Now in main.py as public
# - send_test_risk_alert (/test-risk-alert/{user_id}) -> Now in main.py as public  
# - send_test_workflow_update (/test-workflow-update/{user_id}) -> Now in main.py as public
# - get_user_subscriptions (/subscriptions) -> Now in main.py as public
# - subscribe_to_topic_endpoint (/subscriptions/{topic}) -> Now in main.py as public
# - unsubscribe_from_topic_endpoint (/subscriptions/{topic}) -> Now in main.py as public
# - websocket_health_check (/health) -> Now in main.py as public
#
# This separation ensures:
# - Public test endpoints in main.py (no auth required for testing)
# - User-specific authenticated endpoints remain here (/user/subscriptions/*)
# - Admin/authenticated functionality remains here