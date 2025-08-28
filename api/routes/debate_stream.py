# api/websockets/debate_stream.py
"""
Real-Time Debate Streaming System
===============================
WebSocket implementation for streaming multi-agent debates in real-time
with sophisticated connection management and event distribution.
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import uuid
from dataclasses import asdict

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.routing import APIRouter

from api.routes.auth import get_current_user
from services.mcp_client import get_mcp_client
from mcp.agent_communication import debate_communication_hub, MessageType
from db.models import DebateStage

class AgentMessage:
    def __init__(self, content="", sender="", message_type=None, debate_id=None):
        self.content = content
        self.sender = sender
        self.message_type = message_type
        self.debate_id = debate_id

logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    """Types of events streamed during debates"""
    DEBATE_STARTED = "debate_started"
    AGENT_JOINED = "agent_joined"
    POSITION_SUBMITTED = "position_submitted"
    CHALLENGE_ISSUED = "challenge_issued"
    RESPONSE_GIVEN = "response_given"
    EVIDENCE_PRESENTED = "evidence_presented"
    CONSENSUS_UPDATE = "consensus_update"
    STAGE_CHANGED = "stage_changed"
    DEBATE_COMPLETED = "debate_completed"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CONNECTION_STATUS = "connection_status"

class DebateStreamEvent:
    """Structured event for debate streaming"""
    
    def __init__(self, event_type: StreamEventType, debate_id: str, 
                 data: Any = None, agent_id: str = None, timestamp: datetime = None):
        self.event_type = event_type
        self.debate_id = debate_id
        self.agent_id = agent_id
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "debate_id": self.debate_id,
            "agent_id": self.agent_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    

class ConnectionManager:
    """Manages WebSocket connections for debate streaming"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}  # connection_id -> websocket
        self.debate_subscribers: Dict[str, Set[str]] = {}  # debate_id -> set of connection_ids
        self.user_connections: Dict[int, Set[str]] = {}  # user_id -> set of connection_ids
        self.connection_metadata: Dict[str, Dict] = {}  # connection_id -> metadata
        
        # Event handlers
        self.event_handlers: Dict[StreamEventType, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        
        # Performance tracking
        self.message_counts: Dict[str, int] = {}
        self.connection_stats: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int, 
                     debate_id: str = None, connection_params: Dict = None) -> str:
        """Accept new WebSocket connection"""
        
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        
        self.connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "debate_id": debate_id,
            "connected_at": datetime.now(),
            "params": connection_params or {},
            "last_heartbeat": datetime.now()
        }
        
        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Subscribe to specific debate if provided
        if debate_id:
            await self.subscribe_to_debate(connection_id, debate_id)
        
        # Initialize connection stats
        self.connection_stats[connection_id] = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "last_activity": datetime.now()
        }
        
        logger.info(f"WebSocket connection {connection_id} established for user {user_id}")
        
        # Send connection confirmation
        await self.send_to_connection(connection_id, DebateStreamEvent(
            event_type=StreamEventType.CONNECTION_STATUS,
            debate_id=debate_id or "global",
            data={"status": "connected", "connection_id": connection_id}
        ))
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        
        if connection_id in self.connections:
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get("user_id")
            debate_id = metadata.get("debate_id")
            
            # Remove from all subscriptions
            for subscribers in self.debate_subscribers.values():
                subscribers.discard(connection_id)
            
            # Remove from user connections
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Clean up
            del self.connections[connection_id]
            del self.connection_metadata[connection_id]
            if connection_id in self.connection_stats:
                del self.connection_stats[connection_id]
            
            logger.info(f"WebSocket connection {connection_id} disconnected")
    
    async def subscribe_to_debate(self, connection_id: str, debate_id: str):
        """Subscribe connection to debate events"""
        
        if debate_id not in self.debate_subscribers:
            self.debate_subscribers[debate_id] = set()
        
        self.debate_subscribers[debate_id].add(connection_id)
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["debate_id"] = debate_id
        
        logger.info(f"Connection {connection_id} subscribed to debate {debate_id}")
    
    async def unsubscribe_from_debate(self, connection_id: str, debate_id: str):
        """Unsubscribe connection from debate events"""
        
        if debate_id in self.debate_subscribers:
            self.debate_subscribers[debate_id].discard(connection_id)
            
            # Clean up empty subscription sets
            if not self.debate_subscribers[debate_id]:
                del self.debate_subscribers[debate_id]
        
        logger.info(f"Connection {connection_id} unsubscribed from debate {debate_id}")
    
    async def send_to_connection(self, connection_id: str, event: DebateStreamEvent):
        """Send event to specific connection"""
        
        if connection_id not in self.connections:
            return False
        
        websocket = self.connections[connection_id]
        
        try:
            await websocket.send_text(event.to_json())
            
            # Update stats
            if connection_id in self.connection_stats:
                self.connection_stats[connection_id]["messages_sent"] += 1
                self.connection_stats[connection_id]["last_activity"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {e}")
            
            # Update error stats
            if connection_id in self.connection_stats:
                self.connection_stats[connection_id]["errors"] += 1
            
            # Clean up broken connection
            await self.disconnect(connection_id)
            return False
    
    async def broadcast_to_debate(self, debate_id: str, event: DebateStreamEvent):
        """Broadcast event to all subscribers of a debate"""
        
        if debate_id not in self.debate_subscribers:
            return 0
        
        subscribers = self.debate_subscribers[debate_id].copy()  # Copy to avoid modification during iteration
        successful_sends = 0
        failed_connections = []
        
        for connection_id in subscribers:
            success = await self.send_to_connection(connection_id, event)
            if success:
                successful_sends += 1
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
        
        logger.debug(f"Broadcast to debate {debate_id}: {successful_sends}/{len(subscribers)} successful")
        return successful_sends
    
    async def broadcast_to_user(self, user_id: int, event: DebateStreamEvent):
        """Broadcast event to all connections of a user"""
        
        if user_id not in self.user_connections:
            return 0
        
        connections = self.user_connections[user_id].copy()
        successful_sends = 0
        
        for connection_id in connections:
            success = await self.send_to_connection(connection_id, event)
            if success:
                successful_sends += 1
        
        return successful_sends
    
    async def handle_client_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming message from client"""
        
        if connection_id not in self.connection_stats:
            return
        
        self.connection_stats[connection_id]["messages_received"] += 1
        self.connection_stats[connection_id]["last_activity"] = datetime.now()
        
        message_type = message.get("type")
        
        if message_type == "heartbeat":
            await self._handle_heartbeat(connection_id, message)
        elif message_type == "subscribe":
            await self._handle_subscription_request(connection_id, message)
        elif message_type == "unsubscribe":
            await self._handle_unsubscription_request(connection_id, message)
        elif message_type == "request_status":
            await self._handle_status_request(connection_id, message)
        else:
            logger.warning(f"Unknown message type from connection {connection_id}: {message_type}")
    
    async def _handle_heartbeat(self, connection_id: str, message: Dict[str, Any]):
        """Handle heartbeat from client"""
        
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["last_heartbeat"] = datetime.now()
        
        # Send heartbeat response
        await self.send_to_connection(connection_id, DebateStreamEvent(
            event_type=StreamEventType.HEARTBEAT,
            debate_id="system",
            data={"type": "pong", "timestamp": datetime.now().isoformat()}
        ))
    
    async def _handle_subscription_request(self, connection_id: str, message: Dict[str, Any]):
        """Handle subscription request"""
        
        debate_id = message.get("debate_id")
        if debate_id:
            await self.subscribe_to_debate(connection_id, debate_id)
            
            # Send confirmation
            await self.send_to_connection(connection_id, DebateStreamEvent(
                event_type=StreamEventType.CONNECTION_STATUS,
                debate_id=debate_id,
                data={"status": "subscribed", "debate_id": debate_id}
            ))
    
    async def _handle_unsubscription_request(self, connection_id: str, message: Dict[str, Any]):
        """Handle unsubscription request"""
        
        debate_id = message.get("debate_id")
        if debate_id:
            await self.unsubscribe_from_debate(connection_id, debate_id)
            
            # Send confirmation
            await self.send_to_connection(connection_id, DebateStreamEvent(
                event_type=StreamEventType.CONNECTION_STATUS,
                debate_id=debate_id,
                data={"status": "unsubscribed", "debate_id": debate_id}
            ))
    
    async def _handle_status_request(self, connection_id: str, message: Dict[str, Any]):
        """Handle status request"""
        
        debate_id = message.get("debate_id")
        
        # Get debate status (would integrate with actual debate engine)
        status_data = {
            "connection_id": connection_id,
            "subscribed_debates": [
                debate_id for debate_id, subscribers in self.debate_subscribers.items()
                if connection_id in subscribers
            ],
            "connection_stats": self.connection_stats.get(connection_id, {}),
            "server_time": datetime.now().isoformat()
        }
        
        await self.send_to_connection(connection_id, DebateStreamEvent(
            event_type=StreamEventType.CONNECTION_STATUS,
            debate_id=debate_id or "system",
            data=status_data
        ))
    
    def get_connection_count(self, debate_id: str = None) -> int:
        """Get number of active connections"""
        
        if debate_id:
            return len(self.debate_subscribers.get(debate_id, set()))
        else:
            return len(self.connections)
    
    def get_debate_analytics(self, debate_id: str) -> Dict[str, Any]:
        """Get analytics for a specific debate"""
        
        subscribers = self.debate_subscribers.get(debate_id, set())
        
        return {
            "active_connections": len(subscribers),
            "total_messages_sent": sum(
                self.connection_stats.get(conn_id, {}).get("messages_sent", 0)
                for conn_id in subscribers
            ),
            "engagement_metrics": self._calculate_engagement_metrics(subscribers)
        }
    
    def _calculate_engagement_metrics(self, connection_ids: Set[str]) -> Dict[str, float]:
        """Calculate engagement metrics for connections"""
        
        if not connection_ids:
            return {"average_activity": 0.0, "response_rate": 0.0}
        
        now = datetime.now()
        total_activity = 0
        active_connections = 0
        
        for conn_id in connection_ids:
            if conn_id in self.connection_stats:
                stats = self.connection_stats[conn_id]
                last_activity = stats.get("last_activity")
                
                if last_activity:
                    minutes_since_activity = (now - last_activity).total_seconds() / 60
                    if minutes_since_activity < 30:  # Active within 30 minutes
                        active_connections += 1
                        total_activity += stats.get("messages_received", 0)
        
        return {
            "average_activity": total_activity / max(len(connection_ids), 1),
            "response_rate": active_connections / len(connection_ids)
        }

# Global connection manager
connection_manager = ConnectionManager()

class DebateStreamingService:
    """Service for managing debate streaming and event distribution"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.event_processors: Dict[str, Callable] = {}
        self.setup_event_processors()
    
    def setup_event_processors(self):
        """Set up event processing for different debate events"""
        
        # Register debate communication hub handler
        async def handle_agent_message(message: AgentMessage):
            """Convert agent message to stream event"""
            
            event_type_map = {
                MessageType.POSITION: StreamEventType.POSITION_SUBMITTED,
                MessageType.CHALLENGE: StreamEventType.CHALLENGE_ISSUED,
                MessageType.RESPONSE: StreamEventType.RESPONSE_GIVEN,
                MessageType.EVIDENCE: StreamEventType.EVIDENCE_PRESENTED,
                MessageType.CONSENSUS_CHECK: StreamEventType.CONSENSUS_UPDATE,
                MessageType.FINAL_STATEMENT: StreamEventType.STAGE_CHANGED
            }
            
            event_type = event_type_map.get(message.message_type, StreamEventType.DEBATE_STARTED)
            
            stream_event = DebateStreamEvent(
                event_type=event_type,
                debate_id=getattr(message, 'debate_id', 'unknown'),
                agent_id=message.sender_id,
                data={
                    "message_type": message.message_type.value,
                    "content": message.content,
                    "confidence_score": message.confidence_score,
                    "evidence_count": len(message.evidence_sources) if message.evidence_sources else 0,
                    "round_number": message.round_number,
                    "stage": message.debate_stage.value
                },
                timestamp=message.timestamp
            )
            
            await self.broadcast_event(stream_event)
        
        # Register with debate communication hub
        #debate_communication_hub.register_broadcast_handler(handle_agent_message)
    
    async def broadcast_event(self, event: DebateStreamEvent):
        """Broadcast event to appropriate subscribers"""
        
        # Broadcast to debate subscribers
        await self.connection_manager.broadcast_to_debate(event.debate_id, event)
        
        # Process event-specific logic
        await self._process_special_events(event)
    
    async def _process_special_events(self, event: DebateStreamEvent):
        """Handle special event processing logic"""
        
        if event.event_type == StreamEventType.DEBATE_COMPLETED:
            await self._handle_debate_completion(event)
        elif event.event_type == StreamEventType.ERROR:
            await self._handle_error_event(event)
    
    async def _handle_debate_completion(self, event: DebateStreamEvent):
        """Handle debate completion event"""
        
        # Send final summary to all subscribers
        summary_event = DebateStreamEvent(
            event_type=StreamEventType.CONNECTION_STATUS,
            debate_id=event.debate_id,
            data={
                "status": "debate_completed",
                "final_results": event.data,
                "unsubscribing_in": 60  # Seconds
            }
        )
        
        await self.connection_manager.broadcast_to_debate(event.debate_id, summary_event)
        
        # Schedule cleanup
        asyncio.create_task(self._cleanup_completed_debate(event.debate_id, delay=60))
    
    async def _cleanup_completed_debate(self, debate_id: str, delay: int = 60):
        """Clean up debate subscriptions after completion"""
        
        await asyncio.sleep(delay)
        
        # Unsubscribe all connections from completed debate
        if debate_id in self.connection_manager.debate_subscribers:
            subscribers = self.connection_manager.debate_subscribers[debate_id].copy()
            
            for connection_id in subscribers:
                await self.connection_manager.unsubscribe_from_debate(connection_id, debate_id)
        
        logger.info(f"Cleaned up completed debate {debate_id}")
    
    async def _handle_error_event(self, event: DebateStreamEvent):
        """Handle error events"""
        
        logger.error(f"Debate error in {event.debate_id}: {event.data}")
        
        # Notify all subscribers of error
        error_notification = DebateStreamEvent(
            event_type=StreamEventType.CONNECTION_STATUS,
            debate_id=event.debate_id,
            data={
                "status": "error_occurred",
                "error_details": event.data,
                "recovery_actions": ["refresh_connection", "check_status"]
            }
        )
        
        await self.connection_manager.broadcast_to_debate(event.debate_id, error_notification)

# Global streaming service
streaming_service = DebateStreamingService(connection_manager)

# FastAPI WebSocket Routes
router = APIRouter()

@router.websocket("/ws/debates/{debate_id}")
async def websocket_debate_stream(
    websocket: WebSocket,
    debate_id: str,
    current_user=Depends(get_current_user)
):
    """WebSocket endpoint for streaming specific debate"""
    
    connection_id = await connection_manager.connect(
        websocket=websocket,
        user_id=current_user.id,
        debate_id=debate_id,
        connection_params={"stream_type": "specific_debate"}
    )
    
    try:
        # Send initial debate status
        await connection_manager.send_to_connection(connection_id, DebateStreamEvent(
            event_type=StreamEventType.CONNECTION_STATUS,
            debate_id=debate_id,
            data={"status": "connected_to_debate", "debate_id": debate_id}
        ))
        
        # Handle incoming messages
        while True:
            try:
                message_text = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_data = json.loads(message_text)
                await connection_manager.handle_client_message(connection_id, message_data)
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await connection_manager.send_to_connection(connection_id, DebateStreamEvent(
                    event_type=StreamEventType.HEARTBEAT,
                    debate_id=debate_id,
                    data={"type": "server_heartbeat"}
                ))
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from connection {connection_id}")
                continue
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {e}")
    finally:
        await connection_manager.disconnect(connection_id)

@router.websocket("/ws/debates/all")
async def websocket_all_debates_stream(
    websocket: WebSocket,
    current_user=Depends(get_current_user)
):
    """WebSocket endpoint for streaming all user's debates"""
    
    connection_id = await connection_manager.connect(
        websocket=websocket,
        user_id=current_user.id,
        debate_id=None,
        connection_params={"stream_type": "all_debates"}
    )
    
    try:
        # Send connection confirmation
        await connection_manager.send_to_connection(connection_id, DebateStreamEvent(
            event_type=StreamEventType.CONNECTION_STATUS,
            debate_id="global",
            data={"status": "connected_to_all_debates", "user_id": current_user.id}
        ))
        
        # Handle incoming messages
        while True:
            try:
                message_text = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                message_data = json.loads(message_text)
                await connection_manager.handle_client_message(connection_id, message_data)
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await connection_manager.send_to_connection(connection_id, DebateStreamEvent(
                    event_type=StreamEventType.HEARTBEAT,
                    debate_id="global",
                    data={"type": "server_heartbeat"}
                ))
            except json.JSONDecodeError:
                continue
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {e}")
    finally:
        await connection_manager.disconnect(connection_id)

# Utility functions for integration

async def notify_debate_start(debate_id: str, participants: List[str], query: str):
    """Notify all relevant connections that a debate has started"""
    
    event = DebateStreamEvent(
        event_type=StreamEventType.DEBATE_STARTED,
        debate_id=debate_id,
        data={
            "participants": participants,
            "query": query,
            "start_time": datetime.now().isoformat()
        }
    )
    
    await streaming_service.broadcast_event(event)

async def notify_stage_change(debate_id: str, new_stage: DebateStage, round_number: int):
    """Notify connections of debate stage changes"""
    
    event = DebateStreamEvent(
        event_type=StreamEventType.STAGE_CHANGED,
        debate_id=debate_id,
        data={
            "new_stage": new_stage.value,
            "round_number": round_number,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    await streaming_service.broadcast_event(event)

async def notify_debate_completion(debate_id: str, results: Dict[str, Any]):
    """Notify connections of debate completion"""
    
    event = DebateStreamEvent(
        event_type=StreamEventType.DEBATE_COMPLETED,
        debate_id=debate_id,
        data={
            "results": results,
            "completion_time": datetime.now().isoformat()
        }
    )
    
    await streaming_service.broadcast_event(event)

async def get_streaming_analytics() -> Dict[str, Any]:
    """Get analytics about streaming performance"""
    
    return {
        "total_connections": connection_manager.get_connection_count(),
        "active_debates": len(connection_manager.debate_subscribers),
        "total_users": len(connection_manager.user_connections),
        "connection_stats": {
            "total_messages_sent": sum(
                stats.get("messages_sent", 0) 
                for stats in connection_manager.connection_stats.values()
            ),
            "total_errors": sum(
                stats.get("errors", 0) 
                for stats in connection_manager.connection_stats.values()
            ),
        }
    }


# Health check for streaming system
async def health_check() -> Dict[str, Any]:
    """Health check for streaming system"""
    
    return {
        "status": "healthy",
        "active_connections": connection_manager.get_connection_count(),
        "active_debates": len(connection_manager.debate_subscribers),
        "last_heartbeat": datetime.now().isoformat(),
        "system_load": "normal"  # Could implement actual load monitoring
    }

# Export main components
__all__ = [
    "router", "ConnectionManager", "DebateStreamingService", "DebateStreamEvent",
    "StreamEventType", "connection_manager", "streaming_service",
    "notify_debate_start", "notify_stage_change", "notify_debate_completion",
    "get_streaming_analytics", "health_check"
]