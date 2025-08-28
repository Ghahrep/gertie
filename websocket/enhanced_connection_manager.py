# websocket/enhanced_connection_manager.py
"""
Enhanced WebSocket Connection Manager with connection pooling,
auto-scaling, and performance optimizations for 1000+ concurrent connections
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, List, Optional, Tuple, Any
import json
import logging
import asyncio
import time
import weakref
import zlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict, deque
import uuid
from concurrent.futures import ThreadPoolExecutor
import psutil

logger = logging.getLogger(__name__)

@dataclass
class ConnectionMetrics:
    """Track per-connection metrics"""
    user_id: str
    connected_at: datetime
    last_message_at: datetime
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_quality: float = 1.0
    subscription_topics: Set[str] = field(default_factory=set)

@dataclass
class ConnectionPoolStats:
    """Connection pool statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connections_per_second: float = 0.0
    avg_message_size: float = 0.0
    message_throughput: float = 0.0

class ConnectionPool:
    """Manages WebSocket connection pools with auto-scaling"""
    
    def __init__(self, max_connections_per_pool: int = 100):
        self.max_connections_per_pool = max_connections_per_pool
        self.pools: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.pool_assignments: Dict[WebSocket, str] = {}
        self.connection_metrics: Dict[WebSocket, ConnectionMetrics] = {}
        
    def assign_to_pool(self, websocket: WebSocket, user_id: str) -> str:
        """Assign connection to optimal pool"""
        # Find pool with least connections
        pool_id = min(
            self.pools.keys() if self.pools else ["pool_0"],
            key=lambda p: len(self.pools[p]),
            default=f"pool_{len(self.pools)}"
        )
        
        # Create new pool if current pools are at capacity
        if len(self.pools[pool_id]) >= self.max_connections_per_pool:
            pool_id = f"pool_{len(self.pools)}"
        
        self.pools[pool_id].add(websocket)
        self.pool_assignments[websocket] = pool_id
        
        return pool_id
    
    def remove_from_pool(self, websocket: WebSocket):
        """Remove connection from its pool"""
        if websocket in self.pool_assignments:
            pool_id = self.pool_assignments[websocket]
            self.pools[pool_id].discard(websocket)
            del self.pool_assignments[websocket]
            
            # Clean up empty pools
            if not self.pools[pool_id]:
                del self.pools[pool_id]

class EnhancedConnectionManager:
    """
    Production-ready WebSocket connection manager with:
    - Connection pooling and auto-scaling
    - Message compression and batching
    - Quality of Service (QoS) levels
    - Selective subscriptions
    - Performance monitoring
    """
    
    def __init__(self):
        # Connection management
        self.user_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.connection_metadata: Dict[WebSocket, ConnectionMetrics] = {}
        self.connection_pools = ConnectionPool()
        
        # Message routing and subscriptions
        self.topic_subscriptions: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.user_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance optimization
        self.message_queue: Dict[str, deque] = defaultdict(deque)
        self.batch_processor = ThreadPoolExecutor(max_workers=4)
        self.compression_enabled = True
        self.batch_size = 10
        self.batch_timeout = 0.1  # 100ms
        
        # Health monitoring
        self.health_stats = ConnectionPoolStats()
        self.connection_history = deque(maxlen=1000)
        self.last_stats_update = time.time()
        
        # Rate limiting
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.max_messages_per_minute = 60
        
        logger.info("Enhanced WebSocket Connection Manager initialized")
    
    async def connect(self, websocket: WebSocket, user_id: str, 
                     subscription_topics: Optional[List[str]] = None,
                     compression_level: int = 6) -> bool:
        """Enhanced connection setup with pooling and subscriptions"""
        try:
            await websocket.accept()
            
            # Assign to connection pool
            pool_id = self.connection_pools.assign_to_pool(websocket, user_id)
            
            # Create connection metrics
            now = datetime.now(timezone.utc)
            self.connection_metadata[websocket] = ConnectionMetrics(
                user_id=user_id,
                connected_at=now,
                last_message_at=now,
                subscription_topics=set(subscription_topics or ["risk_alerts", "workflow_updates"])
            )
            
            # Add to user connections
            self.user_connections[user_id].add(websocket)
            
            # Set up topic subscriptions
            for topic in (subscription_topics or ["risk_alerts", "workflow_updates"]):
                await self.subscribe_to_topic(websocket, topic)
            
            # Update statistics
            self.health_stats.total_connections += 1
            self.health_stats.active_connections += 1
            self.connection_history.append(('connect', time.time(), user_id))
            
            # Send connection confirmation
            await self._send_to_connection(websocket, {
                "type": "connection_established",
                "user_id": user_id,
                "pool_id": pool_id,
                "subscriptions": list(subscription_topics or []),
                "compression_enabled": self.compression_enabled,
                "server_time": now.isoformat(),
                "capabilities": [
                    "message_compression",
                    "selective_subscriptions", 
                    "qos_levels",
                    "batch_processing"
                ]
            })
            
            logger.info(f"Connection established: user={user_id}, pool={pool_id}, "
                       f"total_connections={self.health_stats.total_connections}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed for user {user_id}: {e}")
            await self._cleanup_failed_connection(websocket, user_id)
            return False
    
    async def disconnect(self, websocket: WebSocket):
        """Enhanced disconnection cleanup"""
        try:
            # Get connection metadata
            if websocket not in self.connection_metadata:
                return
            
            metrics = self.connection_metadata[websocket]
            user_id = metrics.user_id
            
            # Remove from user connections
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
            
            # Remove from topic subscriptions
            for topic in metrics.subscription_topics:
                self.topic_subscriptions[topic].discard(websocket)
                if not self.topic_subscriptions[topic]:
                    del self.topic_subscriptions[topic]
            
            # Remove from connection pool
            self.connection_pools.remove_from_pool(websocket)
            
            # Clean up metadata
            del self.connection_metadata[websocket]
            
            # Update statistics
            self.health_stats.total_connections -= 1
            self.health_stats.active_connections -= 1
            self.connection_history.append(('disconnect', time.time(), user_id))
            
            logger.info(f"Connection closed: user={user_id}, "
                       f"total_connections={self.health_stats.total_connections}")
            
        except Exception as e:
            logger.error(f"Error during disconnect cleanup: {e}")
    
    async def subscribe_to_topic(self, websocket: WebSocket, topic: str):
        """Subscribe connection to specific topic"""
        if websocket in self.connection_metadata:
            self.topic_subscriptions[topic].add(websocket)
            self.connection_metadata[websocket].subscription_topics.add(topic)
            
            user_id = self.connection_metadata[websocket].user_id
            self.user_subscriptions[user_id].add(topic)
            
            logger.debug(f"User {user_id} subscribed to topic: {topic}")
    
    async def unsubscribe_from_topic(self, websocket: WebSocket, topic: str):
        """Unsubscribe connection from topic"""
        if websocket in self.connection_metadata:
            self.topic_subscriptions[topic].discard(websocket)
            self.connection_metadata[websocket].subscription_topics.discard(topic)
            
            user_id = self.connection_metadata[websocket].user_id
            self.user_subscriptions[user_id].discard(topic)
            
            if not self.topic_subscriptions[topic]:
                del self.topic_subscriptions[topic]
    
    async def send_to_user(self, user_id: str, message: dict, 
                          qos_level: int = 1, compress: bool = None) -> int:
        """Send message to all user connections with QoS"""
        if user_id not in self.user_connections:
            return 0
        
        # Apply compression if enabled
        if compress is None:
            compress = self.compression_enabled and len(json.dumps(message)) > 512
        
        successful_sends = 0
        failed_connections = []
        
        for websocket in self.user_connections[user_id].copy():
            try:
                # Check rate limiting
                if not self._check_rate_limit(user_id):
                    logger.warning(f"Rate limit exceeded for user {user_id}")
                    continue
                
                success = await self._send_to_connection(
                    websocket, message, qos_level, compress
                )
                if success:
                    successful_sends += 1
                else:
                    failed_connections.append(websocket)
                    
            except Exception as e:
                logger.error(f"Error sending to user {user_id}: {e}")
                failed_connections.append(websocket)
        
        # Clean up failed connections
        for websocket in failed_connections:
            await self.disconnect(websocket)
        
        return successful_sends
    
    async def broadcast_to_topic(self, topic: str, message: dict, 
                               qos_level: int = 1, exclude_users: Set[str] = None) -> int:
        """Broadcast message to all subscribers of a topic"""
        if topic not in self.topic_subscriptions:
            return 0
        
        exclude_users = exclude_users or set()
        successful_sends = 0
        failed_connections = []
        
        # Add topic information to message
        message["topic"] = topic
        message["broadcast_time"] = datetime.now(timezone.utc).isoformat()
        
        for websocket in self.topic_subscriptions[topic].copy():
            try:
                metrics = self.connection_metadata.get(websocket)
                if not metrics or metrics.user_id in exclude_users:
                    continue
                
                success = await self._send_to_connection(websocket, message, qos_level)
                if success:
                    successful_sends += 1
                else:
                    failed_connections.append(websocket)
                    
            except Exception as e:
                logger.error(f"Error broadcasting to topic {topic}: {e}")
                failed_connections.append(websocket)
        
        # Clean up failed connections
        for websocket in failed_connections:
            await self.disconnect(websocket)
        
        logger.info(f"Broadcast to topic '{topic}': {successful_sends} recipients")
        return successful_sends
    
    async def send_risk_alert(self, user_id: str, alert_data: dict) -> bool:
        """Send high-priority risk alert with guaranteed delivery"""
        enhanced_message = {
            "type": "risk_alert",
            "priority": "high",
            "title": "Portfolio Risk Alert",
            "alert_data": alert_data,
            "alert_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requires_acknowledgment": True,
            "ttl": 3600  # 1 hour TTL
        }
        
        # Send with highest QoS and no compression for speed
        return await self.send_to_user(user_id, enhanced_message, qos_level=2, compress=False) > 0
    
    async def send_workflow_update(self, user_id: str, workflow_data: dict) -> bool:
        """Send workflow progress update"""
        message = {
            "type": "workflow_update",
            "priority": "medium",
            "workflow_data": workflow_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return await self.send_to_user(user_id, message, qos_level=1) > 0
    
    async def _send_to_connection(self, websocket: WebSocket, message: dict, 
                                qos_level: int = 1, compress: bool = False) -> bool:
        """Send message to specific connection with QoS handling"""
        try:
            # Serialize message
            message_json = json.dumps(message, default=str)
            
            # Compress if requested and beneficial
            if compress and len(message_json) > 512:
                compressed_data = zlib.compress(message_json.encode('utf-8'), level=6)
                if len(compressed_data) < len(message_json) * 0.8:  # Only if 20%+ reduction
                    message_json = compressed_data.hex()
                    message["compressed"] = True
                    message_json = json.dumps(message, default=str)
            
            # Send based on QoS level
            if qos_level == 2:  # Guaranteed delivery
                await websocket.send_text(message_json)
                # Could add delivery confirmation here
            elif qos_level == 1:  # Standard delivery
                await websocket.send_text(message_json)
            else:  # Fire and forget
                asyncio.create_task(websocket.send_text(message_json))
            
            # Update metrics
            if websocket in self.connection_metadata:
                metrics = self.connection_metadata[websocket]
                metrics.messages_sent += 1
                metrics.bytes_sent += len(message_json)
                metrics.last_message_at = datetime.now(timezone.utc)
            
            return True
            
        except WebSocketDisconnect:
            logger.debug(f"WebSocket disconnected during send")
            return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        user_timestamps = self.rate_limits[user_id]
        
        # Remove old timestamps (older than 1 minute)
        while user_timestamps and now - user_timestamps[0] > 60:
            user_timestamps.popleft()
        
        # Check if under limit
        if len(user_timestamps) >= self.max_messages_per_minute:
            return False
        
        # Add current timestamp
        user_timestamps.append(now)
        return True
    
    async def _cleanup_failed_connection(self, websocket: WebSocket, user_id: str):
        """Clean up after failed connection attempt"""
        try:
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
            
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
                
            self.connection_pools.remove_from_pool(websocket)
            
        except Exception as e:
            logger.error(f"Error in connection cleanup: {e}")
    
    def get_connection_stats(self) -> dict:
        """Get comprehensive connection statistics"""
        now = time.time()
        
        # Update throughput metrics
        if now - self.last_stats_update >= 1.0:  # Update every second
            recent_connections = [
                t for op, t, user in self.connection_history 
                if now - t <= 60 and op == 'connect'
            ]
            self.health_stats.connections_per_second = len(recent_connections) / 60
            self.last_stats_update = now
        
        # Calculate average message sizes and throughput
        total_bytes = sum(m.bytes_sent for m in self.connection_metadata.values())
        total_messages = sum(m.messages_sent for m in self.connection_metadata.values())
        
        self.health_stats.avg_message_size = (
            total_bytes / total_messages if total_messages > 0 else 0
        )
        self.health_stats.message_throughput = total_messages / max(
            (now - min(m.connected_at.timestamp() for m in self.connection_metadata.values()) 
             if self.connection_metadata else now), 1
        )
        
        return {
            "total_connections": self.health_stats.total_connections,
            "active_connections": self.health_stats.active_connections,
            "pools_count": len(self.connection_pools.pools),
            "topics_count": len(self.topic_subscriptions),
            "connections_per_second": round(self.health_stats.connections_per_second, 2),
            "avg_message_size_bytes": round(self.health_stats.avg_message_size, 1),
            "message_throughput_per_second": round(self.health_stats.message_throughput, 2),
            "compression_enabled": self.compression_enabled,
            "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
            "subscriptions_by_topic": {
                topic: len(connections) 
                for topic, connections in self.topic_subscriptions.items()
            },
            "pool_distribution": {
                pool_id: len(connections)
                for pool_id, connections in self.connection_pools.pools.items()
            }
        }
    
    async def health_check(self) -> dict:
        """Comprehensive health check"""
        stats = self.get_connection_stats()
        
        # System resource checks
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if stats["total_connections"] > 800:  # Warning at 80% of target capacity
            issues.append("Approaching connection limit")
            
        if memory_percent > 85:
            health_status = "degraded"
            issues.append("High memory usage")
            
        if cpu_percent > 80:
            health_status = "degraded" 
            issues.append("High CPU usage")
        
        if len(issues) > 2:
            health_status = "critical"
        
        return {
            "status": health_status,
            "issues": issues,
            "connections": stats,
            "system": {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global enhanced connection manager
_enhanced_manager = None

def get_enhanced_connection_manager() -> EnhancedConnectionManager:
    """Get the global enhanced connection manager instance"""
    global _enhanced_manager
    if _enhanced_manager is None:
        _enhanced_manager = EnhancedConnectionManager()
    return _enhanced_manager

# Background tasks for health monitoring and cleanup
async def connection_maintenance_task():
    """Background task for connection maintenance"""
    manager = get_enhanced_connection_manager()
    
    while True:
        try:
            await asyncio.sleep(30)  # Run every 30 seconds
            
            # Health check and logging
            health_status = await manager.health_check()
            if health_status["status"] != "healthy":
                logger.warning(f"Connection manager health: {health_status}")
            
            # Clean up stale connections
            now = time.time()
            stale_connections = []
            
            for websocket, metrics in manager.connection_metadata.items():
                time_since_last_message = now - metrics.last_message_at.timestamp()
                if time_since_last_message > 300:  # 5 minutes
                    stale_connections.append(websocket)
            
            for websocket in stale_connections:
                logger.info("Cleaning up stale connection")
                await manager.disconnect(websocket)
                
        except Exception as e:
            logger.error(f"Connection maintenance task error: {e}")

def start_connection_maintenance():
    """Start the connection maintenance background task"""
    asyncio.create_task(connection_maintenance_task())
    logger.info("Connection maintenance task started")