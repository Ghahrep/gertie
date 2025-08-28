# mcp/performance_dashboard.py
"""
Performance Monitoring Dashboard
===============================
Real-time monitoring dashboard for agent performance, circuit breakers,
and system health with web-based visualization.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from typing import Dict, List, Set
from datetime import datetime, timedelta
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, enhanced_registry):
        self.registry = enhanced_registry
        self.websocket_connections: Set[WebSocket] = set()
        self.monitoring_active = False
        self.monitoring_task = None
        self.update_interval = 2.0  # seconds
        
    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Performance monitoring stopped")
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add new WebSocket connection"""
        await websocket.accept()
        self.websocket_connections.add(websocket)
        
        # Send initial data
        await self._send_initial_data(websocket)
        logger.info(f"WebSocket connection added. Total: {len(self.websocket_connections)}")
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.websocket_connections.discard(websocket)
        logger.info(f"WebSocket connection removed. Total: {len(self.websocket_connections)}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect performance data
                performance_data = await self._collect_performance_data()
                
                # Broadcast to all connected clients
                if self.websocket_connections:
                    await self._broadcast_data(performance_data)
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_performance_data(self) -> Dict:
        """Collect comprehensive performance data"""
        
        # System overview
        system_status = self.registry.get_system_status()
        
        # Individual agent performance
        agent_details = []
        for agent_id in self.registry.agents.keys():
            agent_status = self.registry.get_agent_status(agent_id)
            if agent_status:
                agent_details.append(agent_status)
        
        # Circuit breaker status
        circuit_status = self.registry.get_circuit_breaker_status()
        
        # System resilience metrics
        resilience_metrics = self.registry.get_system_resilience_metrics()
        
        # Calculate trending metrics
        trending_data = self._calculate_trending_metrics(agent_details)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_overview": system_status,
            "agent_performance": agent_details,
            "circuit_breakers": circuit_status,
            "resilience_metrics": resilience_metrics,
            "trending": trending_data,
            "alerts": self._generate_performance_alerts(agent_details, system_status)
        }
    
    def _calculate_trending_metrics(self, agent_details: List[Dict]) -> Dict:
        """Calculate trending performance metrics"""
        
        if not agent_details:
            return {"success_rate_trend": [], "response_time_trend": [], "load_trend": []}
        
        # Simple trending - in production would use time series data
        current_time = datetime.now()
        
        avg_success_rate = sum(a["success_rate"] for a in agent_details) / len(agent_details)
        avg_response_time = sum(a["avg_response_time"] for a in agent_details) / len(agent_details)
        total_load = sum(a["current_load"] for a in agent_details)
        
        return {
            "success_rate_trend": [
                {"time": (current_time - timedelta(minutes=i)).isoformat(), "value": avg_success_rate}
                for i in range(10, 0, -1)
            ],
            "response_time_trend": [
                {"time": (current_time - timedelta(minutes=i)).isoformat(), "value": avg_response_time}
                for i in range(10, 0, -1)
            ],
            "load_trend": [
                {"time": (current_time - timedelta(minutes=i)).isoformat(), "value": total_load}
                for i in range(10, 0, -1)
            ]
        }
    
    def _generate_performance_alerts(self, agent_details: List[Dict], system_status: Dict) -> List[Dict]:
        """Generate performance alerts"""
        alerts = []
        
        # System-level alerts
        if system_status["unhealthy_agents"] > 0:
            alerts.append({
                "level": "warning" if system_status["unhealthy_agents"] == 1 else "error",
                "message": f"{system_status['unhealthy_agents']} unhealthy agents detected",
                "timestamp": datetime.now().isoformat(),
                "type": "system_health"
            })
        
        if system_status["average_success_rate"] < 0.8:
            alerts.append({
                "level": "warning",
                "message": f"System success rate below 80%: {system_status['average_success_rate']:.1%}",
                "timestamp": datetime.now().isoformat(),
                "type": "performance"
            })
        
        # Agent-level alerts
        for agent in agent_details:
            if agent["consecutive_failures"] >= 3:
                alerts.append({
                    "level": "error",
                    "message": f"Agent {agent['agent_id']} has {agent['consecutive_failures']} consecutive failures",
                    "timestamp": datetime.now().isoformat(),
                    "type": "agent_failure"
                })
            
            if agent["current_load"] >= agent["max_concurrent"]:
                alerts.append({
                    "level": "warning", 
                    "message": f"Agent {agent['agent_id']} at maximum capacity ({agent['current_load']}/{agent['max_concurrent']})",
                    "timestamp": datetime.now().isoformat(),
                    "type": "capacity"
                })
        
        return alerts
    
    async def _send_initial_data(self, websocket: WebSocket):
        """Send initial dashboard data to new connection"""
        try:
            initial_data = await self._collect_performance_data()
            await websocket.send_text(json.dumps({
                "type": "initial_data",
                "data": initial_data
            }))
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def _broadcast_data(self, data: Dict):
        """Broadcast data to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = json.dumps({
            "type": "performance_update",
            "data": data
        })
        
        # Send to all connections, remove failed ones
        failed_connections = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send to WebSocket: {e}")
                failed_connections.add(websocket)
        
        # Remove failed connections
        self.websocket_connections -= failed_connections
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary for API"""
        system_status = self.registry.get_system_status()
        
        # Calculate key metrics
        healthy_ratio = system_status["healthy_agents"] / max(system_status["total_agents"], 1)
        load_per_agent = system_status["total_active_jobs"] / max(system_status["healthy_agents"], 1)
        
        return {
            "system_health": "healthy" if healthy_ratio > 0.8 else "degraded" if healthy_ratio > 0.5 else "critical",
            "total_agents": system_status["total_agents"],
            "healthy_agents": system_status["healthy_agents"],
            "success_rate": system_status["average_success_rate"],
            "active_jobs": system_status["total_active_jobs"],
            "avg_load_per_agent": load_per_agent,
            "timestamp": datetime.now().isoformat()
        }

def create_dashboard_html() -> str:
    """Create HTML dashboard for performance monitoring"""
    
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent Performance Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric { display: flex; justify-content: space-between; align-items: center; margin: 10px 0; }
            .metric-value { font-size: 24px; font-weight: bold; }
            .healthy { color: #28a745; }
            .warning { color: #ffc107; }
            .error { color: #dc3545; }
            .chart-container { height: 300px; }
            .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
            .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; }
            .alert-error { background: #f8d7da; border: 1px solid #f5c6cb; }
            #connection-status { position: fixed; top: 10px; right: 10px; padding: 5px 10px; border-radius: 4px; }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div id="connection-status" class="disconnected">Connecting...</div>
        
        <h1>Agent Performance Dashboard</h1>
        
        <div class="dashboard">
            <!-- System Overview -->
            <div class="card">
                <h3>System Overview</h3>
                <div class="metric">
                    <span>Total Agents:</span>
                    <span id="total-agents" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Healthy Agents:</span>
                    <span id="healthy-agents" class="metric-value healthy">-</span>
                </div>
                <div class="metric">
                    <span>Success Rate:</span>
                    <span id="success-rate" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Active Jobs:</span>
                    <span id="active-jobs" class="metric-value">-</span>
                </div>
            </div>
            
            <!-- Circuit Breaker Status -->
            <div class="card">
                <h3>Circuit Breaker Status</h3>
                <div id="circuit-breaker-list">
                    <!-- Circuit breakers will be populated here -->
                </div>
            </div>
            
            <!-- Performance Trends -->
            <div class="card">
                <h3>Success Rate Trend</h3>
                <div class="chart-container">
                    <canvas id="success-rate-chart"></canvas>
                </div>
            </div>
            
            <!-- Load Distribution -->
            <div class="card">
                <h3>Agent Load Distribution</h3>
                <div class="chart-container">
                    <canvas id="load-chart"></canvas>
                </div>
            </div>
            
            <!-- Alerts -->
            <div class="card">
                <h3>Active Alerts</h3>
                <div id="alerts-container">
                    <!-- Alerts will be populated here -->
                </div>
            </div>
            
            <!-- Agent Details -->
            <div class="card">
                <h3>Agent Details</h3>
                <div id="agent-details">
                    <!-- Agent details will be populated here -->
                </div>
            </div>
        </div>
        
        <script>
            class PerformanceDashboard {
                constructor() {
                    this.websocket = null;
                    this.charts = {};
                    this.connect();
                    this.initializeCharts();
                }
                
                connect() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/performance`;
                    
                    this.websocket = new WebSocket(wsUrl);
                    
                    this.websocket.onopen = () => {
                        document.getElementById('connection-status').textContent = 'Connected';
                        document.getElementById('connection-status').className = 'connected';
                    };
                    
                    this.websocket.onclose = () => {
                        document.getElementById('connection-status').textContent = 'Disconnected';
                        document.getElementById('connection-status').className = 'disconnected';
                        // Reconnect after 5 seconds
                        setTimeout(() => this.connect(), 5000);
                    };
                    
                    this.websocket.onmessage = (event) => {
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    };
                }
                
                handleMessage(message) {
                    if (message.type === 'initial_data' || message.type === 'performance_update') {
                        this.updateDashboard(message.data);
                    }
                }
                
                initializeCharts() {
                    // Success Rate Chart
                    const successCtx = document.getElementById('success-rate-chart').getContext('2d');
                    this.charts.successRate = new Chart(successCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Success Rate %',
                                data: [],
                                borderColor: '#28a745',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: { y: { beginAtZero: true, max: 100 } }
                        }
                    });
                    
                    // Load Distribution Chart
                    const loadCtx = document.getElementById('load-chart').getContext('2d');
                    this.charts.loadDistribution = new Chart(loadCtx, {
                        type: 'bar',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Current Load',
                                data: [],
                                backgroundColor: '#007bff'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false
                        }
                    });
                }
                
                updateDashboard(data) {
                    // Update system overview
                    document.getElementById('total-agents').textContent = data.system_overview.total_agents;
                    document.getElementById('healthy-agents').textContent = data.system_overview.healthy_agents;
                    document.getElementById('success-rate').textContent = 
                        (data.system_overview.average_success_rate * 100).toFixed(1) + '%';
                    document.getElementById('active-jobs').textContent = data.system_overview.total_active_jobs;
                    
                    // Update success rate chart
                    if (data.trending && data.trending.success_rate_trend) {
                        const labels = data.trending.success_rate_trend.map(point => 
                            new Date(point.time).toLocaleTimeString()
                        );
                        const values = data.trending.success_rate_trend.map(point => point.value * 100);
                        
                        this.charts.successRate.data.labels = labels;
                        this.charts.successRate.data.datasets[0].data = values;
                        this.charts.successRate.update('none');
                    }
                    
                    // Update load distribution chart
                    if (data.agent_performance) {
                        const labels = data.agent_performance.map(agent => agent.agent_id);
                        const loads = data.agent_performance.map(agent => agent.current_load);
                        
                        this.charts.loadDistribution.data.labels = labels;
                        this.charts.loadDistribution.data.datasets[0].data = loads;
                        this.charts.loadDistribution.update('none');
                    }
                    
                    // Update alerts
                    this.updateAlerts(data.alerts || []);
                    
                    // Update circuit breaker status
                    this.updateCircuitBreakers(data.circuit_breakers || {});
                    
                    // Update agent details
                    this.updateAgentDetails(data.agent_performance || []);
                }
                
                updateAlerts(alerts) {
                    const container = document.getElementById('alerts-container');
                    if (alerts.length === 0) {
                        container.innerHTML = '<p>No active alerts</p>';
                        return;
                    }
                    
                    container.innerHTML = alerts.map(alert => 
                        `<div class="alert alert-${alert.level}">
                            <strong>${alert.type}:</strong> ${alert.message}
                         </div>`
                    ).join('');
                }
                
                updateCircuitBreakers(circuitBreakers) {
                    const container = document.getElementById('circuit-breaker-list');
                    const entries = Object.entries(circuitBreakers);
                    
                    if (entries.length === 0) {
                        container.innerHTML = '<p>No circuit breakers active</p>';
                        return;
                    }
                    
                    container.innerHTML = entries.map(([agentId, status]) => 
                        `<div class="metric">
                            <span>${agentId}:</span>
                            <span class="${status.is_available ? 'healthy' : 'error'}">
                                ${status.state} (${(status.health_score * 100).toFixed(0)}%)
                            </span>
                         </div>`
                    ).join('');
                }
                
                updateAgentDetails(agents) {
                    const container = document.getElementById('agent-details');
                    
                    if (agents.length === 0) {
                        container.innerHTML = '<p>No agents registered</p>';
                        return;
                    }
                    
                    container.innerHTML = agents.map(agent => 
                        `<div style="margin: 10px 0; padding: 10px; border-left: 3px solid ${agent.is_healthy ? '#28a745' : '#dc3545'}">
                            <strong>${agent.agent_id}</strong><br>
                            Load: ${agent.current_load}/${agent.max_concurrent} | 
                            Success: ${(agent.success_rate * 100).toFixed(1)}% | 
                            Response: ${agent.avg_response_time.toFixed(2)}s
                         </div>`
                    ).join('');
                }
            }
            
            // Initialize dashboard when page loads
            document.addEventListener('DOMContentLoaded', () => {
                new PerformanceDashboard();
            });
        </script>
    </body>
    </html>
    """

# Global performance monitor instance
performance_monitor = None

def setup_performance_monitoring(enhanced_registry):
    """Setup performance monitoring with the enhanced registry"""
    global performance_monitor
    performance_monitor = PerformanceMonitor(enhanced_registry)
    return performance_monitor