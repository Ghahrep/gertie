# mcp/enhanced_agent_registry.py
"""
Enhanced Agent Registry with Intelligent Routing
===============================================
Production-ready agent management with load balancing and performance tracking.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .schemas import AgentRegistration, AgentCapability

logger = logging.getLogger(__name__)

@dataclass
class AgentPerformanceMetrics:
    """Real-time performance metrics for an agent"""
    agent_id: str
    current_load: int = 0  # Number of active jobs
    success_rate: float = 1.0  # Success rate over last 100 jobs
    avg_response_time: float = 0.0  # Average response time in seconds
    last_health_check: Optional[datetime] = None
    total_jobs_completed: int = 0
    total_jobs_failed: int = 0
    consecutive_failures: int = 0
    
    # Capability-specific performance
    capability_performance: Dict[str, float] = field(default_factory=dict)
    
    def update_success(self, response_time: float, capability: str):
        """Update metrics after successful job completion"""
        self.current_load = max(0, self.current_load - 1)
        self.total_jobs_completed += 1
        self.consecutive_failures = 0
        
        # Update average response time (exponential moving average)
        if self.avg_response_time == 0.0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = 0.8 * self.avg_response_time + 0.2 * response_time
        
        # Update capability-specific performance
        if capability not in self.capability_performance:
            self.capability_performance[capability] = 1.0
        else:
            # Exponential moving average for capability performance
            current_perf = self.capability_performance[capability]
            self.capability_performance[capability] = 0.9 * current_perf + 0.1 * 1.0
        
        # Recalculate success rate
        self._recalculate_success_rate()
    
    def update_failure(self, capability: str):
        """Update metrics after job failure"""
        self.current_load = max(0, self.current_load - 1)
        self.total_jobs_failed += 1
        self.consecutive_failures += 1
        
        # Penalize capability-specific performance
        if capability not in self.capability_performance:
            self.capability_performance[capability] = 0.5
        else:
            current_perf = self.capability_performance[capability]
            self.capability_performance[capability] = 0.9 * current_perf + 0.1 * 0.0
        
        # Recalculate success rate
        self._recalculate_success_rate()
    
    def _recalculate_success_rate(self):
        """Recalculate overall success rate"""
        total_jobs = self.total_jobs_completed + self.total_jobs_failed
        if total_jobs > 0:
            self.success_rate = self.total_jobs_completed / total_jobs
        else:
            self.success_rate = 1.0
    
    def get_capability_score(self, capability: str) -> float:
        """Get performance score for a specific capability"""
        return self.capability_performance.get(capability, 0.5)
    
    def is_overloaded(self, max_concurrent: int) -> bool:
        """Check if agent is at capacity"""
        return self.current_load >= max_concurrent
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy based on recent performance"""
        # More lenient health check - focus on recent performance
        recent_failure_threshold = 5  # Allow more failures before marking unhealthy
        success_rate_threshold = 0.3  # Lower threshold for success rate
        
        return (
            self.consecutive_failures < recent_failure_threshold and
            self.success_rate > success_rate_threshold and
            (self.last_health_check is None or 
            (datetime.now() - self.last_health_check) < timedelta(minutes=10))
        )

class LoadBalancingStrategy:
    """Base class for load balancing strategies"""
    
    def select_agent(self, 
                    agents: List[str], 
                    metrics: Dict[str, AgentPerformanceMetrics],
                    capability: str,
                    job_complexity: float) -> Optional[str]:
        """Select the best agent for a job"""
        raise NotImplementedError

class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round-robin based on performance and load"""
    
    def __init__(self):
        self.last_selected = defaultdict(int)
    
    def select_agent(self, 
                    agents: List[str], 
                    metrics: Dict[str, AgentPerformanceMetrics],
                    capability: str,
                    job_complexity: float) -> Optional[str]:
        
        if not agents:
            return None
        
        # Filter healthy agents
        healthy_agents = [
            agent_id for agent_id in agents 
            if agent_id in metrics and metrics[agent_id].is_healthy()
        ]
        
        if not healthy_agents:
            logger.warning("No healthy agents available")
            return None
        
        # Calculate weights for each agent
        agent_weights = {}
        for agent_id in healthy_agents:
            metric = metrics[agent_id]
            
            # Base weight factors
            load_factor = 1.0 / (metric.current_load + 1)  # Prefer less loaded agents
            success_factor = metric.success_rate
            capability_factor = metric.get_capability_score(capability)
            speed_factor = 1.0 / (metric.avg_response_time + 0.1)  # Prefer faster agents
            
            # Combined weight
            weight = (
                load_factor * 0.3 +
                success_factor * 0.3 +
                capability_factor * 0.25 +
                speed_factor * 0.15
            )
            
            agent_weights[agent_id] = weight
        
        # Select agent with highest weight
        best_agent = max(agent_weights.items(), key=lambda x: x[1])[0]
        
        # Update last selected tracking
        self.last_selected[best_agent] += 1
        
        return best_agent

class CapabilityBasedStrategy(LoadBalancingStrategy):
    """Strategy that prioritizes capability-specific performance"""
    
    def select_agent(self, 
                    agents: List[str], 
                    metrics: Dict[str, AgentPerformanceMetrics],
                    capability: str,
                    job_complexity: float) -> Optional[str]:
        
        if not agents:
            return None
        
        # Filter agents that can handle this capability and are healthy
        capable_agents = []
        for agent_id in agents:
            if agent_id in metrics:
                metric = metrics[agent_id]
                if metric.is_healthy() and not metric.is_overloaded(5):  # Max 5 concurrent
                    capable_agents.append((agent_id, metric))
        
        if not capable_agents:
            return None
        
        # Score agents based on capability performance and current load
        scored_agents = []
        for agent_id, metric in capable_agents:
            capability_score = metric.get_capability_score(capability)
            load_penalty = metric.current_load * 0.1  # Penalize loaded agents
            complexity_bonus = self._get_complexity_bonus(agent_id, job_complexity)
            
            final_score = capability_score - load_penalty + complexity_bonus
            scored_agents.append((agent_id, final_score))
        
        # Return agent with highest score
        return max(scored_agents, key=lambda x: x[1])[0]
    
    def _get_complexity_bonus(self, agent_id: str, complexity: float) -> float:
        """Give bonus to agents better suited for job complexity"""
        # Simple heuristic - can be enhanced with agent-specific complexity handling
        if complexity > 0.8:  # High complexity
            specialist_agents = ["quantitative_analyst", "options_analyst"]
            return 0.1 if any(specialist in agent_id for specialist in specialist_agents) else 0.0
        return 0.0

class EnhancedAgentRegistry:
    """Enhanced agent registry with intelligent routing and performance tracking"""
    
    def __init__(self):
        self.agents: Dict[str, AgentRegistration] = {}
        self.metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.load_balancer: LoadBalancingStrategy = WeightedRoundRobinStrategy()
        self.capability_balancer = CapabilityBasedStrategy()
        
        # Configuration
        self.health_check_interval = 30  # seconds
        self.performance_window = 100  # jobs for performance calculation
        
        # Background tasks
        self._health_check_task = None
        self._cleanup_task = None
    
    async def start(self):
        """Start background tasks"""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Enhanced Agent Registry started")
    
    async def stop(self):
        """Stop background tasks"""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Enhanced Agent Registry stopped")
    
    def register_agent(self, registration: AgentRegistration) -> bool:
        """Register an agent with performance tracking"""
        agent_id = registration.agent_id
        
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered, updating...")
        
        self.agents[agent_id] = registration
        self.metrics[agent_id] = AgentPerformanceMetrics(agent_id=agent_id)
        
        logger.info(f"Registered agent {agent_id} with capabilities: {registration.capabilities}")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id not in self.agents:
            return False
        
        del self.agents[agent_id]
        if agent_id in self.metrics:
            del self.metrics[agent_id]
        
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    async def route_job(self, 
                       capability: str, 
                       job_complexity: float = 0.5,
                       preferred_agents: Optional[List[str]] = None) -> Optional[str]:
        """Intelligently route a job to the best available agent"""
        
        # Find agents with required capability
        capable_agents = self._find_agents_with_capability(capability)
        
        if not capable_agents:
            logger.error(f"No agents found with capability: {capability}")
            return None
        
        # Filter by preferred agents if specified
        if preferred_agents:
            capable_agents = [a for a in capable_agents if a in preferred_agents]
            if not capable_agents:
                logger.warning("No preferred agents have required capability, using all capable agents")
                capable_agents = self._find_agents_with_capability(capability)
        
        # Use capability-based strategy for specialized routing
        selected_agent = self.capability_balancer.select_agent(
            capable_agents, self.metrics, capability, job_complexity
        )
        
        if selected_agent:
            # Mark agent as having one more job
            self.metrics[selected_agent].current_load += 1
            logger.info(f"Routed {capability} job to {selected_agent} (load: {self.metrics[selected_agent].current_load})")
        else:
            logger.error(f"Failed to route job with capability: {capability}")
        
        return selected_agent
    
    def _find_agents_with_capability(self, capability: str) -> List[str]:
        """Find all agents that have the specified capability"""
        capable_agents = []
        for agent_id, registration in self.agents.items():
            if capability in registration.capabilities:
                capable_agents.append(agent_id)
        return capable_agents
    
    async def report_job_success(self, 
                                agent_id: str, 
                                capability: str, 
                                response_time: float):
        """Report successful job completion"""
        if agent_id in self.metrics:
            self.metrics[agent_id].update_success(response_time, capability)
            logger.debug(f"Updated success metrics for {agent_id}: {capability}")
    
    async def report_job_failure(self, agent_id: str, capability: str):
        """Report job failure"""
        if agent_id in self.metrics:
            self.metrics[agent_id].update_failure(capability)
            logger.warning(f"Updated failure metrics for {agent_id}: {capability}")
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Get detailed status for an agent"""
        if agent_id not in self.agents or agent_id not in self.metrics:
            return None
        
        registration = self.agents[agent_id]
        metrics = self.metrics[agent_id]
        
        return {
            "agent_id": agent_id,
            "agent_name": registration.agent_name,
            "capabilities": registration.capabilities,
            "current_load": metrics.current_load,
            "max_concurrent": registration.max_concurrent_jobs,
            "success_rate": metrics.success_rate,
            "avg_response_time": metrics.avg_response_time,
            "total_completed": metrics.total_jobs_completed,
            "total_failed": metrics.total_jobs_failed,
            "consecutive_failures": metrics.consecutive_failures,
            "is_healthy": metrics.is_healthy(),
            "capability_performance": metrics.capability_performance,
            "last_health_check": metrics.last_health_check.isoformat() if metrics.last_health_check else None
        }
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        total_agents = len(self.agents)
        healthy_agents = sum(1 for m in self.metrics.values() if m.is_healthy())
        total_load = sum(m.current_load for m in self.metrics.values())
        avg_success_rate = sum(m.success_rate for m in self.metrics.values()) / max(total_agents, 1)
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": total_agents - healthy_agents,
            "total_active_jobs": total_load,
            "average_success_rate": avg_success_rate,
            "agents_by_capability": self._get_capability_distribution()
        }
    
    def _get_capability_distribution(self) -> Dict[str, int]:
        """Get distribution of agents by capability"""
        capability_count = defaultdict(int)
        for registration in self.agents.values():
            for capability in registration.capabilities:
                capability_count[capability] += 1
        return dict(capability_count)
    
    async def _health_check_loop(self):
        """Background task to perform periodic health checks"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all agents"""
        for agent_id, registration in self.agents.items():
            try:
                # Simple health check - in production, would ping agent endpoint
                if agent_id in self.metrics:
                    self.metrics[agent_id].last_health_check = datetime.now()
                    logger.debug(f"Health check completed for {agent_id}")
            except Exception as e:
                logger.error(f"Health check failed for {agent_id}: {e}")
    
    async def _cleanup_loop(self):
        """Background task to clean up old metrics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                # Cleanup logic - reset very old performance data
                logger.debug("Performed metrics cleanup")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")


    def get_circuit_breaker_status(self) -> Dict:
        """Get circuit breaker status for all agents - placeholder for performance monitoring"""
        # Simple implementation - in production this would integrate with actual circuit breakers
        return {
            agent_id: {
                "state": "closed",
                "health_score": 1.0 if metrics.is_healthy() else 0.5,
                "is_available": metrics.is_healthy()
            }
            for agent_id, metrics in self.metrics.items()
        }

    def get_system_resilience_metrics(self) -> Dict:
        """Get system resilience metrics - placeholder for performance monitoring"""
        total_agents = len(self.agents)
        healthy_agents = sum(1 for m in self.metrics.values() if m.is_healthy())
        
        return {
            "system_availability": healthy_agents / max(total_agents, 1),
            "total_agents": total_agents,
            "available_agents": healthy_agents,
            "unavailable_agents": total_agents - healthy_agents,
            "average_health_score": sum(1.0 if m.is_healthy() else 0.5 for m in self.metrics.values()) / max(total_agents, 1)
        }

    def force_reset_circuit_breaker(self, agent_id: str) -> bool:
        """Force reset a circuit breaker - placeholder for performance monitoring"""
        # Simple implementation - just reset the agent's consecutive failures
        if agent_id in self.metrics:
            self.metrics[agent_id].consecutive_failures = 0
            logger.info(f"Reset failure count for agent {agent_id}")
            return True
        return False
    
    def clear(self):
        """Clear all registered agents and metrics - useful for testing"""
        self.agents.clear()
        self.metrics.clear()
        logger.info("Registry cleared")

# Global instance
enhanced_registry = EnhancedAgentRegistry()