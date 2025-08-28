# mcp/circuit_breaker.py
"""
Circuit Breaker and Failover System
===================================
Production-ready circuit breakers to handle agent failures and provide
automatic failover with intelligent recovery.
"""

import time
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5          # Failures before opening circuit
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout_seconds: int = 60          # Time to wait before trying half-open
    reset_timeout_seconds: int = 300    # Time to reset failure count
    max_failures_window: int = 100     # Window size for failure tracking

@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring"""
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_timeouts: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: List[Dict] = field(default_factory=list)

class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker for agent reliability"""
    
    def __init__(self, agent_id: str, config: Optional[CircuitBreakerConfig] = None):
        self.agent_id = agent_id
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.failure_times: List[datetime] = []
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.stats.total_requests += 1
                raise CircuitBreakerError(f"Circuit breaker is OPEN for agent {self.agent_id}")
        
        # Attempt to execute the function
        self.stats.total_requests += 1
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful execution"""
        self.stats.total_successes += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()
        
        logger.debug(f"Circuit breaker success for {self.agent_id}: {self.stats.consecutive_successes} consecutive")
    
    async def _on_failure(self, error: Exception):
        """Handle failed execution"""
        self.stats.total_failures += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = datetime.now()
        
        # Track recent failures
        now = datetime.now()
        self.failure_times.append(now)
        
        # Clean old failures outside the window
        cutoff = now - timedelta(seconds=self.config.reset_timeout_seconds)
        self.failure_times = [ft for ft in self.failure_times if ft > cutoff]
        
        # Check if we should open the circuit
        if (self.state == CircuitState.CLOSED and 
            len(self.failure_times) >= self.config.failure_threshold):
            self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        
        logger.warning(f"Circuit breaker failure for {self.agent_id}: {len(self.failure_times)} recent failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.state != CircuitState.OPEN:
            return False
        
        time_since_open = datetime.now() - self.last_state_change
        return time_since_open.total_seconds() >= self.config.timeout_seconds
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        if self.state != CircuitState.OPEN:
            self._log_state_change(self.state, CircuitState.OPEN)
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            logger.error(f"Circuit breaker OPENED for agent {self.agent_id}")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        if self.state != CircuitState.HALF_OPEN:
            self._log_state_change(self.state, CircuitState.HALF_OPEN)
            self.state = CircuitState.HALF_OPEN
            self.last_state_change = datetime.now()
            self.stats.consecutive_successes = 0
            logger.info(f"Circuit breaker HALF_OPEN for agent {self.agent_id} - testing recovery")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        if self.state != CircuitState.CLOSED:
            self._log_state_change(self.state, CircuitState.CLOSED)
            self.state = CircuitState.CLOSED
            self.last_state_change = datetime.now()
            self.failure_times.clear()  # Reset failure history
            logger.info(f"Circuit breaker CLOSED for agent {self.agent_id} - service recovered")
    
    def _log_state_change(self, from_state: CircuitState, to_state: CircuitState):
        """Log state changes for monitoring"""
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "from_state": from_state.value,
            "to_state": to_state.value,
            "consecutive_failures": self.stats.consecutive_failures,
            "consecutive_successes": self.stats.consecutive_successes
        }
        self.stats.state_changes.append(change_record)
        
        # Keep only recent state changes
        if len(self.stats.state_changes) > 50:
            self.stats.state_changes = self.stats.state_changes[-25:]
    
    def get_health_score(self) -> float:
        """Calculate health score (0.0 = unhealthy, 1.0 = healthy)"""
        if self.stats.total_requests == 0:
            return 1.0
        
        # Base score from success rate
        success_rate = self.stats.total_successes / self.stats.total_requests
        
        # Penalty for current state
        state_penalty = {
            CircuitState.CLOSED: 0.0,
            CircuitState.HALF_OPEN: 0.2,
            CircuitState.OPEN: 0.5
        }
        
        # Recent failure penalty
        recent_failure_penalty = min(len(self.failure_times) * 0.1, 0.3)
        
        health_score = success_rate - state_penalty[self.state] - recent_failure_penalty
        return max(0.0, min(1.0, health_score))
    
    def is_available(self) -> bool:
        """Check if agent is available for requests"""
        return self.state != CircuitState.OPEN
    
    def get_stats(self) -> Dict:
        """Get comprehensive circuit breaker statistics"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "health_score": self.get_health_score(),
            "is_available": self.is_available(),
            "stats": {
                "total_requests": self.stats.total_requests,
                "total_failures": self.stats.total_failures,
                "total_successes": self.stats.total_successes,
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "recent_failures": len(self.failure_times)
            },
            "last_state_change": self.last_state_change.isoformat(),
            "recent_state_changes": self.stats.state_changes[-5:] if self.stats.state_changes else []
        }

class FailoverManager:
    """Manages failover between agents with circuit breaker integration"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failover_chains: Dict[str, List[str]] = {}  # capability -> ordered list of agents
        self.agent_priorities: Dict[str, Dict[str, int]] = {}  # agent_id -> {capability: priority}
    
    def register_agent_for_failover(self, agent_id: str, capabilities: List[str], priorities: Optional[Dict[str, int]] = None):
        """Register an agent for failover management"""
        
        # Create circuit breaker
        self.circuit_breakers[agent_id] = CircuitBreaker(agent_id)
        
        # Set up priorities
        agent_priorities = priorities or {}
        self.agent_priorities[agent_id] = {}
        
        for capability in capabilities:
            priority = agent_priorities.get(capability, 5)  # Default priority
            self.agent_priorities[agent_id][capability] = priority
            
            # Add to failover chain
            if capability not in self.failover_chains:
                self.failover_chains[capability] = []
            
            if agent_id not in self.failover_chains[capability]:
                self.failover_chains[capability].append(agent_id)
        
        # Sort failover chains by priority
        self._sort_failover_chains()
        
        logger.info(f"Registered {agent_id} for failover with capabilities: {capabilities}")
    
    def _sort_failover_chains(self):
        """Sort failover chains by agent priority"""
        for capability, agents in self.failover_chains.items():
            agents.sort(key=lambda agent_id: self.agent_priorities.get(agent_id, {}).get(capability, 5))
    
    async def execute_with_failover(self, capability: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with automatic failover"""
        
        if capability not in self.failover_chains:
            raise ValueError(f"No agents registered for capability: {capability}")
        
        available_agents = self.failover_chains[capability]
        last_error = None
        
        for agent_id in available_agents:
            circuit_breaker = self.circuit_breakers[agent_id]
            
            if not circuit_breaker.is_available():
                logger.warning(f"Skipping {agent_id} - circuit breaker is OPEN")
                continue
            
            try:
                logger.debug(f"Attempting {capability} with {agent_id}")
                
                # Execute with circuit breaker protection
                result = await circuit_breaker.call(func, agent_id, *args, **kwargs)
                
                logger.info(f"Successfully executed {capability} with {agent_id}")
                return result
                
            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker prevented execution on {agent_id}: {e}")
                last_error = e
                continue
                
            except Exception as e:
                logger.error(f"Execution failed on {agent_id}: {e}")
                last_error = e
                continue
        
        # All agents failed
        raise Exception(f"All agents failed for capability {capability}. Last error: {last_error}")
    
    def get_failover_status(self, capability: str) -> Dict:
        """Get failover status for a capability"""
        
        if capability not in self.failover_chains:
            return {"error": f"No agents registered for {capability}"}
        
        agents = self.failover_chains[capability]
        agent_statuses = []
        
        for agent_id in agents:
            if agent_id in self.circuit_breakers:
                cb_stats = self.circuit_breakers[agent_id].get_stats()
                priority = self.agent_priorities.get(agent_id, {}).get(capability, 5)
                
                agent_statuses.append({
                    "agent_id": agent_id,
                    "priority": priority,
                    "circuit_breaker": cb_stats
                })
        
        available_agents = [s for s in agent_statuses if s["circuit_breaker"]["is_available"]]
        
        return {
            "capability": capability,
            "total_agents": len(agent_statuses),
            "available_agents": len(available_agents),
            "primary_agent": available_agents[0]["agent_id"] if available_agents else None,
            "agent_details": agent_statuses
        }
    
    def get_system_resilience_metrics(self) -> Dict:
        """Get overall system resilience metrics"""
        
        total_agents = len(self.circuit_breakers)
        available_agents = sum(1 for cb in self.circuit_breakers.values() if cb.is_available())
        
        # Calculate capability coverage
        capability_coverage = {}
        for capability, agents in self.failover_chains.items():
            available_for_capability = sum(
                1 for agent_id in agents 
                if agent_id in self.circuit_breakers and self.circuit_breakers[agent_id].is_available()
            )
            capability_coverage[capability] = {
                "total_agents": len(agents),
                "available_agents": available_for_capability,
                "coverage_ratio": available_for_capability / len(agents) if agents else 0.0
            }
        
        # Overall health scores
        health_scores = [cb.get_health_score() for cb in self.circuit_breakers.values()]
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.0
        
        return {
            "system_availability": available_agents / total_agents if total_agents else 0.0,
            "total_agents": total_agents,
            "available_agents": available_agents,
            "unavailable_agents": total_agents - available_agents,
            "average_health_score": avg_health,
            "capability_coverage": capability_coverage,
            "total_capabilities": len(self.failover_chains)
        }
    
    def force_circuit_reset(self, agent_id: str) -> bool:
        """Force reset a circuit breaker (for admin/debugging)"""
        if agent_id in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[agent_id]
            circuit_breaker._transition_to_closed()
            logger.info(f"Manually reset circuit breaker for {agent_id}")
            return True
        return False
    
    def get_all_circuit_breaker_stats(self) -> Dict[str, Dict]:
        """Get stats for all circuit breakers"""
        return {
            agent_id: cb.get_stats() 
            for agent_id, cb in self.circuit_breakers.items()
        }

# Global failover manager instance
failover_manager = FailoverManager()