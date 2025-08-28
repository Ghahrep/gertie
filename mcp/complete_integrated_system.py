# mcp/complete_integrated_system.py
"""
Complete Integrated Debate System
=================================
Simplified integrated system that works with your existing database infrastructure.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Simplified imports with error handling
try:
    from .enhanced_agent_registry import enhanced_registry
except ImportError as e:
    logger.warning(f"enhanced_agent_registry import failed: {e}")
    enhanced_registry = None

try:
    from .consensus_builder import ConsensusBuilder
except ImportError as e:
    logger.warning(f"ConsensusBuilder import failed: {e}")
    ConsensusBuilder = None

try:
    from .circuit_breaker import failover_manager, CircuitBreakerError
except ImportError as e:
    logger.warning(f"Circuit breaker import failed: {e}")
    failover_manager = None
    CircuitBreakerError = Exception

class CompleteIntegratedDebateSystem:
    """Simplified integrated system for database-backed debates"""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.consensus_builder = ConsensusBuilder() if ConsensusBuilder else None
        
        # Active debate tracking
        self.active_debates: Dict[str, Dict] = {}
        
        # Initialize circuit breakers if available
        if failover_manager:
            self._setup_circuit_breakers()
    
    def _setup_circuit_breakers(self):
        """Initialize circuit breakers for agent types"""
        
        agent_capabilities = {
            "quantitative_analyst": ["risk_analysis", "portfolio_optimization", "statistical_modeling"],
            "market_intelligence": ["market_timing", "trend_analysis", "sentiment_analysis"],
            "tax_strategist": ["tax_optimization", "tax_loss_harvesting", "after_tax_analysis"],
            "options_analyst": ["options_pricing", "volatility_analysis", "hedge_strategies"],
            "economic_data": ["macro_analysis", "economic_forecasting", "fed_policy_analysis"],
            "security_screener": ["security_screening", "factor_analysis", "stock_selection"]
        }
        
        # Register agents with failover manager
        for agent_id, capabilities in agent_capabilities.items():
            priorities = {cap: 5 for cap in capabilities}  # Default priority
            
            try:
                failover_manager.register_agent_for_failover(
                    agent_id=agent_id,
                    capabilities=capabilities,
                    priorities=priorities
                )
            except Exception as e:
                logger.error(f"Failed to register {agent_id} with failover manager: {e}")
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Complete integrated debate system initialized")
    
    async def create_debate(self, 
                           topic: str, 
                           preferred_agents: List[str] = None,
                           portfolio_context: Dict = None,
                           debate_params: Dict = None) -> Dict[str, Any]:
        """Create debate with available protection"""
        
        debate_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        
        try:
            # Select agents with health awareness
            available_agents = await self._select_available_agents(
                preferred_agents or [], topic
            )
            
            if len(available_agents) < 2:
                return {
                    "success": False,
                    "error": "Insufficient agents available for debate",
                    "available_agents": len(available_agents)
                }
            
            # Track the debate
            self.active_debates[debate_id] = {
                "job_id": job_id,
                "debate_id": debate_id,
                "topic": topic,
                "agents": available_agents,
                "status": "in_progress",
                "created_at": datetime.now(),
                "portfolio_context": portfolio_context or {}
            }
            
            return {
                "success": True,
                "job_id": job_id,
                "debate_id": debate_id,
                "status": "started",
                "message": "Multi-agent debate initiated",
                "participating_agents": [agent["agent_id"] for agent in available_agents],
                "agent_details": available_agents,
                "topic": topic,
                "resilience_metrics": self._get_system_resilience(),
                "estimated_duration_minutes": 10
            }
            
        except Exception as e:
            logger.error(f"Failed to create debate {debate_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id,
                "debate_id": debate_id
            }
    
    async def _select_available_agents(self, preferred_agents: List[str], topic: str) -> List[Dict]:
        """Select available agents with compatibility across registry formats"""
        
        available_agents = []
        
        # Use compatibility function to get agents
        if enhanced_registry:
            try:
                all_agents = await self._get_available_agents_compatible(enhanced_registry)
                
                # Filter by preferences if specified
                if preferred_agents:
                    filtered_agents = [
                        agent for agent in all_agents 
                        if agent.get("agent_id") in preferred_agents
                    ]
                    if filtered_agents:
                        all_agents = filtered_agents
                
                # Check circuit breaker health if available
                for agent in all_agents:
                    agent_id = agent.get("agent_id")
                    health_score = 1.0  # Default healthy
                    
                    if failover_manager and agent_id in failover_manager.circuit_breakers:
                        cb = failover_manager.circuit_breakers[agent_id]
                        if cb.is_available() and cb.get_health_score() > 0.3:
                            health_score = cb.get_health_score()
                        else:
                            continue  # Skip unhealthy agents
                    
                    available_agents.append({
                        "agent_id": agent_id,
                        "agent_name": agent.get("agent_name", agent_id),
                        "capabilities": agent.get("capabilities", []),
                        "health_score": health_score
                    })
                    
            except Exception as e:
                logger.error(f"Error getting agents from registry: {e}")
        
        # Fallback: use basic agent list
        if not available_agents:
            fallback_agents = preferred_agents or [
                "quantitative_analyst",
                "market_intelligence", 
                "tax_strategist", 
                "options_analyst",
                "economic_data"
            ]
            
            available_agents = [
                {
                    "agent_id": agent_id,
                    "agent_name": agent_id.replace("_", " ").title(),
                    "capabilities": ["general_analysis"],
                    "health_score": 1.0
                }
                for agent_id in fallback_agents
            ]
        
        return available_agents[:5]  # Limit to 5 agents
    
    async def _get_available_agents_compatible(self, registry) -> List[Dict]:
        """Compatibility function to get agents from any registry format"""
        
        available_agents = []
        
        try:
            # Method 1: Try get_available_agents() if it exists
            if hasattr(registry, 'get_available_agents'):
                agents = await registry.get_available_agents()
                return agents
            
            # Method 2: Try accessing agents attribute directly
            elif hasattr(registry, 'agents') and registry.agents:
                for agent_id, agent_info in registry.agents.items():
                    agent_data = {
                        "agent_id": agent_id,
                        "agent_name": getattr(agent_info, 'agent_name', agent_id),
                        "agent_type": getattr(agent_info, 'agent_type', 'unknown'),
                        "capabilities": getattr(agent_info, 'capabilities', []),
                        "description": getattr(agent_info, 'description', 'No description'),
                        "status": "available"
                    }
                    available_agents.append(agent_data)
            
            # Method 3: Try get_all_agents() method
            elif hasattr(registry, 'get_all_agents'):
                agents = registry.get_all_agents()
                if agents:
                    for agent_id, agent_info in agents.items():
                        agent_data = {
                            "agent_id": agent_id,
                            "agent_name": getattr(agent_info, 'agent_name', agent_id),
                            "capabilities": getattr(agent_info, 'capabilities', []),
                            "status": "available"
                        }
                        available_agents.append(agent_data)
            
            # Method 4: Inspect registry to see what's available
            else:
                logger.info(f"Registry attributes: {dir(registry)}")
                # Try to find any method that returns agents
                for attr in dir(registry):
                    if 'agent' in attr.lower() and not attr.startswith('_'):
                        logger.info(f"Found registry method: {attr}")
        
        except Exception as e:
            logger.error(f"Error getting agents from registry: {e}")
        
        return available_agents
    
    async def get_debate_status(self, debate_id: str) -> Dict[str, Any]:
        """Get debate status"""
        
        if debate_id not in self.active_debates:
            return {"error": "Debate not found"}
        
        debate_info = self.active_debates[debate_id]
        
        return {
            "debate_id": debate_id,
            "job_id": debate_info["job_id"],
            "status": debate_info["status"],
            "topic": debate_info["topic"],
            "participating_agents": [agent["agent_id"] for agent in debate_info["agents"]],
            "created_at": debate_info["created_at"].isoformat(),
            "system_resilience": self._get_system_resilience()
        }
    
    def _get_system_resilience(self) -> Dict:
        """Get system resilience metrics"""
        
        if failover_manager:
            try:
                return failover_manager.get_system_resilience_metrics()
            except Exception as e:
                logger.error(f"Error getting resilience metrics: {e}")
        
        return {
            "system_availability": 1.0,
            "total_agents": 5,
            "available_agents": 5,
            "status": "basic_operation"
        }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health"""
        
        health_info = {
            "active_debates": len(self.active_debates),
            "debate_statuses": {},
            "system_resilience": self._get_system_resilience()
        }
        
        # Count debate statuses
        for debate_info in self.active_debates.values():
            status = debate_info["status"]
            health_info["debate_statuses"][status] = health_info["debate_statuses"].get(status, 0) + 1
        
        # Add circuit breaker info if available
        if failover_manager:
            try:
                health_info["circuit_breakers"] = failover_manager.get_all_circuit_breaker_stats()
            except Exception as e:
                logger.error(f"Error getting circuit breaker stats: {e}")
        
        # Add agent registry info if available  
        if enhanced_registry:
            try:
                health_info["agent_registry"] = {"status": "available"}
            except Exception as e:
                logger.error(f"Error getting registry health: {e}")
        
        return health_info
    
    async def shutdown(self):
        """Shutdown the system"""
        self.active_debates.clear()
        logger.info("Complete integrated debate system shutdown")

# Global instance
complete_integrated_system = None

async def initialize_complete_integrated_system(mcp_client):
    """Initialize the complete system"""
    global complete_integrated_system
    
    try:
        complete_integrated_system = CompleteIntegratedDebateSystem(mcp_client)
        await complete_integrated_system.initialize()
        return complete_integrated_system
    except Exception as e:
        logger.error(f"Failed to initialize complete integrated system: {e}")
        return None