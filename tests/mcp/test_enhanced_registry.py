# test_enhanced_registry.py
import asyncio
import pytest
from mcp.enhanced_agent_registry import EnhancedAgentRegistry, AgentPerformanceMetrics
from mcp.schemas import AgentRegistration

@pytest.mark.asyncio
async def test_agent_registration_and_routing():
    registry = EnhancedAgentRegistry()
    await registry.start()
    
    # Register test agents
    agent1 = AgentRegistration(
        agent_id="test_agent_1",
        agent_name="Test Agent 1",
        agent_type="TestAgent",
        capabilities=["risk_analysis", "portfolio_analysis"]
    )
    
    agent2 = AgentRegistration(
        agent_id="test_agent_2", 
        agent_name="Test Agent 2",
        agent_type="TestAgent",
        capabilities=["risk_analysis", "tax_optimization"]
    )
    
    registry.register_agent(agent1)
    registry.register_agent(agent2)
    
    # Test routing
    selected_agent = await registry.route_job("risk_analysis", job_complexity=0.5)
    assert selected_agent in ["test_agent_1", "test_agent_2"]
    
    # Test performance tracking
    await registry.report_job_success(selected_agent, "risk_analysis", 1.5)
    
    status = registry.get_agent_status(selected_agent)
    assert status["success_rate"] == 1.0
    assert status["current_load"] == 0  # Should decrease after success
    
    await registry.stop()

if __name__ == "__main__":
    asyncio.run(test_agent_registration_and_routing())