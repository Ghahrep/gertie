# tests/mcp/test_enhanced_registry_detailed.py
"""
Comprehensive tests for Enhanced Agent Registry
"""
import asyncio
import pytest
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mcp.enhanced_agent_registry import EnhancedAgentRegistry, AgentPerformanceMetrics
from mcp.schemas import AgentRegistration

class TestEnhancedAgentRegistry:
    """Comprehensive test suite for enhanced agent registry"""
    
    @pytest.fixture
    async def registry(self):
        """Setup test registry"""
        reg = EnhancedAgentRegistry()
        await reg.start()
        yield reg
        await reg.stop()
    
    @pytest.fixture
    def sample_agents(self):
        """Sample agents for testing"""
        return [
            AgentRegistration(
                agent_id="quantitative_analyst",
                agent_name="Quantitative Analyst",
                agent_type="QuantAgent",
                capabilities=["risk_analysis", "portfolio_analysis", "statistical_modeling"],
                max_concurrent_jobs=3
            ),
            AgentRegistration(
                agent_id="market_intelligence", 
                agent_name="Market Intelligence Agent",
                agent_type="MarketAgent",
                capabilities=["market_analysis", "trend_analysis", "sentiment_analysis"],
                max_concurrent_jobs=5
            ),
            AgentRegistration(
                agent_id="tax_strategist",
                agent_name="Tax Strategy Agent", 
                agent_type="TaxAgent",
                capabilities=["tax_optimization", "regulatory_compliance"],
                max_concurrent_jobs=2
            )
        ]
    
    async def test_agent_registration(self, registry, sample_agents):
        """Test agent registration functionality"""
        print("\n=== Testing Agent Registration ===")
        
        for agent in sample_agents:
            success = registry.register_agent(agent)
            assert success, f"Failed to register agent {agent.agent_id}"
            print(f"✅ Registered: {agent.agent_id}")
        
        # Verify agents are registered
        system_status = registry.get_system_status()
        assert system_status["total_agents"] == 3
        print(f"✅ System has {system_status['total_agents']} registered agents")
        
        # Test capability distribution
        capability_dist = system_status["agents_by_capability"]
        print(f"✅ Capability distribution: {capability_dist}")
        
        assert "risk_analysis" in capability_dist
        assert capability_dist["risk_analysis"] == 1
    
    async def test_job_routing(self, registry, sample_agents):
        """Test intelligent job routing"""
        print("\n=== Testing Job Routing ===")
        
        # Register agents
        for agent in sample_agents:
            registry.register_agent(agent)
        
        # Test routing for different capabilities
        test_cases = [
            ("risk_analysis", "quantitative_analyst"),
            ("market_analysis", "market_intelligence"), 
            ("tax_optimization", "tax_strategist")
        ]
        
        for capability, expected_agent_type in test_cases:
            selected_agent = await registry.route_job(capability, job_complexity=0.5)
            assert selected_agent is not None, f"No agent selected for {capability}"
            assert expected_agent_type in selected_agent, f"Wrong agent type selected for {capability}"
            print(f"✅ {capability} routed to {selected_agent}")
        
        # Test routing for non-existent capability
        selected_agent = await registry.route_job("non_existent_capability")
        assert selected_agent is None, "Should return None for non-existent capability"
        print("✅ Correctly handled non-existent capability")
    
    async def test_performance_tracking(self, registry, sample_agents):
        """Test performance metrics tracking"""
        print("\n=== Testing Performance Tracking ===")
        
        # Register agents
        for agent in sample_agents:
            registry.register_agent(agent)
        
        agent_id = "quantitative_analyst"
        
        # Get initial metrics
        initial_status = registry.get_agent_status(agent_id)
        print(f"Initial status: success_rate={initial_status['success_rate']}, load={initial_status['current_load']}")
        
        # Route a job (increases load)
        selected_agent = await registry.route_job("risk_analysis")
        assert selected_agent == agent_id
        
        # Check load increased
        status_after_routing = registry.get_agent_status(agent_id)
        assert status_after_routing["current_load"] == 1
        print(f"✅ Load increased to {status_after_routing['current_load']}")
        
        # Report successful completion
        await registry.report_job_success(agent_id, "risk_analysis", response_time=2.5)
        
        # Check metrics updated
        status_after_success = registry.get_agent_status(agent_id)
        assert status_after_success["current_load"] == 0  # Load should decrease
        assert status_after_success["success_rate"] == 1.0
        assert status_after_success["total_completed"] == 1
        print(f"✅ Success metrics updated: completed={status_after_success['total_completed']}")
        
        # Test failure tracking
        await registry.route_job("risk_analysis")  # Increase load again
        await registry.report_job_failure(agent_id, "risk_analysis")
        
        status_after_failure = registry.get_agent_status(agent_id)
        assert status_after_failure["total_failed"] == 1
        assert status_after_failure["success_rate"] == 0.5  # 1 success, 1 failure
        print(f"✅ Failure metrics updated: failed={status_after_failure['total_failed']}, success_rate={status_after_failure['success_rate']}")
    
    async def test_load_balancing(self, registry, sample_agents):
        """Test load balancing across agents"""
        print("\n=== Testing Load Balancing ===")
        
        # Register multiple agents with same capability
        agents_with_risk = [
            AgentRegistration(
                agent_id=f"risk_agent_{i}",
                agent_name=f"Risk Agent {i}",
                agent_type="RiskAgent",
                capabilities=["risk_analysis"],
                max_concurrent_jobs=2
            ) for i in range(3)
        ]
        
        for agent in agents_with_risk:
            registry.register_agent(agent)
        
        # Route multiple jobs and track distribution
        job_distribution = {}
        for i in range(9):  # Route 9 jobs
            selected_agent = await registry.route_job("risk_analysis", job_complexity=0.3)
            job_distribution[selected_agent] = job_distribution.get(selected_agent, 0) + 1
            print(f"Job {i+1} routed to {selected_agent}")
        
        print(f"Job distribution: {job_distribution}")
        
        # Verify jobs were distributed (not all to one agent)
        assert len(job_distribution) > 1, "Jobs should be distributed across multiple agents"
        
        # Check that no agent is severely overloaded
        max_jobs = max(job_distribution.values())
        min_jobs = min(job_distribution.values())
        assert max_jobs - min_jobs <= 2, "Load should be reasonably balanced"
        print("✅ Load balancing working correctly")
    
    async def test_health_monitoring(self, registry, sample_agents):
        """Test health monitoring functionality"""
        print("\n=== Testing Health Monitoring ===")
        
        # Register agents
        for agent in sample_agents:
            registry.register_agent(agent)
        
        # All agents should start healthy
        system_status = registry.get_system_status()
        assert system_status["healthy_agents"] == 3
        assert system_status["unhealthy_agents"] == 0
        print(f"✅ All {system_status['healthy_agents']} agents healthy")
        
        # Simulate agent failures to test health degradation
        agent_id = "quantitative_analyst"
        for i in range(4):  # Cause multiple consecutive failures
            await registry.route_job("risk_analysis")
            await registry.report_job_failure(agent_id, "risk_analysis")
        
        # Check if agent becomes unhealthy
        agent_status = registry.get_agent_status(agent_id)
        print(f"Agent {agent_id} consecutive failures: {agent_status['consecutive_failures']}")
        print(f"Agent {agent_id} is healthy: {agent_status['is_healthy']}")
        
        # System status should reflect unhealthy agent
        system_status_after_failures = registry.get_system_status()
        print(f"System status: healthy={system_status_after_failures['healthy_agents']}, unhealthy={system_status_after_failures['unhealthy_agents']}")

async def run_all_tests():
    """Run all tests manually"""
    print("🧪 Starting Enhanced Agent Registry Tests\n")
    
    registry = EnhancedAgentRegistry()
    await registry.start()
    
    try:
        # Create sample agents
        sample_agents = [
            AgentRegistration(
                agent_id="quantitative_analyst",
                agent_name="Quantitative Analyst",
                agent_type="QuantAgent", 
                capabilities=["risk_analysis", "portfolio_analysis"],
                max_concurrent_jobs=3
            ),
            AgentRegistration(
                agent_id="market_intelligence",
                agent_name="Market Intelligence Agent",
                agent_type="MarketAgent",
                capabilities=["market_analysis", "trend_analysis"],
                max_concurrent_jobs=5
            ),
            AgentRegistration(
                agent_id="tax_strategist",
                agent_name="Tax Strategy Agent",
                agent_type="TaxAgent", 
                capabilities=["tax_optimization"],
                max_concurrent_jobs=2
            )
        ]
        
        test_instance = TestEnhancedAgentRegistry()
        
        # Run tests
        await test_instance.test_agent_registration(registry, sample_agents)
        await test_instance.test_job_routing(registry, sample_agents)
        await test_instance.test_performance_tracking(registry, sample_agents)
        await test_instance.test_load_balancing(registry, sample_agents)
        await test_instance.test_health_monitoring(registry, sample_agents)
        
        print("\n🎉 All tests completed successfully!")
        
        # Print final system status
        final_status = registry.get_system_status()
        print(f"\nFinal System Status:")
        print(f"  Total Agents: {final_status['total_agents']}")
        print(f"  Healthy Agents: {final_status['healthy_agents']}")
        print(f"  Active Jobs: {final_status['total_active_jobs']}")
        print(f"  Average Success Rate: {final_status['average_success_rate']:.2%}")
        
    finally:
        await registry.stop()

if __name__ == "__main__":
    asyncio.run(run_all_tests())