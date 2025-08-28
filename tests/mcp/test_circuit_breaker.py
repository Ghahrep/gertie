# test_circuit_breaker.py
"""
Comprehensive Circuit Breaker Testing
====================================
Test the circuit breaker and failover system functionality.
"""

import asyncio
import sys
import os
from datetime import datetime
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from mcp.circuit_breaker import CircuitBreaker, CircuitBreakerError, FailoverManager, CircuitBreakerConfig

class MockAgent:
    """Mock agent for testing circuit breaker functionality"""
    
    def __init__(self, agent_id: str, failure_rate: float = 0.0):
        self.agent_id = agent_id
        self.failure_rate = failure_rate
        self.call_count = 0
    
    async def execute_task(self, task_data: dict) -> dict:
        """Mock task execution with configurable failure rate"""
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate failures based on failure rate
        if random.random() < self.failure_rate:
            raise Exception(f"Mock failure from {self.agent_id} on call {self.call_count}")
        
        return {
            "agent_id": self.agent_id,
            "result": f"Success from {self.agent_id}",
            "call_count": self.call_count,
            "timestamp": datetime.now().isoformat()
        }

async def test_circuit_breaker_basic():
    """Test basic circuit breaker functionality"""
    print("\n=== Testing Basic Circuit Breaker ===")
    
    # Create reliable agent
    reliable_agent = MockAgent("reliable_agent", failure_rate=0.0)
    circuit_breaker = CircuitBreaker("reliable_agent")
    
    # Test successful execution
    print("1. Testing successful execution...")
    for i in range(3):
        result = await circuit_breaker.call(reliable_agent.execute_task, {"task": f"test_{i}"})
        print(f"   Success {i+1}: {result['result']}")
    
    stats = circuit_breaker.get_stats()
    print(f"   Stats: {stats['stats']['total_successes']} successes, {stats['stats']['total_failures']} failures")
    print(f"   State: {stats['state']}, Health Score: {stats['health_score']:.2f}")
    
    # Create failing agent
    print("\n2. Testing circuit breaker opening on failures...")
    failing_agent = MockAgent("failing_agent", failure_rate=1.0)  # Always fails
    failing_circuit = CircuitBreaker("failing_agent", CircuitBreakerConfig(failure_threshold=3))
    
    failure_count = 0
    for i in range(6):
        try:
            result = await failing_circuit.call(failing_agent.execute_task, {"task": f"test_{i}"})
            print(f"   Unexpected success {i+1}: {result['result']}")
        except CircuitBreakerError as e:
            print(f"   Circuit breaker blocked call {i+1}: {str(e)}")
        except Exception as e:
            failure_count += 1
            print(f"   Execution failure {i+1}: {str(e)}")
    
    failing_stats = failing_circuit.get_stats()
    print(f"   Final State: {failing_stats['state']}, Health Score: {failing_stats['health_score']:.2f}")
    print(f"   Total failures: {failing_stats['stats']['total_failures']}")

async def test_circuit_breaker_recovery():
    """Test circuit breaker recovery (half-open to closed)"""
    print("\n=== Testing Circuit Breaker Recovery ===")
    
    # Create initially failing agent that recovers
    recovering_agent = MockAgent("recovering_agent", failure_rate=1.0)
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=1  # Short timeout for testing
    )
    recovery_circuit = CircuitBreaker("recovering_agent", config)
    
    print("1. Causing failures to open circuit...")
    for i in range(4):
        try:
            await recovery_circuit.call(recovering_agent.execute_task, {"task": f"fail_{i}"})
        except Exception as e:
            print(f"   Failure {i+1}: Circuit open" if "Circuit breaker is OPEN" in str(e) else f"   Failure {i+1}: {str(e)}")
    
    print(f"   Circuit state after failures: {recovery_circuit.state.value}")
    
    # Wait for timeout and test recovery
    print("\n2. Waiting for timeout and testing recovery...")
    await asyncio.sleep(1.2)  # Wait for timeout
    
    # Agent recovers (reduce failure rate)
    recovering_agent.failure_rate = 0.0
    
    print("   Attempting recovery calls...")
    for i in range(3):
        try:
            result = await recovery_circuit.call(recovering_agent.execute_task, {"task": f"recover_{i}"})
            print(f"   Recovery success {i+1}: {result['result']}")
            print(f"   Circuit state: {recovery_circuit.state.value}")
        except Exception as e:
            print(f"   Recovery attempt {i+1} failed: {str(e)}")
    
    final_stats = recovery_circuit.get_stats()
    print(f"   Final state: {final_stats['state']}, Health score: {final_stats['health_score']:.2f}")

async def test_failover_manager():
    """Test the failover manager functionality"""
    print("\n=== Testing Failover Manager ===")
    
    failover_manager = FailoverManager()
    
    # Create mock agents with different reliability
    agents = {
        "primary_agent": MockAgent("primary_agent", failure_rate=0.0),
        "backup_agent": MockAgent("backup_agent", failure_rate=0.0),
        "unreliable_agent": MockAgent("unreliable_agent", failure_rate=0.8)
    }
    
    # Register agents for failover
    print("1. Registering agents for failover...")
    failover_manager.register_agent_for_failover("primary_agent", ["risk_analysis"], {"risk_analysis": 1})
    failover_manager.register_agent_for_failover("backup_agent", ["risk_analysis"], {"risk_analysis": 2})
    failover_manager.register_agent_for_failover("unreliable_agent", ["risk_analysis"], {"risk_analysis": 3})
    
    # Mock execution function
    async def mock_execute_analysis(agent_id: str, task_data: dict):
        agent = agents[agent_id]
        return await agent.execute_task(task_data)
    
    print("2. Testing successful failover execution...")
    for i in range(3):
        try:
            result = await failover_manager.execute_with_failover(
                "risk_analysis", mock_execute_analysis, {"analysis": f"test_{i}"}
            )
            print(f"   Success {i+1}: {result['result']}")
        except Exception as e:
            print(f"   Execution {i+1} failed: {str(e)}")
    
    print("\n3. Testing failover when primary agent fails...")
    # Make primary agent unreliable
    agents["primary_agent"].failure_rate = 1.0
    
    for i in range(5):
        try:
            result = await failover_manager.execute_with_failover(
                "risk_analysis", mock_execute_analysis, {"analysis": f"failover_test_{i}"}
            )
            print(f"   Failover success {i+1}: {result['result']}")
        except Exception as e:
            print(f"   Failover execution {i+1} failed: {str(e)}")
    
    print("\n4. Failover status report...")
    status = failover_manager.get_failover_status("risk_analysis")
    print(f"   Capability: {status['capability']}")
    print(f"   Total agents: {status['total_agents']}")
    print(f"   Available agents: {status['available_agents']}")
    print(f"   Primary agent: {status['primary_agent']}")
    
    # System resilience metrics
    print("\n5. System resilience metrics...")
    metrics = failover_manager.get_system_resilience_metrics()
    print(f"   System availability: {metrics['system_availability']:.1%}")
    print(f"   Average health score: {metrics['average_health_score']:.2f}")
    print(f"   Capability coverage: {metrics['capability_coverage']}")

async def test_performance_under_load():
    """Test circuit breaker performance under load"""
    print("\n=== Testing Performance Under Load ===")
    
    # Create agent with intermittent failures
    load_agent = MockAgent("load_agent", failure_rate=0.3)
    load_circuit = CircuitBreaker("load_agent", CircuitBreakerConfig(failure_threshold=5))
    
    print("1. Running 20 concurrent requests...")
    tasks = []
    for i in range(20):
        task = asyncio.create_task(
            load_circuit.call(load_agent.execute_task, {"request": f"load_test_{i}"})
        )
        tasks.append(task)
    
    # Execute all tasks and collect results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successes = sum(1 for r in results if not isinstance(r, Exception))
    circuit_blocks = sum(1 for r in results if isinstance(r, CircuitBreakerError))
    other_failures = sum(1 for r in results if isinstance(r, Exception) and not isinstance(r, CircuitBreakerError))
    
    print(f"   Results: {successes} successes, {circuit_blocks} circuit blocks, {other_failures} other failures")
    
    final_stats = load_circuit.get_stats()
    print(f"   Final circuit state: {final_stats['state']}")
    print(f"   Health score: {final_stats['health_score']:.2f}")
    print(f"   Total requests: {final_stats['stats']['total_requests']}")

async def test_circuit_breaker_configuration():
    """Test different circuit breaker configurations"""
    print("\n=== Testing Circuit Breaker Configuration ===")
    
    # Test with different configurations
    configs = [
        ("Conservative", CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1)),
        ("Aggressive", CircuitBreakerConfig(failure_threshold=8, timeout_seconds=5)),
        ("Fast Recovery", CircuitBreakerConfig(failure_threshold=3, success_threshold=1))
    ]
    
    for config_name, config in configs:
        print(f"\n1. Testing {config_name} configuration...")
        test_agent = MockAgent(f"test_agent_{config_name.lower()}", failure_rate=0.7)
        test_circuit = CircuitBreaker(f"test_{config_name.lower()}", config)
        
        # Execute requests until circuit opens
        attempt_count = 0
        while test_circuit.state.value != "open" and attempt_count < 15:
            attempt_count += 1
            try:
                await test_circuit.call(test_agent.execute_task, {"test": attempt_count})
            except Exception:
                pass
        
        stats = test_circuit.get_stats()
        print(f"   {config_name}: Circuit opened after {attempt_count} attempts")
        print(f"   State: {stats['state']}, Failures: {stats['stats']['total_failures']}")

async def run_all_circuit_breaker_tests():
    """Run comprehensive circuit breaker test suite"""
    print("🔧 Starting Circuit Breaker Test Suite")
    print("=" * 60)
    
    try:
        await test_circuit_breaker_basic()
        await test_circuit_breaker_recovery()
        await test_failover_manager()
        await test_performance_under_load()
        await test_circuit_breaker_configuration()
        
        print("\n" + "=" * 60)
        print("✅ All Circuit Breaker Tests Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all_circuit_breaker_tests())