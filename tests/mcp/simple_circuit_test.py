# simple_circuit_test.py
"""Simple circuit breaker test"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from mcp.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    print("✅ Circuit breaker imports successful")
    
    async def simple_test():
        # Create a circuit breaker
        cb = CircuitBreaker("test_agent")
        print(f"✅ Circuit breaker created: {cb.agent_id}")
        
        # Test basic functionality
        async def mock_function():
            return {"status": "success"}
        
        result = await cb.call(mock_function)
        print(f"✅ Function execution successful: {result}")
        
        stats = cb.get_stats()
        print(f"✅ Stats retrieved: state={stats['state']}, health={stats['health_score']:.2f}")
    
    asyncio.run(simple_test())
    print("✅ All basic tests passed")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Need to create circuit_breaker.py first")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()