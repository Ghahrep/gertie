# test_enhanced_mcp_client.py
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from services.mcp_client import MCPClient
from mcp.schemas import JobRequest

async def test_enhanced_client():
    """Test the enhanced MCP client features"""
    print("Testing Enhanced MCP Client Features")
    print("=" * 50)
    
    client = MCPClient()
    
    try:
        await client.start()
        print("✅ Enhanced MCP Client started")
        
        # Test health check with caching
        print("\n1. Testing health check with caching...")
        health1 = await client.health_check()
        health2 = await client.health_check()  # Should be cached
        print(f"   Health status: {health1.status}")
        
        # Check client statistics
        stats = client.get_client_statistics()
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"   Circuit breaker: {stats['circuit_breaker_state']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        
        # Test job submission with monitoring
        print("\n2. Testing job submission...")
        job_request = JobRequest(
            query="Test portfolio analysis",
            context={"test": True},
            priority=5
        )
        
        job_response = await client.submit_job(job_request)
        print(f"   Job submitted: {job_response.job_id}")
        
        # Check status a few times
        for i in range(3):
            status = await client.get_job_status(job_response.job_id)
            print(f"   Status check {i+1}: {status.status if status else 'Not found'}")
            await asyncio.sleep(0.5)
        
        # Final statistics
        final_stats = client.get_client_statistics()
        print(f"\n3. Final Statistics:")
        print(f"   Total requests: {final_stats['total_requests']}")
        print(f"   Success rate: {final_stats['success_rate']:.1%}")
        print(f"   Cache hits: {final_stats['cache_hits']}")
        print(f"   Avg response time: {final_stats['avg_response_time_ms']:.1f}ms")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await client.close()
        print("\n✅ Enhanced MCP Client test completed")

if __name__ == "__main__":
    asyncio.run(test_enhanced_client())