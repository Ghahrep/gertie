# test_security_screener.py
import asyncio
from agents.security_screener_agent import SecurityScreenerAgent

async def test_screener():
    print("🔍 Testing SecurityScreenerAgent...")
    agent = SecurityScreenerAgent()
    
    # Test 1: General screening
    print("\n1. Testing general screening...")
    result1 = await agent.run("find quality growth stocks", {})
    print(f"   Success: {result1['success']}")
    print(f"   Agent: {result1['agent_used']}")
    
    # Test 2: Factor analysis
    print("\n2. Testing factor analysis...")
    result2 = await agent.run("analyze value stocks with good quality scores", {})
    print(f"   Success: {result2['success']}")
    
    # Test 3: Health check
    print("\n3. Testing health check...")
    health = await agent.health_check()
    print(f"   Status: {health['status']}")
    
    print("\n✅ SecurityScreener MCP migration test complete!")

if __name__ == "__main__":
    asyncio.run(test_screener())