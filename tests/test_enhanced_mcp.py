import asyncio
import logging
from typing import Dict
from agents.mcp_base_agent import MCPBaseAgent

# Create a simple test agent
class TestAgent(MCPBaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="test_agent",
            agent_name="Test Agent",
            capabilities=["risk_analysis", "portfolio_analysis"]
        )
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "analysis_type": capability,
            "result": "test successful",
            "confidence_score": 0.85
        }

async def test_enhanced_mcp():
    """Test the enhanced MCP base agent"""
    logging.basicConfig(level=logging.INFO)
    
    agent = TestAgent()
    
    # Test basic run method
    result = await agent.run("analyze portfolio risk", {})
    print(f"✅ Run test: {result['success']}")
    print(f"📝 Summary: {result['summary'][:50]}...")
    
    # Test health check
    health = await agent.health_check()
    print(f"🏥 Health: {health['status']}")
    
    # Test error handling
    try:
        error_result = await agent.run("unsupported query type", {})
        print(f"🚨 Error handling: {error_result['success']}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_mcp())