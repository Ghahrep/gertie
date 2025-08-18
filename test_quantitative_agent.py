# test_quantitative_agent.py
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.quantitative_analyst_mcp import QuantitativeAnalystAgent

async def test_quantitative_agent():
    print("🧪 Testing Enhanced QuantitativeAnalyst Agent...")
    
    agent = QuantitativeAnalystAgent()
    print(f"✅ Agent created: {agent.agent_name}")
    print(f"🎯 Capabilities: {agent.capabilities}")
    
    # Test data
    test_data = {
        "portfolio_data": {
            "holdings": [
                {"symbol": "AAPL", "shares": 100, "current_price": 150.25},
                {"symbol": "GOOGL", "shares": 50, "current_price": 2800.50},
                {"symbol": "TSLA", "shares": 25, "current_price": 900.75}
            ],
            "total_value": 305518.75
        },
        "query": "Analyze portfolio risk and provide stress testing"
    }
    
    # Test risk analysis
    result = await agent.execute_capability("risk_analysis", test_data, {})
    
    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
    else:
        print(f"✅ Risk analysis completed!")
        print(f"📊 Risk level: {result.get('risk_summary', {}).get('risk_level', 'Unknown')}")
        print(f"🎯 Methods used: {result.get('analysis_methods_used', [])}")
        print(f"⚠️  Risk factors: {result.get('risk_summary', {}).get('key_risk_factors', [])}")
    
    # Test query interpretation
    interp_result = await agent.execute_capability("query_interpretation", test_data, {})
    print(f"🤖 Query interpretation: {interp_result.get('analysis_types', [])}")
    
    print("🎉 Agent test completed!")

if __name__ == "__main__":
    asyncio.run(test_quantitative_agent())