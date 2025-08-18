# register_agents_with_mcp.py
import asyncio
from agents.quantitative_analyst_mcp import QuantitativeAnalystAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent

async def register_agents():
    print("🤖 Registering agents with MCP...")
    
    # Create agents
    quant_agent = QuantitativeAnalystAgent()
    market_agent = MarketIntelligenceAgent()
    
    try:
        # Initialize HTTP sessions first
        print("🔌 Initializing agent sessions...")
        await quant_agent.start()
        print("✅ Quantitative agent session initialized")
        
        await market_agent.start()
        print("✅ Market intelligence agent session initialized")
        
        print("🎉 Both agents successfully registered with MCP!")
        
        # Let them run for a moment to test job handling
        print("⏱️ Letting agents run for 5 seconds to test connectivity...")
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f"❌ Error during agent startup: {str(e)}")
    
    finally:
        # Clean shutdown
        print("🧹 Shutting down agents...")
        await quant_agent.stop()
        await market_agent.stop()
        print("✅ Agents shutdown complete")

if __name__ == "__main__":
    asyncio.run(register_agents())