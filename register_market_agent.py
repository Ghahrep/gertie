# register_market_agent.py
import asyncio
from agents.market_intelligence_agent import MarketIntelligenceAgent

async def register_market_agent():
    print("🌐 Registering Market Intelligence Agent...")
    
    try:
        agent = MarketIntelligenceAgent()
        print(f"🎯 Agent capabilities: {agent.capabilities}")
        
        # Initialize and register
        await agent.start()
        print("✅ Market Intelligence Agent registered successfully!")
        
        # Let it run briefly
        await asyncio.sleep(3)
        
        # Don't stop it - let it keep running for job processing
        print("🔄 Agent running and ready for jobs...")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(register_market_agent())