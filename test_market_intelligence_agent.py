# test_market_intelligence_agent.py
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.market_intelligence_agent import MarketIntelligenceAgent

async def test_market_intelligence_agent():
    print("🧪 Testing Market Intelligence Agent...")
    
    agent = MarketIntelligenceAgent()
    print(f"✅ Agent created: {agent.agent_name}")
    print(f"🎯 Capabilities: {agent.capabilities}")
    
    # Test data with portfolio
    test_data = {
        "portfolio_data": {
            "holdings": [
                {"symbol": "AAPL", "shares": 100, "current_price": 150.25},
                {"symbol": "GOOGL", "shares": 50, "current_price": 2800.50},
                {"symbol": "SPY", "shares": 200, "current_price": 445.20}
            ],
            "total_value": 405518.75
        },
        "query": "Provide market intelligence and timing signals"
    }
    
    print("\n📡 Testing Real-Time Data Fetching...")
    data_result = await agent.execute_capability("real_time_data", test_data, {})
    if "error" in data_result:
        print(f"❌ Data fetch failed: {data_result['error']}")
    else:
        print(f"✅ Real-time data fetched successfully")
        print(f"📊 VIX Level: {data_result.get('market_data', {}).get('volatility', {}).get('VIX', 'N/A'):.1f}")
        print(f"📈 SPY Price: ${data_result.get('market_data', {}).get('indices', {}).get('SPY', {}).get('price', 'N/A'):.2f}")
    
    print("\n🎛️ Testing Regime Detection...")
    regime_result = await agent.execute_capability("regime_detection", test_data, {})
    if "error" in regime_result:
        print(f"❌ Regime detection failed: {regime_result['error']}")
    else:
        print(f"✅ Regime detection completed")
        current_regime = regime_result.get('current_regime', {})
        print(f"🎯 Current Regime: {current_regime.get('regime', 'Unknown')}")
        print(f"📝 Description: {current_regime.get('description', 'N/A')}")
    
    print("\n📰 Testing News Correlation...")
    news_result = await agent.execute_capability("news_correlation", test_data, {})
    if "error" in news_result:
        print(f"❌ News correlation failed: {news_result['error']}")
    else:
        print(f"✅ News correlation completed")
        portfolio_sentiment = news_result.get('portfolio_sentiment', {})
        print(f"😊 Portfolio Sentiment: {portfolio_sentiment.get('impact', 'neutral').title()}")
        print(f"📊 Sentiment Score: {portfolio_sentiment.get('score', 0.5):.2f}")
        print(f"📰 Key Events: {len(news_result.get('key_news_events', []))}")
    
    print("\n⏰ Testing Market Timing Signals...")
    timing_result = await agent.execute_capability("market_timing", test_data, {})
    if "error" in timing_result:
        print(f"❌ Timing signals failed: {timing_result['error']}")
    else:
        print(f"✅ Timing signals generated")
        overall_signal = timing_result.get('overall_signal', {})
        print(f"🎯 Signal: {overall_signal.get('signal', 'hold').upper()}")
        print(f"💪 Strength: {overall_signal.get('strength', 'neutral').title()}")
        print(f"🎯 Confidence: {overall_signal.get('confidence', 0.5):.1%}")
        recommendations = timing_result.get('recommendations', [])
        if recommendations:
            print(f"💡 Top Recommendation: {recommendations[0].get('description', 'N/A')}")
    
    print("\n🎭 Testing Comprehensive Intelligence...")
    intel_result = await agent.execute_capability("market_intelligence", test_data, {})
    if "error" in intel_result:
        print(f"❌ Comprehensive intelligence failed: {intel_result['error']}")
    else:
        print(f"✅ Comprehensive intelligence completed")
        summary = intel_result.get('intelligence_summary', {})
        print(f"🔮 Market Outlook: {summary.get('market_outlook', 'N/A')}")
        print(f"🎯 Recommended Action: {summary.get('recommended_action', 'hold').upper()}")
        print(f"📊 Confidence: {summary.get('confidence_level', 0.5):.1%}")
    
    print("\n🎉 Market Intelligence Agent test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_market_intelligence_agent())