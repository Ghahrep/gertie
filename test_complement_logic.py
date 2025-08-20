# test_complement_logic.py
import json
import logging
from agents.security_screener_agent import SecurityScreenerAgent

# Configure logging to see the agent's output
logging.basicConfig(level=logging.INFO)

def run_complement_test():
    """
    Tests the portfolio complement functionality of the SecurityScreenerAgent.
    """
    print("\n--- Testing Portfolio Complement Logic ---")
    
    # 1. Initialize the agent. 
    # Since the data is cached from the last run, this should be nearly instant.
    try:
        screener = SecurityScreenerAgent()
        print("✅ Agent initialized successfully (data loaded from cache).")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return

    # 2. Define a mock portfolio heavily tilted towards GROWTH stocks.
    # This context object mimics what the main application would provide to the agent.
    # The 'market_value' is used for weighting the factor exposures.
    mock_growth_portfolio_context = {
        "holdings_with_values": [
            # The agent expects a list of dictionaries with 'ticker' and 'market_value'
            {"ticker": "NVDA", "market_value": 25000},
            {"ticker": "TSLA", "market_value": 18000},
            {"ticker": "AVGO", "market_value": 12000},
        ]
    }
    
    print("\n[INFO] Created a mock portfolio heavily tilted towards GROWTH stocks (NVDA, TSLA, AVGO).")
    print("[EXPECTATION] The recommended stocks should be strong in VALUE and QUALITY to provide balance.")

    # 3. Run a query asking for complementary stocks
    query = "Find 5 stocks to complement and diversify my portfolio"
    
    results = screener.run(query, context=mock_growth_portfolio_context)

    # 4. Print the results for verification
    print("\n--- Agent Response ---")
    if results.get("success"):
        # The summary is the most important part here, as it shows the portfolio analysis
        print(results.get("summary"))
        
        print("\n--- Top 3 Recommendations (raw data) ---")
        print(json.dumps(results.get("recommendations", [])[:3], indent=4))
        print("\n✅ Complement test completed successfully!")
    else:
        print(f"\n❌ Test failed. Error: {results.get('error')}")

if __name__ == "__main__":
    run_complement_test()