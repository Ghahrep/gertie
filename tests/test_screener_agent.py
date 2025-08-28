# test_screener_agent.py
import json
import logging
from agents.security_screener_agent import SecurityScreenerAgent

# Configure logging to see the agent's output
logging.basicConfig(level=logging.INFO)

def run_test():
    """
    Initializes and tests the SecurityScreenerAgent.
    """
    print("\n--- Testing SecurityScreenerAgent ---")
    
    # 1. Initialize the agent
    # This will automatically trigger the S&P 500 fetch in the __init__ method.
    try:
        screener = SecurityScreenerAgent()
        print("✅ Agent initialized successfully.")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return

    # 2. Verify the screening universe size
    universe_size = len(screener.screening_universe)
    print(f"\n[INFO] Screening universe size: {universe_size} tickers.")
    if universe_size > 400:
        print("✅ Successfully fetched the S&P 500 ticker list.")
    else:
        print("⚠️ Fetched a smaller fallback list. Check network or Wikipedia page format.")

    # 3. Run a sample query
    print("\n--- Running a sample factor-based screening query ---")
    query = "Find me the top 3 quality stocks"
    # For this test, context can be empty as it doesn't involve portfolio complement
    context = {} 
    
    results = screener.run(query, context)

    # 4. Pretty-print the results
    print("\n--- Agent Response ---")
    print(json.dumps(results, indent=4))
    
    if results.get("success"):
        print("\n✅ Test completed successfully!")
        recommendations = results.get("recommendations", [])
        assert len(recommendations) == 3, "Expected 3 recommendations"
        print(f"✅ Verified that {len(recommendations)} recommendations were returned.")
    else:
        print(f"\n❌ Test failed. Error: {results.get('error')}")

if __name__ == "__main__":
    run_test()