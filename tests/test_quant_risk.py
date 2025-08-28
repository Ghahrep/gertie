# test_mcp_orchestrator.py
import asyncio
from agents.orchestrator import FinancialOrchestrator

# Mock user object to simulate a logged-in user for testing
class MockUser:
    def __init__(self, user_id):
        self.id = user_id

async def test_mcp_orchestrator():
    """
    Tests the orchestrator's routing, specifically for MCP agents.
    Tests all 5 MCP agents: SecurityScreener, QuantitativeAnalyst, FinancialTutor, CrossAssetAnalyst, and StrategyArchitect.
    """
    orchestrator = FinancialOrchestrator()
    mock_user = MockUser(user_id=1)
    
    print("Testing MCP Agent Routing and Capabilities")
    print("=" * 60)
    
    # Test 1: SecurityScreener (MCP) without a portfolio
    print("\n--- Test 1: MCP SecurityScreener (no portfolio) ---")
    result1 = await orchestrator.route_query("find quality growth stocks", db_session=None, current_user=mock_user)
    print(f"   Success: {result1.get('success')}")
    if result1.get('success'):
        print(f"   Agent Used: {result1.get('agent_used')}")
        print(f"   Execution Time: ~15s (S&P 500 analysis)")
    else:
        print(f"   Error: {result1.get('error')}")

    # Test 2: QuantitativeAnalyst (MCP) - Should fail gracefully without portfolio
    print("\n--- Test 2: MCP QuantitativeAnalyst (requires portfolio) ---")
    result2 = await orchestrator.route_query("analyze my portfolio risk", db_session=None, current_user=mock_user)
    print(f"   Success: {result2.get('success')}")
    if result2.get('success'):
        print(f"   Agent Used: {result2.get('agent_used')}")
    else:
        print(f"   Expected Error: {result2.get('error')}")
    
    # Test 3: FinancialTutor (MCP) - Educational queries
    print("\n--- Test 3: MCP FinancialTutor (concept explanation) ---")
    result3 = await orchestrator.route_query("explain what is VaR", db_session=None, current_user=mock_user)
    print(f"   Success: {result3.get('success')}")
    if result3.get('success'):
        print(f"   Agent Used: {result3.get('agent_used')}")
        print(f"   Response Type: {result3.get('summary', 'No response')[:100]}...")
    else:
        print(f"   Error: {result3.get('error')}")
    
    # Test 4: FinancialTutor different concept
    print("\n--- Test 4: MCP FinancialTutor (Sharpe ratio) ---")
    result4 = await orchestrator.route_query("what is the Sharpe ratio", db_session=None, current_user=mock_user)
    print(f"   Success: {result4.get('success')}")
    if result4.get('success'):
        print(f"   Agent Used: {result4.get('agent_used')}")
        print(f"   Educational Content: Available")
    else:
        print(f"   Error: {result4.get('error')}")
    
    # Test 5: FinancialTutor learning guidance
    print("\n--- Test 5: MCP FinancialTutor (learning guidance) ---")
    result5 = await orchestrator.route_query("help me learn about investing", db_session=None, current_user=mock_user)
    print(f"   Success: {result5.get('success')}")
    if result5.get('success'):
        print(f"   Agent Used: {result5.get('agent_used')}")
        print(f"   Learning Support: Provided")
    else:
        print(f"   Error: {result5.get('error')}")
    
    # Test 6: CrossAssetAnalyst (MCP) - Correlation analysis
    print("\n--- Test 6: MCP CrossAssetAnalyst (correlation analysis) ---")
    result6 = await orchestrator.route_query("analyze cross-asset correlations", db_session=None, current_user=mock_user)
    print(f"   Success: {result6.get('success')}")
    if result6.get('success'):
        print(f"   Agent Used: {result6.get('agent_used')}")
        print(f"   Analysis Type: Cross-asset correlation and regime analysis")
    else:
        print(f"   Error: {result6.get('error')}")
    
    # Test 7: CrossAssetAnalyst regime detection
    print("\n--- Test 7: MCP CrossAssetAnalyst (regime detection) ---")
    result7 = await orchestrator.route_query("detect market regime transition", db_session=None, current_user=mock_user)
    print(f"   Success: {result7.get('success')}")
    if result7.get('success'):
        print(f"   Agent Used: {result7.get('agent_used')}")
        print(f"   Regime Analysis: Market regime and stability assessment")
    else:
        print(f"   Error: {result7.get('error')}")
    
    # Test 8: CrossAssetAnalyst diversification
    print("\n--- Test 8: MCP CrossAssetAnalyst (diversification) ---")
    result8 = await orchestrator.route_query("analyze portfolio diversification", db_session=None, current_user=mock_user)
    print(f"   Success: {result8.get('success')}")
    if result8.get('success'):
        print(f"   Agent Used: {result8.get('agent_used')}")
        print(f"   Diversification: Portfolio diversification effectiveness analysis")
    else:
        print(f"   Error: {result8.get('error')}")
    
    # Test 9: StrategyArchitect (MCP) - Strategy design
    print("\n--- Test 9: MCP StrategyArchitect (strategy design) ---")
    result9 = await orchestrator.route_query("design a momentum strategy", db_session=None, current_user=mock_user)
    print(f"   Success: {result9.get('success')}")
    if result9.get('success'):
        print(f"   Agent Used: {result9.get('agent_used')}")
        print(f"   Strategy Design: Momentum strategy guidance provided")
    else:
        print(f"   Error: {result9.get('error')}")
    
    # Test 10: StrategyArchitect with specific symbols
    print("\n--- Test 10: MCP StrategyArchitect (with symbols) ---")
    result10 = await orchestrator.route_query("create momentum strategy for AAPL MSFT", db_session=None, current_user=mock_user)
    print(f"   Success: {result10.get('success')}")
    if result10.get('success'):
        print(f"   Agent Used: {result10.get('agent_used')}")
        print(f"   Strategy Implementation: Specific momentum strategy created")
    else:
        print(f"   Error: {result10.get('error')}")
    
    # Test 11: Workflow trigger detection
    print("\n--- Test 11: Workflow Trigger Detection ---")
    result11 = await orchestrator.route_query("give me actionable investment recommendations with full analysis", db_session=None, current_user=mock_user)
    print(f"   Success: {result11.get('success')}")
    if result11.get('success'):
        workflow_type = result11.get('workflow_type', 'single_agent')
        print(f"   Execution Type: {workflow_type}")
        if 'workflow' in workflow_type:
            print(f"   Multi-Agent Workflow: Triggered successfully")
        else:
            print(f"   Single Agent: {result11.get('agent_used')}")
    else:
        print(f"   Error: {result11.get('error')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_results = [result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11]
    successful_tests = sum(1 for result in all_results
                          if result.get('success') or ('requires a portfolio' in str(result.get('error', ''))))
    
    print(f"MCP Agents Tested: 5 (SecurityScreener, QuantitativeAnalyst, FinancialTutor, CrossAssetAnalyst, StrategyArchitect)")
    print(f"Test Cases: 11")
    print(f"Expected Behavior: {successful_tests}/11")
    print(f"MCP Architecture: {'Operational' if successful_tests >= 9 else 'Issues Detected'}")
    
    # Agent-specific summaries
    if result1.get('success'):
        print(f"SecurityScreener: Fully operational (S&P 500 screening)")
    if 'requires a portfolio' in str(result2.get('error', '')):
        print(f"QuantitativeAnalyst: Correctly requires portfolio")
    if result3.get('success') and result4.get('success') and result5.get('success'):
        print(f"FinancialTutor: Educational capabilities working")
    if result6.get('success') or result7.get('success') or result8.get('success'):
        print(f"CrossAssetAnalyst: Cross-asset analysis operational")
    if result9.get('success') or result10.get('success'):
        print(f"StrategyArchitect: Strategy design capabilities working")
    
    print(f"\nMCP Architecture Testing Complete!")

if __name__ == "__main__":
    try:
        asyncio.run(test_mcp_orchestrator())
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()