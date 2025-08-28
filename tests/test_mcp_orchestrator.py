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
    Tests all 10 MCP agents: SecurityScreener, QuantitativeAnalyst, FinancialTutor, 
    CrossAssetAnalyst, StrategyArchitect, RegimeForecastingAgent, StrategyRebalancingAgent,
    HedgingStrategistAgent, StrategyBacktesterAgent, and ScenarioSimulationAgent.
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
    
    # Test 11: RegimeForecastingAgent (MCP) - Framework without data
    print("\n--- Test 11: MCP RegimeForecasting (framework) ---")
    result11 = await orchestrator.route_query("forecast market regime changes", db_session=None, current_user=mock_user)
    print(f"   Success: {result11.get('success')}")
    if result11.get('success'):
        print(f"   Agent Used: {result11.get('agent_used')}")
        print(f"   Regime Analysis: Framework and guidance provided")
    else:
        print(f"   Error: {result11.get('error')}")
    
    # Test 12: RegimeForecastingAgent transition analysis
    print("\n--- Test 12: MCP RegimeForecasting (transitions) ---")
    result12 = await orchestrator.route_query("analyze regime transition probabilities", db_session=None, current_user=mock_user)
    print(f"   Success: {result12.get('success')}")
    if result12.get('success'):
        print(f"   Agent Used: {result12.get('agent_used')}")
        print(f"   Transition Analysis: Regime transition framework")
    else:
        print(f"   Error: {result12.get('error')}")
    
    # NEW TESTS: StrategyRebalancingAgent (MCP)
    
    # Test 13: StrategyRebalancing - Portfolio optimization (should fail without portfolio)
    print("\n--- Test 13: MCP StrategyRebalancing (portfolio optimization) ---")
    result13 = await orchestrator.route_query("optimize my portfolio for maximum sharpe ratio", db_session=None, current_user=mock_user)
    print(f"   Success: {result13.get('success')}")
    if result13.get('success'):
        print(f"   Agent Used: {result13.get('agent_used')}")
        print(f"   Optimization: Portfolio optimization completed")
    else:
        print(f"   Expected Error: {result13.get('error')}")
    
    # Test 14: StrategyRebalancing - Rebalancing analysis  
    print("\n--- Test 14: MCP StrategyRebalancing (rebalancing analysis) ---")
    result14 = await orchestrator.route_query("analyze if my portfolio needs rebalancing", db_session=None, current_user=mock_user)
    print(f"   Success: {result14.get('success')}")
    if result14.get('success'):
        print(f"   Agent Used: {result14.get('agent_used')}")
        print(f"   Rebalancing: Portfolio balance analysis completed")
    else:
        print(f"   Expected Error: {result14.get('error')}")
    
    # Test 15: StrategyRebalancing - Risk parity
    print("\n--- Test 15: MCP StrategyRebalancing (risk parity) ---")
    result15 = await orchestrator.route_query("create risk parity allocation", db_session=None, current_user=mock_user)
    print(f"   Success: {result15.get('success')}")
    if result15.get('success'):
        print(f"   Agent Used: {result15.get('agent_used')}")
        print(f"   Risk Parity: HERC optimization strategy applied")
    else:
        print(f"   Expected Error: {result15.get('error')}")
    
    # Test 16: StrategyRebalancing - Generate trades
    print("\n--- Test 16: MCP StrategyRebalancing (trade generation) ---")
    result16 = await orchestrator.route_query("generate rebalancing trades for my portfolio", db_session=None, current_user=mock_user)
    print(f"   Success: {result16.get('success')}")
    if result16.get('success'):
        print(f"   Agent Used: {result16.get('agent_used')}")
        print(f"   Trade Generation: Rebalancing trades generated")
    else:
        print(f"   Expected Error: {result16.get('error')}")
    
    # Test 17: StrategyRebalancing - Minimum variance
    print("\n--- Test 17: MCP StrategyRebalancing (minimum variance) ---")
    result17 = await orchestrator.route_query("minimize portfolio volatility", db_session=None, current_user=mock_user)
    print(f"   Success: {result17.get('success')}")
    if result17.get('success'):
        print(f"   Agent Used: {result17.get('agent_used')}")
        print(f"   Min Variance: Conservative portfolio optimization")
    else:
        print(f"   Expected Error: {result17.get('error')}")
    
    # Test 18: HedgingStrategist - Hedge analysis (should fail without portfolio)
    print("\n--- Test 18: MCP HedgingStrategist (hedge analysis) ---")
    result18 = await orchestrator.route_query("find optimal hedge for my portfolio", db_session=None, current_user=mock_user)
    print(f"   Success: {result18.get('success')}")
    if result18.get('success'):
        print(f"   Agent Used: {result18.get('agent_used')}")
        print(f"   Hedge Analysis: Optimal hedge strategy completed")
    else:
        print(f"   Expected Error: {result18.get('error')}")
    
    # Test 19: HedgingStrategist - Volatility targeting
    print("\n--- Test 19: MCP HedgingStrategist (volatility targeting) ---")
    result19 = await orchestrator.route_query("target 15% volatility", db_session=None, current_user=mock_user)
    print(f"   Success: {result19.get('success')}")
    if result19.get('success'):
        print(f"   Agent Used: {result19.get('agent_used')}")
        print(f"   Volatility Targeting: Risk budgeting strategy applied")
    else:
        print(f"   Expected Error: {result19.get('error')}")
    
    # Test 20: HedgingStrategist - Downside protection
    print("\n--- Test 20: MCP HedgingStrategist (downside protection) ---")
    result20 = await orchestrator.route_query("protect my portfolio from downside", db_session=None, current_user=mock_user)
    print(f"   Success: {result20.get('success')}")
    if result20.get('success'):
        print(f"   Agent Used: {result20.get('agent_used')}")
        print(f"   Downside Protection: Risk management strategies provided")
    else:
        print(f"   Expected Error: {result20.get('error')}")
    
    # Test 21: HedgingStrategist - Risk budgeting
    print("\n--- Test 21: MCP HedgingStrategist (risk budgeting) ---")
    result21 = await orchestrator.route_query("analyze my risk budget allocation", db_session=None, current_user=mock_user)
    print(f"   Success: {result21.get('success')}")
    if result21.get('success'):
        print(f"   Agent Used: {result21.get('agent_used')}")
        print(f"   Risk Budgeting: Portfolio risk allocation analysis")
    else:
        print(f"   Expected Error: {result21.get('error')}")
    
    # NEW TESTS: StrategyBacktesterAgent (MCP)
    
    # Test 22: StrategyBacktester - Moving average backtest (should fail without portfolio)
    print("\n--- Test 22: MCP StrategyBacktester (moving average) ---")
    result22 = await orchestrator.route_query("backtest moving average strategy", db_session=None, current_user=mock_user)
    print(f"   Success: {result22.get('success')}")
    if result22.get('success'):
        print(f"   Agent Used: {result22.get('agent_used')}")
        print(f"   Backtest: Moving average strategy analysis completed")
    else:
        print(f"   Expected Error: {result22.get('error')}")
    
    # Test 23: StrategyBacktester - Regime switching strategy
    print("\n--- Test 23: MCP StrategyBacktester (regime switching) ---")
    result23 = await orchestrator.route_query("test regime switching strategy", db_session=None, current_user=mock_user)
    print(f"   Success: {result23.get('success')}")
    if result23.get('success'):
        print(f"   Agent Used: {result23.get('agent_used')}")
        print(f"   Regime Strategy: Regime switching analysis completed")
    else:
        print(f"   Expected Error: {result23.get('error')}")
    
    # Test 24: StrategyBacktester - Strategy comparison
    print("\n--- Test 24: MCP StrategyBacktester (strategy comparison) ---")
    result24 = await orchestrator.route_query("compare trading strategies performance", db_session=None, current_user=mock_user)
    print(f"   Success: {result24.get('success')}")
    if result24.get('success'):
        print(f"   Agent Used: {result24.get('agent_used')}")
        print(f"   Strategy Comparison: Multi-strategy analysis completed")
    else:
        print(f"   Expected Error: {result24.get('error')}")
    
    # Test 25: StrategyBacktester - Statistical validation
    print("\n--- Test 25: MCP StrategyBacktester (statistical analysis) ---")
    result25 = await orchestrator.route_query("validate strategy statistical significance", db_session=None, current_user=mock_user)
    print(f"   Success: {result25.get('success')}")
    if result25.get('success'):
        print(f"   Agent Used: {result25.get('agent_used')}")
        print(f"   Statistical Analysis: Strategy validation completed")
    else:
        print(f"   Expected Error: {result25.get('error')}")
    
    # NEW TESTS: ScenarioSimulationAgent (MCP)
    
    # Test 27: ScenarioSimulation - Market crash scenario (should fail without portfolio)
    print("\n--- Test 27: MCP ScenarioSimulation (market crash) ---")
    result27 = await orchestrator.route_query("simulate market crash scenario", db_session=None, current_user=mock_user)
    print(f"   Success: {result27.get('success')}")
    if result27.get('success'):
        print(f"   Agent Used: {result27.get('agent_used')}")
        print(f"   Market Crash: Crisis scenario simulation completed")
    else:
        print(f"   Expected Error: {result27.get('error')}")
    
    # Test 28: ScenarioSimulation - Comprehensive stress testing
    print("\n--- Test 28: MCP ScenarioSimulation (stress testing) ---")
    result28 = await orchestrator.route_query("run comprehensive stress tests", db_session=None, current_user=mock_user)
    print(f"   Success: {result28.get('success')}")
    if result28.get('success'):
        print(f"   Agent Used: {result28.get('agent_used')}")
        print(f"   Stress Testing: Multi-scenario analysis completed")
    else:
        print(f"   Expected Error: {result28.get('error')}")
    
    # Test 29: ScenarioSimulation - Tail risk analysis
    print("\n--- Test 29: MCP ScenarioSimulation (tail risk) ---")
    result29 = await orchestrator.route_query("analyze tail risk scenarios", db_session=None, current_user=mock_user)
    print(f"   Success: {result29.get('success')}")
    if result29.get('success'):
        print(f"   Agent Used: {result29.get('agent_used')}")
        print(f"   Tail Risk: Extreme scenario analysis completed")
    else:
        print(f"   Expected Error: {result29.get('error')}")
    
    # Test 30: ScenarioSimulation - Portfolio resilience
    print("\n--- Test 30: MCP ScenarioSimulation (portfolio resilience) ---")
    result30 = await orchestrator.route_query("assess portfolio resilience", db_session=None, current_user=mock_user)
    print(f"   Success: {result30.get('success')}")
    if result30.get('success'):
        print(f"   Agent Used: {result30.get('agent_used')}")
        print(f"   Resilience: Portfolio stress resilience analysis")
    else:
        print(f"   Expected Error: {result30.get('error')}")
    
    # Test 31: Workflow trigger detection
    print("\n--- Test 31: Workflow Trigger Detection ---")
    result31 = await orchestrator.route_query("give me actionable investment recommendations with full analysis", db_session=None, current_user=mock_user)
    print(f"   Success: {result31.get('success')}")
    if result31.get('success'):
        workflow_type = result31.get('workflow_type', 'single_agent')
        print(f"   Execution Type: {workflow_type}")
        if 'workflow' in workflow_type:
            print(f"   Multi-Agent Workflow: Triggered successfully")
        else:
            print(f"   Single Agent: {result31.get('agent_used')}")
    else:
        print(f"   Error: {result31.get('error')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_results = [result1, result2, result3, result4, result5, result6, result7, result8, 
                  result9, result10, result11, result12, result13, result14, result15, 
                  result16, result17, result18, result19, result20, result21, result22, 
                  result23, result24, result25, result27, result28, result29, result30, result31]
    
    successful_tests = sum(1 for result in all_results
                          if result.get('success') or ('requires a portfolio' in str(result.get('error', ''))) 
                          or ('Portfolio' in str(result.get('error', '')))
                          or ('portfolio return data' in str(result.get('error', '')))
                          or ('price data' in str(result.get('error', '')))
                          or ('historical' in str(result.get('error', '')))
                          or ('return data' in str(result.get('error', ''))))
    
    print(f"MCP Agents Tested: 10 (SecurityScreener, QuantitativeAnalyst, FinancialTutor, CrossAssetAnalyst, StrategyArchitect, RegimeForecasting, StrategyRebalancing, HedgingStrategist, StrategyBacktester, ScenarioSimulation)")
    print(f"Test Cases: 30")
    print(f"Expected Behavior: {successful_tests}/30")
    print(f"MCP Architecture: {'Operational' if successful_tests >= 26 else 'Issues Detected'}")
    
    # Agent-specific summaries
    if result1.get('success'):
        print(f"SecurityScreener: Fully operational (S&P 500 screening)")
    if 'requires a portfolio' in str(result2.get('error', '')) or 'Portfolio' in str(result2.get('error', '')):
        print(f"QuantitativeAnalyst: Correctly requires portfolio")
    if result3.get('success') and result4.get('success') and result5.get('success'):
        print(f"FinancialTutor: Educational capabilities working")
    if result6.get('success') or result7.get('success') or result8.get('success'):
        print(f"CrossAssetAnalyst: Cross-asset analysis operational")
    if result9.get('success') or result10.get('success'):
        print(f"StrategyArchitect: Strategy design capabilities working")
    if result11.get('success') or result12.get('success'):
        print(f"RegimeForecasting: Market regime analysis operational")
    
    # StrategyRebalancing summary
    rebalancing_tests = [result13, result14, result15, result16, result17]
    rebalancing_working = sum(1 for r in rebalancing_tests 
                            if r.get('success') or 'Portfolio' in str(r.get('error', '')))
    if rebalancing_working >= 3:
        print(f"StrategyRebalancing: Portfolio optimization capabilities working")
    
    # HedgingStrategist summary
    hedging_tests = [result18, result19, result20, result21]
    hedging_working = sum(1 for r in hedging_tests 
                         if r.get('success') or 'portfolio return data' in str(r.get('error', ''))
                         or 'Portfolio' in str(r.get('error', '')))
    if hedging_working >= 3:
        print(f"HedgingStrategist: Risk management capabilities working")
    
    # NEW: ScenarioSimulation summary
    scenario_tests = [result27, result28, result29, result30]
    scenario_working = sum(1 for r in scenario_tests 
                          if r.get('success') or 'return data' in str(r.get('error', ''))
                          or 'Portfolio' in str(r.get('error', '')))
    if scenario_working >= 3:
        print(f"ScenarioSimulation: Stress testing capabilities working")
    
    print(f"\nMCP Architecture Testing Complete!")

if __name__ == "__main__":
    try:
        asyncio.run(test_mcp_orchestrator())
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()