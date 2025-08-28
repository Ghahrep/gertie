# test_three_new_mcp_agents.py
"""
Test script for the three newly migrated MCP agents:
- BehavioralFinanceAgent
- EconomicDataAgent (already MCP compatible)
- TaxStrategistAgent

This script validates MCP capabilities and integration.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_behavioral_finance_agent():
    """Test BehavioralFinanceAgent MCP capabilities"""
    print("\n" + "="*60)
    print("TESTING BEHAVIORAL FINANCE AGENT (MCP)")
    print("="*60)
    
    try:
        from agents.behavioral_finance_agent import BehavioralFinanceAgent
        agent = BehavioralFinanceAgent()
        
        # Test 1: Health Check
        print("\n--- Test 1: Health Check ---")
        health = await agent.health_check()
        print(f"Health Status: {health.get('status')}")
        print(f"Capabilities: {health.get('capabilities', [])}")
        
        # Test 2: Bias Identification
        print("\n--- Test 2: Bias Identification ---")
        test_data = {
            "query": "I'm certain that tech stocks will always outperform bonds. Everyone says so and I've never seen them lose money."
        }
        test_context = {"chat_history": [
            {"content": "I always buy stocks at their 52-week high because that's when they're proven winners"},
            {"content": "I can't afford to lose any money on this investment, it's too risky"},
            {"content": "All the popular stocks on social media are doing great"}
        ]}
        
        result = await agent.run(test_data["query"], context=test_context)
        print(f"Success: {result.get('success')}")
        print(f"Agent: {result.get('agent_used')}")
        if result.get('summary'):
            summary_preview = result['summary'][:200] + "..." if len(result['summary']) > 200 else result['summary']
            print(f"Summary Preview: {summary_preview}")
        
        # Test 3: Behavioral Pattern Analysis
        print("\n--- Test 3: Behavioral Pattern Analysis ---")
        pattern_data = {"query": "analyze my investment behavior patterns"}
        result = await agent.run(pattern_data["query"], context=test_context)
        print(f"Success: {result.get('success')}")
        print(f"Analysis Type: {result.get('analysis_type', 'N/A')}")
        
        # Test 4: Investment Behavior Assessment
        print("\n--- Test 4: Investment Behavior Assessment ---")
        assessment_data = {
            "query": "assess my overall investment decision-making",
            "portfolio_data": {
                "holdings": [
                    {"symbol": "AAPL", "current_value": 10000, "asset_class": "individual_stocks"},
                    {"symbol": "SPY", "current_value": 5000, "asset_class": "etf"}
                ]
            }
        }
        result = await agent.run(assessment_data["query"], context=test_context)
        print(f"Success: {result.get('success')}")
        
        print("\n✅ BehavioralFinanceAgent MCP tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ BehavioralFinanceAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_economic_data_agent():
    """Test EconomicDataAgent MCP capabilities"""
    print("\n" + "="*60)
    print("TESTING ECONOMIC DATA AGENT (MCP)")
    print("="*60)
    
    try:
        from agents.economic_data_agent import EconomicDataAgent
        agent = EconomicDataAgent()
        
        # Test 1: Health Check
        print("\n--- Test 1: Health Check ---")
        health = await agent.health_check()
        print(f"Health Status: {health.get('status')}")
        print(f"Capabilities: {len(agent.capabilities)} capabilities")
        
        # Test 2: Economic Indicators Analysis
        print("\n--- Test 2: Economic Indicators Analysis ---")
        result = await agent.run("analyze current economic indicators", context={})
        print(f"Success: {result.get('success')}")
        print(f"Agent: {result.get('agent_used')}")
        print(f"Economic Outlook: {result.get('economic_outlook', 'N/A')}")
        print(f"Data Source: {result.get('data_source', 'N/A')}")
        
        # Test 3: Federal Reserve Policy Assessment
        print("\n--- Test 3: Fed Policy Assessment ---")
        result = await agent.run("assess current Fed policy stance", context={})
        print(f"Success: {result.get('success')}")
        print(f"Fed Funds Rate: {result.get('federal_funds_rate', 'N/A')}%")
        print(f"Policy Stance: {result.get('policy_stance', 'N/A')}")
        
        # Test 4: Global Correlations Analysis
        print("\n--- Test 4: Global Correlations Analysis ---")
        result = await agent.run("analyze global asset correlations", context={})
        print(f"Success: {result.get('success')}")
        print(f"Diversification Score: {result.get('diversification_score', 'N/A')}/10")
        
        # Test 5: Yield Curve Analysis
        print("\n--- Test 5: Yield Curve Analysis ---")
        result = await agent.run("analyze the current yield curve", context={})
        print(f"Success: {result.get('success')}")
        print(f"Curve Status: {result.get('status', 'N/A')}")
        print(f"10Y-2Y Spread: {result.get('10y_2y_spread', 'N/A')}")
        
        print("\n✅ EconomicDataAgent MCP tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ EconomicDataAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_tax_strategist_agent():
    """Test TaxStrategistAgent MCP capabilities"""
    print("\n" + "="*60)
    print("TESTING TAX STRATEGIST AGENT (MCP)")
    print("="*60)
    
    try:
        from agents.tax_strategist_agent import TaxStrategistAgent
        agent = TaxStrategistAgent()
        
        # Test 1: Health Check
        print("\n--- Test 1: Health Check ---")
        health = await agent.health_check()
        print(f"Health Status: {health.get('status')}")
        print(f"Capabilities: {len(agent.capabilities)} capabilities")
        
        # Sample portfolio data for testing
        sample_portfolio = {
            "holdings": [
                {
                    "symbol": "AAPL",
                    "current_price": 180.0,
                    "cost_basis": 200.0,
                    "shares": 100,
                    "current_value": 18000,
                    "holding_period": 400,
                    "annual_return_pct": 0.12,
                    "dividend_yield": 0.005,
                    "asset_class": "individual_stocks"
                },
                {
                    "symbol": "TSLA", 
                    "current_price": 250.0,
                    "cost_basis": 180.0,
                    "shares": 50,
                    "current_value": 12500,
                    "holding_period": 200,
                    "annual_return_pct": 0.15,
                    "dividend_yield": 0.0,
                    "asset_class": "individual_stocks"
                }
            ]
        }
        
        sample_tax_context = {
            "marginal_tax_rate": 0.24,
            "annual_income": 120000
        }
        
        sample_accounts = [
            {
                "type": "taxable",
                "holdings": [
                    {"asset_class": "bonds", "current_value": 10000},
                    {"asset_class": "reits", "current_value": 5000}
                ]
            },
            {
                "type": "traditional_401k", 
                "holdings": [
                    {"asset_class": "individual_stocks", "current_value": 15000}
                ]
            }
        ]
        
        # Test 2: Tax-Loss Harvesting Analysis
        print("\n--- Test 2: Tax-Loss Harvesting Analysis ---")
        test_data = {
            "portfolio_data": sample_portfolio,
            "tax_context": sample_tax_context
        }
        result = await agent.run("analyze tax-loss harvesting opportunities", context=test_data)
        print(f"Success: {result.get('success')}")
        print(f"Agent: {result.get('agent_used')}")
        opportunities_count = result.get('opportunities_count', 0)
        total_benefit = result.get('total_potential_benefit', 0)
        print(f"Opportunities Found: {opportunities_count}")
        print(f"Total Tax Benefit: ${total_benefit:.2f}")
        
        # Test 3: Asset Location Optimization
        print("\n--- Test 3: Asset Location Optimization ---")
        asset_data = {
            "accounts": sample_accounts,
            "tax_context": sample_tax_context
        }
        result = await agent.run("optimize my asset location across accounts", context=asset_data)
        print(f"Success: {result.get('success')}")
        efficiency_score = result.get('current_efficiency_score', 0)
        projected_savings = result.get('projected_annual_savings', 0)
        print(f"Current Efficiency: {efficiency_score:.1f}/100")
        print(f"Projected Savings: ${projected_savings:.2f}")
        
        # Test 4: After-Tax Analysis
        print("\n--- Test 4: After-Tax Analysis ---")
        result = await agent.run("analyze my portfolio's after-tax returns", context=test_data)
        print(f"Success: {result.get('success')}")
        if result.get('portfolio_summary'):
            portfolio_summary = result['portfolio_summary']
            print(f"Tax Efficiency: {portfolio_summary.get('tax_efficiency', 0):.2%}")
            print(f"Tax Drag: {portfolio_summary.get('tax_drag', 0):.2%}")
        
        # Test 5: Year-End Tax Planning
        print("\n--- Test 5: Year-End Tax Planning ---")
        year_end_data = {
            **test_data,
            "income_data": {"annual_income": 120000, "age": 35}
        }
        result = await agent.run("generate year-end tax strategy", context=year_end_data)
        print(f"Success: {result.get('success')}")
        print(f"Tax Year: {result.get('tax_year', 'N/A')}")
        total_savings = result.get('estimated_total_savings', 0)
        print(f"Estimated Total Savings: ${total_savings:.2f}")
        
        # Test 6: Comprehensive Tax Optimization
        print("\n--- Test 6: Comprehensive Tax Optimization ---")
        result = await agent.run("perform comprehensive tax optimization", context=year_end_data)
        print(f"Success: {result.get('success')}")
        comprehensive_savings = result.get('total_estimated_savings', 0)
        print(f"Comprehensive Savings: ${comprehensive_savings:.2f}")
        
        print("\n✅ TaxStrategistAgent MCP tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ TaxStrategistAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_orchestrator_integration():
    """Test orchestrator integration with new agents"""
    print("\n" + "="*60)
    print("TESTING ORCHESTRATOR INTEGRATION")
    print("="*60)
    
    try:
        from agents.orchestrator import FinancialOrchestrator
        orchestrator = FinancialOrchestrator()
        
        print("\n--- Orchestrator Agent Registry ---")
        print(f"Total Agents: {len(orchestrator.roster)}")
        print(f"MCP Agents: {len(orchestrator.mcp_agents)}")
        print("MCP Agent List:", list(orchestrator.mcp_agents))
        
        # Test routing to new agents
        test_queries = [
            ("analyze my investment biases", "BehavioralFinanceAgent"),
            ("what's the current economic outlook", "EconomicDataAgent"), 
            ("optimize my taxes", "TaxStrategistAgent"),
            ("help with tax planning", "TaxStrategistAgent"),
            ("assess Fed policy", "EconomicDataAgent"),
            ("identify behavioral patterns", "BehavioralFinanceAgent")
        ]
        
        print("\n--- Routing Tests ---")
        for query, expected_agent in test_queries:
            # Test orchestrator routing logic
            clean_query = query.lower().replace(".", "").replace("?", "")
            query_words = set(clean_query.split())
            selected_agent = orchestrator._classify_query_enhanced(query_words, query)
            
            print(f"Query: '{query}'")
            print(f"Expected: {expected_agent} | Actual: {selected_agent}")
            
            # Verify the selected agent exists in roster
            if selected_agent and selected_agent in orchestrator.roster:
                agent_instance = orchestrator.roster[selected_agent]
                print(f"✅ Agent '{selected_agent}' found - {agent_instance.agent_name}")
            else:
                print(f"❌ Agent '{selected_agent}' not found in roster")
            print()
        
        print("✅ Orchestrator integration tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Orchestrator integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all MCP agent tests"""
    print("🚀 STARTING MCP AGENT MIGRATION TESTS")
    print("Testing three newly migrated agents...")
    
    results = []
    
    # Test each agent individually
    results.append(await test_behavioral_finance_agent())
    results.append(await test_economic_data_agent()) 
    results.append(await test_tax_strategist_agent())
    results.append(await test_orchestrator_integration())
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    agent_names = ["BehavioralFinanceAgent", "EconomicDataAgent", "TaxStrategistAgent", "Orchestrator Integration"]
    
    for i, (agent_name, result) in enumerate(zip(agent_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{agent_name}: {status}")
    
    total_passed = sum(results)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 ALL MCP AGENT MIGRATIONS SUCCESSFUL!")
        print("\nNext Steps:")
        print("1. Update orchestrator.py to add these agents to mcp_agents set")
        print("2. Test routing inconsistencies in the orchestrator")
        print("3. Update handoff documentation with new agent count")
    else:
        print(f"\n⚠️  {len(results) - total_passed} tests failed. Review errors above.")
        
    return total_passed == len(results)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()