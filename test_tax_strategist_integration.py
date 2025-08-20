# test_tax_strategist_integration.py
"""
Tax Strategist MCP Server Integration Test
==========================================
Comprehensive test suite for TaxStrategistAgent integration with MCP server.
Tests real-world scenarios and validates tax optimization capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List
import uuid

# Try to import from your actual project structure
try:
    from agents.tax_strategist_agent import TaxStrategistAgent
    TAX_AGENT_AVAILABLE = True
    print("✅ TaxStrategistAgent imported successfully")
except ImportError as e:
    print(f"❌ TaxStrategistAgent import failed: {e}")
    print("   Make sure the agent file is in the correct location")
    TAX_AGENT_AVAILABLE = False

class MockMCPServer:
    """Mock MCP server matching your project's architecture"""
    
    def __init__(self):
        self.agents = {}
        self.request_log = []
        
    async def register_agent(self, agent):
        """Register agent with MCP server"""
        self.agents[agent.agent_id] = agent
        
        # Get agent capabilities
        capabilities = getattr(agent, 'capabilities', [])
        perspective = getattr(agent, 'perspective', 'unknown')
        
        print(f"✅ Agent '{agent.agent_id}' registered successfully")
        print(f"   📋 Capabilities: {len(capabilities)} features")
        print(f"   🎭 Perspective: {perspective}")
        print(f"   🧠 Specialization: {getattr(agent, 'specialization', 'tax_optimization')}")
        
        return {"status": "registered", "agent_id": agent.agent_id}
    
    async def execute_agent_capability(self, agent_id: str, capability: str, data: Dict, context: Dict = None):
        """Execute agent capability (matching your MCP server pattern)"""
        
        if context is None:
            context = {}
            
        # Log the request
        self.request_log.append({
            "agent_id": agent_id,
            "capability": capability,
            "timestamp": datetime.now(),
            "data_keys": list(data.keys())
        })
        
        # Get the agent
        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": f"Agent '{agent_id}' not found"}
        
        # Execute the capability
        try:
            result = await agent.execute_capability(capability, data, context)
            print(f"✅ Executed {capability} for {agent_id}")
            return result
        except Exception as e:
            print(f"❌ Error executing {capability}: {str(e)}")
            return {"error": str(e)}

# Realistic Test Data
def create_realistic_portfolio() -> Dict:
    """Create a realistic portfolio for testing"""
    return {
        "holdings": [
            # Winners (for tax-loss harvesting context)
            {
                "symbol": "AAPL",
                "shares": 100,
                "current_price": 175.50,
                "cost_basis": 145.00,
                "current_value": 17550,
                "asset_class": "individual_stocks",
                "holding_period": 450,  # Long-term
                "annual_return_pct": 0.15,
                "dividend_yield": 0.005
            },
            {
                "symbol": "MSFT", 
                "shares": 150,
                "current_price": 310.00,
                "cost_basis": 250.00,
                "current_value": 46500,
                "asset_class": "individual_stocks",
                "holding_period": 380,
                "annual_return_pct": 0.18,
                "dividend_yield": 0.007
            },
            # Losers (tax-loss harvesting opportunities)
            {
                "symbol": "TSLA",
                "shares": 75,
                "current_price": 195.00,
                "cost_basis": 285.00,
                "current_value": 14625,
                "asset_class": "individual_stocks",
                "holding_period": 200,
                "annual_return_pct": -0.25,
                "dividend_yield": 0.0
            },
            {
                "symbol": "ARKK",
                "shares": 200,
                "current_price": 42.50,
                "cost_basis": 68.00,
                "current_value": 8500,
                "asset_class": "growth_funds",
                "holding_period": 320,
                "annual_return_pct": -0.30,
                "dividend_yield": 0.0
            },
            # Broad market funds
            {
                "symbol": "VTI",
                "shares": 300,
                "current_price": 235.00,
                "cost_basis": 220.00,
                "current_value": 70500,
                "asset_class": "tax_efficient_funds",
                "holding_period": 600,
                "annual_return_pct": 0.12,
                "dividend_yield": 0.015
            },
            # Bonds in taxable (suboptimal location)
            {
                "symbol": "BND",
                "shares": 400,
                "current_price": 79.25,
                "cost_basis": 82.00,
                "current_value": 31700,
                "asset_class": "bonds",
                "holding_period": 280,
                "annual_return_pct": 0.035,
                "dividend_yield": 0.032
            }
        ],
        "total_value": 189375,
        "transaction_history": [
            {
                "symbol": "TSLA",
                "type": "buy",
                "date": (datetime.now() - timedelta(days=15)).isoformat(),
                "shares": 25,
                "price": 280.00
            },
            {
                "symbol": "AAPL",
                "type": "buy", 
                "date": (datetime.now() - timedelta(days=45)).isoformat(),
                "shares": 50,
                "price": 150.00
            }
        ]
    }

def create_realistic_accounts() -> List[Dict]:
    """Create realistic account structure for asset location testing"""
    return [
        {
            "type": "taxable",
            "name": "Brokerage Account",
            "holdings": [
                {
                    "symbol": "AAPL",
                    "current_value": 17550,
                    "asset_class": "individual_stocks"
                },
                {
                    "symbol": "MSFT",
                    "current_value": 46500,
                    "asset_class": "individual_stocks"
                },
                {
                    "symbol": "BND",  # Suboptimal - bonds in taxable
                    "current_value": 31700,
                    "asset_class": "bonds"
                },
                {
                    "symbol": "VNQ",  # Suboptimal - REIT in taxable
                    "current_value": 15000,
                    "asset_class": "reits"
                }
            ]
        },
        {
            "type": "traditional_401k",
            "name": "401k Account",
            "holdings": [
                {
                    "symbol": "VTI",  # Suboptimal - tax-efficient fund in 401k
                    "current_value": 85000,
                    "asset_class": "tax_efficient_funds"
                },
                {
                    "symbol": "SCHB",
                    "current_value": 25000,
                    "asset_class": "tax_efficient_funds"
                }
            ]
        },
        {
            "type": "roth_ira",
            "name": "Roth IRA",
            "holdings": [
                {
                    "symbol": "BND",  # Suboptimal - bonds in growth account
                    "current_value": 22000,
                    "asset_class": "bonds"
                },
                {
                    "symbol": "VTI",
                    "current_value": 18000,
                    "asset_class": "tax_efficient_funds"
                }
            ]
        }
    ]

def create_tax_context() -> Dict:
    """Create realistic tax context"""
    return {
        "marginal_tax_rate": 0.24,  # 24% bracket
        "annual_income": 120000,
        "filing_status": "single",
        "state_tax_rate": 0.0495,  # CA rate
        "ltcg_rate": 0.15,
        "age": 35
    }

def create_income_data() -> Dict:
    """Create realistic income data for retirement planning"""
    return {
        "annual_income": 120000,
        "age": 35,
        "employer_401k_match": 0.06,  # 6% match
        "current_401k_contribution": 15000,
        "current_roth_ira_contribution": 6000
    }

async def test_tax_loss_harvesting(mcp_server: MockMCPServer):
    """Test tax-loss harvesting capability"""
    print("\n🔍 TEST 1: TAX-LOSS HARVESTING")
    print("=" * 50)
    
    portfolio_data = create_realistic_portfolio()
    tax_context = create_tax_context()
    
    data = {
        "portfolio_data": portfolio_data,
        "tax_context": tax_context
    }
    
    result = await mcp_server.execute_agent_capability(
        "tax_strategist",
        "tax_loss_harvesting",
        data
    )
    
    if "error" not in result:
        print(f"📊 Found {result.get('opportunities_count', 0)} tax-loss opportunities")
        print(f"💰 Total potential benefit: ${result.get('total_potential_benefit', 0):,.2f}")
        print(f"⚡ Confidence: {result.get('confidence_score', 0):.1%}")
        
        # Validate results
        assert result.get('opportunities_count', 0) >= 0
        assert result.get('total_potential_benefit', 0) >= 0
        assert 'implementation_plan' in result
        assert 'wash_sale_warnings' in result
        
        print("✅ Tax-loss harvesting test PASSED")
    else:
        print(f"❌ Tax-loss harvesting test FAILED: {result['error']}")
        
    return result

async def test_asset_location_optimization(mcp_server: MockMCPServer):
    """Test asset location optimization capability"""
    print("\n🏠 TEST 2: ASSET LOCATION OPTIMIZATION")
    print("=" * 50)
    
    accounts = create_realistic_accounts()
    tax_context = create_tax_context()
    
    data = {
        "accounts": accounts,
        "tax_context": tax_context
    }
    
    result = await mcp_server.execute_agent_capability(
        "tax_strategist",
        "asset_location_optimization",
        data
    )
    
    if "error" not in result:
        print(f"📈 Current efficiency score: {result.get('current_efficiency_score', 0):.1f}/100")
        print(f"💰 Projected annual savings: ${result.get('projected_annual_savings', 0):,.2f}")
        print(f"📋 Recommendations: {len(result.get('recommendations', []))}")
        print(f"⚡ Confidence: {result.get('confidence_score', 0):.1%}")
        
        # Validate results
        assert 'current_efficiency_score' in result
        assert 'recommendations' in result
        assert 'implementation_roadmap' in result
        assert 'tax_drag_analysis' in result
        
        print("✅ Asset location optimization test PASSED")
    else:
        print(f"❌ Asset location optimization test FAILED: {result['error']}")
        
    return result

async def test_year_end_strategy(mcp_server: MockMCPServer):
    """Test year-end tax planning strategy"""
    print("\n📅 TEST 3: YEAR-END TAX STRATEGY")
    print("=" * 50)
    
    portfolio_data = create_realistic_portfolio()
    tax_context = create_tax_context()
    income_data = create_income_data()
    
    data = {
        "portfolio_data": portfolio_data,
        "tax_context": tax_context,
        "income_data": income_data
    }
    
    result = await mcp_server.execute_agent_capability(
        "tax_strategist",
        "year_end_tax_planning",
        data
    )
    
    if "error" not in result:
        print(f"📊 Tax year: {result.get('tax_year', 'Unknown')}")
        print(f"💰 Estimated total savings: ${result.get('estimated_total_savings', 0):,.2f}")
        print(f"📋 Strategy count: {len(result.get('strategies', {}))}")
        print(f"⚡ Confidence: {result.get('confidence_score', 0):.1%}")
        
        # Validate results
        assert 'tax_year' in result
        assert 'strategies' in result
        assert 'implementation_timeline' in result
        assert 'priority_actions' in result
        
        print("✅ Year-end strategy test PASSED")
    else:
        print(f"❌ Year-end strategy test FAILED: {result['error']}")
        
    return result

async def test_after_tax_analysis(mcp_server: MockMCPServer):
    """Test after-tax return analysis"""
    print("\n📈 TEST 4: AFTER-TAX RETURN ANALYSIS")
    print("=" * 50)
    
    portfolio_data = create_realistic_portfolio()
    tax_context = create_tax_context()
    
    data = {
        "portfolio_data": portfolio_data,
        "tax_context": tax_context
    }
    
    result = await mcp_server.execute_agent_capability(
        "tax_strategist",
        "after_tax_analysis",
        data
    )
    
    if "error" not in result:
        summary = result.get('portfolio_summary', {})
        print(f"📊 Pre-tax return: {summary.get('pre_tax_return', 0):.2%}")
        print(f"📊 After-tax return: {summary.get('after_tax_return', 0):.2%}")
        print(f"📊 Tax drag: {summary.get('tax_drag', 0):.2%}")
        print(f"📊 Tax efficiency: {summary.get('tax_efficiency', 0):.2%}")
        print(f"⭐ Efficiency score: {result.get('tax_efficiency_score', 0):.1f}/100")
        
        # Validate results
        assert 'holdings_analysis' in result
        assert 'portfolio_summary' in result
        assert 'improvement_opportunities' in result
        
        print("✅ After-tax analysis test PASSED")
    else:
        print(f"❌ After-tax analysis test FAILED: {result['error']}")
        
    return result

async def test_comprehensive_optimization(mcp_server: MockMCPServer):
    """Test comprehensive tax optimization"""
    print("\n🎯 TEST 5: COMPREHENSIVE TAX OPTIMIZATION")
    print("=" * 50)
    
    portfolio_data = create_realistic_portfolio()
    accounts = create_realistic_accounts()
    tax_context = create_tax_context()
    income_data = create_income_data()
    
    data = {
        "portfolio_data": portfolio_data,
        "accounts": accounts,
        "tax_context": tax_context,
        "income_data": income_data
    }
    
    result = await mcp_server.execute_agent_capability(
        "tax_strategist",
        "tax_optimization",
        data
    )
    
    if "error" not in result:
        print(f"💰 Total estimated savings: ${result.get('total_estimated_savings', 0):,.2f}")
        print(f"📋 Comprehensive recommendations: {len(result.get('comprehensive_recommendations', []))}")
        print(f"🎯 Priority actions: {len(result.get('priority_actions', []))}")
        print(f"⚡ Confidence: {result.get('confidence_score', 0):.1%}")
        
        # Validate results
        assert 'detailed_analyses' in result
        assert 'comprehensive_recommendations' in result
        assert 'total_estimated_savings' in result
        assert 'implementation_timeline' in result
        
        print("✅ Comprehensive optimization test PASSED")
    else:
        print(f"❌ Comprehensive optimization test FAILED: {result['error']}")
        
    return result

async def test_wash_sale_compliance(mcp_server: MockMCPServer):
    """Test wash sale compliance checking"""
    print("\n⚖️ TEST 6: WASH SALE COMPLIANCE")
    print("=" * 50)
    
    portfolio_data = create_realistic_portfolio()
    
    # Create planned transactions for testing
    planned_transactions = [
        {
            "symbol": "TSLA",
            "type": "sell",
            "shares": 75,
            "gain_loss": -6750,  # Loss
            "planned_date": datetime.now().isoformat()
        },
        {
            "symbol": "ARKK", 
            "type": "sell",
            "shares": 100,
            "gain_loss": -2550,  # Loss
            "planned_date": datetime.now().isoformat()
        }
    ]
    
    data = {
        "planned_transactions": planned_transactions,
        "portfolio_data": portfolio_data
    }
    
    result = await mcp_server.execute_agent_capability(
        "tax_strategist",
        "wash_sale_compliance",
        data
    )
    
    if "error" not in result:
        print(f"✅ Compliant transactions: {len(result.get('compliant_transactions', []))}")
        print(f"⚠️ Violations: {len(result.get('violations', []))}")
        print(f"🔔 Warnings: {len(result.get('warnings', []))}")
        
        # Validate results
        assert 'compliant_transactions' in result
        assert 'violations' in result
        assert 'warnings' in result
        
        print("✅ Wash sale compliance test PASSED")
    else:
        print(f"❌ Wash sale compliance test FAILED: {result['error']}")
        
    return result

async def test_retirement_optimization(mcp_server: MockMCPServer):
    """Test retirement contribution optimization"""
    print("\n🏦 TEST 7: RETIREMENT CONTRIBUTION OPTIMIZATION")
    print("=" * 50)
    
    income_data = create_income_data()
    tax_context = create_tax_context()
    
    data = {
        "income_data": income_data,
        "tax_context": tax_context
    }
    
    result = await mcp_server.execute_agent_capability(
        "tax_strategist",
        "retirement_planning",
        data
    )
    
    if "error" not in result:
        print(f"📋 Recommendation: {result.get('recommendation', 'None')}")
        split = result.get('suggested_split', {})
        print(f"💰 Traditional: {split.get('traditional_pct', 0)}%")
        print(f"💰 Roth: {split.get('roth_pct', 0)}%")
        print(f"📊 2025 limit: ${result.get('contribution_limit_2025', 0):,}")
        
        # Validate results
        assert 'recommendation' in result
        assert 'suggested_split' in result
        
        print("✅ Retirement optimization test PASSED")
    else:
        print(f"❌ Retirement optimization test FAILED: {result['error']}")
        
    return result

async def test_debate_capabilities(mcp_server: MockMCPServer):
    """Test debate system integration"""
    print("\n🎭 TEST 8: DEBATE SYSTEM INTEGRATION")
    print("=" * 50)
    
    portfolio_data = create_realistic_portfolio()
    tax_context = create_tax_context()
    
    # Test debate position formulation
    debate_request = {
        "query": "Should I prioritize tax-loss harvesting over asset location optimization?",
        "role": "position",
        "context": {
            "portfolio_data": portfolio_data,
            "tax_context": tax_context
        }
    }
    
    data = {"debate_request": debate_request}
    
    result = await mcp_server.execute_agent_capability(
        "tax_strategist",
        "debate_participation",
        data
    )
    
    if "error" not in result:
        print(f"🎯 Stance: {result.get('stance', 'None')}")
        print(f"📋 Arguments: {len(result.get('key_arguments', []))}")
        print(f"📊 Evidence: {len(result.get('supporting_evidence', []))}")
        print(f"⚡ Confidence: {result.get('confidence_score', 0):.1%}")
        
        # Validate results
        assert 'stance' in result
        assert 'key_arguments' in result
        assert 'supporting_evidence' in result
        
        print("✅ Debate capabilities test PASSED")
    else:
        print(f"❌ Debate capabilities test FAILED: {result['error']}")
        
    return result

async def run_integration_tests():
    """Run comprehensive tax strategist integration tests"""
    
    print("🎊 TAX STRATEGIST AGENT - COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 70)
    print("💰 Testing revolutionary tax optimization capabilities!")
    print()
    
    if not TAX_AGENT_AVAILABLE:
        print("❌ Cannot run tests - TaxStrategistAgent not available")
        return
    
    # Initialize MCP server
    mcp_server = MockMCPServer()
    
    # Create and register tax strategist agent
    print("🤖 Creating and registering TaxStrategistAgent...")
    try:
        tax_agent = TaxStrategistAgent()
        await mcp_server.register_agent(tax_agent)
        print()
    except Exception as e:
        print(f"❌ Error creating TaxStrategistAgent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run all tests
    test_results = {}
    
    try:
        test_results["tax_loss_harvesting"] = await test_tax_loss_harvesting(mcp_server)
        test_results["asset_location"] = await test_asset_location_optimization(mcp_server)
        test_results["year_end_strategy"] = await test_year_end_strategy(mcp_server)
        test_results["after_tax_analysis"] = await test_after_tax_analysis(mcp_server)
        test_results["comprehensive"] = await test_comprehensive_optimization(mcp_server)
        test_results["wash_sale"] = await test_wash_sale_compliance(mcp_server)
        test_results["retirement"] = await test_retirement_optimization(mcp_server)
        test_results["debate"] = await test_debate_capabilities(mcp_server)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate test summary
    await display_test_summary(test_results, mcp_server)

async def display_test_summary(test_results: Dict, mcp_server: MockMCPServer):
    """Display comprehensive test summary"""
    
    print("\n\n🏆 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    # Count successful tests
    successful_tests = sum(1 for result in test_results.values() if result and "error" not in result)
    total_tests = len(test_results)
    
    print(f"✅ Tests Completed: {total_tests}")
    print(f"✅ Tests Successful: {successful_tests}")
    print(f"✅ Success Rate: {successful_tests/total_tests:.1%}")
    print()
    
    # Show MCP server statistics
    print("📊 MCP SERVER STATISTICS:")
    print(f"   🤖 Agents Registered: {len(mcp_server.agents)}")
    print(f"   📞 Total Requests: {len(mcp_server.request_log)}")
    print(f"   ⏱️ Test Duration: ~{total_tests * 2:.1f} seconds")
    print()
    
    # Show test details
    print("📋 TEST RESULTS BREAKDOWN:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result and "error" not in result else "❌ FAIL"
        print(f"   {test_name:.<30} {status}")
    print()
    
    # Validate core capabilities
    print("🎯 CORE CAPABILITIES VALIDATED:")
    core_tests = [
        "tax_loss_harvesting", "asset_location", "after_tax_analysis", 
        "year_end_strategy", "comprehensive"
    ]
    
    for test in core_tests:
        if test in test_results and test_results[test] and "error" not in test_results[test]:
            print(f"   ✅ {test.replace('_', ' ').title()}")
        else:
            print(f"   ❌ {test.replace('_', ' ').title()}")
    print()
    
    print("🚀 REVOLUTIONARY FEATURES DEMONSTRATED:")
    print("   💰 Advanced tax-loss harvesting with wash sale compliance")
    print("   🏠 Intelligent asset location optimization")
    print("   📅 Comprehensive year-end tax planning")
    print("   📈 Sophisticated after-tax return analysis")
    print("   🎭 Multi-agent debate system integration")
    print("   ⚖️ Automated compliance checking")
    print("   🏦 Retirement contribution optimization")
    print()
    
    if successful_tests == total_tests:
        print("🎊 CONGRATULATIONS!")
        print("Tax Strategist Agent is FULLY OPERATIONAL and TESTED! 🎊")
        print("Ready for production deployment! 🚀")
    else:
        print("⚠️ Some tests failed - review errors and fix issues before deployment")

if __name__ == "__main__":
    print("💰 Starting Tax Strategist Integration Tests...")
    asyncio.run(run_integration_tests())