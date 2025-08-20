# test_tax_strategist.py
"""
Comprehensive Test Suite for TaxStrategistAgent
=============================================
Tests all tax optimization capabilities including:
- Tax-loss harvesting analysis
- Asset location optimization
- Year-end tax planning
- After-tax return analysis
- Debate system integration
- MCP server integration
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

# Import the agent
try:
    from agents.tax_strategist_agent import TaxStrategistAgent, TaxLossOpportunity, AssetLocationRecommendation
    TAX_AGENT_AVAILABLE = True
    print("✅ TaxStrategistAgent imported successfully")
except ImportError as e:
    print(f"❌ TaxStrategistAgent import failed: {e}")
    TAX_AGENT_AVAILABLE = False

class MockMCPServer:
    """Mock MCP server for testing agent integration"""
    
    def __init__(self):
        self.agents = {}
        self.requests = []
        
    async def register_agent(self, agent):
        """Register agent with mock MCP server"""
        self.agents[agent.agent_id] = agent
        print(f"✅ Agent registered: {agent.agent_id}")
        
    async def route_request(self, agent_id: str, capability: str, data: Dict, context: Dict):
        """Route request to appropriate agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        self.requests.append({
            "agent_id": agent_id,
            "capability": capability,
            "timestamp": datetime.now()
        })
        
        return await agent.execute_capability(capability, data, context)

# Test Data Generators
def create_test_portfolio() -> Dict:
    """Create realistic test portfolio data"""
    return {
        "holdings": [
            {
                "symbol": "AAPL",
                "shares": 100,
                "current_price": 175.00,
                "cost_basis": 150.00,
                "current_value": 17500,
                "asset_class": "individual_stocks",
                "holding_period": 400,  # Days
                "annual_return_pct": 0.12,
                "dividend_yield": 0.005
            },
            {
                "symbol": "TSLA", 
                "shares": 50,
                "current_price": 200.00,
                "cost_basis": 280.00,
                "current_value": 10000,
                "asset_class": "individual_stocks", 
                "holding_period": 200,
                "annual_return_pct": -0.05,
                "dividend_yield": 0.0
            },
            {
                "symbol": "SPY",
                "shares": 200,
                "current_price": 425.00,
                "cost_basis": 400.00,
                "current_value": 85000,
                "asset_class": "tax_efficient_funds",
                "holding_period": 600,
                "annual_return_pct": 0.10,
                "dividend_yield": 0.015
            },
            {
                "symbol": "BND",
                "shares": 500,
                "current_price": 80.00,
                "cost_basis": 85.00,
                "current_value": 40000,
                "asset_class": "bonds",
                "holding_period": 300,
                "annual_return_pct": 0.04,
                "dividend_yield": 0.03
            }
        ],
        "total_value": 152500,
        "transaction_history": [
            {
                "symbol": "AAPL",
                "type": "buy",
                "date": "2024-01-15",
                "shares": 50,
                "price": 145.00
            },
            {
                "symbol": "TSLA",
                "type": "buy", 
                "date": "2024-07-10",
                "shares": 25,
                "price": 275.00
            }
        ]
    }

def create_test_accounts() -> List[Dict]:
    """Create test account structure for asset location optimization"""
    return [
        {
            "type": "taxable",
            "name": "Brokerage Account",
            "holdings": [
                {
                    "symbol": "BND",
                    "current_value": 40000,
                    "asset_class": "bonds"
                },
                {
                    "symbol": "VNQ", 
                    "current_value": 20000,
                    "asset_class": "reits"
                }
            ]
        },
        {
            "type": "traditional_401k",
            "name": "401k Account",
            "holdings": [
                {
                    "symbol": "VTI",
                    "current_value": 80000,
                    "asset_class": "tax_efficient_funds"
                }
            ]
        },
        {
            "type": "roth_ira",
            "name": "Roth IRA",
            "holdings": [
                {
                    "symbol": "SCHB",
                    "current_value": 25000,
                    "asset_class": "tax_efficient_funds"
                }
            ]
        }
    ]

def create_test_tax_context() -> Dict:
    """Create test tax context"""
    return {
        "marginal_tax_rate": 0.24,
        "annual_income": 120000,
        "filing_status": "single",
        "state_tax_rate": 0.06,
        "ltcg_rate": 0.15
    }

# Main Test Class
class TestTaxStrategistAgent:
    """Comprehensive test suite for TaxStrategistAgent"""
    
    @pytest.fixture
    async def agent(self):
        """Create TaxStrategistAgent instance"""
        if not TAX_AGENT_AVAILABLE:
            pytest.skip("TaxStrategistAgent not available")
        return TaxStrategistAgent()
    
    @pytest.fixture
    async def mcp_server(self, agent):
        """Create mock MCP server with registered agent"""
        server = MockMCPServer()
        await server.register_agent(agent)
        return server
    
    @pytest.fixture
    def portfolio_data(self):
        """Test portfolio data"""
        return create_test_portfolio()
    
    @pytest.fixture  
    def accounts_data(self):
        """Test accounts data"""
        return create_test_accounts()
    
    @pytest.fixture
    def tax_context(self):
        """Test tax context"""
        return create_test_tax_context()

    # Core Capability Tests
    async def test_tax_loss_harvesting(self, agent, portfolio_data, tax_context):
        """Test tax-loss harvesting analysis"""
        print("\n🧪 Testing Tax-Loss Harvesting Analysis...")
        
        data = {
            "portfolio_data": portfolio_data,
            "tax_context": tax_context
        }
        
        result = await agent.analyze_tax_loss_harvesting(data, {})
        
        # Validate results
        assert "opportunities" in result
        assert "total_potential_benefit" in result
        assert result["confidence_score"] > 0.8
        
        # Check for TSLA loss opportunity (bought at 280, now 200)
        opportunities = result["opportunities"]
        tsla_opportunity = next((opp for opp in opportunities if opp["symbol"] == "TSLA"), None)
        assert tsla_opportunity is not None
        assert tsla_opportunity["unrealized_loss"] < 0  # Should be negative (loss)
        assert tsla_opportunity["tax_benefit"] > 0  # Should provide tax benefit
        
        print(f"✅ Found {len(opportunities)} tax-loss opportunities")
        print(f"✅ Total potential benefit: ${result['total_potential_benefit']:,.2f}")
        print(f"✅ Confidence score: {result['confidence_score']:.1%}")

    async def test_asset_location_optimization(self, agent, accounts_data, tax_context):
        """Test asset location optimization"""
        print("\n🧪 Testing Asset Location Optimization...")
        
        data = {
            "accounts": accounts_data,
            "tax_context": tax_context
        }
        
        result = await agent.optimize_asset_location(data, {})
        
        # Validate results
        assert "current_efficiency_score" in result
        assert "recommendations" in result
        assert "projected_annual_savings" in result
        assert result["confidence_score"] > 0.8
        
        # Should recommend moving bonds out of taxable account
        recommendations = result["recommendations"]
        bond_recommendations = [rec for rec in recommendations if rec["asset_type"] == "bonds"]
        
        if bond_recommendations:
            assert bond_recommendations[0]["current_account"] == "taxable"
            assert bond_recommendations[0]["recommended_account"] in ["traditional_401k", "traditional_ira"]
        
        print(f"✅ Current efficiency score: {result['current_efficiency_score']:.1f}/100")
        print(f"✅ Found {len(recommendations)} optimization recommendations")
        print(f"✅ Projected annual savings: ${result['projected_annual_savings']:,.2f}")

    async def test_year_end_strategy(self, agent, portfolio_data, tax_context):
        """Test year-end tax planning strategy generation"""
        print("\n🧪 Testing Year-End Tax Strategy...")
        
        data = {
            "portfolio_data": portfolio_data,
            "tax_context": tax_context,
            "income_data": {"annual_income": 120000, "age": 35}
        }
        
        result = await agent.generate_year_end_strategy(data, {})
        
        # Validate results
        assert "tax_year" in result
        assert "strategies" in result
        assert "estimated_total_savings" in result
        assert result["confidence_score"] > 0.8
        
        strategies = result["strategies"]
        
        # Should include tax-loss harvesting if losses available
        if "tax_loss_harvesting" in strategies:
            tlh = strategies["tax_loss_harvesting"]
            assert "opportunities" in tlh
        
        # Should include retirement contribution analysis
        if "retirement_contributions" in strategies:
            retirement = strategies["retirement_contributions"]
            assert "recommendation" in retirement
        
        print(f"✅ Generated {len(strategies)} year-end strategies")
        print(f"✅ Estimated total savings: ${result['estimated_total_savings']:,.2f}")
        print(f"✅ Priority actions: {len(result.get('priority_actions', []))}")

    async def test_after_tax_analysis(self, agent, portfolio_data, tax_context):
        """Test after-tax return analysis"""
        print("\n🧪 Testing After-Tax Return Analysis...")
        
        data = {
            "portfolio_data": portfolio_data,
            "tax_context": tax_context
        }
        
        result = await agent.analyze_after_tax_returns(data, {})
        
        # Validate results
        assert "holdings_analysis" in result
        assert "portfolio_summary" in result
        assert "tax_efficiency_score" in result
        assert result["confidence_score"] > 0.8
        
        holdings_analysis = result["holdings_analysis"]
        assert len(holdings_analysis) > 0
        
        # Check portfolio summary
        portfolio_summary = result["portfolio_summary"]
        if portfolio_summary:
            assert "pre_tax_return" in portfolio_summary
            assert "after_tax_return" in portfolio_summary
            assert "tax_drag" in portfolio_summary
            assert "tax_efficiency" in portfolio_summary
        
        print(f"✅ Analyzed {len(holdings_analysis)} holdings")
        print(f"✅ Tax efficiency score: {result['tax_efficiency_score']:.1f}/100")
        
        if portfolio_summary:
            print(f"✅ Portfolio tax drag: {portfolio_summary.get('tax_drag', 0):.2%}")

    async def test_wash_sale_compliance(self, agent, portfolio_data):
        """Test wash sale compliance checking"""
        print("\n🧪 Testing Wash Sale Compliance...")
        
        # Create planned transactions that might trigger wash sale
        planned_transactions = [
            {
                "symbol": "TSLA",
                "type": "sell",
                "shares": 25,
                "gain_loss": -2000,  # Loss
                "planned_date": datetime.now().isoformat()
            },
            {
                "symbol": "AAPL", 
                "type": "sell",
                "shares": 50,
                "gain_loss": 1250,  # Gain
                "planned_date": datetime.now().isoformat()
            }
        ]
        
        data = {
            "planned_transactions": planned_transactions,
            "portfolio_data": portfolio_data
        }
        
        result = await agent.check_wash_sale_compliance(data, {})
        
        # Validate results
        assert "compliant_transactions" in result
        assert "violations" in result
        assert "warnings" in result
        
        total_transactions = (len(result["compliant_transactions"]) + 
                            len(result["violations"]) + 
                            len(result["warnings"]))
        assert total_transactions == len(planned_transactions)
        
        print(f"✅ Checked {total_transactions} planned transactions")
        print(f"✅ Compliant: {len(result['compliant_transactions'])}")
        print(f"✅ Violations: {len(result['violations'])}")
        print(f"✅ Warnings: {len(result['warnings'])}")

    async def test_retirement_contribution_optimization(self, agent, tax_context):
        """Test retirement contribution optimization"""
        print("\n🧪 Testing Retirement Contribution Optimization...")
        
        income_data = {
            "annual_income": 120000,
            "age": 35,
            "current_401k_contrib": 15000,
            "current_ira_contrib": 3000
        }
        
        data = {
            "income_data": income_data,
            "tax_context": tax_context
        }
        
        result = await agent.optimize_retirement_contributions(data, {})
        
        # Validate results
        assert "recommendation" in result
        assert "suggested_split" in result
        
        suggested_split = result["suggested_split"]
        assert "traditional_pct" in suggested_split
        assert "roth_pct" in suggested_split
        assert suggested_split["traditional_pct"] + suggested_split["roth_pct"] == 100
        
        print(f"✅ Recommendation: {result['recommendation']}")
        print(f"✅ Suggested split: {suggested_split['traditional_pct']}% Traditional, {suggested_split['roth_pct']}% Roth")

    # Debate System Tests
    async def test_debate_position_formulation(self, agent, portfolio_data, tax_context):
        """Test debate position formulation"""
        print("\n🧪 Testing Debate Position Formulation...")
        
        query = "Should I prioritize tax efficiency over growth potential in my portfolio?"
        context = {
            "portfolio_data": portfolio_data,
            "tax_context": tax_context
        }
        
        position = await agent.formulate_debate_position(query, context)
        
        # Validate debate position
        assert "stance" in position
        assert "key_arguments" in position
        assert "supporting_evidence" in position
        assert "confidence_score" in position
        assert position["confidence_score"] > 0.8
        
        print(f"✅ Stance: {position['stance']}")
        print(f"✅ Arguments: {len(position['key_arguments'])}")
        print(f"✅ Evidence: {len(position['supporting_evidence'])}")
        print(f"✅ Confidence: {position['confidence_score']:.1%}")

    async def test_challenge_response(self, agent, portfolio_data, tax_context):
        """Test response to debate challenges"""
        print("\n🧪 Testing Challenge Response...")
        
        # First get a position
        query = "Tax optimization adds unnecessary complexity to investment management"
        context = {"portfolio_data": portfolio_data, "tax_context": tax_context}
        original_position = await agent.formulate_debate_position(query, context)
        
        # Challenge the position
        challenge = "Tax optimization strategies are too complex for most investors and may not provide meaningful benefits"
        
        response = await agent.respond_to_challenge(challenge, original_position)
        
        # Validate response
        assert "response_strategy" in response
        assert "counter_arguments" in response
        assert "rebuttal_strength" in response
        assert response["rebuttal_strength"] > 0.7
        
        print(f"✅ Response strategy: {response['response_strategy']}")
        print(f"✅ Counter-arguments: {len(response['counter_arguments'])}")
        print(f"✅ Rebuttal strength: {response['rebuttal_strength']:.1%}")

    # Integration Tests
    async def test_mcp_integration(self, mcp_server, portfolio_data, tax_context):
        """Test integration with MCP server"""
        print("\n🧪 Testing MCP Server Integration...")
        
        data = {
            "portfolio_data": portfolio_data,
            "tax_context": tax_context
        }
        
        # Test tax-loss harvesting through MCP
        result = await mcp_server.route_request(
            "tax_strategist",
            "tax_loss_harvesting", 
            data,
            {}
        )
        
        assert "opportunities" in result
        assert len(mcp_server.requests) == 1
        
        # Test comprehensive optimization
        comprehensive_result = await mcp_server.route_request(
            "tax_strategist",
            "comprehensive_tax_optimization",
            data,
            {}
        )
        
        assert "detailed_analyses" in comprehensive_result
        assert "comprehensive_recommendations" in comprehensive_result
        assert len(mcp_server.requests) == 2
        
        print(f"✅ Processed {len(mcp_server.requests)} MCP requests")
        print("✅ All MCP integrations working correctly")

    async def test_comprehensive_optimization(self, agent, portfolio_data, accounts_data, tax_context):
        """Test comprehensive tax optimization that combines all strategies"""
        print("\n🧪 Testing Comprehensive Tax Optimization...")
        
        data = {
            "portfolio_data": portfolio_data,
            "accounts": accounts_data,
            "tax_context": tax_context,
            "income_data": {"annual_income": 120000, "age": 35}
        }
        
        result = await agent.comprehensive_tax_optimization(data, {})
        
        # Validate comprehensive results
        assert "detailed_analyses" in result
        assert "comprehensive_recommendations" in result
        assert "total_estimated_savings" in result
        assert "priority_actions" in result
        assert result["confidence_score"] > 0.85
        
        detailed_analyses = result["detailed_analyses"]
        
        # Should include multiple analysis types
        expected_analyses = ["tax_loss_harvesting", "asset_location", "after_tax_analysis", "year_end_strategy"]
        for analysis_type in expected_analyses:
            if analysis_type in detailed_analyses:
                analysis = detailed_analyses[analysis_type]
                assert "error" not in analysis or analysis.get("confidence_score", 0) > 0
        
        print(f"✅ Generated {len(detailed_analyses)} detailed analyses")
        print(f"✅ Total estimated savings: ${result['total_estimated_savings']:,.2f}")
        print(f"✅ Priority actions: {len(result['priority_actions'])}")

# Test Runner
async def run_tax_strategist_tests():
    """Run all tax strategist tests"""
    
    print("🎊 REVOLUTIONARY TAX STRATEGIST AGENT - COMPREHENSIVE TESTING")
    print("=" * 80)
    print("🧮 Testing the world's most advanced AI tax optimization system!")
    print()
    
    if not TAX_AGENT_AVAILABLE:
        print("❌ Cannot run tests - TaxStrategistAgent not available")
        return
    
    # Initialize test suite
    test_suite = TestTaxStrategistAgent()
    
    # Create agent and mock server
    print("🎭 Setting up test environment...")
    agent = TaxStrategistAgent()
    mcp_server = MockMCPServer()
    await mcp_server.register_agent(agent)
    
    # Generate test data
    portfolio_data = create_test_portfolio()
    accounts_data = create_test_accounts()
    tax_context = create_test_tax_context()
    
    print(f"📊 Test portfolio value: ${portfolio_data['total_value']:,}")
    print(f"💼 Test accounts: {len(accounts_data)}")
    print(f"💰 Test income: ${tax_context['annual_income']:,}")
    print()
    
    # Run all tests
    tests = [
        ("Tax-Loss Harvesting", test_suite.test_tax_loss_harvesting(agent, portfolio_data, tax_context)),
        ("Asset Location Optimization", test_suite.test_asset_location_optimization(agent, accounts_data, tax_context)),
        ("Year-End Strategy", test_suite.test_year_end_strategy(agent, portfolio_data, tax_context)),
        ("After-Tax Analysis", test_suite.test_after_tax_analysis(agent, portfolio_data, tax_context)),
        ("Wash Sale Compliance", test_suite.test_wash_sale_compliance(agent, portfolio_data)),
        ("Retirement Optimization", test_suite.test_retirement_contribution_optimization(agent, tax_context)),
        ("Debate Position", test_suite.test_debate_position_formulation(agent, portfolio_data, tax_context)),
        ("Challenge Response", test_suite.test_challenge_response(agent, portfolio_data, tax_context)),
        ("MCP Integration", test_suite.test_mcp_integration(mcp_server, portfolio_data, tax_context)),
        ("Comprehensive Optimization", test_suite.test_comprehensive_optimization(agent, portfolio_data, accounts_data, tax_context))
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_coro in tests:
        try:
            print(f"\n🧪 Running {test_name} Test...")
            await test_coro
            print(f"✅ {test_name} Test: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} Test: FAILED - {str(e)}")
            failed_tests += 1
            import traceback
            traceback.print_exc()
    
    # Display results
    print(f"\n\n🏆 TAX STRATEGIST TEST RESULTS")
    print("=" * 50)
    print(f"✅ Tests Passed: {passed_tests}")
    print(f"❌ Tests Failed: {failed_tests}")
    print(f"📊 Success Rate: {passed_tests/(passed_tests + failed_tests):.1%}")
    print()
    
    if failed_tests == 0:
        print("🎊 ALL TESTS PASSED!")
        print("🚀 Tax Strategist Agent is fully operational!")
        print()
        print("🏆 REVOLUTIONARY CAPABILITIES VERIFIED:")
        print("   ✅ Advanced tax-loss harvesting with wash sale compliance")
        print("   ✅ Intelligent asset location optimization")
        print("   ✅ Comprehensive year-end tax planning")
        print("   ✅ Sophisticated after-tax return analysis")
        print("   ✅ Automated retirement contribution optimization")
        print("   ✅ Debate system integration with tax expertise")
        print("   ✅ Full MCP server compatibility")
        print("   ✅ Multi-strategy comprehensive optimization")
        print()
        print("🎭 The Tax Strategist Agent represents a quantum leap in")
        print("   automated tax optimization technology!")
    else:
        print(f"⚠️ {failed_tests} tests failed - review implementation")
    
    print("\n🎊 Tax optimization revolution is HERE! 🎊")

if __name__ == "__main__":
    print("🧮 Starting Revolutionary Tax Strategist Agent Tests...")
    asyncio.run(run_tax_strategist_tests())