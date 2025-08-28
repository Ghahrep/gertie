# in agents/scenario_simulation_agent.py
from typing import Dict, Any, Optional, List
import re

from agents.mcp_base_agent import MCPBaseAgent
from tools.scenario_tools import apply_market_shock
from tools.risk_tools import calculate_risk_metrics
from tools.nlp_tools import summarize_analysis_results

class ScenarioSimulationAgent(MCPBaseAgent):
    """
    MCP-compatible scenario simulation agent that models portfolio performance 
    under multiple macroeconomic and geopolitical stress scenarios.
    """
    
    def __init__(self):
        # Define MCP capabilities
        capabilities = [
            "stress_testing",
            "scenario_modeling", 
            "crisis_simulation",
            "tail_risk_analysis",
            "portfolio_resilience",
            "shock_analysis"
        ]
        
        super().__init__(
            agent_id="ScenarioSimulationAgent",
            agent_name="Scenario Simulation Specialist",
            capabilities=capabilities,
            max_concurrent_jobs=3
        )
        
        # Initialize scenario tools
        self.tools = [apply_market_shock, calculate_risk_metrics, summarize_analysis_results]
        self.tool_map = {
            "ApplyMarketShock": apply_market_shock,
            "CalculateRiskMetrics": calculate_risk_metrics,
            "SummarizeAnalysisResults": summarize_analysis_results,
        }
        
        # Predefined scenario templates
        self.scenario_templates = self._initialize_scenario_templates()
    
    @property
    def name(self) -> str: 
        return "Scenario Simulation Specialist"

    @property
    def purpose(self) -> str: 
        return "Simulates portfolio performance under various stress scenarios and crisis conditions."

    def _initialize_scenario_templates(self) -> Dict[str, Dict]:
        """Initialize predefined stress scenarios"""
        return {
            "market_crash": {
                "type": "market_crash",
                "impact_pct": -0.30,
                "description": "Global market crash (30% decline)",
                "recovery_time": 18,
                "correlation_increase": 0.85
            },
            "interest_rate_spike": {
                "type": "interest_rate_spike", 
                "impact_pct": -0.15,
                "description": "Rapid interest rate increase (200 bps)",
                "recovery_time": 12,
                "bond_impact": -0.25
            },
            "recession": {
                "type": "recession",
                "impact_pct": -0.25,
                "description": "Economic recession scenario",
                "recovery_time": 24,
                "earnings_impact": -0.40
            },
            "inflation_surge": {
                "type": "inflation_surge",
                "impact_pct": -0.10,
                "description": "Rapid inflation increase",
                "recovery_time": 15,
                "real_return_erosion": -0.05
            },
            "geopolitical_crisis": {
                "type": "geopolitical_crisis",
                "impact_pct": -0.20,
                "description": "Major geopolitical event",
                "recovery_time": 9,
                "volatility_spike": 2.5
            },
            "liquidity_crisis": {
                "type": "liquidity_crisis",
                "impact_pct": -0.35,
                "description": "Credit/liquidity freeze",
                "recovery_time": 30,
                "spread_widening": 0.15
            }
        }

    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Execute specific scenario simulation capability"""
        
        try:
            # Check for required portfolio data
            if not context or "returns" not in context:
                return {
                    "success": False,
                    "error": "missing_portfolio",
                    "user_message": "Scenario simulation requires portfolio return data. Please ensure you have an active portfolio with return history."
                }
            
            query = data.get("query", "")
            
            if capability == "stress_testing":
                return await self._comprehensive_stress_test(query, context)
            elif capability == "scenario_modeling":
                return await self._model_specific_scenario(query, context)
            elif capability == "crisis_simulation":
                return await self._simulate_crisis_scenario(query, context)
            elif capability == "tail_risk_analysis":
                return await self._analyze_tail_risks(query, context)
            elif capability == "portfolio_resilience":
                return await self._assess_portfolio_resilience(query, context)
            elif capability == "shock_analysis":
                return await self._analyze_market_shocks(query, context)
            else:
                return await self._general_scenario_analysis(query, context)
                
        except Exception as e:
            return {
                "success": False,
                "error": "execution_error",
                "user_message": f"Scenario simulation encountered an issue: {str(e)}"
            }

    async def _comprehensive_stress_test(self, query: str, context: Dict) -> Dict[str, Any]:
        """Execute comprehensive stress testing across multiple scenarios"""
        
        # Run multiple stress scenarios
        stress_results = {}
        key_scenarios = ["market_crash", "interest_rate_spike", "recession"]
        
        for scenario_name in key_scenarios:
            scenario = self.scenario_templates[scenario_name]
            result = await self._execute_scenario(scenario, context)
            if result.get("success"):
                stress_results[scenario_name] = result["result"]
        
        if not stress_results:
            return {
                "success": False,
                "error": "stress_test_failed",
                "user_message": "Unable to execute stress tests. Please check your portfolio data."
            }
        
        # Analyze worst-case scenario
        worst_case = min(stress_results.items(), 
                        key=lambda x: x[1]["portfolio_impact"])
        
        # Generate comprehensive assessment
        assessment = self._generate_stress_assessment(stress_results, worst_case)
        
        return {
            "success": True,
            "result": {
                "stress_test_results": stress_results,
                "worst_case_scenario": worst_case[0],
                "worst_case_impact": worst_case[1]["portfolio_impact"],
                "resilience_score": assessment["resilience_score"],
                "recommendations": assessment["recommendations"],
                "recovery_estimates": assessment["recovery_estimates"]
            },
            "confidence": 0.82,
            "agent_used": self.agent_id
        }

    async def _model_specific_scenario(self, query: str, context: Dict) -> Dict[str, Any]:
        """Model a specific scenario based on query"""
        
        # Extract scenario type from query
        scenario_name = self._identify_scenario_type(query)
        
        if not scenario_name:
            return {
                "success": False,
                "error": "scenario_not_identified",
                "user_message": "Please specify a scenario type (market crash, recession, interest rate spike, etc.)."
            }
        
        scenario = self.scenario_templates[scenario_name]
        
        # Execute the specific scenario
        result = await self._execute_scenario(scenario, context)
        
        if not result.get("success"):
            return result
        
        return {
            "success": True,
            "result": {
                "scenario_type": scenario_name,
                "scenario_description": scenario["description"],
                "portfolio_impact": result["result"]["portfolio_impact"],
                "risk_metrics": result["result"]["risk_metrics"],
                "recovery_time": scenario["recovery_time"],
                "mitigation_strategies": self._suggest_mitigation_strategies(scenario_name)
            },
            "confidence": 0.85,
            "agent_used": self.agent_id
        }

    async def _simulate_crisis_scenario(self, query: str, context: Dict) -> Dict[str, Any]:
        """Simulate severe crisis scenarios"""
        
        # Focus on severe crisis scenarios
        crisis_scenarios = ["market_crash", "liquidity_crisis", "geopolitical_crisis"]
        crisis_results = {}
        
        for scenario_name in crisis_scenarios:
            scenario = self.scenario_templates[scenario_name]
            result = await self._execute_scenario(scenario, context)
            if result.get("success"):
                crisis_results[scenario_name] = result["result"]
        
        if not crisis_results:
            return {
                "success": False,
                "error": "crisis_simulation_failed",
                "user_message": "Unable to simulate crisis scenarios."
            }
        
        # Analyze crisis impact
        max_loss = max(r["portfolio_impact"] for r in crisis_results.values())
        avg_recovery = sum(self.scenario_templates[s]["recovery_time"] 
                          for s in crisis_results.keys()) / len(crisis_results)
        
        return {
            "success": True,
            "result": {
                "crisis_scenarios": crisis_results,
                "maximum_loss": max_loss,
                "average_recovery_months": avg_recovery,
                "crisis_preparedness": "High" if max_loss > -0.25 else "Medium" if max_loss > -0.15 else "Low",
                "emergency_recommendations": self._generate_crisis_recommendations(max_loss)
            },
            "confidence": 0.78,
            "agent_used": self.agent_id
        }

    async def _analyze_tail_risks(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze tail risk scenarios and extreme events"""
        
        # Extreme scenarios for tail risk analysis
        extreme_scenarios = {
            "black_swan": {
                "type": "black_swan",
                "impact_pct": -0.50,
                "description": "Extreme tail event (50% decline)",
                "probability": 0.01
            },
            "systemic_collapse": {
                "type": "systemic_collapse", 
                "impact_pct": -0.60,
                "description": "Financial system collapse",
                "probability": 0.005
            },
            "hyperinflation": {
                "type": "hyperinflation",
                "impact_pct": -0.40,
                "description": "Currency/hyperinflation crisis",
                "probability": 0.02
            }
        }
        
        tail_results = {}
        for scenario_name, scenario in extreme_scenarios.items():
            result = await self._execute_scenario(scenario, context)
            if result.get("success"):
                tail_results[scenario_name] = {
                    **result["result"],
                    "probability": scenario["probability"],
                    "expected_loss": result["result"]["portfolio_impact"] * scenario["probability"]
                }
        
        # Calculate tail risk metrics
        var_99 = min(r["portfolio_impact"] for r in tail_results.values())
        expected_tail_loss = sum(r["expected_loss"] for r in tail_results.values())
        
        return {
            "success": True,
            "result": {
                "tail_scenarios": tail_results,
                "value_at_risk_99": var_99,
                "expected_tail_loss": expected_tail_loss,
                "tail_risk_rating": "Extreme" if var_99 < -0.45 else "High" if var_99 < -0.30 else "Moderate",
                "hedging_recommendations": self._suggest_tail_risk_hedging(var_99)
            },
            "confidence": 0.75,
            "agent_used": self.agent_id
        }

    async def _assess_portfolio_resilience(self, query: str, context: Dict) -> Dict[str, Any]:
        """Assess overall portfolio resilience across scenarios"""
        
        # Test resilience across all scenario types
        resilience_tests = {}
        total_impact = 0
        scenario_count = 0
        
        for scenario_name, scenario in self.scenario_templates.items():
            result = await self._execute_scenario(scenario, context)
            if result.get("success"):
                impact = result["result"]["portfolio_impact"]
                recovery_time = scenario["recovery_time"]
                
                resilience_tests[scenario_name] = {
                    "impact": impact,
                    "recovery_months": recovery_time,
                    "resilience_score": self._calculate_resilience_score(impact, recovery_time)
                }
                
                total_impact += abs(impact)
                scenario_count += 1
        
        if scenario_count == 0:
            return {
                "success": False,
                "error": "resilience_test_failed",
                "user_message": "Unable to assess portfolio resilience."
            }
        
        # Calculate overall resilience metrics
        avg_impact = total_impact / scenario_count
        overall_resilience = sum(r["resilience_score"] for r in resilience_tests.values()) / scenario_count
        
        # Generate resilience rating
        if overall_resilience > 0.8:
            resilience_rating = "Excellent"
        elif overall_resilience > 0.6:
            resilience_rating = "Good" 
        elif overall_resilience > 0.4:
            resilience_rating = "Fair"
        else:
            resilience_rating = "Poor"
        
        return {
            "success": True,
            "result": {
                "resilience_tests": resilience_tests,
                "overall_resilience_score": overall_resilience,
                "resilience_rating": resilience_rating,
                "average_impact": avg_impact,
                "resilience_recommendations": self._generate_resilience_recommendations(overall_resilience),
                "strongest_scenarios": [k for k, v in resilience_tests.items() if v["resilience_score"] > 0.7],
                "weakest_scenarios": [k for k, v in resilience_tests.items() if v["resilience_score"] < 0.4]
            },
            "confidence": 0.88,
            "agent_used": self.agent_id
        }

    async def _analyze_market_shocks(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze impact of specific market shocks"""
        
        # Extract shock parameters from query
        shock_type = self._identify_scenario_type(query)
        
        if not shock_type:
            # Default to market crash analysis
            shock_type = "market_crash"
        
        scenario = self.scenario_templates[shock_type]
        result = await self._execute_scenario(scenario, context)
        
        if not result.get("success"):
            return result
        
        # Enhanced shock analysis
        shock_analysis = {
            "immediate_impact": result["result"]["portfolio_impact"],
            "volatility_change": result["result"]["risk_metrics"].get("volatility", 0) * 1.5,
            "correlation_effects": "Correlations increase to 0.85+ during shock",
            "liquidity_impact": "Moderate" if abs(scenario["impact_pct"]) < 0.25 else "Severe",
            "sector_effects": self._analyze_sector_effects(shock_type)
        }
        
        return {
            "success": True,
            "result": {
                "shock_type": shock_type,
                "shock_analysis": shock_analysis,
                "recovery_timeline": scenario["recovery_time"],
                "adaptation_strategies": self._suggest_adaptation_strategies(shock_type),
                "monitoring_metrics": self._suggest_monitoring_metrics(shock_type)
            },
            "confidence": 0.83,
            "agent_used": self.agent_id
        }

    async def _general_scenario_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle general scenario analysis queries"""
        
        # Default to comprehensive stress testing
        return await self._comprehensive_stress_test(query, context)

    async def _execute_scenario(self, scenario: Dict, context: Dict) -> Dict[str, Any]:
        """Execute a specific scenario simulation"""
        
        try:
            # Apply market shock
            shocked_returns = self.tool_map["ApplyMarketShock"].invoke({
                "returns": context["returns"],
                "shock_scenario": scenario
            })
            
            if shocked_returns is None:
                return {
                    "success": False,
                    "error": "shock_application_failed"
                }
            
            # Calculate stressed portfolio returns
            weights = context.get("weights", None)
            if weights is not None:
                stressed_portfolio_returns = (shocked_returns * weights).sum(axis=1)
            else:
                # Equal weight assumption
                stressed_portfolio_returns = shocked_returns.mean(axis=1)
            
            # Calculate risk metrics on shocked data
            stressed_metrics = self.tool_map["CalculateRiskMetrics"].invoke({
                "portfolio_returns": stressed_portfolio_returns
            })
            
            if not stressed_metrics:
                return {
                    "success": False,
                    "error": "risk_calculation_failed"
                }
            
            # Calculate portfolio impact
            original_returns = context["returns"]
            if weights is not None:
                original_portfolio_returns = (original_returns * weights).sum(axis=1)
            else:
                original_portfolio_returns = original_returns.mean(axis=1)
            
            original_value = original_portfolio_returns.mean()
            stressed_value = stressed_portfolio_returns.mean()
            portfolio_impact = (stressed_value - original_value) / abs(original_value) if original_value != 0 else 0
            
            return {
                "success": True,
                "result": {
                    "portfolio_impact": portfolio_impact,
                    "risk_metrics": stressed_metrics,
                    "stressed_returns": stressed_portfolio_returns
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": "scenario_execution_failed",
                "details": str(e)
            }

    def _identify_scenario_type(self, query: str) -> Optional[str]:
        """Identify scenario type from query"""
        
        query_lower = query.lower()
        
        scenario_keywords = {
            "market_crash": ["market crash", "crash", "market decline", "bear market"],
            "interest_rate_spike": ["interest rate", "rate spike", "fed hiking", "monetary tightening"],
            "recession": ["recession", "economic downturn", "gdp decline", "economic crisis"],
            "inflation_surge": ["inflation", "hyperinflation", "price surge", "currency debasement"],
            "geopolitical_crisis": ["geopolitical", "war", "conflict", "political crisis"],
            "liquidity_crisis": ["liquidity crisis", "credit freeze", "funding crisis", "banking crisis"]
        }
        
        for scenario_name, keywords in scenario_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return scenario_name
        
        return None

    def _generate_stress_assessment(self, stress_results: Dict, worst_case: tuple) -> Dict:
        """Generate stress test assessment"""
        
        impacts = [result["portfolio_impact"] for result in stress_results.values()]
        avg_impact = sum(impacts) / len(impacts)
        
        # Calculate resilience score (0-1, higher is better)
        resilience_score = max(0, 1 + avg_impact / 0.5)  # Normalize around -50% worst case
        
        recommendations = []
        if resilience_score < 0.5:
            recommendations.append("Consider increasing defensive allocations")
            recommendations.append("Implement hedging strategies for tail risk protection")
        if abs(worst_case[1]["portfolio_impact"]) > 0.3:
            recommendations.append("Diversify across uncorrelated asset classes")
            
        recovery_estimates = {
            scenario: self.scenario_templates[scenario]["recovery_time"]
            for scenario in stress_results.keys()
        }
        
        return {
            "resilience_score": resilience_score,
            "recommendations": recommendations,
            "recovery_estimates": recovery_estimates
        }

    def _suggest_mitigation_strategies(self, scenario_name: str) -> List[str]:
        """Suggest mitigation strategies for specific scenarios"""
        
        strategies = {
            "market_crash": [
                "Increase allocation to defensive assets (bonds, utilities)",
                "Implement put option hedging strategies",
                "Maintain higher cash reserves for opportunities"
            ],
            "interest_rate_spike": [
                "Reduce duration risk in bond allocations", 
                "Consider floating rate instruments",
                "Hedge with interest rate derivatives"
            ],
            "recession": [
                "Focus on recession-resistant sectors",
                "Increase quality factor allocation",
                "Consider counter-cyclical investments"
            ],
            "inflation_surge": [
                "Increase real asset allocation (commodities, REITs)",
                "Consider inflation-protected securities",
                "Reduce long-duration fixed income"
            ]
        }
        
        return strategies.get(scenario_name, ["Increase diversification", "Implement risk management strategies"])

    def _generate_crisis_recommendations(self, max_loss: float) -> List[str]:
        """Generate recommendations based on crisis impact"""
        
        if max_loss < -0.4:
            return [
                "URGENT: Implement immediate hedging strategies",
                "Consider reducing risk exposure significantly", 
                "Maintain substantial cash reserves",
                "Diversify across crisis-resistant asset classes"
            ]
        elif max_loss < -0.25:
            return [
                "Consider moderate hedging strategies",
                "Increase defensive allocation",
                "Monitor crisis indicators closely"
            ]
        else:
            return [
                "Current portfolio shows good crisis resilience",
                "Continue monitoring risk metrics",
                "Consider minor defensive adjustments"
            ]

    def _suggest_tail_risk_hedging(self, var_99: float) -> List[str]:
        """Suggest tail risk hedging strategies"""
        
        if var_99 < -0.4:
            return [
                "Implement tail risk hedging with put spreads",
                "Consider VIX-based hedging instruments",
                "Allocate to crisis-alpha strategies"
            ]
        else:
            return [
                "Monitor tail risk metrics regularly",
                "Consider modest portfolio insurance",
                "Maintain diversified risk sources"
            ]

    def _calculate_resilience_score(self, impact: float, recovery_time: int) -> float:
        """Calculate resilience score for a scenario"""
        
        # Score based on impact (lower impact = higher score) and recovery time (shorter = higher score)
        impact_score = max(0, 1 + impact / 0.5)  # Normalize around -50%
        recovery_score = max(0, 1 - recovery_time / 36)  # Normalize around 36 months
        
        return (impact_score + recovery_score) / 2

    def _generate_resilience_recommendations(self, overall_resilience: float) -> List[str]:
        """Generate recommendations based on resilience score"""
        
        if overall_resilience < 0.4:
            return [
                "PRIORITY: Implement comprehensive risk management framework",
                "Significantly increase portfolio diversification",
                "Consider professional risk assessment consultation"
            ]
        elif overall_resilience < 0.7:
            return [
                "Enhance risk management practices",
                "Consider additional hedging strategies",
                "Monitor portfolio stress metrics regularly"
            ]
        else:
            return [
                "Portfolio shows strong resilience",
                "Continue current risk management approach",
                "Regular stress testing recommended"
            ]

    def _analyze_sector_effects(self, shock_type: str) -> Dict[str, str]:
        """Analyze sector-specific effects of shocks"""
        
        sector_effects = {
            "market_crash": {
                "Technology": "Severe impact due to high valuations",
                "Healthcare": "Moderate impact - defensive characteristics",
                "Financials": "Severe impact from credit concerns"
            },
            "interest_rate_spike": {
                "Utilities": "Severe impact from rate sensitivity",
                "REITs": "Severe impact from higher discount rates",
                "Banks": "Mixed impact - higher rates vs credit concerns"
            },
            "recession": {
                "Consumer Discretionary": "Severe impact from reduced spending",
                "Energy": "High impact from demand destruction",
                "Consumer Staples": "Low impact - defensive nature"
            }
        }
        
        return sector_effects.get(shock_type, {"Overall Market": "Broad-based impact expected"})

    def _suggest_adaptation_strategies(self, shock_type: str) -> List[str]:
        """Suggest adaptation strategies for specific shocks"""
        
        strategies = {
            "market_crash": [
                "Implement systematic rebalancing during decline",
                "Gradually increase equity exposure as valuations improve",
                "Focus on quality companies with strong balance sheets"
            ],
            "interest_rate_spike": [
                "Ladder bond maturities to reinvest at higher rates",
                "Shift to variable rate instruments",
                "Consider rate-hedged equity strategies"
            ]
        }
        
        return strategies.get(shock_type, [
            "Monitor market conditions closely",
            "Adjust portfolio allocation based on new environment",
            "Maintain disciplined approach to risk management"
        ])

    def _suggest_monitoring_metrics(self, shock_type: str) -> List[str]:
        """Suggest key metrics to monitor for specific shocks"""
        
        metrics = {
            "market_crash": ["VIX volatility index", "Credit spreads", "Market correlation"],
            "interest_rate_spike": ["Yield curve shape", "Duration exposure", "Rate volatility"],
            "recession": ["Economic indicators", "Earnings revisions", "Credit quality metrics"]
        }
        
        return metrics.get(shock_type, ["Portfolio volatility", "Correlation measures", "Risk metrics"])

    def _generate_summary(self, result: Dict, capability: str, execution_time: float) -> str:
        """Generate user-friendly summary for scenario results"""
        
        if not result.get("success"):
            error_type = result.get("error", "unknown")
            if error_type == "missing_portfolio":
                return "Portfolio Required: Scenario simulation requires portfolio return data. Please ensure you have an active portfolio with return history."
            else:
                return f"Error: {result.get('user_message', 'Scenario simulation encountered an issue.')}"
        
        data = result.get("result", {})
        
        if capability == "stress_testing":
            worst_case = data.get("worst_case_scenario", "unknown")
            worst_impact = data.get("worst_case_impact", 0)
            resilience_score = data.get("resilience_score", 0)
            
            return f"""Comprehensive Stress Testing Complete ({execution_time:.1f}s)

**Worst Case Scenario**: {worst_case.replace('_', ' ').title()}
**Maximum Portfolio Impact**: {worst_impact:.1%}
**Resilience Score**: {resilience_score:.1f}/1.0

**Assessment**: {"Strong resilience" if resilience_score > 0.7 else "Moderate resilience" if resilience_score > 0.4 else "Weak resilience"} across tested scenarios.

**Key Recommendations**:
{chr(10).join([f"• {rec}" for rec in data.get("recommendations", [])])}"""

        elif capability == "crisis_simulation":
            max_loss = data.get("maximum_loss", 0)
            preparedness = data.get("crisis_preparedness", "Unknown")
            
            return f"""Crisis Simulation Analysis ({execution_time:.1f}s)

**Maximum Crisis Loss**: {max_loss:.1%}
**Crisis Preparedness**: {preparedness}

**Emergency Preparedness**: {"Well prepared" if max_loss > -0.15 else "Moderately prepared" if max_loss > -0.25 else "Needs improvement"}

**Crisis Response Plan**:
{chr(10).join([f"• {rec}" for rec in data.get("emergency_recommendations", [])])}"""

        else:
            scenario_type = data.get("scenario_type", "scenario")
            impact = data.get("portfolio_impact", 0) or data.get("immediate_impact", 0)
            
            return f"""Scenario Analysis Complete ({execution_time:.1f}s)

**Scenario**: {scenario_type.replace('_', ' ').title()}
**Portfolio Impact**: {impact:.1%}

**Analysis**: Portfolio shows {"strong resilience" if abs(impact) < 0.15 else "moderate resilience" if abs(impact) < 0.30 else "significant vulnerability"} to this scenario."""

    async def _health_check_capability(self, capability: str) -> Dict:
        """Health check for specific scenario capability"""
        
        capability_checks = {
            "stress_testing": {"status": "operational", "response_time": 0.8},
            "scenario_modeling": {"status": "operational", "response_time": 0.6},
            "crisis_simulation": {"status": "operational", "response_time": 1.0},
            "tail_risk_analysis": {"status": "operational", "response_time": 0.9},
            "portfolio_resilience": {"status": "operational", "response_time": 1.2},
            "shock_analysis": {"status": "operational", "response_time": 0.7}
        }
        
        return capability_checks.get(capability, {"status": "unknown", "response_time": 0.8})

    # Legacy method for backward compatibility
    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Legacy run method for backward compatibility"""
        
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        if not context or "returns" not in context:
            return {"success": False, "error": "Could not run simulation. Portfolio data is missing."}

        # Parse query to define shock scenario
        query = user_query.lower()
        scenario = {}
        if "interest rate" in query or "rate spike" in query:
            scenario = {'type': 'interest_rate_spike', 'impact_pct': -0.05}
        elif "market crash" in query or "downturn" in query:
            scenario = {'type': 'market_crash', 'impact_pct': -0.20}
        else:
            return {"success": False, "error": "Please specify a valid scenario (e.g., 'market crash' or 'interest rate spike')."}

        # Three-step tool chain execution
        # Apply market shock
        shocked_returns = self.tool_map["ApplyMarketShock"].invoke({
            "returns": context["returns"],
            "shock_scenario": scenario
        })
        if shocked_returns is None:
            return {"success": False, "error": "Failed to apply market shock."}

        # Calculate risk metrics on shocked data
        weights = context.get("weights")
        stressed_portfolio_returns = (shocked_returns * weights).sum(axis=1)
        stressed_metrics = self.tool_map["CalculateRiskMetrics"].invoke({
            "portfolio_returns": stressed_portfolio_returns
        })
        if not stressed_metrics:
            return {"success": False, "error": "Failed to calculate risk metrics on shocked data."}

        # Summarize results using NLP tool
        summary = self.tool_map["SummarizeAnalysisResults"].invoke({
            "analysis_type": f"Scenario Simulation Report ({scenario['type']})",
            "data": stressed_metrics
        })

        stressed_metrics['summary'] = summary
        stressed_metrics['agent_used'] = self.name
        return stressed_metrics