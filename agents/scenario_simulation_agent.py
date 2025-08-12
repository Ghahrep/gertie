# in agents/scenario_simulation_agent.py
from typing import Dict, Any, Optional
from agents.base_agent import BaseFinancialAgent
from tools.scenario_tools import apply_market_shock
from tools.risk_tools import calculate_risk_metrics
from tools.nlp_tools import summarize_analysis_results

class ScenarioSimulationAgent(BaseFinancialAgent):
    """
    Models portfolio performance under multiple macroeconomic and geopolitical scenarios.
    """
    @property
    def name(self) -> str: return "ScenarioSimulationAgent"

    @property
    def purpose(self) -> str: return "Simulates how a portfolio might perform under various stress scenarios."

    def __init__(self):
        tools = [
            apply_market_shock,
            calculate_risk_metrics,
            summarize_analysis_results,
        ]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        if not context or "returns" not in context:
            return {"success": False, "error": "Could not run simulation. Portfolio data is missing."}

        # 1. Parse the query to define a shock scenario
        query = user_query.lower()
        scenario = {}
        if "interest rate" in query or "rate spike" in query:
            scenario = {'type': 'interest_rate_spike', 'impact_pct': -0.05}
        elif "market crash" in query or "downturn" in query:
            scenario = {'type': 'market_crash', 'impact_pct': -0.20}
        else:
            return {"success": False, "error": "Please specify a valid scenario (e.g., 'market crash' or 'interest rate spike')."}

        # --- Three-Step Tool Chain ---
        # 2. Apply the shock to the portfolio's historical returns
        shocked_returns = self.tool_map["ApplyMarketShock"].invoke({
            "returns": context["returns"],
            "shock_scenario": scenario
        })
        if shocked_returns is None:
            return {"success": False, "error": "Failed to apply market shock."}

        # 3. Calculate risk metrics on the NEW, shocked data
        weights = context.get("weights")
        stressed_portfolio_returns = (shocked_returns * weights).sum(axis=1)
        stressed_metrics = self.tool_map["CalculateRiskMetrics"].invoke({
            "portfolio_returns": stressed_portfolio_returns
        })
        if not stressed_metrics:
            return {"success": False, "error": "Failed to calculate risk metrics on shocked data."}

        # 4. Summarize the results using the NLP tool
        summary = self.tool_map["SummarizeAnalysisResults"].invoke({
            "analysis_type": f"Scenario Simulation Report ({scenario['type']})",
            "data": stressed_metrics
        })

        stressed_metrics['summary'] = summary
        stressed_metrics['agent_used'] = self.name
        return stressed_metrics