# in agents/quantitative_analyst.py
from typing import Dict, Any, Optional

# Import our custom agent base class and the decorated tools
from agents.base_agent import BaseFinancialAgent
from tools.risk_tools import calculate_risk_metrics
from tools.strategy_tools import perform_factor_analysis

# The placeholder data function has been DELETED.

class QuantitativeAnalystAgent(BaseFinancialAgent):
    """
    ### UPGRADED: This agent now operates on real, user-specific portfolio data
    provided by the orchestrator.
    """
    @property
    def name(self) -> str:
        return "QuantitativeAnalyst"

    @property
    def purpose(self) -> str:
        return "Performs detailed statistical analysis on portfolios, including risk metrics and factor analysis."

    def __init__(self):
        tools = [
            calculate_risk_metrics,
            perform_factor_analysis,
        ]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Processes a user query to perform a quantitative analysis using the
        portfolio context provided by the orchestrator.
        """
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")

        # --- Agent "Thinking" Process ---
        # 1. Check if the necessary context was provided by the orchestrator
        if not context or "portfolio_returns" not in context:
            return {"success": False, "error": "Could not perform analysis. Portfolio data is missing from the context."}

        # 2. Use simple keyword-based routing to select the right tool
        query = user_query.lower()
        tool_to_use = None
        tool_args = {}

        if "risk" in query or "report" in query:
            tool_to_use = self.tool_map["CalculateRiskMetrics"]
            tool_args["portfolio_returns"] = context["portfolio_returns"]
        elif "factor" in query or "alpha" in query:
            # Factor analysis also needs factor data, which we should add to the data handler later.
            # For now, we'll assume it's part of the context or skip.
            return {"success": False, "error": "Factor analysis requires market factor data, which is not yet implemented."}

        if not tool_to_use:
            return {"success": False, "error": "I'm sorry, I don't have a tool for that specific analysis."}

        # 3. Execute the selected tool with the real data from the context
        print(f"Executing tool '{tool_to_use.name}' with real user portfolio data...")
        result = tool_to_use.run(tool_args)

        # 4. Format and return the result
        if result:
            result['summary'] = f"Successfully completed the '{tool_to_use.name}' analysis on your portfolio."
            return result
        else:
            return {"success": False, "error": f"The '{tool_to_use.name}' tool failed to execute."}