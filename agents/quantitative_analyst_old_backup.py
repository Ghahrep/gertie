# agents/quantitative_analyst.py

from typing import Dict, Any, Optional
import pandas as pd

from agents.base_agent import BaseFinancialAgent
from tools.risk_tools import calculate_risk_metrics, calculate_tail_risk_copula
from tools.strategy_tools import perform_factor_analysis
from tools.nlp_tools import summarize_analysis_results

class QuantitativeAnalystAgent(BaseFinancialAgent):
    """
    A multi-talented analyst that performs and summarizes detailed statistical analysis.
    """
    @property
    def name(self) -> str: return "QuantitativeAnalyst"

    @property
    def purpose(self) -> str: return "Performs and summarizes detailed statistical analysis on portfolios."

    def __init__(self):
        tools = [
            calculate_risk_metrics,
            perform_factor_analysis,
            calculate_tail_risk_copula,
            summarize_analysis_results,
        ]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")

        if not context or "portfolio_returns" not in context:
            return {"success": False, "error": "Could not perform analysis. Portfolio data is missing."}

        # --- THIS IS THE CORE FIX ---
        # We will now handle all logic within a try/except block to ensure
        # a properly formatted response is always returned to the API route.

        try:
            query = user_query.lower()
            
            # This logic remains the same
            if "stress test" in query or "tail risk" in query or "copula" in query:
                # ... (stress test logic is unchanged)
                 if "returns" not in context or context["returns"].shape[1] < 2:
                    return {"success": False, "error": "Stress testing requires a portfolio with at least two assets."}

                 print("Executing tool 'CalculateTailRiskCopula'...")
                 simulations = self.tool_map["CalculateTailRiskCopula"].invoke({"returns": context["returns"]})
                 
                 if simulations is None or simulations.empty:
                     return {"success": False, "error": "Tail-risk stress test failed to generate scenarios."}

                 weights = context.get("weights", pd.Series(dtype=float))
                 stressed_returns = (simulations * weights).sum(axis=1)
                 analysis_result = self.tool_map["CalculateRiskMetrics"].invoke({"portfolio_returns": stressed_returns})
                 analysis_type = "Portfolio Tail-Risk Stress Test Report"

            else: # Default to a general risk report
                print("Executing tool 'CalculateRiskMetrics'...")
                analysis_result = self.tool_map["CalculateRiskMetrics"].invoke({"portfolio_returns": context["portfolio_returns"]})
                analysis_type = "Portfolio Risk Report"

            if not analysis_result:
                return {"success": False, "error": "The financial metrics calculation failed."}

            print("Executing tool 'SummarizeAnalysisResults'...")
            summary_text = self.tool_map["SummarizeAnalysisResults"].invoke({
                "analysis_type": analysis_type,
                "data": analysis_result 
            })

            # --- CRITICAL CHANGE HERE ---
            # We now construct a final, successful dictionary that the API expects.
            return {
                "success": True,
                "summary": summary_text,
                "data": analysis_result, # Include the raw data for the frontend
                "agent_used": self.name
            }

        except Exception as e:
            # If anything goes wrong in the process, we catch it and return a formatted error.
            print(f"An unexpected error occurred in the QuantitativeAnalystAgent: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"An internal error occurred in the analyst agent: {str(e)}"}