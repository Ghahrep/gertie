# in agents/quantitative_analyst.py
from typing import Dict, Any, Optional
import pandas as pd

from agents.base_agent import BaseFinancialAgent
from tools.risk_tools import calculate_risk_metrics, calculate_tail_risk_copula
from tools.strategy_tools import perform_factor_analysis
from tools.nlp_tools import summarize_analysis_results

class QuantitativeAnalystAgent(BaseFinancialAgent):
    """
    ### FINAL VERSION: A multi-talented analyst using modern .invoke() calls.
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

        query = user_query.lower()
        
        # Handle specific requests for stress testing
        if "stress test" in query or "tail risk" in query or "copula" in query:
            if "returns" not in context or context["returns"].shape[1] < 2:
                 return {"success": False, "error": "Stress testing requires a portfolio with at least two assets."}

            print("Executing tool 'CalculateTailRiskCopula'...")
            # ### REFINEMENT: Use .invoke() instead of .run() ###
            simulations = self.tool_map["CalculateTailRiskCopula"].invoke({"returns": context["returns"]})
            
            if simulations is None or simulations.empty:
                return {"success": False, "error": "Tail-risk stress test failed to generate scenarios."}

            weights = context.get("weights", pd.Series(dtype=float))
            stressed_returns = (simulations * weights).sum(axis=1)
            stressed_metrics = self.tool_map["CalculateRiskMetrics"].invoke({"portfolio_returns": stressed_returns})

            if not stressed_metrics:
                return {"success": False, "error": "Could not calculate risk metrics on the stress test results."}
            
            print("Executing tool 'SummarizeAnalysisResults' on stressed data...")
            summary_text = self.tool_map["SummarizeAnalysisResults"].invoke({
                "analysis_type": "Portfolio Tail-Risk Stress Test Report",
                "data": stressed_metrics 
            })

            stressed_metrics['summary'] = summary_text
            stressed_metrics['agent_used'] = self.name
            return stressed_metrics
            
        # Handle requests for a general risk report
        elif "risk" in query or "report" in query or "analyze" in query or "metrics" in query:
            print("Executing tool 'CalculateRiskMetrics'...")
            risk_result = self.tool_map["CalculateRiskMetrics"].invoke({"portfolio_returns": context["portfolio_returns"]})
            if not risk_result:
                return {"success": False, "error": "The 'CalculateRiskMetrics' tool failed."}

            print("Executing tool 'SummarizeAnalysisResults'...")
            summary_text = self.tool_map["SummarizeAnalysisResults"].invoke({
                "analysis_type": "Portfolio Risk Report",
                "data": risk_result 
            })
            
            risk_result['summary'] = summary_text
            risk_result['agent_used'] = self.name
            return risk_result
            
        else:
            return {"success": False, "error": "I'm a quantitative analyst. Please ask for a specific analysis like a 'risk report' or 'stress test'."}