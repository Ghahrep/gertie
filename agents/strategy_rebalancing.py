# in agents/strategy_rebalancing.py
from typing import Dict, Any, Optional, Set
import re # <-- THE FIX: Add this import
import pandas as pd

from agents.base_agent import BaseFinancialAgent
from tools.strategy_tools import optimize_portfolio, generate_trade_orders

class StrategyRebalancingAgent(BaseFinancialAgent):
    """
    Operates on real user portfolio data from the context to generate rebalancing plans.
    """
    @property
    def name(self) -> str: return "StrategyRebalancingAgent"

    @property
    def purpose(self) -> str: return "Optimizes portfolios and generates actionable trade plans."

    def __init__(self):
        tools = [optimize_portfolio, generate_trade_orders]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")

        if not context or "prices" not in context:
            return {"success": False, "error": "Could not rebalance. Portfolio data is missing."}

        # Use robust word-set matching for intent parsing
        clean_query = re.sub(r'[^\w\s]', '', user_query.lower())
        query_words = set(clean_query.split())
        
        objective = 'MaximizeSharpe'  # Default objective

        min_vol_keywords: Set[Set[str]] = {frozenset(["minimize", "volatility"]), frozenset(["minimize", "risk"]), frozenset(["safest"])}
        herc_keywords: Set[Set[str]] = {frozenset(["herc"]), frozenset(["risk", "parity"]), frozenset(["diversify", "risk"])}

        if any(keyword_set.issubset(query_words) for keyword_set in min_vol_keywords):
            objective = 'MinimizeVolatility'
        elif any(keyword_set.issubset(query_words) for keyword_set in herc_keywords):
            objective = 'HERC'
        
        print(f"Inferred objective: {objective}")

        opt_result = self.tool_map["OptimizePortfolio"].invoke({
            "asset_prices": context["prices"],
            "objective": objective
        })
        if not opt_result or 'optimal_weights' not in opt_result:
            return {"success": False, "error": "The portfolio optimization step failed."}
        
        current_holdings_dict = { h.asset.ticker: h.market_value for h in context.get("holdings_with_values", []) }
        total_value = context.get("total_value", 0)

        trade_result = self.tool_map["GenerateTradeOrders"].invoke({
            "current_holdings": current_holdings_dict,
            "target_weights": opt_result["optimal_weights"],
            "total_portfolio_value": total_value
        })
        if not trade_result or 'trades' not in trade_result:
            return {"success": False, "error": "The trade generation step failed."}

        trades = trade_result['trades']
        summary = "No rebalancing is necessary." if not trades else (
            f"### Portfolio Rebalancing Plan\n**Objective:** {objective.title()}\n\n**Recommended Trades:**\n" +
            "\n".join([f"- **{t['action']} ${t['amount_usd']:,.2f} of {t['ticker']}**" for t in trades])
        )

        return {"success": True, "summary": summary, "optimization_results": opt_result, "trade_plan": trades, "agent_used": self.name}