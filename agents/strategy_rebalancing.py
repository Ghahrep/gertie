# in agents/strategy_rebalancing.py
from typing import Dict, Any, Optional

from agents.base_agent import BaseFinancialAgent
from tools.strategy_tools import optimize_portfolio, generate_trade_orders

class StrategyRebalancingAgent(BaseFinancialAgent):
    """
    ### UPGRADED: Operates on real user portfolio data from the context.
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

        # 1. Parse objective from query
        query = user_query.lower()
        objective = 'MaximizeSharpe'
        min_vol_keywords = ["minimize risk", "minimize volatility", "safest", "low risk"]
        herc_keywords = ["herc", "hierarchical", "risk parity", "diversify risk"]
        if any(keyword in query for keyword in min_vol_keywords):
            objective = 'MinimizeVolatility'
        elif any(keyword in query for keyword in herc_keywords):
            objective = 'HERC'
        
        print(f"Inferred objective: {objective}")

        # 2. Call Optimizer Tool with data from context
        opt_result = self.tool_map["OptimizePortfolio"].run({
            "asset_prices": context["prices"],
            "objective": objective
        })
        if not opt_result or 'optimal_weights' not in opt_result:
            return {"success": False, "error": "The portfolio optimization step failed."}

        # 3. Call Trade Generator Tool with data from context
        # We need to transform the holdings list into the simple {ticker: value} format for the tool
        current_holdings_dict = {
            h.asset.ticker: h.market_value for h in context["holdings_with_values"]
        }
        trade_result = self.tool_map["GenerateTradeOrders"].run({
            "current_holdings": current_holdings_dict,
            "target_weights": opt_result["optimal_weights"],
            "total_portfolio_value": context["total_value"]
        })
        if not trade_result or 'trades' not in trade_result:
            return {"success": False, "error": "The trade generation step failed."}

        # 4. Format the final output
        trades = trade_result['trades']
        summary = "No rebalancing needed." if not trades else (
            f"### Portfolio Rebalancing Plan\n**Objective:** {objective.title()}\n\n**Recommended Trades:**\n" +
            "\n".join([f"- **{t['action']} ${t['amount_usd']:,.2f} of {t['ticker']}**" for t in trades])
        )

        return {"success": True, "summary": summary, "optimization_results": opt_result, "trade_plan": trades}