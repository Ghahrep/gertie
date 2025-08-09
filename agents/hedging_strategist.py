# in agents/hedging_strategist.py
import yfinance as yf
from typing import Dict, Any, Optional

from agents.base_agent import BaseFinancialAgent
from tools.risk_tools import calculate_volatility_budget
from tools.strategy_tools import find_optimal_hedge

class HedgingStrategistAgent(BaseFinancialAgent):
    """
    ### UPGRADED: Operates on real user portfolio data from the context.
    """
    @property
    def name(self) -> str: return "HedgingStrategist"

    @property
    def purpose(self) -> str: return "Provides tactical risk management advice."

    def __init__(self):
        tools = [find_optimal_hedge, calculate_volatility_budget]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        if not context or "portfolio_returns" not in context:
            return {"success": False, "error": "Could not provide hedging advice. Portfolio data is missing."}

        query = user_query.lower()
        tool_to_use, tool_args = None, {"portfolio_returns": context["portfolio_returns"]}

        # 1. Parse query for intent and parameters
        if "hedge" in query or "protect" in query:
            tool_to_use = self.tool_map["FindOptimalHedge"]
            # Fetch data for a common hedge instrument (inverse S&P 500 ETF)
            print("Fetching market data for hedge instrument 'SH'...")
            hedge_returns = yf.download('SH', period="1y", auto_adjust=True)['Close'].pct_change()
            tool_args['hedge_instrument_returns'] = hedge_returns
        elif "volatility" in query or "target" in query:
            tool_to_use = self.tool_map["CalculateVolatilityBudget"]
            try:
                target_vol_str = [word for word in query.replace('%',' ').split() if word.replace('.','').isdigit()][0]
                tool_args['target_volatility'] = float(target_vol_str) / 100.0
            except IndexError:
                return {"success": False, "error": "Please specify a target volatility (e.g., 'target 15%')."}

        if not tool_to_use: return {"success": False, "error": "I can only help with hedging or volatility targeting."}

        # 2. Execute tool
        result = tool_to_use.run(tool_args)

        # 3. Format output
        if result:
            # ... (summary logic is the same as before, no changes needed)
            if tool_to_use.name == "FindOptimalHedge":
                result['summary'] = (f"To hedge your portfolio with 'SH', the optimal ratio is {result['optimal_hedge_ratio']:.4f}. "
                                     f"This is expected to reduce daily volatility by {result['volatility_reduction_pct']:.2f}%.")
            elif tool_to_use.name == "CalculateVolatilityBudget":
                result['summary'] = (f"To achieve your target of {result['target_annual_volatility']:.1%}, "
                                     f"allocate {result['risky_asset_weight']:.1%} to your portfolio and "
                                     f"{result['risk_free_asset_weight']:.1%} to a risk-free asset.")
        else:
            result = {"success": False, "error": f"The '{tool_to_use.name}' tool failed."}
        
        return result