# in agents/strategy_architect.py
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseFinancialAgent
# ### UPGRADE: Import both strategy design tools ###
from tools.strategy_tools import design_mean_reversion_strategy, design_momentum_strategy

class StrategyArchitectAgent(BaseFinancialAgent):
    """
    ### UPGRADED: Can now design both mean-reversion and momentum strategies.
    """
    @property
    def name(self) -> str: return "StrategyArchitect"
        
    @property
    def purpose(self) -> str: return "Designs new investment strategies based on quantitative signals."

    def __init__(self):
        tools = [
            design_mean_reversion_strategy,
            design_momentum_strategy, # <-- Add new tool
        ]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        query = user_query.lower()
        tool_to_use = None
        
        # ### UPGRADE: Smarter tool selection ###
        if "mean-reversion" in query:
            tool_to_use = self.tool_map["DesignMeanReversionStrategy"]
        elif "momentum" in query:
            tool_to_use = self.tool_map["DesignMomentumStrategy"]
        
        if not tool_to_use:
            return {"success": False, "error": "Please specify a strategy to design (e.g., 'mean-reversion' or 'momentum')."}

        universe = [word for word in user_query.replace(",", "").split() if word.isupper() and len(word) > 1]
        if not universe:
            return {"success": False, "error": "Please specify a universe of stocks (e.g., AAPL, MSFT)."}

        print(f"Fetching market data for universe: {universe}")
        try:
            price_data = yf.download(universe, period="1y", auto_adjust=True, progress=False)['Close']
            if isinstance(price_data, pd.Series):
                price_data = price_data.to_frame(name=universe[0])
        except Exception as e:
            return {"success": False, "error": f"Could not retrieve price data: {e}"}
            
        result = tool_to_use.invoke({"asset_prices": price_data})
        
        if result.get("success"):
            result['summary'] = (f"Analyzed {len(universe)} stocks for a {result['strategy_type']} strategy. "
                               f"Found {len(result['candidates'])} promising candidates.")
        else:
             result['summary'] = "Could not find any suitable candidates in the specified universe."
        
        result['agent_used'] = self.name
        return result