# in agents/strategy_architect.py
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseFinancialAgent
from tools.strategy_tools import design_mean_reversion_strategy

class StrategyArchitectAgent(BaseFinancialAgent):
    """
    ### UPGRADED: Fetches live data for the universe specified in the user query.
    """
    @property
    def name(self) -> str: return "StrategyArchitect"
        
    @property
    def purpose(self) -> str: return "Designs new investment strategies based on quantitative signals."

    def __init__(self):
        tools = [design_mean_reversion_strategy]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        # 1. Parse the user query to extract the universe
        if "mean-reversion" not in user_query.lower():
            return {"success": False, "error": "Sorry, I can only design mean-reversion strategies."}
        
        universe = [word for word in user_query.replace(",", "").split() if word.isupper() and len(word) > 1]
        if not universe:
            return {"success": False, "error": "Please specify a universe of stocks (e.g., AAPL, MSFT)."}

        # 2. Fetch live data for the specified universe
        print(f"Fetching market data for universe: {universe}")
        try:
            price_data = yf.download(universe, period="1y", auto_adjust=True)['Close']
            if isinstance(price_data, pd.Series):
                price_data = price_data.to_frame(name=universe[0])
        except Exception as e:
            return {"success": False, "error": f"Could not retrieve price data: {e}"}
            
        # 3. Execute the tool with the fetched price data.
        result = self.tool_map["DesignMeanReversionStrategy"].run({
            "price_data": price_data,
            "hurst_threshold": 0.45
        })
        
        # 4. Format the output
        if result.get("success"):
            result['summary'] = (f"Analyzed {len(universe)} stocks for mean-reversion. "
                               f"Found {len(result['candidates'])} promising candidates.")
        else:
             result['summary'] = "Could not find any suitable candidates in the specified universe."
        return result