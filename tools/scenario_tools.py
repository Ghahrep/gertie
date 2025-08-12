# in tools/scenario_tools.py
import pandas as pd
from typing import Dict, Any
from langchain.tools import tool

@tool("ApplyMarketShock")
def apply_market_shock(
    returns: pd.DataFrame, 
    shock_scenario: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies a predefined shock to a historical returns series.
    'shock_scenario' dict should contain 'type' and parameters.
    Example: {'type': 'market_crash', 'impact_pct': -0.20}
    """
    shocked_returns = returns.copy()
    shock_type = shock_scenario.get("type", "none")
    
    print(f"Applying shock type: {shock_type}")

    if shock_type == "market_crash":
        impact = shock_scenario.get("impact_pct", -0.20)
        # Simulate a sudden one-day drop
        shock_day = shocked_returns.sample(1).index
        shocked_returns.loc[shock_day] = impact
    
    elif shock_type == "interest_rate_spike":
        impact = shock_scenario.get("impact_pct", -0.05)
        # Simulate a negative shock to interest-rate sensitive assets (like bonds, if identifiable)
        # For a simple model, we'll apply a smaller, broad shock.
        shocked_returns = shocked_returns + (impact / len(shocked_returns)) # Gradual shock

    # Add more scenarios here in the future
    
    return shocked_returns