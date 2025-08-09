# in core/data_handler.py
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any

def get_market_data_for_portfolio(holdings: List[Any], period: str = "1y") -> Dict[str, Any]:
    """
    Takes a list of SQLAlchemy Holding objects, fetches market data,
    and returns a rich context object with prices, returns, and market values.
    """
    # Extract tickers from the Holding objects. Each holding has an 'asset' relationship.
    tickers = [h.asset.ticker for h in holdings]
    if not tickers:
        return {
            "holdings": [], "prices": pd.DataFrame(), "returns": pd.DataFrame(),
            "portfolio_returns": pd.Series(dtype=float), "total_value": 0
        }

    try:
        prices = yf.download(tickers, period=period, auto_adjust=True)['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        
        returns = prices.pct_change().dropna()
        
        # Calculate current market values and add them to the holding objects
        latest_prices = prices.iloc[-1]
        for h in holdings:
            h.market_value = h.shares * latest_prices.get(h.asset.ticker, 0)
        
        total_value = sum(h.market_value for h in holdings)
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(dtype=float)
        if total_value > 0:
            weights = [h.market_value / total_value for h in holdings]
            # Ensure returns columns are in the same order as weights
            aligned_returns = returns[tickers]
            portfolio_returns = (aligned_returns * weights).sum(axis=1)

        return {
            "holdings_with_values": holdings,
            "prices": prices,
            "returns": returns,
            "portfolio_returns": portfolio_returns,
            "total_value": total_value
        }
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return {} # Return an empty dict on failure