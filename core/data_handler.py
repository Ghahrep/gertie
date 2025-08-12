# in core/data_handler.py
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any

def get_market_data_for_portfolio(holdings: List[Any], period: str = "1y") -> Dict[str, Any]:
    """
    Takes a list of SQLAlchemy Holding objects, fetches market data,
    and returns a rich context object with prices, returns, weights, and market values.
    """
    tickers = [h.asset.ticker for h in holdings]
    if not tickers:
        return {
            "holdings_with_values": [], "prices": pd.DataFrame(), "returns": pd.DataFrame(),
            "portfolio_returns": pd.Series(dtype=float), "total_value": 0, "weights": pd.Series(dtype=float)
        }

    try:
        prices = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        
        returns = prices.pct_change().dropna()
        
        latest_prices = prices.iloc[-1]
        for h in holdings:
            h.market_value = h.shares * latest_prices.get(h.asset.ticker, 0)
        
        total_value = sum(h.market_value for h in holdings)
        
        # ### IMPROVEMENT: Store weights as a pandas Series for robust, label-based math ###
        weights_series = pd.Series(
            {h.asset.ticker: h.market_value / total_value if total_value > 0 else 0 for h in holdings}
        )
        
        portfolio_returns = pd.Series(dtype=float)
        if total_value > 0:
            aligned_returns = returns[weights_series.index] # Ensure column order matches
            portfolio_returns = (aligned_returns * weights_series).sum(axis=1)

        return {
            "holdings_with_values": holdings,
            "prices": prices,
            "returns": returns,
            "portfolio_returns": portfolio_returns,
            "total_value": total_value,
            "weights": weights_series # <-- ### THE FIX: Add weights to the returned dictionary ###
        }
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return {}