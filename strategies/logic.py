# in strategies/logic.py
import pandas as pd

# --- Library of Reusable Strategy Logic Functions ---

def moving_average_crossover_logic(prices: pd.DataFrame, regimes: pd.Series) -> pd.Series:
    """A simple moving average crossover strategy. Ignores the regime."""
    close = prices['Close']
    fast_ma = close.rolling(50).mean()
    slow_ma = close.rolling(200).mean()
    # Signal is 1 (long) when fast MA > slow MA, otherwise 0 (flat)
    signal = (fast_ma > slow_ma).astype(int)
    return signal.dropna()

def regime_switching_logic(prices: pd.DataFrame, regimes: pd.Series) -> pd.Series:
    """A more advanced strategy that uses a M/A crossover ONLY in the low-volatility regime."""
    close = prices['Close']
    fast_ma = close.rolling(50).mean()
    slow_ma = close.rolling(200).mean()
    
    crossover_signal = (fast_ma > slow_ma)
    # Get the regime state (0 = low vol, 1 = high vol)
    low_vol_regime = (regimes == 0)
    
    # Only go long if the crossover is true AND we are in the low volatility regime.
    final_signal = (crossover_signal & low_vol_regime).astype(int)
    return final_signal.dropna()