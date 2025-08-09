
"""
Portfolio Strategy & Optimization Module (Revised)
==================================================

A production-grade toolkit for designing, optimizing, and backtesting quantitative
investment strategies. This module uses industry-standard libraries to replace
all previous placeholder logic.

Key Functions:
- optimize_portfolio: Full implementation using PyPortfolioOpt.
- perform_factor_analysis: Full implementation using statsmodels.
- run_backtest: Full implementation using the backtesting.py library.
- run_regime_aware_backtest: An advanced backtester that makes strategies regime-aware.
- design_mean_reversion_strategy: A pure calculation tool for screening assets.

Dependencies:
    pip install numpy pandas pypfopt statsmodels backtesting

Author: Quant Platform Development
Revised: Friday, August 8, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional

# --- Tool Imports ---
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field # Using v1 for broader compatibility
from .fractal_tools import calculate_hurst
from .risk_tools import calculate_drawdowns # We can use tools from other modules
from strategies.logic import moving_average_crossover_logic, regime_switching_logic

# --- Core Library Imports ---
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
    from pypfopt.hierarchical_portfolio import HRPOpt
except ImportError:
    print("Warning: pypfopt not found. Optimization functions will be unavailable. `pip install PyPortfolioOpt`")
    EfficientFrontier = None
    HRPOpt = None

try:
    import statsmodels.api as sm
except ImportError:
    print("Warning: statsmodels not found. Factor analysis will be unavailable. `pip install statsmodels`")
    sm = None

try:
    from backtesting import Backtest, Strategy
except ImportError:
    print("Warning: backtesting.py not found. Backtesting will be unavailable. `pip install backtesting.py`")
    Backtest = None



### --- FULLY IMPLEMENTED CORE FUNCTIONS --- ###
@tool("OptimizePortfolio")
def optimize_portfolio(
    asset_prices: pd.DataFrame, 
    objective: str = 'MaximizeSharpe',
    risk_free_rate: float = 0.02
) -> Optional[Dict[str, Any]]:
    """
    ### IMPROVEMENT: Expanded to support HERC (Hierarchical Risk Parity).
    Constructs an optimal portfolio based on a specified objective function.
    Objectives: 'MaximizeSharpe', 'MinimizeVolatility', 'HERC'.
    """
    if EfficientFrontier is None or HRPOpt is None: 
        raise ImportError("PyPortfolioOpt is not installed.")
    if asset_prices.empty: return None

    # HERC uses returns directly, while MVO uses prices to calculate mu and S
    returns = expected_returns.returns_from_prices(asset_prices)

    try:
        if objective == 'HERC':
            # --- HERC / Risk Parity Logic ---
            hrp = HRPOpt(returns)
            hrp.optimize()
            clean_weights = hrp.clean_weights()
            
            # HRP doesn't provide performance, so we calculate it afterwards
            perf = hrp.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            expected_return, annual_vol, sharpe = perf

        elif objective in ['MaximizeSharpe', 'MinimizeVolatility']:
            # --- Mean-Variance Optimization Logic (existing) ---
            mu = expected_returns.mean_historical_return(asset_prices)
            S = risk_models.sample_cov(asset_prices)
            ef = EfficientFrontier(mu, S)
            
            if objective == 'MaximizeSharpe':
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == 'MinimizeVolatility':
                ef.min_volatility()
                
            clean_weights = ef.clean_weights()
            expected_return, annual_vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        
        else:
            raise ValueError(f"Objective '{objective}' not recognized.")
            
        return {
            "objective_used": objective, # ### IMPROVEMENT: Return the objective used for clarity
            "optimal_weights": clean_weights,
            "expected_performance": {
                "annual_return_pct": expected_return * 100,
                "annual_volatility_pct": annual_vol * 100,
                "sharpe_ratio": sharpe
            }
        }
    except Exception as e:
        print(f"Portfolio optimization failed for objective '{objective}': {e}")
        return None

@tool("PerformFactorAnalysis")
def perform_factor_analysis(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame
) -> Optional[Dict[str, float]]:
    """
    ### IMPROVEMENT: Fully implemented with statsmodels.
    Performs a factor analysis on portfolio returns using specified factors.
    """
    if sm is None: raise ImportError("statsmodels is not installed.")
    
    try:
        # 1. Align data and calculate portfolio excess returns
        data = pd.DataFrame({'portfolio': portfolio_returns}).join(factor_returns).dropna()
        if 'RF' not in data.columns: raise ValueError("factor_returns must include 'RF' column.")
        
        y = data['portfolio'] - data['RF']
        X = data.drop(columns=['portfolio', 'RF'])
        X = sm.add_constant(X) # Add constant for alpha calculation
        
        # 2. Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # 3. Extract results and annualize alpha
        params = model.params.to_dict()
        alpha_daily = params.pop('const')
        alpha_annual_pct = alpha_daily * 252 * 100
        
        return {
            "alpha_annual_pct": alpha_annual_pct,
            "factor_betas": params,
            "r_squared_adj": model.rsquared_adj,
            "p_values": model.pvalues.to_dict()
        }
    except Exception as e:
        print(f"Factor analysis failed: {e}")
        return None

@tool("RunBacktest")
def run_backtest(
    prices: pd.DataFrame,
    strategy_logic: Callable[[pd.DataFrame], pd.Series],
    initial_cash: float = 100_000,
    commission: float = 0.001
) -> Optional[Dict[str, Any]]:
    """
    ### DEFINITIVE FIX: Manually builds a clean, JSON-safe result dictionary.
    """
    if Backtest is None: raise ImportError("backtesting.py is not installed.")

    # ... (signal calculation and Strategy class definition are unchanged) ...
    signals = strategy_logic(prices)
    data = prices.copy()
    data['Signal'] = signals.reindex(data.index, method='ffill').fillna(0)

    class SignalFollowingStrategy(Strategy):
        def init(self): pass
        def next(self):
            signal = self.data.Signal[-1]
            if signal == 1 and not self.position:
                self.buy()
            elif signal == 0 and self.position:
                self.position.close()

    try:
        bt = Backtest(data, SignalFollowingStrategy, cash=initial_cash, commission=commission)
        stats = bt.run()
        
        # ### THE DEFINITIVE FIX: Manually build a clean result dictionary ###
        # We iterate through the stats Series and exclude the non-serializable objects.
        performance_summary = {
            key: value for key, value in stats.items()
            if key not in ['_strategy', '_equity_curve', '_trades']
        }
        
        # We can still provide the equity and trade data, but converted to a clean JSON string
        equity_curve_json = stats._equity_curve.reset_index().to_json(orient='records', date_format='iso')
        trades_json = stats._trades.reset_index().to_json(orient='records', date_format='iso')

        return {
            "performance_summary": performance_summary,
            "equity_curve_json": equity_curve_json,
            "trades_json": trades_json
        }
    except Exception as e:
        print(f"Backtest failed: {e}")
        return None

@tool("DesignMeanReversionStrategy")
def design_mean_reversion_strategy(
    price_data: pd.DataFrame, ### IMPROVEMENT: Pure function, takes data as input.
    hurst_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    ### IMPROVEMENT: Refactored to be a pure calculation tool.
    Screens a universe of assets for mean-reverting candidates based on Hurst exponent.
    """
    if price_data.empty: return {"success": False, "error": "Price data is empty."}

    hurst_results = []
    for ticker in price_data.columns:
        try:
            price_series = price_data[ticker].dropna()
            if len(price_series) < 100: continue
            
            # ### FIX: Correctly call the refactored `calculate_hurst`
            hurst_dict = calculate_hurst(price_series)
            if hurst_dict:
                h_value = hurst_dict['hurst_exponent']
                hurst_results.append({'ticker': ticker, 'hurst': round(h_value, 4)})
        except Exception as e:
            print(f"Could not process {ticker} for Hurst calculation: {e}")
            continue

    candidates = [res for res in hurst_results if res['hurst'] < hurst_threshold]
    sorted_candidates = sorted(candidates, key=lambda x: x['hurst'])
    
    if not sorted_candidates:
        return { "success": False, "error": "No suitable mean-reverting assets found." }
    
    return {
        "success": True,
        "strategy_type": "Mean-Reversion",
        "candidates": sorted_candidates,
    }

### --- NEW STRATEGIC FUNCTIONS --- ###
@tool("RunRegimeAwareBacktest")
def run_regime_aware_backtest(
    prices: pd.DataFrame,
    regime_series: pd.Series,
    strategy_name: str,
    initial_cash: float = 100_000,
    commission: float = 0.001
) -> Optional[Dict[str, Any]]:
    """
    ### DEFINITIVE FIX: Manually builds a clean, JSON-safe result dictionary.
    """
    if Backtest is None: raise ImportError("backtesting.py is not installed.")

    # ... (strategy lookup and signal calculation are unchanged) ...
    strategies = {
        "moving_average_crossover": moving_average_crossover_logic,
        "regime_switching_crossover": regime_switching_logic,
    }
    strategy_logic = strategies.get(strategy_name)
    if not strategy_logic:
        raise ValueError(f"Strategy '{strategy_name}' not found in the tool.")

    signals = strategy_logic(prices, regime_series)
    data = prices.copy()
    data['Signal'] = signals.reindex(data.index, method='ffill').fillna(0)
    
    class SignalFollowingStrategy(Strategy):
        def init(self): pass
        def next(self):
            signal = self.data.Signal[-1]
            if signal == 1 and not self.position:
                self.buy()
            elif signal == 0 and self.position:
                self.position.close()
            
    try:
        bt = Backtest(data, SignalFollowingStrategy, cash=initial_cash, commission=commission)
        stats = bt.run()
        
        # ### THE DEFINITIVE FIX: Manually build a clean result dictionary ###
        performance_summary = {
            key: value for key, value in stats.items()
            if key not in ['_strategy', '_equity_curve', '_trades']
        }
        
        equity_curve_json = stats._equity_curve.reset_index().to_json(orient='records', date_format='iso')
        trades_json = stats._trades.reset_index().to_json(orient='records', date_format='iso')

        return {
            "performance_summary": performance_summary,
            "equity_curve_json": equity_curve_json,
            "trades_json": trades_json
        }
    except Exception as e:
        print(f"Regime-aware backtest failed: {e}")
        return None

# The functions `find_optimal_hedge` and `generate_trade_orders` from the previous step
# are already well-designed pure functions and belong in this file as well.
@tool("FindOptimalHedge")
def find_optimal_hedge(
    portfolio_returns: pd.Series, 
    hedge_instrument_returns: pd.Series
) -> Optional[Dict[str, float]]:
    """
    Calculates the optimal hedge ratio to minimize portfolio variance.
    The hedge ratio (beta) is Cov(portfolio, hedge) / Var(hedge).
    """
    # ... the function body remains the same ...
    if portfolio_returns.empty or hedge_instrument_returns.empty:
        return None
        
    try:
        df = pd.DataFrame({'portfolio': portfolio_returns, 'hedge': hedge_instrument_returns}).dropna()
        
        if len(df) < 30:
            print("Warning: Insufficient overlapping data for a reliable hedge ratio.")
            return None
            
        covariance = df['portfolio'].cov(df['hedge'])
        hedge_variance = df['hedge'].var()
        
        if hedge_variance == 0:
            return None

        hedge_ratio = -covariance / hedge_variance
        
        original_vol = df['portfolio'].std()
        hedged_portfolio_returns = df['portfolio'] + hedge_ratio * df['hedge']
        hedged_vol = hedged_portfolio_returns.std()
        vol_reduction = original_vol - hedged_vol
        
        return {
            'optimal_hedge_ratio': round(hedge_ratio, 4),
            'original_daily_vol': round(original_vol, 5),
            'hedged_daily_vol': round(hedged_vol, 5),
            'volatility_reduction_pct': round((vol_reduction / original_vol) * 100, 2) if original_vol > 0 else 0
        }
    except Exception as e:
        print(f"Error finding optimal hedge: {e}")
        return None

@tool("GenerateTradeOrders")
def generate_trade_orders(
    current_holdings: Dict[str, float], 
    target_weights: Dict[str, float], 
    total_portfolio_value: float
) -> Dict[str, Any]:
    """
    ### IMPROVEMENT: Fully implemented with logic from your previous agent.
    Generates a list of buy/sell orders to move from a current portfolio
    allocation to a target allocation.

    Parameters:
    -----------
    current_holdings : Dict[str, float]
        Dictionary mapping tickers to their current market value.
        Example: {'AAPL': 50000, 'GOOG': 50000}
    target_weights : Dict[str, float]
        Dictionary of target weights for each asset.
        Example: {'AAPL': 0.6, 'GOOG': 0.4}
    total_portfolio_value : float
        The total market value of the portfolio.
    """
    orders = []
    all_assets = set(current_holdings.keys()) | set(target_weights.keys())

    for asset in all_assets:
        current_value = current_holdings.get(asset, 0)
        target_weight = target_weights.get(asset, 0)
        target_value = total_portfolio_value * target_weight
        
        dollar_change = target_value - current_value
        
        # Use a small threshold to avoid tiny, meaningless trades
        if abs(dollar_change) > 1.0: # Ignore trades less than $1.00
            order = {
                'ticker': asset,
                'action': 'BUY' if dollar_change > 0 else 'SELL',
                'amount_usd': round(abs(dollar_change), 2)
            }
            orders.append(order)
            
    return {"trades": orders}