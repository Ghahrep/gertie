"""
Financial Risk Analysis Module (Revised)
========================================

A comprehensive and robust toolkit for institutional-grade portfolio risk analysis.
Functions are designed to produce enriched, dictionary-based outputs for easy
consumption by AI agents and downstream APIs.

Key Functions:
- calculate_risk_metrics: A comprehensive report of VaR, CVaR, Sharpe, etc.
- calculate_drawdowns: Detailed analysis of portfolio drawdowns.
- fit_garch_forecast: Forecasts future volatility with contextual output.
- calculate_beta / calculate_correlation_matrix: Pure calculation tools.
- calculate_tail_risk_copula: Advanced stress-testing via copula simulation.
- generate_risk_sentiment_index: A proprietary score synthesizing multiple risk factors.

Author: Quant Platform Development
Revised: Friday, August 8, 2025
"""

import pandas as pd
import numpy as np
from arch import arch_model
from scipy import stats
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field



# This utility is excellent as-is. No changes needed.
def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall (ES)."""
    if not isinstance(returns, pd.Series):
        raise TypeError("Returns must be a pandas Series")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return np.nan
        
    var_threshold = clean_returns.quantile(1 - confidence_level)
    tail_losses = clean_returns[clean_returns <= var_threshold]
    
    return -var_threshold if tail_losses.empty else -tail_losses.mean()


### --- REFACTORED FUNCTIONS --- ###

def fit_garch_forecast(
    returns: pd.Series, 
    forecast_horizon: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Fit a GARCH(1,1) model and generate volatility forecasts.
    Returns an enriched dictionary with context for AI agents.
    """
    if len(returns) < 100: ### IMPROVEMENT: Increased data requirement for stable GARCH fit.
        print("Warning: Need at least 100 observations for reliable GARCH. Returning None.")
        return None
        
    returns_pct = returns.dropna() * 100
    
    try:
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='t')
        fitted_model = model.fit(disp='off', show_warning=False)
    except Exception as e:
        print(f"GARCH model fitting failed: {e}. Cannot generate forecast.")
        return None

    # Out-of-sample forecast
    forecast = fitted_model.forecast(horizon=forecast_horizon, reindex=False)
    # Convert annualized variance to daily volatility
    future_vol_values = np.sqrt(forecast.variance).iloc[0].values / 100
    
    future_dates = pd.date_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    forecast_series = pd.Series(future_vol_values, index=future_dates)
    
    current_vol = np.sqrt(fitted_model.conditional_volatility[-1]) / 100
    mean_forecast_vol = forecast_series.mean()
    
    ### IMPROVEMENT: Return a rich dictionary instead of just the series.
    trend = "Increasing" if mean_forecast_vol > current_vol else "Decreasing" if mean_forecast_vol < current_vol else "Stable"
    
    return {
        "forecast_series": forecast_series,
        "current_daily_vol": current_vol,
        "mean_forecast_daily_vol": mean_forecast_vol,
        "volatility_trend": trend,
        "forecast_horizon": forecast_horizon,
        "model_summary": str(fitted_model.summary())
    }

@tool("CalculateRiskMetrics")
def calculate_risk_metrics(
    portfolio_returns: pd.Series, 
    confidence_levels: List[float] = [0.95, 0.99],
    trading_days: int = 252
) -> Optional[Dict[str, Any]]:
    """
    Calculates a comprehensive set of risk and return metrics for a portfolio.
    This function replaces `calculate_multi_level_var` with a more complete report.
    """
    if not isinstance(portfolio_returns, pd.Series) or portfolio_returns.empty:
        return None
        
    try:
        # VaR and CVaR Analysis
        var_analysis = {}
        for confidence in confidence_levels:
            var_value = portfolio_returns.quantile(1 - confidence)
            cvar_value = calculate_cvar(portfolio_returns, confidence)
            var_analysis[f"{int(confidence*100)}%"] = {
                "var": var_value,
                "cvar_expected_shortfall": cvar_value
            }
        
        # Performance Statistics
        daily_vol = portfolio_returns.std()
        annual_vol = daily_vol * np.sqrt(trading_days)
        annual_return = portfolio_returns.mean() * trading_days

        # Risk-Adjusted Ratios
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(trading_days)
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        ### IMPROVEMENT: Integrate drawdown stats directly into the main risk report.
        drawdown_stats = calculate_drawdowns(portfolio_returns)

        return {
            "risk_measures": var_analysis,
            "performance_stats": {
                "annualized_return_pct": annual_return * 100,
                "annualized_volatility_pct": annual_vol * 100,
            },
            "risk_adjusted_ratios": {
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": drawdown_stats['calmar_ratio'] if drawdown_stats else None
            },
            "drawdown_stats": drawdown_stats
        }
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return None

def calculate_correlation_matrix(returns: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    ### IMPROVEMENT: Pure calculation function, replacing the async version.
    The agent is responsible for fetching data and passing it here.
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        return None
    return returns.corr()

def calculate_beta(
    portfolio_returns: pd.Series, 
    market_returns: pd.Series
) -> Optional[float]:
    """
    ### IMPROVEMENT: Pure calculation function, replacing the async version.
    Calculates the beta of a portfolio relative to market returns.
    """
    if portfolio_returns.empty or market_returns.empty:
        return None
    
    # Align data by index
    df = pd.DataFrame({'portfolio': portfolio_returns, 'market': market_returns}).dropna()
    if len(df) < 30: return None # Not enough data for reliable beta
        
    covariance = df['portfolio'].cov(df['market'])
    market_variance = df['market'].var()
    
    return covariance / market_variance if market_variance > 0 else None


### --- NEW ESSENTIAL FUNCTIONS --- ###


def calculate_drawdowns(returns: pd.Series) -> Optional[Dict[str, Any]]:
    """
    A new, dedicated function to analyze historical portfolio drawdowns.
    Handles both timeseries and simulated (integer-indexed) data.
    """
    if returns.empty: return None

    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max

    max_drawdown = drawdowns.min()
    end_date_idx = drawdowns.idxmin()
    start_date_idx = cumulative_returns.loc[:end_date_idx].idxmax()
    
    # ### THE FIX: Check the type of the index before formatting ###
    if isinstance(start_date_idx, (datetime, pd.Timestamp)):
        start_date_str = start_date_idx.strftime('%Y-%m-%d')
        end_date_str = end_date_idx.strftime('%Y-%m-%d')
    else:
        # It's likely an integer index from a simulation
        start_date_str = f"Day {start_date_idx}"
        end_date_str = f"Day {end_date_idx}"

    annual_return = returns.mean() * 252
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    return {
        "max_drawdown_pct": max_drawdown * 100,
        "start_of_max_drawdown": start_date_str,
        "end_of_max_drawdown": end_date_str,
        "current_drawdown_pct": drawdowns.iloc[-1] * 100,
        "calmar_ratio": calmar_ratio
    }

def generate_risk_sentiment_index(
    risk_metrics: Dict[str, Any],
    correlation_matrix: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    A new, proprietary function to create the "Portfolio Stress Sentiment Index".
    It synthesizes multiple risk factors into a single, intuitive score from 0 (Low Risk) to 100 (High Risk).
    """
    if not risk_metrics: return None
    
    try:
        # --- Component Scoring (0-100 scale) ---
        
        # Volatility Score (normalized by a typical range, e.g., 5% to 40% annual vol)
        vol = risk_metrics['performance_stats']['annualized_volatility_pct']
        vol_score = np.clip((vol - 5) / (40 - 5), 0, 1) * 100
        
        # Tail Risk Score (based on 99% CVaR)
        cvar99 = abs(risk_metrics['risk_measures']['99%']['cvar_expected_shortfall'])
        cvar_score = np.clip((cvar99 - 0.01) / (0.05 - 0.01), 0, 1) * 100
        
        # Correlation Score (based on the average of off-diagonal correlations)
        np.fill_diagonal(correlation_matrix.values, np.nan)
        avg_corr = correlation_matrix.mean().mean()
        corr_score = np.clip(avg_corr / 0.8, 0, 1) * 100
        
        # --- Weighting and Final Index Calculation ---
        # Weights can be adjusted based on market regime or user preference.
        weights = {'volatility': 0.4, 'tail_risk': 0.4, 'correlation': 0.2}
        
        final_score = (
            weights['volatility'] * vol_score +
            weights['tail_risk'] * cvar_score +
            weights['correlation'] * corr_score
        )
        
        if final_score < 33:
            sentiment = "Calm"
        elif final_score < 66:
            sentiment = "Uneasy"
        else:
            sentiment = "Stressed"

        return {
            "stress_index_score": int(final_score),
            "sentiment": sentiment,
            "component_scores": {
                "volatility": int(vol_score),
                "tail_risk": int(cvar_score),
                "correlation": int(corr_score)
            }
        }
    except Exception as e:
        print(f"Could not generate risk sentiment index: {e}")
        return None

@tool("CalculateVolatilityBudget")
def calculate_volatility_budget(
    portfolio_returns: pd.Series, 
    target_volatility: float,
    trading_days: int = 252
) -> Optional[Dict[str, float]]:
    """
    Calculates the allocation between a risky portfolio and a risk-free asset
    to achieve a specific target volatility level.
    """
    if not isinstance(portfolio_returns, pd.Series) or portfolio_returns.empty:
        return None
        
    try:
        # Calculate current annualized volatility of the risky portfolio
        current_vol = portfolio_returns.std() * np.sqrt(trading_days)
        
        if current_vol == 0:
            # If portfolio has no risk, it's all "risk-free"
            return {'risky_asset_weight': 0.0, 'risk_free_asset_weight': 1.0}

        # Calculate the required weight in the risky asset
        weight_risky = target_volatility / current_vol
        
        # We cap the weight at 1.5 (50% leverage) and floor at 0 for practical advice.
        weight_risky = np.clip(weight_risky, 0, 1.5)
        
        weight_risk_free = 1 - weight_risky
        
        return {
            'risky_asset_weight': round(weight_risky, 4),
            'risk_free_asset_weight': round(weight_risk_free, 4),
            'current_annual_volatility': round(current_vol, 4),
            'target_annual_volatility': round(target_volatility, 4)
        }
    except Exception as e:
        print(f"Error calculating volatility budget: {e}")
        return None

@tool("CalculateTailRiskCopula")
def calculate_tail_risk_copula(
    returns: pd.DataFrame, 
    n_simulations: int = 10000
) -> Optional[pd.DataFrame]:
    """
    Performs a stress test by simulating returns using a Student's t-copula.
    This method is excellent for capturing "fat tails" and joint extreme events.
    Input should be a pandas DataFrame of daily returns for multiple assets.
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[0] < 100:
        print("Error: Requires at least 100 data points for reliable copula fitting.")
        return None

    try:
        # 1. Fit marginal distributions (Student's t) to each asset's returns
        fitted_marginals = {
            asset: stats.t.fit(returns[asset].dropna())
            for asset in returns.columns
        }

        # 2. Transform historical returns to uniform distributions (pseudo-observations)
        uniform_returns = pd.DataFrame({
            asset: stats.t.cdf(returns[asset].dropna(), *params)
            for asset, params in fitted_marginals.items()
        })

        # 3. Fit the t-copula
        spearman_corr = uniform_returns.corr(method='spearman')
        copula_df = 5  # Lower DF for fatter tails

        # 4. Simulate from the copula
        n_assets = len(returns.columns)
        mvt_rvs = stats.multivariate_t.rvs(
            loc=np.zeros(n_assets), 
            shape=spearman_corr.values, 
            df=copula_df, 
            size=n_simulations
        )
        copula_sims_uniform = stats.t.cdf(mvt_rvs, df=copula_df)
        
        # 5. Transform uniform copula samples back to asset returns
        simulated_returns = pd.DataFrame({
            asset: stats.t.ppf(copula_sims_uniform[:, i], *fitted_marginals[asset])
            for i, asset in enumerate(returns.columns)
        })

        return simulated_returns
    except Exception as e:
        print(f"Error during copula stress testing: {e}")
        return None