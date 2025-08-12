"""
Market Regime Analysis Module (Revised)
=======================================

This module provides robust tools for identifying and forecasting distinct market regimes.
It features standardized HMM outputs for consistent interpretation and has been updated
for seamless interoperability with other analysis modules.

Key Functions:
- detect_hmm_regimes: Uses a Hidden Markov Model with standardized, volatility-sorted outputs.
- analyze_hurst_regimes: Uses a rolling Hurst exponent to classify regimes based on memory.
- forecast_regime_transition_probability: Forecasts regime changes based on a fitted HMM.

Dependencies:
    pip install numpy pandas matplotlib hmmlearn yfinance

Author: Quant Platform Development
Revised: 2025-08-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Dict, Any, Optional
from langchain.tools import tool
import warnings

try:
    from hmmlearn import hmm
except ImportError:
    print("Warning: hmmlearn not found. HMM functions will be unavailable. `pip install hmmlearn`")
    hmm = None

try:
    # Ensure this path is correct relative to your project's root
    from tools.fractal_tools import calculate_hurst
except ImportError:
    print("Warning: fractal_tools.py not found. Hurst regime analysis will be unavailable.")
    calculate_hurst = None

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='hmmlearn')

@tool("DetectHMMRegimes")
def detect_hmm_regimes(
    returns: pd.Series, 
    n_regimes: int = 2, 
    n_init: int = 10 
) -> Optional[Dict[str, Any]]:
    """
    Detect market regimes using a Gaussian HMM with standardized, volatility-sorted output.
    """
    if hmm is None:
        raise ImportError("hmmlearn is not installed.")
    if len(returns) < n_regimes * 25:
        raise ValueError(f"Not enough data for HMM. Has {len(returns)}, needs {n_regimes * 25}")

    returns_clean = returns.dropna()
    feature_matrix = returns_clean.values.reshape(-1, 1)

    try:
        model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full", 
            n_iter=1000,
            # n_init=n_init,  # <-- ### THE FIX: Comment out or delete this line ###
            random_state=42
        )
        model.fit(feature_matrix)
    except Exception as e:
        print(f"HMM model fitting failed: {e}")
        return None

    hidden_states = model.predict(feature_matrix)

    ### IMPROVEMENT: Standardize regime labels by sorting by volatility.
    # This ensures that regime 0 is always the lowest volatility state, etc.
    regime_volatilities = [np.sqrt(model.covars_[i][0][0]) for i in range(n_regimes)]
    vol_sorted_indices = np.argsort(regime_volatilities)
    
    # Create a mapping from old, arbitrary labels to new, sorted labels
    label_map = {old_label: new_label for new_label, old_label in enumerate(vol_sorted_indices)}
    standardized_states = np.array([label_map[s] for s in hidden_states])
    
    regime_series = pd.Series(standardized_states, index=returns_clean.index, name='hmm_regime')
    
    # Create a characteristics dictionary with the new, sorted labels
    characteristics = {}
    for i in range(n_regimes):
        original_index = vol_sorted_indices[i]
        characteristics[i] = {
            'mean_return': model.means_[original_index][0],
            'volatility': np.sqrt(model.covars_[original_index][0][0])
        }
        
    # Reorder the transition matrix to match the new labels
    transition_matrix = model.transmat_[np.ix_(vol_sorted_indices, vol_sorted_indices)]

    ### IMPROVEMENT: Return a single, enriched dictionary.
    return {
        "regime_series": regime_series,
        "regime_characteristics": characteristics,
        "transition_matrix": transition_matrix,
        "current_regime": regime_series.iloc[-1],
        "fitted_model": model # Still useful for advanced analysis
    }

def analyze_hurst_regimes(
    series: pd.Series, 
    window: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Perform rolling Hurst analysis to identify memory-based regimes.

    Parameters:
    -----------
    series : pd.Series
        Price or return series with a datetime index.
    window : int, default=100
        Rolling window size for Hurst calculation.

    Returns:
    --------
    Dict[str, Any] or None
        A dictionary containing the results DataFrame and a summary of regime frequencies.
    """
    if calculate_hurst is None:
        raise ImportError("calculate_hurst function from fractal_tools not available.")
    if len(series) < window:
        return None

    ### IMPROVEMENT: Correctly handle the dictionary output from the new `calculate_hurst`.
    # We apply a lambda function to extract the 'hurst_exponent' value.
    rolling_hurst_values = series.rolling(window).apply(
        lambda x: calculate_hurst(pd.Series(x)).get('hurst_exponent', np.nan),
        raw=False
    ).dropna()

    df = pd.DataFrame({'hurst': rolling_hurst_values})
    
    conditions = [
        df['hurst'] < 0.45,
        (df['hurst'] >= 0.45) & (df['hurst'] <= 0.55),
        df['hurst'] > 0.55
    ]
    choices = ['Mean-Reverting', 'Random Walk', 'Trending']
    df['hurst_regime'] = np.select(conditions, choices, default='N/A')
    
    ### IMPROVEMENT: Add a summary for easier agent interpretation.
    regime_counts = df['hurst_regime'].value_counts(normalize=True).round(4)
    summary = {
        "time_in_mean_reverting_pct": regime_counts.get("Mean-Reverting", 0) * 100,
        "time_in_random_walk_pct": regime_counts.get("Random Walk", 0) * 100,
        "time_in_trending_pct": regime_counts.get("Trending", 0) * 100,
        "current_regime": df['hurst_regime'].iloc[-1]
    }
    
    return {
        "results_df": df,
        "summary": summary
    }
@tool("ForecastRegimeTransitionProbability")
def forecast_regime_transition_probability(
    hmm_results: Dict[str, Any] ### IMPROVEMENT: Take the entire HMM result dictionary as input.
) -> Optional[Dict[str, Any]]:
    """
    Calculates the probability of transitioning from the current regime to all others.

    Parameters:
    -----------
    hmm_results : Dict[str, Any]
        The complete result dictionary returned by `detect_hmm_regimes`.

    Returns:
    --------
    Dict[str, Any] or None
        A dictionary detailing the transition probabilities with added context.
    """
    try:
        trans_mat = hmm_results['transition_matrix']
        current_regime = hmm_results['current_regime']
        characteristics = hmm_results['regime_characteristics']
        
        transition_probs = trans_mat[current_regime, :]
        
        ### IMPROVEMENT: Create a richer, more descriptive output.
        forecast = {
            'from_regime': {
                'index': int(current_regime),
                'volatility': round(characteristics[current_regime]['volatility'], 6)
            },
            'transition_forecast': []
        }
        
        for i, prob in enumerate(transition_probs):
            forecast['transition_forecast'].append({
                'to_regime_index': i,
                'probability': round(prob, 4),
                'volatility': round(characteristics[i]['volatility'], 6)
            })
            
        return forecast
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error forecasting regime transition. Is the input a valid result from detect_hmm_regimes? Details: {e}")
        return None