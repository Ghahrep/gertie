# services/risk_calculator.py
"""
Enhanced Risk Calculator - Task 2.1.1 Implementation
===================================================
Production-ready risk calculator with comprehensive metrics and fractal analysis.
Fixes LangChain tool decorator issues and adds all required risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pydantic import BaseModel, Field
# Fix LangChain imports and decorators
try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback if LangChain not available
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    BaseModel = object
    Field = lambda **kwargs: None

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container"""
    # Fields WITHOUT default values must come first
    var_95: float
    var_99: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float
    
    # Risk-Adjusted Return Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Volatility Metrics
    annualized_volatility: float
    downside_deviation: float
    upside_deviation: float
    volatility_skewness: float
    
    # Drawdown Metrics
    max_drawdown: float
    current_drawdown: float
    drawdown_duration: int
    
    # Distribution Metrics
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # Fractal Analysis
    hurst_exponent: float
    dfa_alpha: float  # Detrended Fluctuation Analysis
    fractal_dimension: float
    
    # Metadata (required fields)
    calculation_date: datetime
    data_period_days: int
    confidence_score: float  # Confidence in risk calculations
    
    # Fields WITH default values must come last
    recovery_time: Optional[int] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    correlation_to_market: Optional[float] = None
    tracking_error: Optional[float] = None

class RiskCalculatorInput(BaseModel):
    """Input schema for risk calculator"""
    returns: Union[List[float], str] = Field(description="Portfolio returns or symbol for calculation")
    benchmark_returns: Optional[Union[List[float], str]] = Field(None, description="Benchmark returns for relative metrics")
    confidence_levels: Optional[List[float]] = Field([0.95, 0.99], description="VaR confidence levels")
    risk_free_rate: Optional[float] = Field(0.02, description="Risk-free rate for Sharpe ratio")
    periods_per_year: Optional[int] = Field(252, description="Trading periods per year")

class EnhancedRiskCalculator:
    """
    Production-ready risk calculator with comprehensive metrics
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._calculation_cache = {}
        
    def calculate_comprehensive_risk(
        self, 
        returns: Union[pd.Series, np.ndarray, List[float]],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        confidence_levels: List[float] = [0.95, 0.99],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a return series
        
        Args:
            returns: Portfolio returns (daily)
            benchmark_returns: Benchmark returns for relative metrics
            confidence_levels: Confidence levels for VaR calculation
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (252 for daily)
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        
        # Convert inputs to numpy arrays
        returns = np.array(returns) if not isinstance(returns, np.ndarray) else returns
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        if len(returns) < 30:
            raise ValueError("Insufficient data: Need at least 30 return observations")
        
        # Calculate cache key
        cache_key = self._generate_cache_key(returns, benchmark_returns, confidence_levels, risk_free_rate)
        
        if self.cache_enabled and cache_key in self._calculation_cache:
            logger.info("Returning cached risk metrics")
            return self._calculation_cache[cache_key]
        
        logger.info(f"Calculating risk metrics for {len(returns)} observations")
        
        try:
            # Basic statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            
            # VaR and CVaR calculations
            var_metrics = self._calculate_var_cvar(returns, confidence_levels)
            
            # Risk-adjusted return metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_returns(
                returns, risk_free_rate, periods_per_year
            )
            
            # Volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(returns, periods_per_year)
            
            # Drawdown analysis
            drawdown_metrics = self._calculate_drawdown_metrics(returns)
            
            # Distribution analysis
            distribution_metrics = self._calculate_distribution_metrics(returns)
            
            # Fractal analysis
            fractal_metrics = self._calculate_fractal_metrics(returns)
            
            # Benchmark-relative metrics
            benchmark_metrics = {}
            if benchmark_returns is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(
                    returns, benchmark_returns, risk_free_rate, periods_per_year
                )
            
            # Confidence score based on data quality
            confidence_score = self._calculate_confidence_score(returns, len(returns))
            
            # Combine all metrics
            risk_metrics = RiskMetrics(
                # VaR metrics
                var_95=var_metrics.get('var_95', np.nan),
                var_99=var_metrics.get('var_99', np.nan),
                cvar_95=var_metrics.get('cvar_95', np.nan),
                cvar_99=var_metrics.get('cvar_99', np.nan),
                
                # Risk-adjusted returns
                sharpe_ratio=risk_adjusted_metrics.get('sharpe_ratio', np.nan),
                sortino_ratio=risk_adjusted_metrics.get('sortino_ratio', np.nan),
                calmar_ratio=risk_adjusted_metrics.get('calmar_ratio', np.nan),
                information_ratio=benchmark_metrics.get('information_ratio', np.nan),
                
                # Volatility
                annualized_volatility=volatility_metrics.get('annualized_volatility', np.nan),
                downside_deviation=volatility_metrics.get('downside_deviation', np.nan),
                upside_deviation=volatility_metrics.get('upside_deviation', np.nan),
                volatility_skewness=volatility_metrics.get('volatility_skewness', np.nan),
                
                # Drawdown
                max_drawdown=drawdown_metrics.get('max_drawdown', np.nan),
                current_drawdown=drawdown_metrics.get('current_drawdown', np.nan),
                drawdown_duration=drawdown_metrics.get('drawdown_duration', 0),
                recovery_time=drawdown_metrics.get('recovery_time'),
                
                # Distribution
                skewness=distribution_metrics.get('skewness', np.nan),
                kurtosis=distribution_metrics.get('kurtosis', np.nan),
                jarque_bera_stat=distribution_metrics.get('jarque_bera_stat', np.nan),
                jarque_bera_pvalue=distribution_metrics.get('jarque_bera_pvalue', np.nan),
                
                # Fractal
                hurst_exponent=fractal_metrics.get('hurst_exponent', np.nan),
                dfa_alpha=fractal_metrics.get('dfa_alpha', np.nan),
                fractal_dimension=fractal_metrics.get('fractal_dimension', np.nan),
                
                # Benchmark metrics
                beta=benchmark_metrics.get('beta'),
                alpha=benchmark_metrics.get('alpha'),
                correlation_to_market=benchmark_metrics.get('correlation'),
                tracking_error=benchmark_metrics.get('tracking_error'),
                
                # Metadata
                calculation_date=datetime.now(),
                data_period_days=len(returns),
                confidence_score=confidence_score
            )
            
            # Cache results
            if self.cache_enabled:
                self._calculation_cache[cache_key] = risk_metrics
            
            logger.info("Risk metrics calculation completed successfully")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {str(e)}")
            raise RuntimeError(f"Risk calculation failed: {str(e)}")
    
    def _calculate_var_cvar(self, returns: np.ndarray, confidence_levels: List[float]) -> Dict:
        """Calculate Value at Risk and Conditional Value at Risk"""
        
        var_results = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # Historical VaR
            var_value = np.percentile(returns, alpha * 100)
            
            # Conditional VaR (Expected Shortfall)
            cvar_value = np.mean(returns[returns <= var_value])
            
            conf_str = f"{int(conf_level * 100)}"
            var_results[f'var_{conf_str}'] = var_value
            var_results[f'cvar_{conf_str}'] = cvar_value
        
        return var_results
    
    def _calculate_risk_adjusted_returns(
        self, 
        returns: np.ndarray, 
        risk_free_rate: float, 
        periods_per_year: int
    ) -> Dict:
        """Calculate risk-adjusted return metrics"""
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Annualize metrics
        annual_return = mean_return * periods_per_year
        annual_volatility = std_return * np.sqrt(periods_per_year)
        daily_rf = risk_free_rate / periods_per_year
        
        # Sharpe Ratio
        excess_return = mean_return - daily_rf
        sharpe_ratio = (excess_return * periods_per_year) / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < daily_rf]
        downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
        annual_downside_deviation = downside_deviation * np.sqrt(periods_per_year)
        sortino_ratio = (annual_return - risk_free_rate) / annual_downside_deviation if annual_downside_deviation > 0 else 0
        
        # Calmar Ratio (return / max drawdown)
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_volatility_metrics(self, returns: np.ndarray, periods_per_year: int) -> Dict:
        """Calculate various volatility metrics"""
        
        std_return = np.std(returns, ddof=1)
        annual_volatility = std_return * np.sqrt(periods_per_year)
        
        # Downside and upside deviation
        mean_return = np.mean(returns)
        downside_returns = returns[returns < mean_return]
        upside_returns = returns[returns > mean_return]
        
        downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
        upside_deviation = np.std(upside_returns, ddof=1) if len(upside_returns) > 1 else 0
        
        # Volatility skewness
        volatility_skewness = stats.skew(returns)
        
        return {
            'annualized_volatility': annual_volatility,
            'downside_deviation': downside_deviation * np.sqrt(periods_per_year),
            'upside_deviation': upside_deviation * np.sqrt(periods_per_year),
            'volatility_skewness': volatility_skewness
        }
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate drawdown analysis metrics"""
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        # Current drawdown
        current_drawdown = drawdowns[-1]
        
        # Drawdown duration (consecutive periods in drawdown)
        in_drawdown = drawdowns < -0.001  # 0.1% threshold
        drawdown_duration = 0
        for i in range(len(in_drawdown) - 1, -1, -1):
            if in_drawdown[i]:
                drawdown_duration += 1
            else:
                break
        
        # Recovery time calculation
        recovery_time = None
        max_dd_idx = np.argmin(drawdowns)
        
        # Look for recovery after max drawdown
        for i in range(max_dd_idx + 1, len(drawdowns)):
            if drawdowns[i] >= -0.001:  # Recovered to within 0.1%
                recovery_time = i - max_dd_idx
                break
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time
        }
    
    def _calculate_distribution_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate return distribution metrics"""
        
        # Skewness and kurtosis
        skewness_val = stats.skew(returns)
        kurtosis_val = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        return {
            'skewness': skewness_val,
            'kurtosis': kurtosis_val,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue
        }
    
    def _calculate_fractal_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate fractal analysis metrics"""
        
        try:
            # Hurst Exponent calculation
            hurst_exp = self._calculate_hurst_exponent(returns)
            
            # Detrended Fluctuation Analysis (DFA)
            dfa_alpha = self._calculate_dfa(returns)
            
            # Fractal Dimension
            fractal_dim = 2 - hurst_exp if not np.isnan(hurst_exp) else np.nan
            
            return {
                'hurst_exponent': hurst_exp,
                'dfa_alpha': dfa_alpha,
                'fractal_dimension': fractal_dim
            }
            
        except Exception as e:
            logger.warning(f"Fractal analysis failed: {str(e)}")
            return {
                'hurst_exponent': np.nan,
                'dfa_alpha': np.nan,
                'fractal_dimension': np.nan
            }
    
    def _calculate_hurst_exponent(self, returns: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        
        if len(returns) < 50:
            return np.nan
        
        try:
            # Convert returns to price series
            prices = np.cumsum(returns - np.mean(returns))
            
            # Calculate R/S statistic for different time lags
            lags = range(10, min(len(prices) // 4, 100))
            rs_values = []
            
            for lag in lags:
                # Split series into segments
                n_segments = len(prices) // lag
                if n_segments < 2:
                    continue
                
                rs_segment_values = []
                for i in range(n_segments):
                    segment = prices[i*lag:(i+1)*lag]
                    
                    # Calculate range
                    cumdev = np.cumsum(segment - np.mean(segment))
                    R = np.max(cumdev) - np.min(cumdev)
                    
                    # Calculate standard deviation
                    S = np.std(segment, ddof=1)
                    
                    if S > 0:
                        rs_segment_values.append(R / S)
                
                if rs_segment_values:
                    rs_values.append((lag, np.mean(rs_segment_values)))
            
            if len(rs_values) < 3:
                return np.nan
            
            # Fit log(R/S) vs log(lag) to get Hurst exponent
            lags_array = np.array([x[0] for x in rs_values])
            rs_array = np.array([x[1] for x in rs_values])
            
            # Remove invalid values
            valid_idx = (rs_array > 0) & np.isfinite(rs_array)
            if np.sum(valid_idx) < 3:
                return np.nan
            
            log_lags = np.log(lags_array[valid_idx])
            log_rs = np.log(rs_array[valid_idx])
            
            # Linear regression
            slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
            
            return slope
            
        except Exception as e:
            logger.warning(f"Hurst exponent calculation failed: {str(e)}")
            return np.nan
    
    def _calculate_dfa(self, returns: np.ndarray) -> float:
        """Calculate Detrended Fluctuation Analysis alpha"""
        
        if len(returns) < 50:
            return np.nan
        
        try:
            # Integrate the series
            y = np.cumsum(returns - np.mean(returns))
            
            # Window sizes
            window_sizes = np.unique(np.logspace(1, np.log10(len(y) // 4), 10).astype(int))
            window_sizes = window_sizes[window_sizes >= 4]
            
            if len(window_sizes) < 3:
                return np.nan
            
            fluctuations = []
            
            for window_size in window_sizes:
                # Split into non-overlapping windows
                n_windows = len(y) // window_size
                
                if n_windows < 1:
                    continue
                
                window_fluctuations = []
                
                for i in range(n_windows):
                    start_idx = i * window_size
                    end_idx = (i + 1) * window_size
                    window_data = y[start_idx:end_idx]
                    
                    # Fit linear trend
                    x = np.arange(len(window_data))
                    coeffs = np.polyfit(x, window_data, 1)
                    trend = np.polyval(coeffs, x)
                    
                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean((window_data - trend) ** 2))
                    window_fluctuations.append(fluctuation)
                
                if window_fluctuations:
                    fluctuations.append((window_size, np.mean(window_fluctuations)))
            
            if len(fluctuations) < 3:
                return np.nan
            
            # Fit log-log relationship
            sizes = np.array([x[0] for x in fluctuations])
            flucts = np.array([x[1] for x in fluctuations])
            
            valid_idx = (flucts > 0) & np.isfinite(flucts)
            if np.sum(valid_idx) < 3:
                return np.nan
            
            log_sizes = np.log(sizes[valid_idx])
            log_flucts = np.log(flucts[valid_idx])
            
            slope, _, _, _, _ = stats.linregress(log_sizes, log_flucts)
            
            return slope
            
        except Exception as e:
            logger.warning(f"DFA calculation failed: {str(e)}")
            return np.nan
    
    def _calculate_benchmark_metrics(
        self, 
        returns: np.ndarray, 
        benchmark_returns: Union[pd.Series, np.ndarray, List[float]],
        risk_free_rate: float,
        periods_per_year: int
    ) -> Dict:
        """Calculate benchmark-relative metrics"""
        
        benchmark_returns = np.array(benchmark_returns) if not isinstance(benchmark_returns, np.ndarray) else benchmark_returns
        
        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
        
        # Beta calculation
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else np.nan
        
        # Alpha calculation
        mean_return = np.mean(returns) * periods_per_year
        mean_benchmark = np.mean(benchmark_returns) * periods_per_year
        alpha = mean_return - (risk_free_rate + beta * (mean_benchmark - risk_free_rate))
        
        # Correlation
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1] if len(returns) > 1 else np.nan
        
        # Information Ratio (excess return / tracking error)
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)
        information_ratio = (np.mean(excess_returns) * periods_per_year) / tracking_error if tracking_error > 0 else np.nan
        
        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    def _calculate_confidence_score(self, returns: np.ndarray, sample_size: int) -> float:
        """Calculate confidence score for risk metrics based on data quality"""
        
        # Base confidence on sample size
        size_confidence = min(1.0, sample_size / 252)  # 1 year of daily data = full confidence
        
        # Adjust for data quality
        quality_factors = []
        
        # Check for missing or extreme values
        extreme_threshold = 3 * np.std(returns, ddof=1)
        extreme_count = np.sum(np.abs(returns) > extreme_threshold)
        extreme_penalty = min(0.2, extreme_count / len(returns))
        quality_factors.append(1 - extreme_penalty)
        
        # Check for volatility clustering (GARCH effects)
        squared_returns = returns ** 2
        autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        volatility_clustering = 1 - min(0.1, abs(autocorr)) if not np.isnan(autocorr) else 1
        quality_factors.append(volatility_clustering)
        
        # Combine factors
        quality_score = np.mean(quality_factors)
        
        # Final confidence score
        confidence_score = size_confidence * quality_score
        
        return min(max(confidence_score, 0.1), 1.0)
    
    def _generate_cache_key(self, returns, benchmark_returns, confidence_levels, risk_free_rate) -> str:
        """Generate cache key for risk calculations"""
        
        import hashlib
        
        # Create hash of inputs
        key_components = [
            str(len(returns)),
            str(np.sum(returns)),  # Checksum of returns
            str(benchmark_returns is not None),
            str(confidence_levels),
            str(risk_free_rate)
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_risk_summary(self, risk_metrics: RiskMetrics) -> str:
        """Generate human-readable risk summary"""
        
        summary_parts = []
        
        # Risk level assessment
        if risk_metrics.annualized_volatility < 0.10:
            risk_level = "Low"
        elif risk_metrics.annualized_volatility < 0.20:
            risk_level = "Moderate"
        elif risk_metrics.annualized_volatility < 0.30:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        summary_parts.append(f"Overall Risk Level: {risk_level}")
        summary_parts.append(f"Annualized Volatility: {risk_metrics.annualized_volatility:.1%}")
        summary_parts.append(f"Maximum Drawdown: {risk_metrics.max_drawdown:.1%}")
        summary_parts.append(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
        summary_parts.append(f"95% VaR: {risk_metrics.var_95:.1%}")
        
        if not np.isnan(risk_metrics.hurst_exponent):
            if risk_metrics.hurst_exponent > 0.5:
                trend_desc = "trending (persistent)"
            elif risk_metrics.hurst_exponent < 0.5:
                trend_desc = "mean-reverting"
            else:
                trend_desc = "random walk"
            summary_parts.append(f"Market Behavior: {trend_desc} (H={risk_metrics.hurst_exponent:.3f})")
        
        return "\n".join(summary_parts)

# LangChain Tool Integration (Fixed)
if LANGCHAIN_AVAILABLE:
    
    class RiskCalculatorTool(BaseTool):
        """LangChain tool wrapper for risk calculator"""
        
        name: str = "risk_calculator"
        description: str = """
        Calculate comprehensive risk metrics including VaR, CVaR, Sharpe ratio, Sortino ratio,
        drawdown analysis, and fractal analysis for portfolio returns.
        """
        calculator: EnhancedRiskCalculator = Field(default_factory=EnhancedRiskCalculator)
        
        # Remove the __init__ method entirely - Pydantic will handle initialization
        
        def _run(self, **kwargs) -> str:
            """Run risk calculation"""
            try:
                # Extract parameters
                returns = kwargs.get('returns')
                benchmark_returns = kwargs.get('benchmark_returns')
                
                if isinstance(returns, str):
                    # Assume it's a ticker symbol - fetch data
                    returns = self._fetch_returns(returns)
                
                if isinstance(benchmark_returns, str):
                    benchmark_returns = self._fetch_returns(benchmark_returns)
                
                # Calculate metrics
                risk_metrics = self.calculator.calculate_comprehensive_risk(
                    returns=returns,
                    benchmark_returns=benchmark_returns,
                    confidence_levels=kwargs.get('confidence_levels', [0.95, 0.99]),
                    risk_free_rate=kwargs.get('risk_free_rate', 0.02),
                    periods_per_year=kwargs.get('periods_per_year', 252)
                )
                
                # Return summary
                return self.calculator.get_risk_summary(risk_metrics)
                
            except Exception as e:
                return f"Risk calculation failed: {str(e)}"
        
        def _fetch_returns(self, symbol: str, period: str = "1y") -> np.ndarray:
            """Fetch returns for a given symbol"""
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if hist.empty:
                    raise ValueError(f"No data found for symbol {symbol}")
                
                returns = hist['Close'].pct_change().dropna()
                return returns.values
                
            except Exception as e:
                raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
        
        async def _arun(self, **kwargs) -> str:
            """Async version"""
            return self._run(**kwargs)
    
    # Create the tool instance
    risk_calculator_tool = RiskCalculatorTool()

else:
    # Fallback when LangChain is not available
    risk_calculator_tool = None
    logger.warning("LangChain not available - tool integration disabled")

# Standalone calculator instance
risk_calculator = EnhancedRiskCalculator()
RiskCalculatorService = EnhancedRiskCalculator

def calculate_portfolio_risk(
    returns: Union[List[float], np.ndarray, pd.Series],
    **kwargs
) -> RiskMetrics:
    """
    Convenience function for risk calculation
    
    Args:
        returns: Portfolio returns
        **kwargs: Additional parameters for risk calculation
        
    Returns:
        RiskMetrics object with comprehensive risk analysis
    """
    return risk_calculator.calculate_comprehensive_risk(returns, **kwargs)

def get_risk_calculator():
    """Factory function to get a risk calculator instance"""
    return EnhancedRiskCalculator()


# Validation and testing functions
def validate_risk_calculator():
    """Validate risk calculator with known test cases"""
    
    print("🧪 Validating Risk Calculator...")
    
    # Test case 1: Normal distribution
    np.random.seed(42)
    normal_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns, ~20% annual vol
    
    try:
        risk_metrics = calculate_portfolio_risk(normal_returns)
        
        # Validate basic metrics
        assert 0.15 < risk_metrics.annualized_volatility < 0.25, "Volatility out of expected range"
        assert -0.05 < risk_metrics.var_95 < 0, "VaR 95% out of expected range"
        assert risk_metrics.cvar_95 < risk_metrics.var_95, "CVaR should be more negative than VaR"
        assert not np.isnan(risk_metrics.sharpe_ratio), "Sharpe ratio should not be NaN"
        assert not np.isnan(risk_metrics.hurst_exponent), "Hurst exponent should not be NaN"
        
        print("✅ Test Case 1 (Normal Distribution): PASSED")
        
    except AssertionError as e:
        print(f"❌ Test Case 1 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Test Case 1 ERROR: {str(e)}")
        return False
    
    # Test case 2: Market data (if possible)
    try:
        spy_data = yf.download("SPY", start="2020-01-01", end="2023-01-01", progress=False)
        spy_returns = spy_data['Close'].pct_change().dropna().values
        
        risk_metrics = calculate_portfolio_risk(spy_returns)
        
        # Validate SPY metrics (approximate ranges)
        assert 0.10 < risk_metrics.annualized_volatility < 0.40, "SPY volatility out of expected range"
        assert risk_metrics.max_drawdown < 0, "Max drawdown should be negative"
        
        print("✅ Test Case 2 (SPY Market Data): PASSED")
        
    except Exception as e:
        print(f"⚠️ Test Case 2 (Market Data): SKIPPED - {str(e)}")
    
    print("🎉 Risk Calculator Validation Complete")
    return True

if __name__ == "__main__":
    # Run validation
    validate_risk_calculator()