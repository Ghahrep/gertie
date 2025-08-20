"""
Risk Calculator Service - FIXED VERSION
=======================================

Fixed to work with LangChain tool decorators in your existing tools.
Integrates with existing risk_tools.py and fractal_tools.py modules.

Author: Quant Platform Development
Created: August 19, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class RiskMetrics:
    """Data class to hold comprehensive risk metrics"""
    volatility: float
    beta: float
    max_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    hurst_exponent: float
    dfa_alpha: float
    risk_score: float
    sentiment_index: int
    regime_volatility: float
    timestamp: datetime

class RiskCalculatorService:
    """
    Centralized risk calculation engine for portfolio monitoring.
    FIXED to work with LangChain tool decorators.
    """
    
    def __init__(self, market_benchmark: str = "SPY"):
        self.market_benchmark = market_benchmark
        self.trading_days_per_year = 252
        
    def calculate_comprehensive_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        market_returns: Optional[pd.Series] = None,
        price_data: Optional[pd.DataFrame] = None
    ) -> Optional[RiskMetrics]:
        """
        Calculate comprehensive risk metrics for a portfolio.
        FIXED to work with your existing LangChain tools.
        """
        try:
            if len(portfolio_returns) < 30:
                print("Warning: Insufficient data for reliable risk calculation")
                return None
                
            # 1. Core Risk Metrics using DIRECT function calls (not LangChain tool calls)
            risk_report = self._calculate_risk_metrics_direct(portfolio_returns)
            
            if not risk_report:
                print("Failed to calculate core risk metrics")
                return None
            
            # 2. Drawdown Analysis - use direct function call
            drawdown_stats = self._calculate_drawdowns_direct(portfolio_returns)
            max_drawdown = drawdown_stats['max_drawdown_pct'] / 100 if drawdown_stats else 0.0
            calmar_ratio = drawdown_stats['calmar_ratio'] if drawdown_stats else 0.0
            
            # 3. Beta Calculation (if market data available)
            beta = 1.0  # Default beta
            if market_returns is not None:
                calculated_beta = self._calculate_beta_direct(portfolio_returns, market_returns)
                beta = calculated_beta if calculated_beta is not None else 1.0
            
            # 4. Fractal Analysis - use direct function calls
            hurst_result = self._safe_calculate_hurst(portfolio_returns)
            hurst_exponent = hurst_result.get('hurst_exponent', 0.5) if hurst_result else 0.5
            
            dfa_result = self._safe_calculate_dfa(portfolio_returns)
            dfa_alpha = dfa_result.get('dfa_alpha', 0.5) if dfa_result else 0.5
            
            # 5. Regime Analysis for volatility context
            regime_volatility = self._calculate_regime_volatility(portfolio_returns)
            
            # 6. Risk Sentiment Index
            sentiment_index = self._calculate_sentiment_index(risk_report, price_data)
            
            # 7. Composite Risk Score
            risk_score = self._calculate_composite_risk_score(
                volatility=risk_report['performance_stats']['annualized_volatility_pct'] / 100,
                max_drawdown=abs(max_drawdown),
                sharpe_ratio=risk_report['risk_adjusted_ratios']['sharpe_ratio'],
                cvar_99=risk_report['risk_measures']['99%']['cvar_expected_shortfall']
            )
            
            # Create RiskMetrics object
            return RiskMetrics(
                volatility=risk_report['performance_stats']['annualized_volatility_pct'] / 100,
                beta=beta,
                max_drawdown=abs(max_drawdown),
                var_95=abs(risk_report['risk_measures']['95%']['var']),
                var_99=abs(risk_report['risk_measures']['99%']['var']),
                cvar_95=risk_report['risk_measures']['95%']['cvar_expected_shortfall'],
                cvar_99=risk_report['risk_measures']['99%']['cvar_expected_shortfall'],
                sharpe_ratio=risk_report['risk_adjusted_ratios']['sharpe_ratio'],
                sortino_ratio=risk_report['risk_adjusted_ratios']['sortino_ratio'],
                calmar_ratio=calmar_ratio,
                hurst_exponent=hurst_exponent,
                dfa_alpha=dfa_alpha,
                risk_score=risk_score,
                sentiment_index=sentiment_index,
                regime_volatility=regime_volatility,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error calculating comprehensive risk metrics: {e}")
            return None
    
    def _calculate_risk_metrics_direct(self, portfolio_returns: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Direct implementation of risk metrics calculation.
        Avoids LangChain tool call issues.
        """
        try:
            confidence_levels = [0.95, 0.99]
            trading_days = self.trading_days_per_year
            
            # VaR and CVaR Analysis
            var_analysis = {}
            for confidence in confidence_levels:
                var_value = portfolio_returns.quantile(1 - confidence)
                cvar_value = self._calculate_cvar_direct(portfolio_returns, confidence)
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

            return {
                "risk_measures": var_analysis,
                "performance_stats": {
                    "annualized_return_pct": annual_return * 100,
                    "annualized_volatility_pct": annual_vol * 100,
                },
                "risk_adjusted_ratios": {
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                }
            }
            
        except Exception as e:
            print(f"Error in direct risk metrics calculation: {e}")
            return None
    
    def _calculate_cvar_direct(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Direct CVaR calculation"""
        try:
            clean_returns = returns.dropna()
            if clean_returns.empty:
                return np.nan
                
            var_threshold = clean_returns.quantile(1 - confidence_level)
            tail_losses = clean_returns[clean_returns <= var_threshold]
            
            return -var_threshold if tail_losses.empty else -tail_losses.mean()
        except:
            return 0.0
    
    def _calculate_drawdowns_direct(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """Direct drawdown calculation"""
        try:
            if returns.empty:
                return None

            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - running_max) / running_max

            max_drawdown = drawdowns.min()
            
            # Handle datetime index
            end_date_idx = drawdowns.idxmin()
            start_date_idx = cumulative_returns.loc[:end_date_idx].idxmax()
            
            if isinstance(start_date_idx, (datetime, pd.Timestamp)):
                start_date_str = start_date_idx.strftime('%Y-%m-%d')
                end_date_str = end_date_idx.strftime('%Y-%m-%d')
            else:
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
        except Exception as e:
            print(f"Error calculating drawdowns: {e}")
            return None
    
    def _calculate_beta_direct(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> Optional[float]:
        """Direct beta calculation"""
        try:
            if portfolio_returns.empty or market_returns.empty:
                return None
            
            # Align data by index
            df = pd.DataFrame({'portfolio': portfolio_returns, 'market': market_returns}).dropna()
            if len(df) < 30:
                return None
                
            covariance = df['portfolio'].cov(df['market'])
            market_variance = df['market'].var()
            
            return covariance / market_variance if market_variance > 0 else None
        except:
            return None
    
    def _safe_calculate_hurst(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """Safely calculate Hurst exponent with error handling"""
        try:
            if len(returns) >= 100:  # Minimum data requirement
                # Import the function directly, not as LangChain tool
                from tools.fractal_tools import calculate_hurst
                # Use invoke method for LangChain tools
                try:
                    return calculate_hurst.invoke({"series": returns})
                except:
                    # Fallback to direct call if invoke doesn't work
                    return calculate_hurst(returns)
        except Exception as e:
            print(f"Hurst calculation failed: {e}")
        return None
    
    def _safe_calculate_dfa(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """Safely calculate DFA with error handling"""
        try:
            if len(returns) >= 100:  # Minimum data requirement
                # Import the function directly, not as LangChain tool
                from tools.fractal_tools import calculate_dfa
                # Use invoke method for LangChain tools
                try:
                    return calculate_dfa.invoke({"series": returns})
                except:
                    # Fallback to direct call if invoke doesn't work
                    return calculate_dfa(returns)
        except Exception as e:
            print(f"DFA calculation failed: {e}")
        return None
    
    def _calculate_regime_volatility(self, returns: pd.Series) -> float:
        """Calculate regime-aware volatility measure"""
        try:
            if len(returns) >= 100:
                # Import the function directly, not as LangChain tool
                from tools.regime_tools import detect_hmm_regimes
                # Use invoke method for LangChain tools
                try:
                    regime_result = detect_hmm_regimes.invoke({
                        "returns": returns, 
                        "n_regimes": 2
                    })
                except:
                    # Fallback: try direct call
                    try:
                        regime_result = detect_hmm_regimes(returns, n_regimes=2)
                    except:
                        regime_result = None
                
                if regime_result:
                    current_regime = regime_result['current_regime']
                    regime_char = regime_result['regime_characteristics']
                    return regime_char[current_regime]['volatility']
        except Exception as e:
            print(f"Regime volatility calculation failed: {e}")
        
        # Fallback to simple rolling volatility
        return returns.rolling(30).std().iloc[-1] * np.sqrt(252)
    
    def _calculate_sentiment_index(
        self, 
        risk_report: Dict[str, Any], 
        price_data: Optional[pd.DataFrame]
    ) -> int:
        """Calculate risk sentiment index"""
        try:
            if price_data is not None and len(price_data.columns) > 1:
                # Direct correlation calculation instead of using LangChain tool
                corr_matrix = price_data.pct_change().corr()
                if corr_matrix is not None:
                    # Simple sentiment calculation
                    np.fill_diagonal(corr_matrix.values, np.nan)
                    avg_corr = corr_matrix.mean().mean()
                    
                    # Calculate sentiment index components
                    vol_pct = risk_report['performance_stats']['annualized_volatility_pct']
                    vol_score = np.clip((vol_pct - 5) / (40 - 5), 0, 1) * 100
                    
                    cvar99 = abs(risk_report['risk_measures']['99%']['cvar_expected_shortfall'])
                    cvar_score = np.clip((cvar99 - 0.01) / (0.05 - 0.01), 0, 1) * 100
                    
                    corr_score = np.clip(avg_corr / 0.8, 0, 1) * 100
                    
                    # Weighted average
                    final_score = (0.4 * vol_score + 0.4 * cvar_score + 0.2 * corr_score)
                    return int(final_score)
        except Exception as e:
            print(f"Sentiment index calculation failed: {e}")
        
        # Fallback sentiment based on volatility
        vol_pct = risk_report['performance_stats']['annualized_volatility_pct']
        if vol_pct < 15:
            return 25  # Low risk
        elif vol_pct < 25:
            return 50  # Medium risk
        else:
            return 75  # High risk
    
    def _calculate_composite_risk_score(
        self,
        volatility: float,
        max_drawdown: float,
        sharpe_ratio: float,
        cvar_99: float
    ) -> float:
        """Calculate a composite risk score from 0-100"""
        try:
            # Normalize each component (0-1 scale)
            vol_score = min(volatility / 0.4, 1.0)  # Cap at 40% volatility
            drawdown_score = min(max_drawdown / 0.5, 1.0)  # Cap at 50% drawdown
            sharpe_score = max(0, min((2 - sharpe_ratio) / 2, 1.0))  # Invert Sharpe (lower is riskier)
            cvar_score = min(abs(cvar_99) / 0.1, 1.0)  # Cap at 10% CVaR
            
            # Weighted average
            weights = {
                'volatility': 0.3,
                'drawdown': 0.3,
                'sharpe': 0.2,
                'cvar': 0.2
            }
            
            composite_score = (
                weights['volatility'] * vol_score +
                weights['drawdown'] * drawdown_score +
                weights['sharpe'] * sharpe_score +
                weights['cvar'] * cvar_score
            )
            
            return round(composite_score * 100, 2)
            
        except Exception as e:
            print(f"Error calculating composite risk score: {e}")
            return 50.0  # Default medium risk
    
    def calculate_portfolio_risk_from_tickers(
        self,
        tickers: List[str],
        weights: List[float],
        lookback_days: int = 252
    ) -> Optional[RiskMetrics]:
        """
        Calculate risk metrics for a portfolio defined by tickers and weights.
        Fetches data automatically using yfinance.
        """
        try:
            if len(tickers) != len(weights):
                raise ValueError("Tickers and weights must have same length")
            
            if abs(sum(weights) - 1.0) > 0.01:
                print("Warning: Weights do not sum to 1.0, normalizing...")
                weights = [w / sum(weights) for w in weights]
            
            # Fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(lookback_days * 1.5))  # Extra buffer
            
            print(f"📊 Fetching data for {tickers} from {start_date.date()} to {end_date.date()}")
            
            price_data = yf.download(
                tickers + [self.market_benchmark], 
                start=start_date, 
                end=end_date,
                progress=False
            )['Close']
            
            if price_data.empty:
                print("Failed to fetch price data")
                return None
            
            print(f"✅ Fetched {len(price_data)} days of price data")
            
            # Calculate returns
            returns_data = price_data.pct_change().dropna()
            
            # Calculate portfolio returns
            portfolio_returns = (returns_data[tickers] * weights).sum(axis=1)
            market_returns = returns_data[self.market_benchmark] if self.market_benchmark in returns_data.columns else None
            
            # Use last 'lookback_days' of data
            portfolio_returns = portfolio_returns.tail(lookback_days)
            if market_returns is not None:
                market_returns = market_returns.tail(lookback_days)
            
            print(f"📈 Calculating risk metrics for {len(portfolio_returns)} days of returns")
            
            # Calculate comprehensive risk metrics
            return self.calculate_comprehensive_risk_metrics(
                portfolio_returns=portfolio_returns,
                market_returns=market_returns,
                price_data=returns_data[tickers].tail(lookback_days)
            )
            
        except Exception as e:
            print(f"Error calculating portfolio risk from tickers: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_risk_metrics(
        self,
        current_metrics: RiskMetrics,
        previous_metrics: RiskMetrics,
        threshold_pct: float = 15.0
    ) -> Dict[str, Any]:
        """Compare current vs previous risk metrics and identify significant changes."""
        try:
            def calculate_pct_change(current: float, previous: float) -> float:
                if previous == 0:
                    return 0.0
                return ((current - previous) / abs(previous)) * 100
            
            changes = {
                'volatility': calculate_pct_change(current_metrics.volatility, previous_metrics.volatility),
                'beta': calculate_pct_change(current_metrics.beta, previous_metrics.beta),
                'max_drawdown': calculate_pct_change(current_metrics.max_drawdown, previous_metrics.max_drawdown),
                'var_99': calculate_pct_change(current_metrics.var_99, previous_metrics.var_99),
                'cvar_99': calculate_pct_change(current_metrics.cvar_99, previous_metrics.cvar_99),
                'risk_score': calculate_pct_change(current_metrics.risk_score, previous_metrics.risk_score),
                'sentiment_index': calculate_pct_change(current_metrics.sentiment_index, previous_metrics.sentiment_index)
            }
            
            # Identify significant changes
            significant_changes = {
                metric: change for metric, change in changes.items()
                if abs(change) >= threshold_pct
            }
            
            # Overall risk direction
            risk_direction = "INCREASED" if current_metrics.risk_score > previous_metrics.risk_score else "DECREASED"
            risk_magnitude = abs(changes['risk_score'])
            
            return {
                'risk_direction': risk_direction,
                'risk_magnitude_pct': risk_magnitude,
                'all_changes': changes,
                'significant_changes': significant_changes,
                'threshold_breached': len(significant_changes) > 0,
                'comparison_timestamp': datetime.now(),
                'time_between_measurements': (current_metrics.timestamp - previous_metrics.timestamp).total_seconds() / 3600
            }
            
        except Exception as e:
            print(f"Error comparing risk metrics: {e}")
            return {
                'error': str(e),
                'threshold_breached': False
            }

# Utility function for easy access
def create_risk_calculator(market_benchmark: str = "SPY") -> RiskCalculatorService:
    """Factory function to create a risk calculator instance"""
    return RiskCalculatorService(market_benchmark=market_benchmark)

# Example usage and testing
if __name__ == "__main__":
    # Test the risk calculator
    calculator = create_risk_calculator()
    
    # Test with sample portfolio
    test_tickers = ['AAPL', 'MSFT']
    test_weights = [0.6, 0.4]
    
    print("Testing Risk Calculator Service...")
    risk_metrics = calculator.calculate_portfolio_risk_from_tickers(
        tickers=test_tickers,
        weights=test_weights,
        lookback_days=100  # Smaller for testing
    )
    
    if risk_metrics:
        print(f"✅ Risk calculation successful!")
        print(f"Portfolio Volatility: {risk_metrics.volatility:.2%}")
        print(f"Beta: {risk_metrics.beta:.2f}")
        print(f"Risk Score: {risk_metrics.risk_score}/100")
        print(f"Sentiment Index: {risk_metrics.sentiment_index}")
    else:
        print("❌ Risk calculation failed")