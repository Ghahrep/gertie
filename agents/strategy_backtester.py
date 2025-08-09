# in agents/strategy_backtester.py
from typing import Dict, Any, Optional
from agents.base_agent import BaseFinancialAgent
from tools.strategy_tools import run_regime_aware_backtest
from tools.regime_tools import detect_hmm_regimes
from strategies.logic import moving_average_crossover_logic, regime_switching_logic

class StrategyBacktesterAgent(BaseFinancialAgent):
    """
    ### UPGRADED: Operates on real user portfolio data from the context.
    """
    @property
    def name(self) -> str: return "StrategyBacktester"

    @property
    def purpose(self) -> str: return "Tests trading strategies against historical data."

    def __init__(self):
        self.strategies = {
            "moving_average_crossover": moving_average_crossover_logic,
            "regime_switching_crossover": regime_switching_logic,
        }
        tools = [run_regime_aware_backtest]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        if not context or "prices" not in context:
            return {"success": False, "error": "Could not backtest. Portfolio data is missing."}

        # 1. Parse query to select strategy
        query, strategy_to_test = user_query.lower(), None
        if "moving average" in query and "regime" not in query:
            strategy_to_test = "moving_average_crossover"
        elif "regime" in query:
            strategy_to_test = "regime_switching_crossover"
        if not strategy_to_test:
            return {"success": False, "error": "Please specify which strategy to backtest."}

        print(f"Selected strategy for testing: {strategy_to_test}")

        # 2. Prepare data from the context
        # The backtester tool needs OHLC data. Our context['prices'] is just 'Close'.
        # We will create the OHLC data needed by the tool.
        prices_df = context['prices'].copy()
        if isinstance(prices_df, pd.Series): # Handle single-asset portfolios
            prices_df = prices_df.to_frame(name=prices_df.name or 'Close')
            prices_df.columns = ['Close']
        else: # Use the first asset for the backtest
            prices_df = prices_df[[prices_df.columns[0]]].rename(columns={prices_df.columns[0]: 'Close'})

        prices_df['Open'] = prices_df['Close']
        prices_df['High'] = prices_df['Close']
        prices_df['Low'] = prices_df['Close']
        prices_df['Volume'] = 1_000_000 # Placeholder volume

        # 3. Generate regimes from the context's returns data
        hmm_results = detect_hmm_regimes(context["portfolio_returns"], n_regimes=2)
        if not hmm_results:
            return {"success": False, "error": "Could not generate market regimes for the backtest."}

        # 4. Execute backtest tool
        result = self.tool_map["RunRegimeAwareBacktest"].run({
            "prices": prices_df,
            "regime_series": hmm_results['regime_series'],
            "strategy_name": strategy_to_test
        })

        # 5. Format output
        if result and result.get("performance_summary"):
            summary_stats = result['performance_summary']
            result['summary'] = (f"Backtest for '{strategy_to_test}' on your portfolio's primary asset complete. "
                               f"Sharpe Ratio: {summary_stats['Sharpe Ratio']:.2f}. "
                               f"Max Drawdown: {summary_stats['Max. Drawdown [%]']:.2f}%.")
        else:
            result = {"success": False, "error": "Backtest failed to execute."}
        return result