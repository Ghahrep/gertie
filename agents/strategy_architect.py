# in agents/strategy_architect.py
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional, List
import asyncio

from agents.mcp_base_agent import MCPBaseAgent
from tools.strategy_tools import design_mean_reversion_strategy, design_momentum_strategy

class StrategyArchitectAgent(MCPBaseAgent):
    """
    Investment strategy design specialist migrated to MCP architecture.
    Designs quantitative strategies including mean-reversion and momentum approaches.
    """
    
    def __init__(self):
        # Define MCP capabilities
        capabilities = [
            "strategy_design",
            "momentum_strategy", 
            "mean_reversion_strategy",
            "backtesting_analysis",
            "signal_generation",
            "debate_participation",
            "consensus_building", 
            "collaborative_analysis"
                ]
        
        super().__init__(
            agent_id="strategy_architect",
            agent_name="StrategyArchitectAgent",
            capabilities=capabilities
        )
        
        # Strategy design tools
        self.tools = [
            design_mean_reversion_strategy,
            design_momentum_strategy,
        ]
        
        # Create tool mapping for strategy types
        self.tool_map = {
            "mean_reversion": design_mean_reversion_strategy,
            "momentum": design_momentum_strategy,
        }
        
        # Strategy types and their characteristics
        self.strategy_types = {
            "momentum": {
                "description": "Trend-following strategy that buys assets with strong upward price movement",
                "indicators": ["moving_average", "rsi", "macd", "breakout"],
                "market_conditions": ["trending", "bull_market", "high_volatility"],
                "risk_factors": ["trend_reversal", "whipsaws", "momentum_crashes"]
            },
            "mean_reversion": {
                "description": "Counter-trend strategy that buys oversold assets expecting price normalization",
                "indicators": ["bollinger_bands", "rsi", "stochastic", "price_deviation"],
                "market_conditions": ["range_bound", "stable_volatility", "mean_reverting"],
                "risk_factors": ["trend_continuation", "fundamental_deterioration", "liquidity_issues"]
            },
            "pairs_trading": {
                "description": "Market-neutral strategy trading price divergences between correlated assets",
                "indicators": ["cointegration", "z_score", "correlation", "spread"],
                "market_conditions": ["any", "market_neutral"],
                "risk_factors": ["correlation_breakdown", "regime_change", "execution_costs"]
            },
            "breakout": {
                "description": "Strategy that captures significant price movements after consolidation periods",
                "indicators": ["support_resistance", "volume", "volatility", "range_compression"],
                "market_conditions": ["low_volatility_preceding", "consolidation"],
                "risk_factors": ["false_breakouts", "gap_risk", "news_events"]
            }
        }
    
    @property
    def name(self) -> str: 
        return "StrategyArchitect"
        
    @property
    def purpose(self) -> str: 
        return "Designs new investment strategies based on quantitative signals."
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Execute MCP capability with routing to appropriate methods"""
        
        try:
            if capability == "strategy_design":
                return await self._design_strategy(data, context)
            elif capability == "momentum_strategy":
                return await self._create_momentum_strategy(data, context)
            elif capability == "mean_reversion_strategy":
                return await self._create_mean_reversion_strategy(data, context)
            elif capability == "backtesting_analysis":
                return await self._analyze_backtest_results(data, context)
            elif capability == "signal_generation":
                return await self._generate_trading_signals(data, context)
            else:
                return {
                    "success": False,
                    "error": f"Unknown capability: {capability}",
                    "error_type": "invalid_capability"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing {capability}: {str(e)}",
                "error_type": "execution_error"
            }
    
    async def _design_strategy(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Design a comprehensive investment strategy based on user requirements"""
        
        query = data.get("query", "").lower()
        
        # Analyze query to determine strategy type
        strategy_type = self._determine_strategy_type(query)
        
        if strategy_type == "unknown":
            # Provide strategy recommendations based on market context
            return await self._recommend_strategy_type(data, context)
        
        # Extract universe from query or context
        universe = self._extract_universe(data.get("query", ""), context)
        
        if not universe:
            return await self._provide_strategy_guidance(strategy_type)
        
        # Design the specific strategy
        if strategy_type in ["momentum", "mean_reversion"]:
            return await self._execute_quantitative_strategy(strategy_type, universe, query)
        else:
            return await self._design_custom_strategy(strategy_type, universe, query)
    
    async def _create_momentum_strategy(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Create a momentum-based trading strategy"""
        
        query = data.get("query", "")
        universe = self._extract_universe(query, context)
        
        if not universe:
            return {
                "success": False,
                "error": "Please specify stock symbols for momentum strategy (e.g., AAPL, MSFT)",
                "error_type": "missing_universe"
            }
        
        # Get market data and design momentum strategy
        price_data = await self._fetch_market_data(universe)
        if price_data is None:
            return {
                "success": False,
                "error": "Could not retrieve market data for specified symbols",
                "error_type": "data_fetch_error"
            }
        
        # Execute momentum strategy design
        result = await self._execute_strategy_tool("momentum", price_data)
        
        # Enhance with momentum-specific insights
        if result.get("success"):
            result["strategy_insights"] = self._generate_momentum_insights(result)
            result["implementation_guidance"] = self._get_momentum_implementation_tips()
        
        return result
    
    async def _create_mean_reversion_strategy(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Create a mean-reversion trading strategy"""
        
        query = data.get("query", "")
        universe = self._extract_universe(query, context)
        
        if not universe:
            return {
                "success": False,
                "error": "Please specify stock symbols for mean-reversion strategy (e.g., AAPL, MSFT)", 
                "error_type": "missing_universe"
            }
        
        # Get market data and design mean-reversion strategy
        price_data = await self._fetch_market_data(universe)
        if price_data is None:
            return {
                "success": False,
                "error": "Could not retrieve market data for specified symbols",
                "error_type": "data_fetch_error"
            }
        
        # Execute mean-reversion strategy design
        result = await self._execute_strategy_tool("mean_reversion", price_data)
        
        # Enhance with mean-reversion specific insights
        if result.get("success"):
            result["strategy_insights"] = self._generate_mean_reversion_insights(result)
            result["implementation_guidance"] = self._get_mean_reversion_implementation_tips()
        
        return result
    
    async def _analyze_backtest_results(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze backtesting results and provide performance insights"""
        
        query = data.get("query", "")
        
        # Check if backtest data is provided in context
        backtest_data = context.get("backtest_results") or data.get("backtest_results")
        
        if not backtest_data:
            return {
                "success": True,
                "analysis": "No specific backtest results provided. Here's a framework for analyzing strategy performance:",
                "framework": {
                    "return_metrics": ["Total Return", "Annualized Return", "CAGR"],
                    "risk_metrics": ["Volatility", "Maximum Drawdown", "VaR"],
                    "risk_adjusted": ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
                    "consistency": ["Win Rate", "Profit Factor", "Average Trade"],
                    "robustness": ["Out-of-sample testing", "Walk-forward analysis"]
                },
                "interpretation_guide": self._get_backtest_interpretation_guide()
            }
        
        # Analyze provided backtest results
        analysis = self._analyze_performance_metrics(backtest_data)
        recommendations = self._generate_strategy_improvements(analysis)
        
        return {
            "success": True,
            "performance_analysis": analysis,
            "recommendations": recommendations,
            "risk_assessment": self._assess_strategy_risks(analysis)
        }
    
    async def _generate_trading_signals(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators"""
        
        query = data.get("query", "").lower()
        universe = self._extract_universe(query, context)
        
        if not universe:
            return {
                "success": True,
                "signal_framework": "Trading signal generation framework",
                "common_signals": {
                    "momentum": ["Moving Average Crossover", "MACD", "RSI Breakout"],
                    "mean_reversion": ["Bollinger Band Mean Reversion", "RSI Oversold/Overbought"],
                    "breakout": ["Support/Resistance Break", "Volume Confirmation"],
                    "trend": ["Trend Line Break", "Channel Breakout"]
                },
                "implementation_notes": "Specify symbols to generate specific trading signals"
            }
        
        # Generate signals for specific universe
        price_data = await self._fetch_market_data(universe)
        if price_data is None:
            return {
                "success": False,
                "error": "Could not retrieve market data for signal generation",
                "error_type": "data_fetch_error"
            }
        
        # Generate various signal types
        signals = {}
        signals["momentum"] = self._generate_momentum_signals(price_data)
        signals["mean_reversion"] = self._generate_mean_reversion_signals(price_data)
        signals["trend"] = self._generate_trend_signals(price_data)
        
        return {
            "success": True,
            "signals": signals,
            "universe": universe,
            "signal_strength": self._assess_signal_strength(signals),
            "recommendations": self._generate_signal_recommendations(signals)
        }
    
    # Helper methods for strategy design and analysis
    
    def _determine_strategy_type(self, query: str) -> str:
        """Determine strategy type from query"""
        
        strategy_keywords = {
            "momentum": ["momentum", "trend", "trending", "breakout", "moving average"],
            "mean_reversion": ["mean-reversion", "reversion", "oversold", "overbought", "contrarian"],
            "pairs_trading": ["pairs", "pair trading", "market neutral", "spread"],
            "breakout": ["breakout", "break out", "resistance", "support"]
        }
        
        for strategy_type, keywords in strategy_keywords.items():
            if any(keyword in query for keyword in keywords):
                return strategy_type
        
        return "unknown"
    
    def _extract_universe(self, query: str, context: Dict) -> List[str]:
        """Extract stock symbols from query or context"""
        
        # First check context for holdings
        holdings = context.get("holdings_with_values", [])
        if holdings:
            universe = [h.get("symbol") for h in holdings if h.get("symbol")]
            if universe:
                return universe[:10]  # Limit to 10 symbols for strategy design
        
        # Extract from query - look for uppercase words that could be symbols
        words = query.replace(",", " ").split()
        universe = [word.upper() for word in words if word.isupper() and 1 < len(word) <= 5]
        
        # Remove common false positives
        false_positives = {"AND", "OR", "THE", "FOR", "WITH", "FROM", "TO"}
        universe = [symbol for symbol in universe if symbol not in false_positives]
        
        return universe[:10] if universe else []
    
    async def _fetch_market_data(self, universe: List[str]) -> Optional[pd.DataFrame]:
        """Fetch market data for strategy design"""
        
        try:
            # Use asyncio to run yfinance download in executor to avoid blocking
            loop = asyncio.get_event_loop()
            price_data = await loop.run_in_executor(
                None, 
                lambda: yf.download(universe, period="1y", auto_adjust=True, progress=False)['Close']
            )
            
            # Handle single symbol case
            if isinstance(price_data, pd.Series):
                price_data = price_data.to_frame(name=universe[0])
            
            return price_data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
    
    async def _execute_strategy_tool(self, strategy_type: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Execute strategy design tool asynchronously"""
        
        tool = self.tool_map.get(strategy_type)
        if not tool:
            return {
                "success": False,
                "error": f"No tool available for {strategy_type} strategy",
                "error_type": "tool_not_found"
            }
        
        try:
            # Run tool in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tool.invoke({"asset_prices": price_data})
            )
            
            # Enhance result with metadata
            if result.get("success"):
                result["universe_size"] = len(price_data.columns)
                result["analysis_period"] = f"{len(price_data)} days"
                result["strategy_type"] = strategy_type
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Strategy design failed: {str(e)}",
                "error_type": "execution_error"
            }
    
    async def _recommend_strategy_type(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Recommend strategy type based on context and market conditions"""
        
        recommendations = []
        
        # Analyze portfolio context if available
        holdings = context.get("holdings_with_values", [])
        if holdings:
            equity_concentration = len([h for h in holdings if h.get("symbol")])
            if equity_concentration > 10:
                recommendations.append({
                    "strategy": "mean_reversion",
                    "rationale": "Large portfolio suitable for mean-reversion across multiple holdings",
                    "priority": "high"
                })
        
        # Market condition-based recommendations
        query = data.get("query", "").lower()
        
        if any(word in query for word in ["volatile", "uncertainty", "risk"]):
            recommendations.append({
                "strategy": "mean_reversion", 
                "rationale": "Mean-reversion strategies often perform better in volatile markets",
                "priority": "medium"
            })
        
        if any(word in query for word in ["growth", "bull", "trending"]):
            recommendations.append({
                "strategy": "momentum",
                "rationale": "Momentum strategies excel in trending markets",
                "priority": "high"
            })
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                {
                    "strategy": "momentum",
                    "rationale": "Trend-following approach suitable for most market conditions",
                    "priority": "medium"
                },
                {
                    "strategy": "mean_reversion",
                    "rationale": "Counter-trend approach for risk-conscious investors",
                    "priority": "medium"
                }
            ]
        
        return {
            "success": True,
            "strategy_recommendations": recommendations,
            "next_steps": "Choose a strategy type and specify stock symbols to begin design",
            "available_strategies": list(self.strategy_types.keys())
        }
    
    async def _provide_strategy_guidance(self, strategy_type: str) -> Dict[str, Any]:
        """Provide guidance for strategy implementation without specific universe"""
        
        strategy_info = self.strategy_types.get(strategy_type, {})
        
        return {
            "success": True,
            "strategy_type": strategy_type,
            "description": strategy_info.get("description", "Custom strategy approach"),
            "key_indicators": strategy_info.get("indicators", []),
            "suitable_conditions": strategy_info.get("market_conditions", []),
            "risk_factors": strategy_info.get("risk_factors", []),
            "implementation_steps": [
                "1. Define universe of stocks to analyze",
                "2. Gather historical price data",
                "3. Calculate technical indicators",
                "4. Define entry/exit rules",
                "5. Backtest strategy performance",
                "6. Optimize parameters",
                "7. Implement risk management"
            ],
            "next_steps": "Provide stock symbols (e.g., AAPL, MSFT) to design specific strategy"
        }
    
    async def _design_custom_strategy(self, strategy_type: str, universe: List[str], query: str) -> Dict[str, Any]:
        """Design custom strategy for non-standard strategy types"""
        
        return {
            "success": True,
            "strategy_type": strategy_type,
            "universe": universe,
            "design_framework": {
                "signal_generation": f"Custom {strategy_type} signals",
                "entry_rules": f"Entry conditions for {strategy_type}",
                "exit_rules": f"Exit conditions for {strategy_type}",
                "risk_management": "Position sizing and stop-loss rules"
            },
            "implementation_note": f"Custom {strategy_type} strategy requires detailed specification of rules and parameters",
            "recommendation": "Consider using momentum or mean-reversion strategies for standardized implementation"
        }
    
    async def _execute_quantitative_strategy(self, strategy_type: str, universe: List[str], query: str) -> Dict[str, Any]:
        """Execute quantitative strategy design for momentum or mean-reversion"""
        
        price_data = await self._fetch_market_data(universe)
        if price_data is None:
            return {
                "success": False,
                "error": "Could not retrieve market data for strategy design",
                "error_type": "data_fetch_error"
            }
        
        result = await self._execute_strategy_tool(strategy_type, price_data)
        
        # Add strategy-specific enhancements
        if result.get("success"):
            result["implementation_checklist"] = self._get_implementation_checklist(strategy_type)
            result["performance_expectations"] = self._get_performance_expectations(strategy_type)
            result["monitoring_metrics"] = self._get_monitoring_metrics(strategy_type)
        
        return result
    
    def _generate_momentum_insights(self, result: Dict) -> List[str]:
        """Generate momentum strategy insights"""
        
        insights = []
        candidates = result.get("candidates", [])
        
        if candidates:
            insights.append(f"Found {len(candidates)} momentum candidates with strong trend signals")
            
            # Analyze candidate characteristics
            avg_score = sum(c.get("score", 0) for c in candidates) / len(candidates)
            insights.append(f"Average momentum score: {avg_score:.2f} indicates {'strong' if avg_score > 0.7 else 'moderate'} trend strength")
        
        insights.extend([
            "Momentum strategies perform best in trending markets",
            "Consider trend confirmation with volume indicators",
            "Implement stop-losses to protect against reversals"
        ])
        
        return insights
    
    def _generate_mean_reversion_insights(self, result: Dict) -> List[str]:
        """Generate mean-reversion strategy insights"""
        
        insights = []
        candidates = result.get("candidates", [])
        
        if candidates:
            insights.append(f"Identified {len(candidates)} mean-reversion opportunities")
            insights.append("Mean-reversion candidates show oversold conditions")
        
        insights.extend([
            "Mean-reversion works best in range-bound markets", 
            "Monitor fundamental health of oversold securities",
            "Use position sizing to manage individual stock risk"
        ])
        
        return insights
    
    def _get_momentum_implementation_tips(self) -> List[str]:
        """Get momentum strategy implementation guidance"""
        
        return [
            "Use trend confirmation indicators (MACD, moving averages)",
            "Implement trailing stops to capture trend reversals",
            "Consider volume confirmation for stronger signals",
            "Monitor sector rotation and market leadership",
            "Diversify across multiple momentum signals"
        ]
    
    def _get_mean_reversion_implementation_tips(self) -> List[str]:
        """Get mean-reversion strategy implementation guidance"""
        
        return [
            "Confirm oversold conditions with multiple indicators",
            "Use fundamental screening to avoid value traps",
            "Implement gradual position building in oversold stocks",
            "Set clear profit targets for mean-reversion trades",
            "Monitor correlation to avoid concentrated positions"
        ]
    
    def _get_implementation_checklist(self, strategy_type: str) -> List[str]:
        """Get implementation checklist for strategy type"""
        
        common_items = [
            "Define position sizing rules",
            "Set risk management parameters", 
            "Establish performance monitoring",
            "Create rebalancing schedule"
        ]
        
        strategy_specific = {
            "momentum": [
                "Confirm trend direction",
                "Set trailing stop levels",
                "Monitor trend strength indicators"
            ],
            "mean_reversion": [
                "Verify mean-reversion signals",
                "Check fundamental quality",
                "Set profit-taking targets"
            ]
        }
        
        return common_items + strategy_specific.get(strategy_type, [])
    
    def _get_performance_expectations(self, strategy_type: str) -> Dict[str, str]:
        """Get performance expectations for strategy type"""
        
        expectations = {
            "momentum": {
                "market_conditions": "Bull markets and trending periods",
                "expected_return": "10-15% annual excess return",
                "volatility": "Moderate to high",
                "drawdowns": "Can experience sharp reversals"
            },
            "mean_reversion": {
                "market_conditions": "Range-bound and volatile markets",
                "expected_return": "8-12% annual excess return", 
                "volatility": "Low to moderate",
                "drawdowns": "Generally smaller but more frequent"
            }
        }
        
        return expectations.get(strategy_type, {})
    
    def _get_monitoring_metrics(self, strategy_type: str) -> List[str]:
        """Get monitoring metrics for strategy type"""
        
        common_metrics = ["Portfolio return", "Sharpe ratio", "Maximum drawdown"]
        
        strategy_metrics = {
            "momentum": ["Trend strength", "Signal decay", "Reversal frequency"],
            "mean_reversion": ["Mean-reversion speed", "Oversold recovery rate", "False signal rate"]
        }
        
        return common_metrics + strategy_metrics.get(strategy_type, [])
    
    def _generate_momentum_signals(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate momentum signals from price data"""
        
        # Simplified momentum signal generation
        signals = {}
        
        for symbol in price_data.columns:
            prices = price_data[symbol].dropna()
            if len(prices) < 50:
                continue
            
            # Simple momentum indicators
            ma_20 = prices.rolling(20).mean()
            ma_50 = prices.rolling(50).mean()
            current_price = prices.iloc[-1]
            
            # Generate signal
            signal_strength = 0
            if current_price > ma_20.iloc[-1]:
                signal_strength += 1
            if ma_20.iloc[-1] > ma_50.iloc[-1]:
                signal_strength += 1
            if current_price > prices.iloc[-20]:  # 20-day momentum
                signal_strength += 1
            
            signals[symbol] = {
                "signal_strength": signal_strength / 3,
                "current_price": current_price,
                "trend": "bullish" if signal_strength >= 2 else "bearish"
            }
        
        return signals
    
    def _generate_mean_reversion_signals(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate mean-reversion signals from price data"""
        
        signals = {}
        
        for symbol in price_data.columns:
            prices = price_data[symbol].dropna()
            if len(prices) < 50:
                continue
            
            # Simple mean-reversion indicators
            ma_20 = prices.rolling(20).mean()
            std_20 = prices.rolling(20).std()
            current_price = prices.iloc[-1]
            
            # Z-score for mean reversion
            z_score = (current_price - ma_20.iloc[-1]) / std_20.iloc[-1]
            
            signals[symbol] = {
                "z_score": z_score,
                "signal": "oversold" if z_score < -1.5 else "overbought" if z_score > 1.5 else "neutral",
                "mean_reversion_strength": abs(z_score) if abs(z_score) > 1 else 0
            }
        
        return signals
    
    def _generate_trend_signals(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trend signals from price data"""
        
        signals = {}
        
        for symbol in price_data.columns:
            prices = price_data[symbol].dropna()
            if len(prices) < 20:
                continue
            
            # Simple trend analysis
            ma_10 = prices.rolling(10).mean()
            ma_20 = prices.rolling(20).mean()
            
            trend_direction = "up" if ma_10.iloc[-1] > ma_20.iloc[-1] else "down"
            trend_strength = abs(ma_10.iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1]
            
            signals[symbol] = {
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "signal": "strong_trend" if trend_strength > 0.02 else "weak_trend"
            }
        
        return signals
    
    def _assess_signal_strength(self, signals: Dict) -> Dict[str, Any]:
        """Assess overall signal strength across different signal types"""
        
        assessment = {}
        
        for signal_type, symbol_signals in signals.items():
            if not symbol_signals:
                continue
            
            strong_signals = 0
            total_signals = len(symbol_signals)
            
            for symbol, signal_data in symbol_signals.items():
                if signal_type == "momentum":
                    if signal_data.get("signal_strength", 0) > 0.6:
                        strong_signals += 1
                elif signal_type == "mean_reversion":
                    if signal_data.get("mean_reversion_strength", 0) > 1.5:
                        strong_signals += 1
                elif signal_type == "trend":
                    if signal_data.get("trend_strength", 0) > 0.02:
                        strong_signals += 1
            
            assessment[signal_type] = {
                "strong_signals": strong_signals,
                "total_signals": total_signals,
                "signal_ratio": strong_signals / total_signals if total_signals > 0 else 0
            }
        
        return assessment
    
    def _generate_signal_recommendations(self, signals: Dict) -> List[str]:
        """Generate recommendations based on signal analysis"""
        
        recommendations = []
        
        # Analyze momentum signals
        momentum_signals = signals.get("momentum", {})
        bullish_momentum = len([s for s in momentum_signals.values() if s.get("trend") == "bullish"])
        
        if bullish_momentum > len(momentum_signals) * 0.6:
            recommendations.append("Strong bullish momentum detected - consider momentum strategy")
        
        # Analyze mean-reversion signals  
        mr_signals = signals.get("mean_reversion", {})
        oversold_count = len([s for s in mr_signals.values() if s.get("signal") == "oversold"])
        
        if oversold_count > len(mr_signals) * 0.3:
            recommendations.append("Multiple oversold opportunities - consider mean-reversion approach")
        
        # Analyze trend signals
        trend_signals = signals.get("trend", {})
        strong_trends = len([s for s in trend_signals.values() if s.get("signal") == "strong_trend"])
        
        if strong_trends > len(trend_signals) * 0.4:
            recommendations.append("Strong trending environment - trend-following strategies favored")
        
        if not recommendations:
            recommendations.append("Mixed signals - consider diversified multi-strategy approach")
        
        return recommendations
    
    def _analyze_performance_metrics(self, backtest_data: Dict) -> Dict[str, Any]:
        """Analyze backtest performance metrics"""
        
        # This would analyze actual backtest data if provided
        # For now, return framework for analysis
        
        return {
            "return_analysis": "Analyze total and risk-adjusted returns",
            "risk_analysis": "Evaluate maximum drawdown and volatility",
            "consistency": "Assess win rate and profit consistency",
            "robustness": "Test across different market conditions"
        }
    
    def _generate_strategy_improvements(self, analysis: Dict) -> List[str]:
        """Generate strategy improvement recommendations"""
        
        return [
            "Optimize entry/exit timing based on backtest results",
            "Adjust position sizing for better risk management",
            "Consider adding filters to reduce false signals",
            "Test strategy across different market regimes"
        ]
    
    def _assess_strategy_risks(self, analysis: Dict) -> Dict[str, Any]:
        """Assess strategy-specific risks"""
        
        return {
            "market_risk": "Strategy sensitivity to market conditions",
            "model_risk": "Risk of strategy overfitting to historical data", 
            "execution_risk": "Implementation and transaction cost impacts",
            "regime_risk": "Performance in different market environments"
        }
    
    def _get_backtest_interpretation_guide(self) -> Dict[str, str]:
        """Get guide for interpreting backtest results"""
        
        return {
            "sharpe_ratio": "> 1.0 is good, > 2.0 is excellent",
            "max_drawdown": "< 10% is low risk, > 20% is high risk",
            "win_rate": "> 50% is positive, but consider profit factor",
            "profit_factor": "> 1.5 indicates profitable strategy",
            "volatility": "Should be appropriate for expected returns"
        }
    
    def _generate_summary(self, result: Dict, capability: str, execution_time: float = 0.0) -> str:
        """Generate custom summary for strategy architect responses"""
        
        if not result.get("success"):
            return f"Unable to complete strategy design: {result.get('error', 'Unknown error')}"
        
        if capability == "strategy_design":
            strategy_type = result.get("strategy_type", "custom")
            universe_size = result.get("universe_size", 0)
            if universe_size > 0:
                return f"**{strategy_type.title()} strategy** designed for {universe_size} securities with implementation guidance"
            else:
                return f"**Strategy design guidance** provided for {strategy_type} approach with framework and recommendations"
            
        elif capability == "momentum_strategy":
            candidates = len(result.get("candidates", []))
            return f"**Momentum strategy** created with {candidates} trend-following candidates and implementation tips"
            
        elif capability == "mean_reversion_strategy":
            candidates = len(result.get("candidates", []))
            return f"**Mean-reversion strategy** designed with {candidates} mean-reverting opportunities and risk guidance"
            
        elif capability == "backtesting_analysis":
            if result.get("performance_analysis"):
                return f"**Backtest analysis** completed with performance metrics and improvement recommendations"
            else:
                return f"**Backtesting framework** provided with analysis methodology and interpretation guide"
            
        elif capability == "signal_generation":
            universe_size = len(result.get("universe", []))
            if universe_size > 0:
                return f"**Trading signals** generated for {universe_size} securities across momentum, mean-reversion, and trend indicators"
            else:
                return f"**Signal generation framework** provided with common indicators and implementation guidance"
            
        else:
            return f"**Strategy architecture** analysis completed successfully in {execution_time:.2f}s"
    
    async def _health_check_capability(self, capability: str) -> bool:
        """Health check for specific capabilities"""
        
        test_data = {
            "strategy_design": {"query": "design momentum strategy"},
            "momentum_strategy": {"query": "create momentum strategy for AAPL"},
            "mean_reversion_strategy": {"query": "design mean reversion strategy"},
            "backtesting_analysis": {"query": "analyze strategy performance"},
            "signal_generation": {"query": "generate trading signals"}
        }
        
        if capability in test_data:
            try:
                result = await self.execute_capability(capability, test_data[capability], {})
                return result.get("success", False)
            except:
                return False
        
        return False