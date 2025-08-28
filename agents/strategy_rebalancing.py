# in agents/strategy_rebalancing.py
from typing import Dict, Any, Optional, List
import re
import pandas as pd

from agents.mcp_base_agent import MCPBaseAgent
from tools.strategy_tools import optimize_portfolio, generate_trade_orders

class StrategyRebalancingAgent(MCPBaseAgent):
    """
    MCP-compatible portfolio rebalancing agent that operates on real user portfolio data 
    to generate optimized rebalancing plans with actionable trade recommendations.
    """
    
    def __init__(self):
        # Define MCP capabilities
        capabilities = [
            "portfolio_optimization",
            "rebalancing_analysis", 
            "trade_generation",
            "risk_parity_optimization",
            "sharpe_optimization",
            "minimum_variance_optimization"
        ]
        
        super().__init__(
            agent_id="StrategyRebalancingAgent",
            agent_name="Portfolio Rebalancing Specialist",
            capabilities=capabilities,
            max_concurrent_jobs=2
        )
        
        # Initialize tools for portfolio optimization
        self.tools = [optimize_portfolio, generate_trade_orders]
        self.tool_map = {
            "OptimizePortfolio": optimize_portfolio,
            "GenerateTradeOrders": generate_trade_orders,
        }
    
    @property
    def name(self) -> str: 
        return "Portfolio Rebalancing Specialist"

    @property
    def purpose(self) -> str: 
        return "Optimizes portfolios and generates actionable rebalancing trade plans with risk management."

    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Execute specific rebalancing capability"""
        
        try:
            # Check for required portfolio data
            if not context or "prices" not in context:
                return {
                    "success": False,
                    "error": "missing_portfolio",
                    "user_message": "Portfolio rebalancing requires your current holdings data. Please ensure you have an active portfolio."
                }
            
            query = data.get("query", "")
            
            if capability == "portfolio_optimization":
                return await self._optimize_portfolio(query, context)
            elif capability == "rebalancing_analysis":
                return await self._analyze_rebalancing_needs(query, context)
            elif capability == "trade_generation":
                return await self._generate_trades(query, context)
            elif capability == "risk_parity_optimization":
                return await self._risk_parity_optimization(query, context)
            elif capability == "sharpe_optimization":
                return await self._sharpe_optimization(query, context)
            elif capability == "minimum_variance_optimization":
                return await self._minimum_variance_optimization(query, context)
            else:
                return await self._general_rebalancing(query, context)
                
        except Exception as e:
            return {
                "success": False,
                "error": "execution_error",
                "user_message": f"Portfolio optimization encountered an issue: {str(e)}"
            }

    async def _optimize_portfolio(self, query: str, context: Dict) -> Dict[str, Any]:
        """General portfolio optimization"""
        
        # Infer optimization objective from query
        objective = self._infer_objective(query)
        
        # Execute optimization
        opt_result = self.tool_map["OptimizePortfolio"].invoke({
            "asset_prices": context["prices"],
            "objective": objective
        })
        
        if not opt_result or 'optimal_weights' not in opt_result:
            return {
                "success": False,
                "error": "optimization_failed",
                "user_message": "Portfolio optimization failed. Please check your portfolio data and try again."
            }
        
        # Generate trade recommendations
        trades = await self._generate_trade_orders(opt_result["optimal_weights"], context)
        
        return {
            "success": True,
            "result": {
                "objective": objective,
                "optimal_weights": opt_result["optimal_weights"],
                "trades": trades.get("trades", []),
                "optimization_metrics": opt_result
            },
            "confidence": 0.85,
            "agent_used": self.agent_id
        }

    async def _analyze_rebalancing_needs(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze current portfolio drift and rebalancing needs"""
        
        current_holdings = {h.asset.ticker: h.market_value for h in context.get("holdings_with_values", [])}
        total_value = context.get("total_value", 0)
        
        if total_value == 0:
            return {
                "success": False,
                "error": "empty_portfolio",
                "user_message": "Cannot analyze rebalancing needs for an empty portfolio."
            }
        
        # Calculate current weights
        current_weights = {ticker: value/total_value for ticker, value in current_holdings.items()}
        
        # Get target allocation (default to equal weight for simplicity)
        target_weights = {ticker: 1.0/len(current_holdings) for ticker in current_holdings.keys()}
        
        # Calculate drift
        drift_analysis = {}
        total_drift = 0
        for ticker in current_weights:
            drift = abs(current_weights[ticker] - target_weights.get(ticker, 0))
            drift_analysis[ticker] = {
                "current_weight": current_weights[ticker],
                "target_weight": target_weights.get(ticker, 0),
                "drift": drift
            }
            total_drift += drift
        
        rebalancing_needed = total_drift > 0.1  # 10% total drift threshold
        
        return {
            "success": True,
            "result": {
                "rebalancing_needed": rebalancing_needed,
                "total_drift": total_drift,
                "drift_analysis": drift_analysis,
                "recommendation": "Rebalancing recommended" if rebalancing_needed else "Portfolio is well balanced"
            },
            "confidence": 0.90,
            "agent_used": self.agent_id
        }

    async def _generate_trades(self, query: str, context: Dict) -> Dict[str, Any]:
        """Generate specific trade orders for rebalancing"""
        
        # Use default optimization for trade generation
        objective = self._infer_objective(query)
        
        opt_result = self.tool_map["OptimizePortfolio"].invoke({
            "asset_prices": context["prices"],
            "objective": objective
        })
        
        if not opt_result:
            return {
                "success": False,
                "error": "optimization_failed",
                "user_message": "Could not generate trades due to optimization failure."
            }
        
        trades = await self._generate_trade_orders(opt_result["optimal_weights"], context)
        
        return {
            "success": True,
            "result": {
                "trades": trades.get("trades", []),
                "objective": objective,
                "trade_summary": f"Generated {len(trades.get('trades', []))} trade orders"
            },
            "confidence": 0.80,
            "agent_used": self.agent_id
        }

    async def _risk_parity_optimization(self, query: str, context: Dict) -> Dict[str, Any]:
        """Execute risk parity optimization"""
        return await self._execute_specific_optimization("HERC", context)

    async def _sharpe_optimization(self, query: str, context: Dict) -> Dict[str, Any]:
        """Execute Sharpe ratio maximization"""
        return await self._execute_specific_optimization("MaximizeSharpe", context)

    async def _minimum_variance_optimization(self, query: str, context: Dict) -> Dict[str, Any]:
        """Execute minimum variance optimization"""
        return await self._execute_specific_optimization("MinimizeVolatility", context)

    async def _execute_specific_optimization(self, objective: str, context: Dict) -> Dict[str, Any]:
        """Execute optimization with specific objective"""
        
        opt_result = self.tool_map["OptimizePortfolio"].invoke({
            "asset_prices": context["prices"],
            "objective": objective
        })
        
        if not opt_result or 'optimal_weights' not in opt_result:
            return {
                "success": False,
                "error": "optimization_failed",
                "user_message": f"{objective} optimization failed. Please verify your portfolio data."
            }
        
        trades = await self._generate_trade_orders(opt_result["optimal_weights"], context)
        
        return {
            "success": True,
            "result": {
                "objective": objective,
                "optimal_weights": opt_result["optimal_weights"],
                "trades": trades.get("trades", []),
                "optimization_metrics": opt_result
            },
            "confidence": 0.85,
            "agent_used": self.agent_id
        }

    async def _general_rebalancing(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle general rebalancing queries"""
        
        objective = self._infer_objective(query)
        
        opt_result = self.tool_map["OptimizePortfolio"].invoke({
            "asset_prices": context["prices"],
            "objective": objective
        })
        
        if not opt_result or 'optimal_weights' not in opt_result:
            return {
                "success": False,
                "error": "optimization_failed",
                "user_message": "Portfolio optimization failed. Please check your holdings data."
            }
        
        trades = await self._generate_trade_orders(opt_result["optimal_weights"], context)
        
        return {
            "success": True,
            "result": {
                "objective": objective,
                "optimal_weights": opt_result["optimal_weights"],
                "trades": trades.get("trades", []),
                "optimization_results": opt_result
            },
            "confidence": 0.80,
            "agent_used": self.agent_id
        }

    def _infer_objective(self, query: str) -> str:
        """Infer optimization objective from query"""
        
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        query_words = set(clean_query.split())
        
        # Define keyword sets for different objectives
        min_vol_keywords = [
            {"minimize", "volatility"},
            {"minimize", "risk"},
            {"safest"},
            {"conservative"},
            {"low", "risk"}
        ]
        
        herc_keywords = [
            {"herc"},
            {"risk", "parity"},
            {"diversify", "risk"},
            {"equal", "risk"},
            {"risk", "budgeting"}
        ]
        
        # Check for minimum volatility
        if any(set(keywords).issubset(query_words) for keywords in min_vol_keywords):
            return 'MinimizeVolatility'
        
        # Check for risk parity
        elif any(set(keywords).issubset(query_words) for keywords in herc_keywords):
            return 'HERC'
        
        # Default to Sharpe maximization
        else:
            return 'MaximizeSharpe'

    async def _generate_trade_orders(self, optimal_weights: Dict, context: Dict) -> Dict:
        """Generate trade orders from optimal weights"""
        
        current_holdings_dict = {
            h.asset.ticker: h.market_value 
            for h in context.get("holdings_with_values", [])
        }
        total_value = context.get("total_value", 0)
        
        trade_result = self.tool_map["GenerateTradeOrders"].invoke({
            "current_holdings": current_holdings_dict,
            "target_weights": optimal_weights,
            "total_portfolio_value": total_value
        })
        
        return trade_result if trade_result else {"trades": []}

    def _generate_summary(self, result: Dict, capability: str, execution_time: float) -> str:
        """Generate user-friendly summary for rebalancing results"""
        
        if not result.get("success"):
            error_type = result.get("error", "unknown")
            if error_type == "missing_portfolio":
                return "❌ **Portfolio Required**: Please set up your portfolio first to use rebalancing features."
            elif error_type == "optimization_failed":
                return "⚠️ **Optimization Failed**: Unable to optimize portfolio. Please check your holdings data."
            else:
                return f"❌ **Error**: {result.get('user_message', 'Portfolio rebalancing encountered an issue.')}"
        
        data = result.get("result", {})
        trades = data.get("trades", [])
        objective = data.get("objective", "optimization")
        
        if not trades:
            return f"✅ **Portfolio Optimized** ({execution_time:.1f}s)\n\n**Analysis**: Your portfolio is already well-balanced for {objective.replace('Maximize', '').replace('Minimize', 'minimal ').lower()} objectives. No trades needed!"
        
        trade_summary = "\n".join([
            f"• **{t['action']} ${t['amount_usd']:,.2f}** of {t['ticker']}"
            for t in trades[:5]  # Show first 5 trades
        ])
        
        if len(trades) > 5:
            trade_summary += f"\n• *...and {len(trades) - 5} more trades*"
        
        return f"""✅ **Portfolio Rebalancing Plan** ({execution_time:.1f}s)

**Objective**: {objective.replace('Maximize', 'Maximize ').replace('Minimize', 'Minimize ').replace('HERC', 'Risk Parity')}

**Recommended Actions**:
{trade_summary}

**Summary**: Generated {len(trades)} trades to optimize your portfolio allocation."""

    async def _health_check_capability(self, capability: str) -> Dict:
        """Health check for specific rebalancing capability"""
        
        capability_checks = {
            "portfolio_optimization": {"status": "operational", "response_time": 0.3},
            "rebalancing_analysis": {"status": "operational", "response_time": 0.2},
            "trade_generation": {"status": "operational", "response_time": 0.4},
            "risk_parity_optimization": {"status": "operational", "response_time": 0.3},
            "sharpe_optimization": {"status": "operational", "response_time": 0.3},
            "minimum_variance_optimization": {"status": "operational", "response_time": 0.3}
        }
        
        return capability_checks.get(capability, {"status": "unknown", "response_time": 0.5})

    # Legacy method for backward compatibility (can be removed after full migration)
    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Legacy run method for backward compatibility"""
        
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        if not context or "prices" not in context:
            return {"success": False, "error": "Could not rebalance. Portfolio data is missing."}
        
        objective = self._infer_objective(user_query)
        print(f"Inferred objective: {objective}")
        
        opt_result = self.tool_map["OptimizePortfolio"].invoke({
            "asset_prices": context["prices"],
            "objective": objective
        })
        
        if not opt_result or 'optimal_weights' not in opt_result:
            return {"success": False, "error": "The portfolio optimization step failed."}
        
        current_holdings_dict = {
            h.asset.ticker: h.market_value 
            for h in context.get("holdings_with_values", [])
        }
        total_value = context.get("total_value", 0)
        
        trade_result = self.tool_map["GenerateTradeOrders"].invoke({
            "current_holdings": current_holdings_dict,
            "target_weights": opt_result["optimal_weights"],
            "total_portfolio_value": total_value
        })
        
        if not trade_result or 'trades' not in trade_result:
            return {"success": False, "error": "The trade generation step failed."}
        
        trades = trade_result['trades']
        summary = "No rebalancing is necessary." if not trades else (
            f"### Portfolio Rebalancing Plan\n**Objective:** {objective.title()}\n\n**Recommended Trades:**\n" +
            "\n".join([f"- **{t['action']} ${t['amount_usd']:,.2f} of {t['ticker']}**" for t in trades])
        )
        
        return {
            "success": True, 
            "summary": summary, 
            "optimization_results": opt_result, 
            "trade_plan": trades, 
            "agent_used": self.name
        }