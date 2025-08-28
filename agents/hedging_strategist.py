# in agents/hedging_strategist.py
import yfinance as yf
from typing import Dict, Any, Optional, List
import re

from agents.mcp_base_agent import MCPBaseAgent
from tools.risk_tools import calculate_volatility_budget
from tools.strategy_tools import find_optimal_hedge

class HedgingStrategistAgent(MCPBaseAgent):
    """
    MCP-compatible hedging strategist that provides tactical risk management advice
    and portfolio protection strategies using hedging instruments.
    """
    
    def __init__(self):
        # Define MCP capabilities
        capabilities = [
            "hedge_analysis",
            "volatility_targeting",
            "downside_protection",
            "risk_budgeting",
            "hedge_instrument_selection",
            "portfolio_insurance",
            "debate_participation",
            "consensus_building", 
            "collaborative_analysis"
        ]
        
        super().__init__(
            agent_id="HedgingStrategistAgent",
            agent_name="Risk Hedging Specialist",
            capabilities=capabilities,
            max_concurrent_jobs=2
        )
        
        # Initialize hedging tools
        self.tools = [find_optimal_hedge, calculate_volatility_budget]
        self.tool_map = {
            "FindOptimalHedge": find_optimal_hedge,
            "CalculateVolatilityBudget": calculate_volatility_budget,
        }
    
    @property
    def name(self) -> str: 
        return "Risk Hedging Specialist"

    @property
    def purpose(self) -> str: 
        return "Provides tactical risk management advice and portfolio hedging strategies."

    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Execute specific hedging capability"""
        
        try:
            # Check for required portfolio data
            if not context or "portfolio_returns" not in context:
                return {
                    "success": False,
                    "error": "missing_portfolio",
                    "user_message": "Hedging strategies require portfolio return data. Please ensure you have an active portfolio with return history."
                }
            
            query = data.get("query", "")
            
            if capability == "hedge_analysis":
                return await self._analyze_hedging_options(query, context)
            elif capability == "volatility_targeting":
                return await self._volatility_targeting_analysis(query, context)
            elif capability == "downside_protection":
                return await self._downside_protection_strategy(query, context)
            elif capability == "risk_budgeting":
                return await self._risk_budgeting_analysis(query, context)
            elif capability == "hedge_instrument_selection":
                return await self._select_hedge_instruments(query, context)
            elif capability == "portfolio_insurance":
                return await self._portfolio_insurance_strategy(query, context)
            else:
                return await self._general_hedging_advice(query, context)
                
        except Exception as e:
            return {
                "success": False,
                "error": "execution_error",
                "user_message": f"Hedging analysis encountered an issue: {str(e)}"
            }

    async def _analyze_hedging_options(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze optimal hedging strategies for the portfolio"""
        
        portfolio_returns = context["portfolio_returns"]
        
        # Fetch hedge instrument data (default to inverse S&P 500 ETF)
        hedge_instrument = self._extract_hedge_instrument(query) or 'SH'
        
        try:
            print(f"Fetching market data for hedge instrument '{hedge_instrument}'...")
            hedge_data = yf.download(hedge_instrument, period="1y", auto_adjust=True)
            hedge_returns = hedge_data['Close'].pct_change()
            
            # Calculate optimal hedge
            result = self.tool_map["FindOptimalHedge"].run({
                "portfolio_returns": portfolio_returns,
                "hedge_instrument_returns": hedge_returns
            })
            
            if not result:
                return {
                    "success": False,
                    "error": "hedge_calculation_failed",
                    "user_message": "Unable to calculate optimal hedge ratio. Please check your portfolio data."
                }
            
            return {
                "success": True,
                "result": {
                    "hedge_instrument": hedge_instrument,
                    "optimal_hedge_ratio": result['optimal_hedge_ratio'],
                    "volatility_reduction": result['volatility_reduction_pct'],
                    "hedge_effectiveness": result.get('correlation', 0),
                    "recommendation": f"Use {result['optimal_hedge_ratio']:.1%} allocation to {hedge_instrument} for optimal protection"
                },
                "confidence": 0.82,
                "agent_used": self.agent_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": "data_fetch_error",
                "user_message": f"Could not fetch data for hedge instrument {hedge_instrument}. Please verify the symbol."
            }

    async def _volatility_targeting_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze volatility targeting strategies"""
        
        portfolio_returns = context["portfolio_returns"]
        target_volatility = self._extract_target_volatility(query)
        
        if not target_volatility:
            return {
                "success": False,
                "error": "missing_target",
                "user_message": "Please specify a target volatility level (e.g., '15%' or '0.15')."
            }
        
        result = self.tool_map["CalculateVolatilityBudget"].run({
            "portfolio_returns": portfolio_returns,
            "target_volatility": target_volatility
        })
        
        if not result:
            return {
                "success": False,
                "error": "calculation_failed",
                "user_message": "Unable to calculate volatility budget. Please check your portfolio data."
            }
        
        return {
            "success": True,
            "result": {
                "target_volatility": target_volatility,
                "risky_asset_weight": result['risky_asset_weight'],
                "risk_free_weight": result['risk_free_asset_weight'],
                "current_volatility": result.get('current_volatility', 0),
                "adjustment_needed": abs(result['risky_asset_weight'] - 1.0) > 0.05,
                "recommendation": f"Allocate {result['risky_asset_weight']:.1%} to risky assets and {result['risk_free_asset_weight']:.1%} to risk-free assets"
            },
            "confidence": 0.85,
            "agent_used": self.agent_id
        }

    async def _downside_protection_strategy(self, query: str, context: Dict) -> Dict[str, Any]:
        """Develop comprehensive downside protection strategy"""
        
        portfolio_returns = context["portfolio_returns"]
        
        # Calculate portfolio risk metrics
        volatility = portfolio_returns.std() * (252 ** 0.5)  # Annualized
        max_drawdown = (portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()).max()
        
        protection_strategies = [
            {
                "strategy": "Put Options",
                "description": "Purchase put options on portfolio or index",
                "effectiveness": "High downside protection",
                "cost": "Premium payment required"
            },
            {
                "strategy": "Inverse ETFs",
                "description": "Allocate portion to inverse market ETFs",
                "effectiveness": "Moderate protection with daily rebalancing",
                "cost": "Drag on performance in up markets"
            },
            {
                "strategy": "Volatility Targeting",
                "description": "Reduce allocation during high volatility periods",
                "effectiveness": "Moderate protection through position sizing",
                "cost": "Lower upside participation"
            }
        ]
        
        return {
            "success": True,
            "result": {
                "current_volatility": volatility,
                "historical_max_drawdown": max_drawdown,
                "protection_strategies": protection_strategies,
                "recommended_allocation": "5-10% to hedging instruments",
                "priority_ranking": ["Put Options", "Volatility Targeting", "Inverse ETFs"]
            },
            "confidence": 0.78,
            "agent_used": self.agent_id
        }

    async def _risk_budgeting_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze risk budgeting across portfolio components"""
        
        portfolio_returns = context["portfolio_returns"]
        
        # Calculate risk contribution metrics
        volatility = portfolio_returns.std() * (252 ** 0.5)
        var_95 = portfolio_returns.quantile(0.05) * (252 ** 0.5)
        
        risk_budget = {
            "total_portfolio_volatility": volatility,
            "value_at_risk_95": var_95,
            "recommended_risk_allocation": {
                "core_holdings": 0.70,  # 70% of risk budget
                "satellite_positions": 0.20,  # 20% of risk budget
                "hedging_instruments": 0.10   # 10% of risk budget
            },
            "risk_limits": {
                "maximum_position_risk": 0.05,  # 5% of total risk
                "correlation_limit": 0.70,
                "volatility_ceiling": volatility * 1.25
            }
        }
        
        return {
            "success": True,
            "result": risk_budget,
            "confidence": 0.80,
            "agent_used": self.agent_id
        }

    async def _select_hedge_instruments(self, query: str, context: Dict) -> Dict[str, Any]:
        """Select appropriate hedging instruments"""
        
        hedge_instruments = [
            {
                "instrument": "SH (ProShares Short S&P 500)",
                "type": "Inverse ETF",
                "correlation": -0.99,
                "liquidity": "High",
                "cost": "Low",
                "use_case": "General market hedging"
            },
            {
                "instrument": "VXX (iPath S&P 500 VIX)",
                "type": "Volatility ETF",
                "correlation": -0.75,
                "liquidity": "High",
                "cost": "Medium",
                "use_case": "Volatility spike protection"
            },
            {
                "instrument": "SPY Put Options",
                "type": "Options",
                "correlation": "Variable",
                "liquidity": "Very High",
                "cost": "Premium dependent",
                "use_case": "Tail risk protection"
            },
            {
                "instrument": "TLT (20+ Year Treasury)",
                "type": "Flight-to-quality",
                "correlation": -0.30,
                "liquidity": "High",
                "cost": "Low",
                "use_case": "Diversification hedge"
            }
        ]
        
        return {
            "success": True,
            "result": {
                "available_instruments": hedge_instruments,
                "recommendation": "Combination of SH and TLT for balanced protection",
                "allocation_suggestion": "3-5% SH, 2-3% TLT for moderate hedging"
            },
            "confidence": 0.75,
            "agent_used": self.agent_id
        }

    async def _portfolio_insurance_strategy(self, query: str, context: Dict) -> Dict[str, Any]:
        """Develop portfolio insurance strategy"""
        
        portfolio_returns = context["portfolio_returns"]
        
        insurance_strategies = {
            "dynamic_hedging": {
                "description": "Adjust hedge ratio based on market conditions",
                "trigger": "Portfolio decline > 5%",
                "action": "Increase hedge allocation to 15-20%",
                "effectiveness": "High, but requires active management"
            },
            "floor_protection": {
                "description": "Set minimum portfolio value floor",
                "trigger": "Portfolio value approaches floor",
                "action": "Increase defensive allocation",
                "effectiveness": "Moderate, limits upside"
            },
            "volatility_breakout": {
                "description": "Increase hedging when volatility spikes",
                "trigger": "VIX > 25 or portfolio vol > 1.5x normal",
                "action": "Add volatility hedging positions",
                "effectiveness": "Good for volatility events"
            }
        }
        
        return {
            "success": True,
            "result": {
                "insurance_strategies": insurance_strategies,
                "recommended_approach": "Dynamic hedging with volatility triggers",
                "monitoring_metrics": ["Portfolio drawdown", "VIX level", "Correlation breakdown"],
                "rebalancing_frequency": "Weekly during normal conditions, daily during stress"
            },
            "confidence": 0.77,
            "agent_used": self.agent_id
        }

    async def _general_hedging_advice(self, query: str, context: Dict) -> Dict[str, Any]:
        """Provide general hedging advice"""
        
        # Determine if this is a hedge-specific or volatility-specific query
        if any(word in query.lower() for word in ["hedge", "protect", "protection", "insurance"]):
            return await self._analyze_hedging_options(query, context)
        elif any(word in query.lower() for word in ["volatility", "target", "vol"]):
            return await self._volatility_targeting_analysis(query, context)
        else:
            return await self._downside_protection_strategy(query, context)

    def _extract_hedge_instrument(self, query: str) -> Optional[str]:
        """Extract hedge instrument symbol from query"""
        # Look for common hedge instrument patterns
        hedge_patterns = [
            r'\b(SH|SPXU|PSQ|DOG|DXD|TWM|EFZ|REW)\b',  # Common inverse ETFs
            r'\b(VXX|UVXY|VIXY|VXZ)\b',  # Volatility instruments
            r'\b([A-Z]{1,5})\s+(?:etf|put|call|option)',  # Ticker with instrument type
        ]
        
        query_upper = query.upper()
        for pattern in hedge_patterns:
            match = re.search(pattern, query_upper)
            if match:
                return match.group(1)
        
        return None

    def _extract_target_volatility(self, query: str) -> Optional[float]:
        """Extract target volatility from query"""
        # Look for percentage or decimal patterns
        vol_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # e.g., "15%" or "12.5%"
            r'(\d*\.\d+)',           # e.g., "0.15"
            r'(\d+)\s*(?:percent|pct)',  # e.g., "15 percent"
        ]
        
        for pattern in vol_patterns:
            match = re.search(pattern, query.lower())
            if match:
                value = float(match.group(1))
                # Convert percentage to decimal if > 1
                return value / 100.0 if value > 1 else value
        
        return None

    def _generate_summary(self, result: Dict, capability: str, execution_time: float) -> str:
        """Generate user-friendly summary for hedging results"""
        
        if not result.get("success"):
            error_type = result.get("error", "unknown")
            if error_type == "missing_portfolio":
                return "❌ **Portfolio Required**: Hedging analysis requires portfolio return data. Please ensure you have an active portfolio."
            elif error_type == "data_fetch_error":
                return "⚠️ **Data Issue**: Unable to fetch hedge instrument data. Please verify the symbol and try again."
            else:
                return f"❌ **Error**: {result.get('user_message', 'Hedging analysis encountered an issue.')}"
        
        data = result.get("result", {})
        
        if capability == "hedge_analysis":
            hedge_ratio = data.get("optimal_hedge_ratio", 0)
            vol_reduction = data.get("volatility_reduction", 0)
            instrument = data.get("hedge_instrument", "hedge instrument")
            
            return f"""✅ **Optimal Hedge Analysis** ({execution_time:.1f}s)

**Recommendation**: Use **{hedge_ratio:.1%}** allocation to {instrument}

**Benefits**:
• **{vol_reduction:.1f}%** volatility reduction expected
• Maintains portfolio upside participation
• Cost-effective downside protection

**Implementation**: Gradually build hedge position over 1-2 weeks to minimize market impact."""

        elif capability == "volatility_targeting":
            target_vol = data.get("target_volatility", 0)
            risky_weight = data.get("risky_asset_weight", 0)
            risk_free_weight = data.get("risk_free_weight", 0)
            
            return f"""✅ **Volatility Targeting Strategy** ({execution_time:.1f}s)

**Target**: {target_vol:.1%} annual volatility

**Recommended Allocation**:
• **{risky_weight:.1%}** risky assets (stocks/growth)
• **{risk_free_weight:.1%}** risk-free assets (bonds/cash)

**Rebalancing**: Monitor monthly and adjust allocation to maintain target volatility."""

        else:
            strategies = data.get("protection_strategies", [])
            if strategies:
                strategy_list = "\n".join([f"• **{s.get('strategy', 'Strategy')}**: {s.get('description', '')}" for s in strategies[:3]])
                return f"""✅ **Risk Management Strategy** ({execution_time:.1f}s)

**Protection Options**:
{strategy_list}

**Recommendation**: Start with 5-10% allocation to hedging instruments based on risk tolerance."""
            else:
                return f"✅ **Hedging Analysis Complete** ({execution_time:.1f}s)\n\nCustom hedging strategy developed based on your portfolio characteristics."

    async def _health_check_capability(self, capability: str) -> Dict:
        """Health check for specific hedging capability"""
        
        capability_checks = {
            "hedge_analysis": {"status": "operational", "response_time": 0.4},
            "volatility_targeting": {"status": "operational", "response_time": 0.3},
            "downside_protection": {"status": "operational", "response_time": 0.2},
            "risk_budgeting": {"status": "operational", "response_time": 0.3},
            "hedge_instrument_selection": {"status": "operational", "response_time": 0.2},
            "portfolio_insurance": {"status": "operational", "response_time": 0.3}
        }
        
        return capability_checks.get(capability, {"status": "unknown", "response_time": 0.5})

    # Legacy method for backward compatibility (can be removed after full migration)
    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Legacy run method for backward compatibility"""
        
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        if not context or "portfolio_returns" not in context:
            return {"success": False, "error": "Could not provide hedging advice. Portfolio data is missing."}

        query = user_query.lower()
        tool_to_use, tool_args = None, {"portfolio_returns": context["portfolio_returns"]}

        # Parse query for intent and parameters
        if "hedge" in query or "protect" in query:
            tool_to_use = self.tool_map["FindOptimalHedge"]
            # Fetch data for a common hedge instrument (inverse S&P 500 ETF)
            print("Fetching market data for hedge instrument 'SH'...")
            hedge_returns = yf.download('SH', period="1y", auto_adjust=True)['Close'].pct_change()
            tool_args['hedge_instrument_returns'] = hedge_returns
        elif "volatility" in query or "target" in query:
            tool_to_use = self.tool_map["CalculateVolatilityBudget"]
            try:
                target_vol_str = [word for word in query.replace('%',' ').split() if word.replace('.','').isdigit()][0]
                tool_args['target_volatility'] = float(target_vol_str) / 100.0
            except IndexError:
                return {"success": False, "error": "Please specify a target volatility (e.g., 'target 15%')."}

        if not tool_to_use: 
            return {"success": False, "error": "I can only help with hedging or volatility targeting."}

        # Execute tool
        result = tool_to_use.run(tool_args)

        # Format output
        if result:
            if tool_to_use.name == "FindOptimalHedge":
                result['summary'] = (f"To hedge your portfolio with 'SH', the optimal ratio is {result['optimal_hedge_ratio']:.4f}. "
                                   f"This is expected to reduce daily volatility by {result['volatility_reduction_pct']:.2f}%.")
            elif tool_to_use.name == "CalculateVolatilityBudget":
                result['summary'] = (f"To achieve your target of {result['target_annual_volatility']:.1%}, "
                                   f"allocate {result['risky_asset_weight']:.1%} to your portfolio and "
                                   f"{result['risk_free_asset_weight']:.1%} to a risk-free asset.")
        else:
            result = {"success": False, "error": f"The '{tool_to_use.name}' tool failed."}
        
        return result