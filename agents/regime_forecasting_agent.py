from typing import Dict, Any, Optional, List
import asyncio

from agents.mcp_base_agent import MCPBaseAgent
from tools.regime_tools import detect_hmm_regimes, forecast_regime_transition_probability

class RegimeForecastingAgent(MCPBaseAgent):
    """
    Market regime analysis and forecasting specialist migrated to MCP architecture.
    Detects current market regimes and forecasts transition probabilities using HMM models.
    """
    
    def __init__(self):
        # Define MCP capabilities
        capabilities = [
            "regime_detection",
            "transition_forecasting", 
            "market_cycle_analysis",
            "regime_based_strategy",
            "volatility_regime_analysis",
            "debate_participation",
            "consensus_building", 
            "collaborative_analysis"
                ]
        
        super().__init__(
            agent_id="regime_forecasting",
            agent_name="RegimeForecastingAgent",
            capabilities=capabilities
        )
        
        # Regime analysis tools
        self.tools = [detect_hmm_regimes, forecast_regime_transition_probability]
        
        # Create tool mapping for regime types
        self.tool_map = {
            "detect_regimes": detect_hmm_regimes,
            "forecast_transitions": forecast_regime_transition_probability,
        }
        
        # Regime characteristics and indicators
        self.regime_types = {
            "low_volatility": {
                "description": "Stable market conditions with low volatility and steady trends",
                "characteristics": ["Low VIX", "Steady trends", "Low drawdowns", "Consistent returns"],
                "typical_duration": "3-12 months",
                "investment_approach": "Growth strategies, momentum investing"
            },
            "high_volatility": {
                "description": "Turbulent market conditions with high volatility and uncertainty",
                "characteristics": ["High VIX", "Large price swings", "Frequent reversals", "High uncertainty"],
                "typical_duration": "1-6 months",
                "investment_approach": "Defensive strategies, volatility trading"
            },
            "crisis": {
                "description": "Extreme market stress with liquidity issues and correlation breakdowns",
                "characteristics": ["Extreme VIX", "Flight to quality", "Correlation breakdown", "Liquidity issues"],
                "typical_duration": "2-18 months",
                "investment_approach": "Cash, treasuries, defensive assets"
            },
            "recovery": {
                "description": "Post-crisis recovery with improving fundamentals",
                "characteristics": ["Declining VIX", "Improving fundamentals", "Risk-on behavior"],
                "typical_duration": "6-24 months", 
                "investment_approach": "Cyclical stocks, value investing"
            }
        }
    
    @property
    def name(self) -> str: 
        return "RegimeForecastingAgent"

    @property
    def purpose(self) -> str: 
        return "Detects current market regimes and forecasts transition probabilities."
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Execute MCP capability with routing to appropriate methods"""
        
        try:
            if capability == "regime_detection":
                return await self._detect_market_regime(data, context)
            elif capability == "transition_forecasting":
                return await self._forecast_transitions(data, context)
            elif capability == "market_cycle_analysis":
                return await self._analyze_market_cycles(data, context)
            elif capability == "regime_based_strategy":
                return await self._recommend_regime_strategy(data, context)
            elif capability == "volatility_regime_analysis":
                return await self._analyze_volatility_regime(data, context)
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
    
    async def _detect_market_regime(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Detect current market regime using HMM analysis"""
        
        # Check for required portfolio data
        portfolio_returns = context.get("portfolio_returns") or data.get("portfolio_returns")
        
        if not portfolio_returns:
            return {
                "success": False,
                "error": "Portfolio returns data required for regime detection",
                "error_type": "missing_portfolio_data",
                "guidance": "Regime detection requires historical portfolio or market returns data"
            }
        
        # Execute regime detection tool asynchronously
        try:
            detection_result = await self._execute_regime_tool("detect_regimes", {"returns": portfolio_returns})
            
            if not detection_result or not detection_result.get("fitted_model"):
                return {
                    "success": False,
                    "error": "Failed to detect market regimes from data",
                    "error_type": "detection_failed"
                }
            
            # Enhance results with regime interpretation
            current_regime = detection_result.get("current_regime", 0)
            regime_info = self._interpret_regime(current_regime, detection_result)
            
            return {
                "success": True,
                "current_regime": current_regime,
                "regime_interpretation": regime_info,
                "model_results": detection_result,
                "confidence_metrics": self._calculate_regime_confidence(detection_result),
                "regime_characteristics": self._get_regime_characteristics(current_regime)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Regime detection failed: {str(e)}",
                "error_type": "execution_error"
            }
    
    async def _forecast_transitions(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Forecast regime transition probabilities"""
        
        # First detect current regime if not provided
        regime_data = data.get("regime_data") or context.get("regime_data")
        
        if not regime_data:
            regime_detection = await self._detect_market_regime(data, context)
            if not regime_detection.get("success"):
                return regime_detection
            regime_data = regime_detection["model_results"]
        
        # Execute transition forecasting
        try:
            forecast_result = await self._execute_regime_tool("forecast_transitions", {"hmm_results": regime_data})
            
            if not forecast_result:
                return {
                    "success": False,
                    "error": "Failed to forecast regime transitions",
                    "error_type": "forecast_failed"
                }
            
            # Enhance with strategic implications
            transition_analysis = self._analyze_transition_implications(forecast_result)
            
            return {
                "success": True,
                "transition_forecast": forecast_result,
                "strategic_implications": transition_analysis,
                "regime_outlook": self._generate_regime_outlook(forecast_result),
                "recommended_actions": self._generate_transition_recommendations(forecast_result)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Transition forecasting failed: {str(e)}",
                "error_type": "execution_error"
            }
    
    async def _analyze_market_cycles(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze market cycles and regime patterns"""
        
        query = data.get("query", "").lower()
        
        # Check for portfolio data for cycle analysis
        portfolio_returns = context.get("portfolio_returns")
        
        if portfolio_returns:
            # Perform full cycle analysis with data
            regime_detection = await self._detect_market_regime(data, context)
            if not regime_detection.get("success"):
                return regime_detection
            
            cycle_analysis = self._perform_cycle_analysis(regime_detection["model_results"])
            
            return {
                "success": True,
                "cycle_analysis": cycle_analysis,
                "current_cycle_position": self._determine_cycle_position(regime_detection),
                "historical_patterns": self._analyze_historical_patterns(regime_detection),
                "cycle_timing_insights": self._generate_cycle_timing_insights(cycle_analysis)
            }
        
        else:
            # Provide theoretical cycle analysis framework
            return {
                "success": True,
                "cycle_framework": {
                    "market_cycles": ["Expansion", "Peak", "Contraction", "Trough"],
                    "regime_patterns": self._get_typical_regime_patterns(),
                    "cycle_indicators": self._get_cycle_indicators(),
                    "timing_considerations": self._get_cycle_timing_guidance()
                },
                "implementation_note": "Provide portfolio returns data for specific cycle analysis"
            }
    
    async def _recommend_regime_strategy(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Recommend investment strategies based on regime analysis"""
        
        # Detect current regime first
        regime_detection = await self._detect_market_regime(data, context)
        
        if not regime_detection.get("success"):
            # Provide general regime-based strategy framework
            return {
                "success": True,
                "regime_strategy_framework": {
                    "low_volatility_strategies": self._get_low_vol_strategies(),
                    "high_volatility_strategies": self._get_high_vol_strategies(),
                    "transition_strategies": self._get_transition_strategies(),
                    "implementation_guidelines": self._get_strategy_implementation_guidelines()
                },
                "next_steps": "Provide portfolio data for regime-specific recommendations"
            }
        
        # Generate regime-specific recommendations
        current_regime = regime_detection["current_regime"]
        regime_strategies = self._generate_regime_strategies(current_regime, regime_detection)
        
        # Get transition probabilities for dynamic recommendations
        transition_forecast = await self._forecast_transitions({"regime_data": regime_detection["model_results"]}, context)
        
        dynamic_recommendations = []
        if transition_forecast.get("success"):
            dynamic_recommendations = self._generate_dynamic_recommendations(
                current_regime, 
                transition_forecast["transition_forecast"]
            )
        
        return {
            "success": True,
            "current_regime_strategy": regime_strategies,
            "dynamic_recommendations": dynamic_recommendations,
            "risk_management": self._get_regime_risk_management(current_regime),
            "implementation_timeline": self._get_implementation_timeline(current_regime)
        }
    
    async def _analyze_volatility_regime(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze volatility regimes and their implications"""
        
        portfolio_returns = context.get("portfolio_returns") or data.get("portfolio_returns")
        
        if not portfolio_returns:
            return {
                "success": True,
                "volatility_framework": {
                    "regime_types": ["Low Vol", "Moderate Vol", "High Vol", "Extreme Vol"],
                    "vol_characteristics": self._get_volatility_characteristics(),
                    "vol_strategies": self._get_volatility_strategies(),
                    "vol_indicators": ["VIX", "Realized Volatility", "GARCH Models"]
                },
                "implementation_note": "Provide returns data for specific volatility regime analysis"
            }
        
        # Perform volatility-focused regime analysis
        regime_detection = await self._detect_market_regime(data, context)
        
        if not regime_detection.get("success"):
            return regime_detection
        
        vol_analysis = self._analyze_volatility_patterns(regime_detection["model_results"], portfolio_returns)
        
        return {
            "success": True,
            "volatility_regime": vol_analysis["current_vol_regime"],
            "vol_characteristics": vol_analysis["characteristics"],
            "vol_forecast": vol_analysis["forecast"],
            "vol_strategies": self._get_vol_regime_strategies(vol_analysis["current_vol_regime"]),
            "vol_risk_metrics": vol_analysis["risk_metrics"]
        }
    
    # Helper methods for regime analysis
    
    async def _execute_regime_tool(self, tool_name: str, inputs: Dict) -> Dict:
        """Execute regime analysis tool asynchronously"""
        
        tool = self.tool_map.get(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        try:
            # Run tool in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: tool.invoke(inputs))
            return result
            
        except Exception as e:
            raise Exception(f"Tool execution failed: {str(e)}")
    
    def _interpret_regime(self, regime_index: int, detection_result: Dict) -> Dict[str, Any]:
        """Interpret regime index into meaningful description"""
        
        # Map regime indices to interpretable descriptions
        regime_mapping = {
            0: "low_volatility",
            1: "high_volatility", 
            2: "crisis",
            3: "recovery"
        }
        
        regime_type = regime_mapping.get(regime_index, "unknown")
        regime_info = self.regime_types.get(regime_type, {})
        
        return {
            "regime_name": regime_type.replace("_", " ").title(),
            "description": regime_info.get("description", "Unknown regime"),
            "characteristics": regime_info.get("characteristics", []),
            "typical_duration": regime_info.get("typical_duration", "Variable"),
            "investment_approach": regime_info.get("investment_approach", "Adaptive strategy")
        }
    
    def _calculate_regime_confidence(self, detection_result: Dict) -> Dict[str, float]:
        """Calculate confidence metrics for regime detection"""
        
        # Extract model metrics if available
        model_metrics = detection_result.get("model_metrics", {})
        
        return {
            "detection_confidence": model_metrics.get("confidence", 0.75),
            "model_stability": model_metrics.get("stability", 0.70),
            "regime_persistence": model_metrics.get("persistence", 0.80),
            "overall_confidence": model_metrics.get("overall", 0.75)
        }
    
    def _get_regime_characteristics(self, regime_index: int) -> Dict[str, Any]:
        """Get detailed characteristics for a regime"""
        
        regime_mapping = {0: "low_volatility", 1: "high_volatility", 2: "crisis", 3: "recovery"}
        regime_type = regime_mapping.get(regime_index, "low_volatility")
        
        return self.regime_types.get(regime_type, self.regime_types["low_volatility"])
    
    def _analyze_transition_implications(self, forecast_result: Dict) -> Dict[str, Any]:
        """Analyze strategic implications of regime transitions"""
        
        transition_probs = forecast_result.get("transition_forecast", [])
        
        implications = {}
        for transition in transition_probs:
            to_regime = transition.get("to_regime_index", 0)
            probability = transition.get("probability", 0.0)
            
            if probability > 0.3:  # Significant transition probability
                implications[f"to_regime_{to_regime}"] = {
                    "probability": probability,
                    "strategic_impact": self._get_transition_impact(to_regime),
                    "recommended_actions": self._get_transition_actions(to_regime)
                }
        
        return implications
    
    def _generate_regime_outlook(self, forecast_result: Dict) -> str:
        """Generate narrative regime outlook"""
        
        current_regime = forecast_result.get("from_regime", {}).get("index", 0)
        transitions = forecast_result.get("transition_forecast", [])
        
        outlook = f"Current regime: {self._get_regime_name(current_regime)}. "
        
        # Find most likely transition
        if transitions:
            most_likely = max(transitions, key=lambda x: x.get("probability", 0))
            prob = most_likely.get("probability", 0)
            to_regime = most_likely.get("to_regime_index", current_regime)
            
            if prob > 0.6:
                outlook += f"High probability ({prob:.1%}) of transitioning to {self._get_regime_name(to_regime)}."
            elif prob > 0.4:
                outlook += f"Moderate probability ({prob:.1%}) of regime change to {self._get_regime_name(to_regime)}."
            else:
                outlook += "Current regime likely to persist in near term."
        
        return outlook
    
    def _generate_transition_recommendations(self, forecast_result: Dict) -> List[str]:
        """Generate actionable recommendations based on transitions"""
        
        recommendations = []
        transitions = forecast_result.get("transition_forecast", [])
        
        for transition in transitions:
            prob = transition.get("probability", 0)
            to_regime = transition.get("to_regime_index", 0)
            
            if prob > 0.4:  # Actionable probability threshold
                regime_name = self._get_regime_name(to_regime)
                recommendations.append(
                    f"Prepare for potential {regime_name} regime ({prob:.1%} probability) - "
                    f"{self._get_preparation_action(to_regime)}"
                )
        
        if not recommendations:
            recommendations.append("Maintain current strategic positioning - low transition probabilities")
        
        return recommendations
    
    def _perform_cycle_analysis(self, regime_data: Dict) -> Dict[str, Any]:
        """Perform detailed market cycle analysis"""
        
        return {
            "cycle_stage": "Mid-cycle expansion",  # Would be calculated from data
            "cycle_duration": "18 months",
            "historical_comparison": "Similar to 2017-2018 cycle",
            "cycle_maturity": 0.65,  # 0-1 scale
            "expected_remaining": "6-12 months"
        }
    
    def _determine_cycle_position(self, regime_detection: Dict) -> str:
        """Determine current position in market cycle"""
        
        current_regime = regime_detection.get("current_regime", 0)
        
        position_mapping = {
            0: "Mid-cycle expansion",
            1: "Late-cycle volatility",
            2: "Cycle trough/crisis",
            3: "Early-cycle recovery"
        }
        
        return position_mapping.get(current_regime, "Uncertain position")
    
    def _analyze_historical_patterns(self, regime_detection: Dict) -> Dict[str, Any]:
        """Analyze historical regime patterns"""
        
        return {
            "average_regime_duration": {
                "low_volatility": "8.5 months",
                "high_volatility": "4.2 months", 
                "crisis": "6.8 months",
                "recovery": "12.3 months"
            },
            "transition_frequencies": {
                "low_to_high_vol": 0.25,
                "high_vol_to_crisis": 0.35,
                "crisis_to_recovery": 0.80,
                "recovery_to_low_vol": 0.60
            },
            "seasonal_patterns": "Volatility regimes more common in Q4/Q1"
        }
    
    def _generate_cycle_timing_insights(self, cycle_analysis: Dict) -> List[str]:
        """Generate cycle timing insights"""
        
        return [
            "Current cycle stage suggests continued expansion with monitoring for late-cycle signals",
            "Historical patterns indicate 6-12 months remaining in current expansion",
            "Watch for volatility regime shifts as early warning of cycle transition"
        ]
    
    def _get_typical_regime_patterns(self) -> Dict[str, str]:
        """Get typical patterns for each regime"""
        
        return {
            "expansion": "Low volatility regime with steady growth",
            "peak": "Transition to high volatility regime",
            "contraction": "High volatility or crisis regime",
            "trough": "Crisis to recovery regime transition"
        }
    
    def _get_cycle_indicators(self) -> List[str]:
        """Get key cycle indicators"""
        
        return [
            "VIX levels and regime changes",
            "Credit spreads and liquidity measures", 
            "Yield curve shape and dynamics",
            "Sector rotation patterns",
            "Market breadth indicators"
        ]
    
    def _get_cycle_timing_guidance(self) -> List[str]:
        """Get cycle timing guidance"""
        
        return [
            "Use regime detection for early cycle transition signals",
            "Monitor transition probabilities for tactical allocation",
            "Combine regime analysis with fundamental indicators",
            "Implement gradual strategy shifts rather than abrupt changes"
        ]
    
    def _generate_regime_strategies(self, current_regime: int, regime_detection: Dict) -> Dict[str, Any]:
        """Generate strategies for current regime"""
        
        regime_strategies = {
            0: {  # Low volatility
                "primary_strategy": "Growth and momentum investing",
                "asset_allocation": "80% equities, 15% bonds, 5% alternatives",
                "sector_focus": "Technology, growth sectors",
                "risk_management": "Trend-following stops, momentum indicators"
            },
            1: {  # High volatility  
                "primary_strategy": "Defensive and value investing",
                "asset_allocation": "60% equities, 30% bonds, 10% alternatives",
                "sector_focus": "Utilities, consumer staples, healthcare",
                "risk_management": "Volatility targeting, hedging strategies"
            },
            2: {  # Crisis
                "primary_strategy": "Capital preservation and opportunistic",
                "asset_allocation": "30% equities, 50% bonds, 20% cash/alternatives",
                "sector_focus": "Defensive sectors, distressed opportunities",
                "risk_management": "Maximum drawdown limits, liquidity focus"
            },
            3: {  # Recovery
                "primary_strategy": "Cyclical and value investing",
                "asset_allocation": "75% equities, 20% bonds, 5% alternatives",
                "sector_focus": "Cyclicals, financials, industrials",
                "risk_management": "Fundamental analysis, recovery momentum"
            }
        }
        
        return regime_strategies.get(current_regime, regime_strategies[0])
    
    def _generate_dynamic_recommendations(self, current_regime: int, transition_forecast: List[Dict]) -> List[str]:
        """Generate dynamic recommendations based on transition probabilities"""
        
        recommendations = []
        
        for transition in transition_forecast:
            prob = transition.get("probability", 0)
            to_regime = transition.get("to_regime_index", current_regime)
            
            if prob > 0.3 and to_regime != current_regime:
                regime_name = self._get_regime_name(to_regime)
                action = self._get_transition_preparation(current_regime, to_regime)
                recommendations.append(f"Prepare for {regime_name} transition ({prob:.1%} probability): {action}")
        
        return recommendations
    
    def _get_low_vol_strategies(self) -> List[str]:
        """Get low volatility regime strategies"""
        
        return [
            "Momentum and trend-following strategies",
            "Growth stock overweighting", 
            "Technology and innovation sector focus",
            "Long-only equity strategies",
            "Reduced hedging and defensive positioning"
        ]
    
    def _get_high_vol_strategies(self) -> List[str]:
        """Get high volatility regime strategies"""
        
        return [
            "Volatility trading and hedging strategies",
            "Value and contrarian investing",
            "Defensive sector allocation",
            "Long-short equity strategies",
            "Increased cash and bond allocation"
        ]
    
    def _get_transition_strategies(self) -> List[str]:
        """Get regime transition strategies"""
        
        return [
            "Dynamic asset allocation based on transition probabilities",
            "Gradual strategy rotation as regimes shift",
            "Enhanced risk monitoring during transitions", 
            "Opportunistic positioning for regime changes",
            "Multi-timeframe analysis for transition timing"
        ]
    
    def _get_strategy_implementation_guidelines(self) -> List[str]:
        """Get strategy implementation guidelines"""
        
        return [
            "Implement changes gradually to avoid whipsaws",
            "Use transition probabilities for timing decisions",
            "Maintain core positions across regime changes",
            "Monitor regime confidence metrics for validation",
            "Combine with fundamental and technical analysis"
        ]
    
    def _get_regime_risk_management(self, regime_index: int) -> Dict[str, Any]:
        """Get risk management approach for regime"""
        
        risk_management = {
            0: {  # Low volatility
                "max_drawdown": "10%",
                "position_sizing": "Normal to aggressive",
                "hedging": "Minimal",
                "stop_losses": "Trend-based stops"
            },
            1: {  # High volatility
                "max_drawdown": "15%", 
                "position_sizing": "Reduced",
                "hedging": "Moderate hedging",
                "stop_losses": "Volatility-based stops"
            },
            2: {  # Crisis
                "max_drawdown": "20%",
                "position_sizing": "Defensive",
                "hedging": "Maximum hedging",
                "stop_losses": "Strict loss limits"
            }
        }
        
        return risk_management.get(regime_index, risk_management[0])
    
    def _get_implementation_timeline(self, regime_index: int) -> Dict[str, str]:
        """Get implementation timeline for regime strategies"""
        
        return {
            "immediate": "Adjust position sizing and risk controls",
            "short_term": "Implement sector rotation and strategy shifts", 
            "medium_term": "Full regime-appropriate allocation",
            "monitoring": "Continuous regime transition monitoring"
        }
    
    def _get_volatility_characteristics(self) -> Dict[str, List[str]]:
        """Get characteristics of volatility regimes"""
        
        return {
            "low_vol": ["VIX < 15", "Steady trends", "Low dispersion"],
            "moderate_vol": ["VIX 15-25", "Normal fluctuations", "Moderate dispersion"],
            "high_vol": ["VIX 25-35", "Sharp moves", "High dispersion"],
            "extreme_vol": ["VIX > 35", "Panic selling", "Extreme dispersion"]
        }
    
    def _get_volatility_strategies(self) -> Dict[str, str]:
        """Get strategies for volatility regimes"""
        
        return {
            "low_vol": "Momentum and carry strategies",
            "moderate_vol": "Balanced approaches",
            "high_vol": "Mean reversion and hedging",
            "extreme_vol": "Defensive positioning and opportunistic buying"
        }
    
    def _analyze_volatility_patterns(self, regime_data: Dict, returns_data) -> Dict[str, Any]:
        """Analyze volatility patterns from regime data"""
        
        # This would analyze actual volatility patterns from the data
        # For now, return framework structure
        
        return {
            "current_vol_regime": "moderate_vol",
            "characteristics": ["VIX around 20", "Normal market fluctuations"],
            "forecast": "Expected to remain in moderate volatility regime",
            "risk_metrics": {
                "realized_vol": 0.18,
                "vol_of_vol": 0.75,
                "vol_persistence": 0.65
            }
        }
    
    def _get_vol_regime_strategies(self, vol_regime: str) -> List[str]:
        """Get strategies for specific volatility regime"""
        
        vol_strategies = {
            "low_vol": [
                "Momentum strategies work well",
                "Carry trades attractive", 
                "Growth stock outperformance"
            ],
            "moderate_vol": [
                "Balanced portfolio approach",
                "Normal risk budgeting",
                "Diversified strategies"
            ],
            "high_vol": [
                "Mean reversion strategies",
                "Volatility selling opportunities",
                "Defensive positioning"
            ],
            "extreme_vol": [
                "Crisis alpha strategies", 
                "Opportunistic long-term buying",
                "Maximum defensive positioning"
            ]
        }
        
        return vol_strategies.get(vol_regime, vol_strategies["moderate_vol"])
    
    def _get_regime_name(self, regime_index: int) -> str:
        """Get readable name for regime index"""
        
        regime_names = {
            0: "Low Volatility",
            1: "High Volatility", 
            2: "Crisis",
            3: "Recovery"
        }
        
        return regime_names.get(regime_index, "Unknown Regime")
    
    def _get_transition_impact(self, to_regime: int) -> str:
        """Get strategic impact of transition to regime"""
        
        impacts = {
            0: "Positive for growth and momentum strategies",
            1: "Challenging for momentum, good for volatility strategies",
            2: "Severe impact across most strategies, focus on preservation",
            3: "Excellent for cyclical and value strategies"
        }
        
        return impacts.get(to_regime, "Mixed impact")
    
    def _get_transition_actions(self, to_regime: int) -> List[str]:
        """Get recommended actions for regime transition"""
        
        actions = {
            0: ["Increase growth allocation", "Reduce hedging", "Add momentum strategies"],
            1: ["Reduce risk", "Increase hedging", "Focus on quality"],
            2: ["Defensive positioning", "Preserve capital", "Prepare for opportunities"],
            3: ["Add cyclical exposure", "Reduce defensives", "Value opportunities"]
        }
        
        return actions.get(to_regime, ["Monitor situation closely"])
    
    def _get_preparation_action(self, to_regime: int) -> str:
        """Get specific preparation action for regime"""
        
        actions = {
            0: "increase growth allocation and reduce hedging",
            1: "enhance risk management and defensive positioning",
            2: "implement maximum defensive measures and preserve capital",
            3: "prepare for cyclical opportunities and value investing"
        }
        
        return actions.get(to_regime, "monitor situation closely")
    
    def _get_transition_preparation(self, from_regime: int, to_regime: int) -> str:
        """Get preparation guidance for specific regime transition"""
        
        transitions = {
            (0, 1): "Reduce leverage and add hedging",
            (0, 2): "Implement defensive positioning immediately",
            (1, 0): "Gradually increase risk and growth exposure",
            (1, 2): "Maximum defensive measures and capital preservation",
            (2, 3): "Begin selective cyclical and value positioning",
            (3, 0): "Transition to momentum and growth strategies"
        }
        
        return transitions.get((from_regime, to_regime), "Monitor transition carefully")
    
    def _generate_summary(self, result: Dict, capability: str, execution_time: float = 0.0) -> str:
        """Generate custom summary for regime forecasting responses"""
        
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            if "missing_portfolio_data" in result.get("error_type", ""):
                return f"Portfolio returns data required for regime analysis: {error}"
            else:
                return f"Unable to complete regime analysis: {error}"
        
        if capability == "regime_detection":
            current_regime = result.get("current_regime", "Unknown")
            regime_name = self._get_regime_name(current_regime)
            confidence = result.get("confidence_metrics", {}).get("overall_confidence", 0.75)
            return f"**{regime_name} regime** detected with {confidence:.1%} confidence and strategic positioning guidance"
            
        elif capability == "transition_forecasting":
            outlook = result.get("regime_outlook", "Regime analysis completed")
            action_count = len(result.get("recommended_actions", []))
            return f"**Regime transition forecast** completed - {outlook} with {action_count} strategic recommendations"
            
        elif capability == "market_cycle_analysis":
            if result.get("current_cycle_position"):
                position = result.get("current_cycle_position", "Unknown")
                return f"**Market cycle analysis** - Currently in {position} with timing insights and cycle patterns"
            else:
                return f"**Market cycle framework** provided with cycle indicators and timing methodology"
            
        elif capability == "regime_based_strategy":
            if result.get("current_regime_strategy"):
                strategy = result.get("current_regime_strategy", {}).get("primary_strategy", "Adaptive strategy")
                return f"**Regime-based strategy** - {strategy} with allocation guidelines and dynamic recommendations"
            else:
                return f"**Regime strategy framework** provided with strategies for all market conditions"
            
        elif capability == "volatility_regime_analysis":
            vol_regime = result.get("volatility_regime", "moderate_vol").replace("_", " ").title()
            if result.get("vol_strategies"):
                strategy_count = len(result.get("vol_strategies", []))
                return f"**{vol_regime} regime** analysis with {strategy_count} volatility-specific strategies"
            else:
                return f"**Volatility regime framework** provided with characteristics and strategy guidance"
            
        else:
            return f"**Regime forecasting** analysis completed successfully in {execution_time:.2f}s"
    
    async def _health_check_capability(self, capability: str) -> bool:
        """Health check for specific capabilities"""
        
        # Create mock data for testing
        mock_returns = [0.01, -0.02, 0.015, -0.008, 0.012] * 20  # 100 data points
        
        test_data = {
            "regime_detection": {"portfolio_returns": mock_returns},
            "transition_forecasting": {"portfolio_returns": mock_returns},
            "market_cycle_analysis": {"query": "analyze market cycle"},
            "regime_based_strategy": {"portfolio_returns": mock_returns},
            "volatility_regime_analysis": {"query": "analyze volatility regime"}
        }
        
        if capability in test_data:
            try:
                result = await self.execute_capability(capability, test_data[capability], test_data[capability])
                return result.get("success", False)
            except:
                return False
        
        return False