# agents/cross_asset_analyst.py

from agents.mcp_base_agent import MCPBaseAgent
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import asyncio

class CrossAssetAnalyst(MCPBaseAgent):
    """
    Cross-Asset Risk Analyst specializing in correlation analysis, regime detection, and diversification scoring.
    Migrated to MCP architecture with enhanced async capabilities.
    """
    
    def __init__(self):
        # Define MCP capabilities
        capabilities = [
            "correlation_analysis",
            "regime_detection", 
            "diversification_analysis",
            "cross_asset_risk_assessment",
            "asset_allocation_optimization",
            "debate_participation",
            "consensus_building", 
            "collaborative_analysis"
        ]
        
        super().__init__(
            agent_id="cross_asset_analyst",
            agent_name="CrossAssetAnalyst",
            capabilities=capabilities
        )
        
        # Asset class definitions for analysis
        self.asset_classes = {
            "equities": ["stocks", "equity", "spy", "vti", "qqq"],
            "bonds": ["bonds", "treasuries", "bnd", "tlt", "agg"],
            "commodities": ["gold", "oil", "gld", "uso", "dbc"],
            "real_estate": ["reits", "reit", "vnq", "iyr"],
            "alternatives": ["crypto", "btc", "bitcoin", "hedge"]
        }
        
        # Correlation regime thresholds
        self.regime_thresholds = {
            "normal": 0.3,
            "elevated": 0.6,
            "crisis": 0.8
        }
    
    @property
    def name(self) -> str:
        return "Cross-Asset Risk Analyst"
    
    @property
    def purpose(self) -> str:
        return "Analyze cross-asset correlations, regime detection, and diversification scoring"
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Execute MCP capability with routing to appropriate methods"""
        
        try:
            if capability == "correlation_analysis":
                return await self._analyze_correlations(data, context)
            elif capability == "regime_detection":
                return await self._detect_regime(data, context)
            elif capability == "diversification_analysis":
                return await self._analyze_diversification(data, context)
            elif capability == "cross_asset_risk_assessment":
                return await self._assess_cross_asset_risk(data, context)
            elif capability == "asset_allocation_optimization":
                return await self._optimize_allocation(data, context)
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
    
    async def _analyze_correlations(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze cross-asset correlations"""
        
        holdings = context.get("holdings_with_values", [])
        query = data.get("query", "").lower()
        
        if not holdings:
            # Provide market-wide correlation analysis
            correlation_matrix = self._generate_market_correlation_analysis()
        else:
            # Analyze portfolio-specific correlations
            correlation_matrix = self._analyze_portfolio_correlations(holdings)
        
        # Detect correlation regime
        avg_correlation = correlation_matrix.get("average_correlation", 0.4)
        regime = self._classify_correlation_regime(avg_correlation)
        
        # Generate insights
        insights = self._generate_correlation_insights(correlation_matrix, regime, query)
        
        return {
            "success": True,
            "correlation_matrix": correlation_matrix,
            "correlation_regime": regime,
            "average_correlation": avg_correlation,
            "insights": insights,
            "recommendations": self._generate_correlation_recommendations(regime, avg_correlation)
        }
    
    async def _detect_regime(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Detect current market regime and transition probability"""
        
        holdings = context.get("holdings_with_values", [])
        
        # Calculate regime indicators
        regime_analysis = self._calculate_regime_indicators(holdings)
        
        # Assess transition probability
        transition_prob = self._calculate_transition_probability(regime_analysis)
        
        # Generate regime forecast
        forecast = self._generate_regime_forecast(regime_analysis, transition_prob)
        
        return {
            "success": True,
            "current_regime": regime_analysis["regime"],
            "regime_stability": regime_analysis["stability_score"],
            "transition_probability": transition_prob,
            "forecast": forecast,
            "indicators": regime_analysis["indicators"],
            "recommendations": self._generate_regime_recommendations(regime_analysis)
        }
    
    async def _analyze_diversification(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze portfolio diversification effectiveness"""
        
        holdings = context.get("holdings_with_values", [])
        
        if not holdings:
            return {
                "success": False,
                "error": "Portfolio holdings required for diversification analysis",
                "error_type": "missing_portfolio"
            }
        
        # Calculate diversification metrics
        diversification_metrics = self._calculate_diversification_metrics(holdings)
        
        # Assess diversification effectiveness
        effectiveness_score = self._assess_diversification_effectiveness(diversification_metrics)
        
        # Generate improvement suggestions
        improvements = self._suggest_diversification_improvements(diversification_metrics, holdings)
        
        return {
            "success": True,
            "diversification_score": effectiveness_score,
            "metrics": diversification_metrics,
            "asset_class_breakdown": self._analyze_asset_class_breakdown(holdings),
            "improvements": improvements,
            "recommendations": self._generate_diversification_recommendations(effectiveness_score)
        }
    
    async def _assess_cross_asset_risk(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Comprehensive cross-asset risk assessment"""
        
        holdings = context.get("holdings_with_values", [])
        query = data.get("query", "").lower()
        
        # Analyze multiple risk dimensions
        risk_assessment = {
            "correlation_risk": await self._assess_correlation_risk(holdings),
            "regime_risk": await self._assess_regime_risk(holdings),
            "concentration_risk": self._assess_concentration_risk(holdings),
            "liquidity_risk": self._assess_liquidity_risk(holdings, query)
        }
        
        # Calculate composite risk score
        composite_risk = self._calculate_composite_risk_score(risk_assessment)
        
        # Generate risk mitigation strategies
        mitigation_strategies = self._generate_risk_mitigation_strategies(risk_assessment)
        
        return {
            "success": True,
            "composite_risk_score": composite_risk,
            "risk_breakdown": risk_assessment,
            "risk_level": self._classify_risk_level(composite_risk),
            "mitigation_strategies": mitigation_strategies,
            "recommendations": self._generate_risk_recommendations(composite_risk, risk_assessment)
        }
    
    async def _optimize_allocation(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Optimize asset allocation for cross-asset efficiency"""
        
        holdings = context.get("holdings_with_values", [])
        
        if not holdings:
            # Provide general allocation guidance
            optimal_allocation = self._generate_default_allocation_model()
        else:
            # Optimize based on current portfolio
            current_allocation = self._analyze_current_allocation(holdings)
            optimal_allocation = self._calculate_optimal_allocation(current_allocation)
        
        # Calculate allocation adjustments
        adjustments = self._calculate_allocation_adjustments(
            holdings, optimal_allocation
        )
        
        # Estimate improvement potential
        improvement_metrics = self._estimate_allocation_improvements(adjustments)
        
        return {
            "success": True,
            "current_allocation": self._analyze_current_allocation(holdings) if holdings else {},
            "optimal_allocation": optimal_allocation,
            "recommended_adjustments": adjustments,
            "improvement_metrics": improvement_metrics,
            "implementation_plan": self._generate_implementation_plan(adjustments)
        }
    
    # Helper methods for analysis calculations
    
    def _generate_market_correlation_analysis(self) -> Dict[str, Any]:
        """Generate market-wide correlation analysis when no portfolio provided"""
        
        # Simulated current market correlations
        return {
            "stock_bond": 0.15,  # Lower correlation (good diversification)
            "stock_commodity": 0.35,  # Moderate correlation
            "stock_reit": 0.65,  # Higher correlation
            "bond_commodity": -0.05,  # Slight negative correlation
            "average_correlation": 0.28,  # Overall low correlation environment
            "regime": "normal_correlation"
        }
    
    def _analyze_portfolio_correlations(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Analyze correlations within portfolio holdings"""
        
        # Classify holdings by asset class
        asset_breakdown = self._classify_holdings_by_asset_class(holdings)
        
        # Estimate correlations based on asset class composition
        total_value = sum(h.get("current_value", 0) for h in holdings)
        
        if total_value == 0:
            return self._generate_market_correlation_analysis()
        
        equity_weight = asset_breakdown.get("equities", 0) / total_value
        
        # Estimate average correlation based on equity concentration
        if equity_weight > 0.9:
            avg_correlation = 0.85  # High correlation (high equity concentration)
        elif equity_weight > 0.7:
            avg_correlation = 0.6   # Moderate correlation
        else:
            avg_correlation = 0.3   # Low correlation (well diversified)
        
        return {
            "average_correlation": avg_correlation,
            "equity_weight": equity_weight,
            "asset_breakdown": asset_breakdown,
            "estimated_correlations": self._estimate_asset_class_correlations(asset_breakdown)
        }
    
    def _classify_holdings_by_asset_class(self, holdings: List[Dict]) -> Dict[str, float]:
        """Classify portfolio holdings by asset class"""
        
        asset_breakdown = {asset_class: 0.0 for asset_class in self.asset_classes.keys()}
        
        for holding in holdings:
            symbol = holding.get("symbol", "").lower()
            value = holding.get("current_value", 0)
            
            # Classify by symbol
            classified = False
            for asset_class, keywords in self.asset_classes.items():
                if any(keyword in symbol for keyword in keywords):
                    asset_breakdown[asset_class] += value
                    classified = True
                    break
            
            # Default to equities if not classified
            if not classified:
                asset_breakdown["equities"] += value
        
        return asset_breakdown
    
    def _classify_correlation_regime(self, avg_correlation: float) -> str:
        """Classify correlation regime based on average correlation"""
        
        if avg_correlation >= self.regime_thresholds["crisis"]:
            return "crisis_correlation"
        elif avg_correlation >= self.regime_thresholds["elevated"]:
            return "elevated_correlation"
        else:
            return "normal_correlation"
    
    def _calculate_regime_indicators(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Calculate various regime indicators"""
        
        if not holdings:
            return {
                "regime": "normal_market",
                "stability_score": 0.75,
                "indicators": {
                    "volatility_regime": "normal",
                    "correlation_regime": "normal", 
                    "liquidity_regime": "normal"
                }
            }
        
        # Analyze portfolio concentration as regime indicator
        total_value = sum(h.get("current_value", 0) for h in holdings)
        asset_breakdown = self._classify_holdings_by_asset_class(holdings)
        
        if total_value > 0:
            equity_concentration = asset_breakdown.get("equities", 0) / total_value
        else:
            equity_concentration = 0.8
        
        # Determine regime based on concentration and market conditions
        if equity_concentration > 0.95:
            regime = "high_risk_regime"
            stability_score = 0.4
        elif equity_concentration > 0.8:
            regime = "elevated_risk_regime"
            stability_score = 0.6
        else:
            regime = "normal_market"
            stability_score = 0.8
        
        return {
            "regime": regime,
            "stability_score": stability_score,
            "equity_concentration": equity_concentration,
            "indicators": {
                "concentration_risk": "high" if equity_concentration > 0.9 else "normal",
                "diversification_level": "low" if equity_concentration > 0.8 else "adequate"
            }
        }
    
    def _calculate_transition_probability(self, regime_analysis: Dict) -> float:
        """Calculate probability of regime transition"""
        
        stability_score = regime_analysis.get("stability_score", 0.75)
        
        # Higher instability = higher transition probability
        return max(0.05, min(0.5, 1 - stability_score))
    
    def _calculate_diversification_metrics(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive diversification metrics"""
        
        total_value = sum(h.get("current_value", 0) for h in holdings)
        
        if total_value == 0:
            return {"error": "No portfolio value found"}
        
        # Asset class breakdown
        asset_breakdown = self._classify_holdings_by_asset_class(holdings)
        asset_weights = {k: v/total_value for k, v in asset_breakdown.items() if v > 0}
        
        # Calculate concentration metrics
        herfindahl_index = sum(w**2 for w in asset_weights.values())
        effective_number_assets = 1 / herfindahl_index if herfindahl_index > 0 else 1
        
        # Diversification ratio (simplified)
        diversification_ratio = min(1.0, effective_number_assets / len(self.asset_classes))
        
        return {
            "asset_weights": asset_weights,
            "herfindahl_index": herfindahl_index,
            "effective_number_assets": effective_number_assets,
            "diversification_ratio": diversification_ratio,
            "concentration_score": 1 - herfindahl_index  # Higher is better
        }
    
    def _assess_diversification_effectiveness(self, metrics: Dict) -> float:
        """Assess overall diversification effectiveness (0-10 scale)"""
        
        if "error" in metrics:
            return 5.0  # Default moderate score
        
        diversification_ratio = metrics.get("diversification_ratio", 0.5)
        concentration_score = metrics.get("concentration_score", 0.5)
        
        # Weighted combination
        effectiveness = (diversification_ratio * 0.6 + concentration_score * 0.4) * 10
        
        return max(1.0, min(10.0, effectiveness))
    
    async def _assess_correlation_risk(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Assess correlation-based risk"""
        
        if not holdings:
            return {"risk_level": "moderate", "score": 0.5}
        
        correlation_analysis = self._analyze_portfolio_correlations(holdings)
        avg_correlation = correlation_analysis.get("average_correlation", 0.4)
        
        # Higher correlation = higher risk
        risk_score = min(1.0, avg_correlation * 1.25)  # Scale to 0-1
        
        if risk_score > 0.8:
            risk_level = "high"
        elif risk_score > 0.5:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "score": risk_score,
            "average_correlation": avg_correlation
        }
    
    async def _assess_regime_risk(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Assess regime transition risk"""
        
        regime_analysis = self._calculate_regime_indicators(holdings)
        transition_prob = self._calculate_transition_probability(regime_analysis)
        
        return {
            "risk_level": "high" if transition_prob > 0.3 else "moderate" if transition_prob > 0.15 else "low",
            "score": transition_prob,
            "regime": regime_analysis["regime"]
        }
    
    def _assess_concentration_risk(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Assess portfolio concentration risk"""
        
        if not holdings:
            return {"risk_level": "low", "score": 0.2}
        
        metrics = self._calculate_diversification_metrics(holdings)
        
        if "error" in metrics:
            return {"risk_level": "moderate", "score": 0.5}
        
        concentration_score = 1 - metrics.get("concentration_score", 0.5)  # Invert for risk
        
        return {
            "risk_level": "high" if concentration_score > 0.7 else "moderate" if concentration_score > 0.4 else "low",
            "score": concentration_score,
            "herfindahl_index": metrics.get("herfindahl_index", 0.5)
        }
    
    def _assess_liquidity_risk(self, holdings: List[Dict], query: str) -> Dict[str, Any]:
        """Assess cross-asset liquidity risk"""
        
        # Simplified liquidity assessment
        if "crisis" in query or "stress" in query:
            return {"risk_level": "high", "score": 0.8, "reason": "Crisis scenarios reduce liquidity"}
        
        return {"risk_level": "moderate", "score": 0.3, "reason": "Normal market liquidity conditions"}
    
    def _calculate_composite_risk_score(self, risk_assessment: Dict) -> float:
        """Calculate composite risk score from individual assessments"""
        
        weights = {
            "correlation_risk": 0.3,
            "regime_risk": 0.25,
            "concentration_risk": 0.25,
            "liquidity_risk": 0.2
        }
        
        composite_score = 0.0
        for risk_type, weight in weights.items():
            risk_data = risk_assessment.get(risk_type, {})
            score = risk_data.get("score", 0.5)
            composite_score += score * weight
        
        return composite_score
    
    def _classify_risk_level(self, composite_score: float) -> str:
        """Classify overall risk level"""
        
        if composite_score > 0.7:
            return "high"
        elif composite_score > 0.4:
            return "moderate"
        else:
            return "low"
    
    def _analyze_current_allocation(self, holdings: List[Dict]) -> Dict[str, float]:
        """Analyze current asset allocation"""
        
        if not holdings:
            return {}
        
        total_value = sum(h.get("current_value", 0) for h in holdings)
        
        if total_value == 0:
            return {}
        
        asset_breakdown = self._classify_holdings_by_asset_class(holdings)
        
        return {k: v/total_value for k, v in asset_breakdown.items() if v > 0}
    
    def _generate_default_allocation_model(self) -> Dict[str, float]:
        """Generate default balanced allocation model"""
        
        return {
            "equities": 0.60,
            "bonds": 0.25,
            "real_estate": 0.10,
            "commodities": 0.05
        }
    
    def _calculate_optimal_allocation(self, current_allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimal allocation based on current portfolio"""
        
        # Risk-based optimization (simplified)
        target_allocation = self._generate_default_allocation_model()
        
        # Adjust based on current allocation
        optimized = {}
        for asset_class, target_weight in target_allocation.items():
            current_weight = current_allocation.get(asset_class, 0.0)
            
            # Gradual adjustment toward target
            adjustment_factor = 0.7  # Don't move all the way to target immediately
            optimized_weight = current_weight + (target_weight - current_weight) * adjustment_factor
            
            optimized[asset_class] = max(0.0, optimized_weight)
        
        # Normalize to sum to 1.0
        total_weight = sum(optimized.values())
        if total_weight > 0:
            optimized = {k: v/total_weight for k, v in optimized.items()}
        
        return optimized
    
    def _calculate_allocation_adjustments(self, holdings: List[Dict], optimal_allocation: Dict[str, float]) -> List[Dict]:
        """Calculate specific allocation adjustments needed"""
        
        if not holdings:
            return [{"action": "build_initial_portfolio", "allocation": optimal_allocation}]
        
        current_allocation = self._analyze_current_allocation(holdings)
        adjustments = []
        
        for asset_class, target_weight in optimal_allocation.items():
            current_weight = current_allocation.get(asset_class, 0.0)
            difference = target_weight - current_weight
            
            if abs(difference) > 0.05:  # Only recommend changes > 5%
                action = "increase" if difference > 0 else "decrease"
                adjustments.append({
                    "asset_class": asset_class,
                    "action": action,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "adjustment": abs(difference)
                })
        
        return adjustments
    
    def _estimate_allocation_improvements(self, adjustments: List[Dict]) -> Dict[str, Any]:
        """Estimate improvements from allocation changes"""
        
        if not adjustments:
            return {"diversification_improvement": 0.0, "risk_reduction": 0.0}
        
        # Simplified improvement estimates
        total_adjustments = sum(adj.get("adjustment", 0) for adj in adjustments)
        
        return {
            "diversification_improvement": min(0.3, total_adjustments * 2),  # Max 30% improvement
            "risk_reduction": min(0.25, total_adjustments * 1.5),  # Max 25% risk reduction
            "expected_return_impact": 0.0  # Neutral assumption
        }
    
    def _generate_implementation_plan(self, adjustments: List[Dict]) -> List[Dict]:
        """Generate step-by-step implementation plan"""
        
        if not adjustments:
            return [{"step": 1, "action": "Portfolio is well-balanced", "priority": "low"}]
        
        plan = []
        
        # Sort by adjustment size (largest first)
        sorted_adjustments = sorted(adjustments, key=lambda x: x.get("adjustment", 0), reverse=True)
        
        for i, adj in enumerate(sorted_adjustments, 1):
            asset_class = adj["asset_class"]
            action = adj["action"]
            adjustment = adj.get("adjustment", 0)
            
            priority = "high" if adjustment > 0.15 else "medium" if adjustment > 0.1 else "low"
            
            plan.append({
                "step": i,
                "action": f"{action.title()} {asset_class} allocation by {adjustment:.1%}",
                "asset_class": asset_class,
                "priority": priority,
                "target_change": adjustment
            })
        
        return plan
    
    # Recommendation generation methods
    
    def _generate_correlation_recommendations(self, regime: str, avg_correlation: float) -> List[str]:
        """Generate correlation-specific recommendations"""
        
        recommendations = []
        
        if regime == "crisis_correlation":
            recommendations.extend([
                "Consider defensive assets with lower correlations to equities",
                "Increase allocation to safe haven assets like treasuries or gold",
                "Monitor correlation breakdowns during market stress"
            ])
        elif regime == "elevated_correlation":
            recommendations.extend([
                "Review portfolio diversification across asset classes",
                "Consider alternative assets with low correlation to traditional assets",
                "Prepare for potential correlation regime shifts"
            ])
        else:
            recommendations.extend([
                "Current correlation environment supports diversification",
                "Maintain balanced allocation across asset classes",
                "Monitor for signs of correlation regime changes"
            ])
        
        return recommendations
    
    def _generate_regime_recommendations(self, regime_analysis: Dict) -> List[str]:
        """Generate regime-specific recommendations"""
        
        regime = regime_analysis.get("regime", "normal_market")
        stability = regime_analysis.get("stability_score", 0.75)
        
        recommendations = []
        
        if regime == "high_risk_regime":
            recommendations.extend([
                "Consider reducing portfolio concentration",
                "Increase allocation to defensive assets",
                "Implement risk management strategies"
            ])
        elif stability < 0.6:
            recommendations.extend([
                "Monitor regime indicators closely",
                "Prepare for potential market transitions",
                "Consider tactical asset allocation adjustments"
            ])
        else:
            recommendations.extend([
                "Current regime appears stable",
                "Maintain strategic asset allocation",
                "Continue monitoring regime indicators"
            ])
        
        return recommendations
    
    def _generate_diversification_recommendations(self, effectiveness_score: float) -> List[str]:
        """Generate diversification-specific recommendations"""
        
        recommendations = []
        
        if effectiveness_score < 4:
            recommendations.extend([
                "Portfolio shows low diversification - consider broader asset allocation",
                "Add uncorrelated asset classes like REITs or commodities",
                "Reduce concentration in single asset class"
            ])
        elif effectiveness_score < 7:
            recommendations.extend([
                "Portfolio diversification is adequate but could be improved",
                "Consider adding alternative assets for better diversification",
                "Monitor correlation changes between holdings"
            ])
        else:
            recommendations.extend([
                "Portfolio shows good diversification across asset classes",
                "Maintain current diversification approach",
                "Continue monitoring for correlation changes"
            ])
        
        return recommendations
    
    def _generate_risk_recommendations(self, composite_risk: float, risk_breakdown: Dict) -> List[str]:
        """Generate comprehensive risk management recommendations"""
        
        recommendations = []
        
        if composite_risk > 0.7:
            recommendations.extend([
                "High cross-asset risk detected - consider immediate portfolio adjustments",
                "Implement risk mitigation strategies across multiple dimensions",
                "Consider professional portfolio review"
            ])
        elif composite_risk > 0.4:
            recommendations.extend([
                "Moderate cross-asset risk - monitor key risk factors",
                "Consider gradual portfolio adjustments",
                "Review risk management approaches"
            ])
        else:
            recommendations.extend([
                "Cross-asset risk appears manageable",
                "Continue monitoring risk factors",
                "Maintain current risk management approach"
            ])
        
        # Add specific recommendations based on highest risk factors
        highest_risk = max(risk_breakdown.items(), key=lambda x: x[1].get("score", 0))
        risk_type, risk_data = highest_risk
        
        if risk_data.get("score", 0) > 0.6:
            if risk_type == "correlation_risk":
                recommendations.append("Address correlation risk through better diversification")
            elif risk_type == "regime_risk":
                recommendations.append("Prepare for potential market regime transitions")
            elif risk_type == "concentration_risk":
                recommendations.append("Reduce portfolio concentration across asset classes")
        
        return recommendations
    
    def _generate_risk_mitigation_strategies(self, risk_assessment: Dict) -> List[Dict]:
        """Generate specific risk mitigation strategies"""
        
        strategies = []
        
        for risk_type, risk_data in risk_assessment.items():
            if risk_data.get("score", 0) > 0.5:
                
                if risk_type == "correlation_risk":
                    strategies.append({
                        "risk_type": risk_type,
                        "strategy": "Diversify across low-correlation asset classes",
                        "implementation": "Add bonds, commodities, or alternative assets",
                        "timeline": "1-3 months"
                    })
                
                elif risk_type == "regime_risk":
                    strategies.append({
                        "risk_type": risk_type,
                        "strategy": "Implement regime-aware allocation",
                        "implementation": "Monitor regime indicators and adjust allocation dynamically",
                        "timeline": "Ongoing"
                    })
                
                elif risk_type == "concentration_risk":
                    strategies.append({
                        "risk_type": risk_type,
                        "strategy": "Reduce portfolio concentration",
                        "implementation": "Spread investments across more asset classes",
                        "timeline": "3-6 months"
                    })
        
        return strategies
    
    def _generate_correlation_insights(self, correlation_matrix: Dict, regime: str, query: str) -> List[str]:
        """Generate actionable correlation insights"""
        
        insights = []
        avg_correlation = correlation_matrix.get("average_correlation", 0.4)
        
        if "crisis" in query or "stress" in query:
            insights.extend([
                "During crisis periods, asset correlations typically increase significantly",
                "Traditional diversification benefits may break down under extreme stress",
                "Safe haven assets like gold and treasuries provide better crisis protection"
            ])
        
        if regime == "crisis_correlation":
            insights.append("Current correlation regime shows elevated systemic risk")
        elif regime == "elevated_correlation":
            insights.append("Correlations are above normal - monitor for further increases")
        
        if avg_correlation > 0.6:
            insights.append("High average correlation suggests limited diversification benefit")
        
        return insights
    
    def _generate_regime_forecast(self, regime_analysis: Dict, transition_prob: float) -> str:
        """Generate regime forecast and outlook"""
        
        current_regime = regime_analysis.get("regime", "normal_market")
        stability = regime_analysis.get("stability_score", 0.75)
        
        if transition_prob > 0.3:
            return f"High probability ({transition_prob:.1%}) of regime transition from {current_regime}"
        elif transition_prob > 0.15:
            return f"Moderate probability ({transition_prob:.1%}) of regime change - monitor indicators"
        else:
            return f"Current {current_regime} appears stable with low transition risk ({transition_prob:.1%})"
    
    def _suggest_diversification_improvements(self, metrics: Dict, holdings: List[Dict]) -> List[Dict]:
        """Suggest specific diversification improvements"""
        
        improvements = []
        
        if "error" in metrics:
            return improvements
        
        asset_weights = metrics.get("asset_weights", {})
        
        # Check for missing asset classes
        missing_classes = []
        for asset_class in self.asset_classes.keys():
            if asset_weights.get(asset_class, 0) < 0.05:  # Less than 5%
                missing_classes.append(asset_class)
        
        for asset_class in missing_classes:
            improvements.append({
                "type": "add_asset_class",
                "asset_class": asset_class,
                "suggested_allocation": "5-15%",
                "benefit": "Improved diversification and reduced correlation"
            })
        
        # Check for over-concentration
        for asset_class, weight in asset_weights.items():
            if weight > 0.8:  # Over 80%
                improvements.append({
                    "type": "reduce_concentration",
                    "asset_class": asset_class,
                    "current_allocation": f"{weight:.1%}",
                    "suggested_allocation": "60-70%",
                    "benefit": "Reduced concentration risk"
                })
        
        return improvements
    
    def _estimate_asset_class_correlations(self, asset_breakdown: Dict) -> Dict[str, float]:
        """Estimate correlations between asset classes"""
        
        # Simplified correlation estimates
        return {
            "equities_bonds": 0.15,
            "equities_commodities": 0.30,
            "equities_real_estate": 0.65,
            "bonds_commodities": -0.05,
            "bonds_real_estate": 0.20,
            "commodities_real_estate": 0.25
        }
    
    def _generate_summary(self, result: Dict, capability: str, execution_time: float = 0.0) -> str:
        """Generate custom summary for cross-asset analysis responses"""
        
        if not result.get("success"):
            return f"Unable to complete cross-asset analysis: {result.get('error', 'Unknown error')}"
        
        if capability == "correlation_analysis":
            regime = result.get("correlation_regime", "normal")
            avg_corr = result.get("average_correlation", 0.0)
            return f"**Cross-asset correlation analysis** completed - {regime} regime with {avg_corr:.2f} average correlation"
            
        elif capability == "regime_detection":
            current_regime = result.get("current_regime", "normal")
            stability = result.get("regime_stability", 0.75)
            return f"**Market regime analysis** - Current: {current_regime} (stability: {stability:.2f})"
            
        elif capability == "diversification_analysis":
            score = result.get("diversification_score", 5.0)
            return f"**Portfolio diversification analysis** - Score: {score:.1f}/10 with improvement recommendations"
            
        elif capability == "cross_asset_risk_assessment":
            risk_level = result.get("risk_level", "moderate")
            risk_score = result.get("composite_risk_score", 0.5)
            return f"**Cross-asset risk assessment** - {risk_level.title()} risk level (score: {risk_score:.2f})"
            
        elif capability == "asset_allocation_optimization":
            num_adjustments = len(result.get("recommended_adjustments", []))
            return f"**Asset allocation optimization** - {num_adjustments} recommended adjustments for improved diversification"
            
        else:
            return f"**Cross-asset analysis** completed successfully in {execution_time:.2f}s"
    
    async def _health_check_capability(self, capability: str) -> bool:
        """Health check for specific capabilities"""
        
        test_data = {
            "correlation_analysis": {"query": "analyze correlations"},
            "regime_detection": {"query": "detect market regime"},
            "diversification_analysis": {"query": "analyze diversification"},
            "cross_asset_risk_assessment": {"query": "assess cross-asset risk"},
            "asset_allocation_optimization": {"query": "optimize allocation"}
        }
        
        if capability in test_data:
            try:
                result = await self.execute_capability(capability, test_data[capability], {"holdings_with_values": []})
                return result.get("success", False)
            except:
                return False
        
        return False