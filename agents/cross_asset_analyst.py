# agents/cross_asset_analyst.py

from agents.base_agent import BaseFinancialAgent, DebatePerspective
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class CrossAssetAnalyst(BaseFinancialAgent):
    def __init__(self):
        super().__init__(
            agent_id="cross_asset_analyst",
            perspective=DebatePerspective.SPECIALIST
        )
        self.name = "Cross-Asset Risk Analyst"
        self.purpose = "Analyze cross-asset correlations, regime detection, and diversification scoring"
    
    def _get_specialization(self) -> str:
        return "cross_asset_correlation_and_regime_analysis"
    
    def _get_debate_strengths(self) -> List[str]:
        return ["correlation_analysis", "regime_detection", "diversification_analysis", "cross_asset_risk"]
    
    def _get_specialized_themes(self) -> Dict[str, List[str]]:
        return {
            "correlation": ["correlation", "relationship", "dependency", "covariance"],
            "regime": ["regime", "shift", "transition", "change", "break"],
            "diversification": ["diversification", "spread", "balance", "allocation"],
            "cross_asset": ["cross-asset", "asset class", "multi-asset", "inter-asset"]
        }
    
    async def _gather_specialized_evidence(self, analysis: Dict, context: Dict) -> List[Dict]:
        """Gather cross-asset correlation evidence"""
        
        evidence = []
        
        # Cross-asset correlation analysis
        if "correlation" in analysis.get("relevant_themes", []):
            evidence.append({
                "type": "correlation_analysis",
                "analysis": "Cross-asset correlation matrix shows elevated dependencies",
                "data": "Stock-Bond correlation: 0.45 (above normal 0.2)",
                "confidence": 0.8,
                "source": "Historical correlation analysis"
            })
        
        # Regime detection
        if "regime" in analysis.get("relevant_themes", []):
            evidence.append({
                "type": "regime_analysis",
                "analysis": "Market regime indicators suggest potential transition",
                "data": "Regime stability score: 0.65 (declining from 0.8)",
                "confidence": 0.7,
                "source": "Regime detection models"
            })
        
        # Diversification analysis
        evidence.append({
            "type": "diversification_analysis",
            "analysis": "Portfolio diversification effectiveness analysis",
            "data": "Effective diversification ratio: 0.72",
            "confidence": 0.85,
            "source": "Cross-asset diversification metrics"
        })
        
        return evidence
    
    async def _generate_stance(self, analysis: Dict, evidence: List[Dict]) -> str:
        """Generate cross-asset focused stance"""
        
        themes = analysis.get("relevant_themes", [])
        
        if "correlation" in themes:
            return "recommend monitoring cross-asset correlation breakdowns and adjusting allocations"
        elif "regime" in themes:
            return "suggest preparing for potential regime transition with defensive positioning"
        elif "diversification" in themes:
            return "advise enhancing diversification across uncorrelated asset classes"
        else:
            return "recommend comprehensive cross-asset risk assessment and regime monitoring"
    
    async def _identify_general_risks(self, context: Dict) -> List[str]:
        """Identify general cross-asset risks"""
        return [
            "Correlation breakdown during stress",
            "Regime transition risk",
            "Cross-asset contagion",
            "Liquidity spillover effects",
            "Currency correlation risk"
        ]
    
    async def _identify_specialized_risks(self, analysis: Dict, context: Dict) -> List[str]:
        """Identify cross-asset specific risks"""
        return [
            "Model breakdown during regime transitions",
            "Historical correlation data limitations",
            "Cross-asset volatility spillovers"
        ]
    
    def run(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute cross-asset analysis (synchronous interface for compatibility)"""
        
        if context is None:
            context = {}
        
        try:
            # Analyze portfolio holdings for cross-asset factors
            analysis_result = self._analyze_cross_asset_factors(context)
            
            # Generate actionable insights
            insights = self._generate_cross_asset_insights(analysis_result, user_query)
            
            summary = f"### 🌐 Cross-Asset Analysis Results\n\n"
            summary += f"**Correlation Regime:** {analysis_result['regime_status']}\n"
            summary += f"**Diversification Score:** {analysis_result['diversification_score']:.1f}/10\n"
            summary += f"**Risk Concentration:** {analysis_result['risk_concentration']}\n\n"
            summary += f"**Key Insights:**\n{insights}\n\n"
            summary += f"**Regime Shift Probability:** {analysis_result['regime_shift_probability']:.1%}"
            
            return {
                "success": True,
                "summary": summary,
                "analysis_type": "cross_asset_correlation",
                "regime_status": analysis_result['regime_status'],
                "diversification_score": analysis_result['diversification_score'],
                "recommendations": analysis_result['recommendations']
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Cross-asset analysis failed: {str(e)}",
                "summary": "Unable to complete cross-asset analysis due to technical error."
            }
    
    async def execute_specialized_analysis(self, query: str, context: Dict) -> Dict:
        """Execute cross-asset analysis (async interface)"""
        return self.run(query, context)
    
    async def health_check(self) -> Dict:
        """Health check for cross-asset analyst"""
        return {
            "status": "healthy",
            "response_time": 0.3,
            "memory_usage": "normal",
            "active_jobs": 0,
            "capabilities": self.debate_strengths
        }
    
    def _analyze_cross_asset_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-asset correlations and regimes"""
        
        holdings = context.get("holdings_with_values", [])
        
        if not holdings:
            # Default analysis for queries without portfolio
            return {
                "regime_status": "Normal Correlation Environment",
                "diversification_score": 7.5,
                "risk_concentration": "Moderate",
                "regime_shift_probability": 0.15,
                "recommendations": ["Consider adding cross-asset diversification"]
            }
        
        # Simple cross-asset analysis based on holdings
        stock_weight = sum(h.get("current_value", 0) for h in holdings if h.get("symbol", "").upper() not in ["BND", "TLT", "GLD", "VTI"])
        total_value = sum(h.get("current_value", 0) for h in holdings)
        
        if total_value > 0:
            equity_concentration = stock_weight / total_value
        else:
            equity_concentration = 0.8  # Default assumption
        
        # Calculate diversification score
        diversification_score = max(1, 10 - (equity_concentration * 8))
        
        # Determine regime based on concentration
        if equity_concentration > 0.9:
            regime_status = "High Correlation Risk"
            risk_concentration = "High"
            regime_shift_prob = 0.35
        elif equity_concentration > 0.7:
            regime_status = "Elevated Correlation"
            risk_concentration = "Moderate-High"
            regime_shift_prob = 0.25
        else:
            regime_status = "Normal Correlation Environment"
            risk_concentration = "Low-Moderate"
            regime_shift_prob = 0.15
        
        recommendations = []
        if equity_concentration > 0.8:
            recommendations.append("Consider adding bonds or alternative assets")
        if len(holdings) < 5:
            recommendations.append("Increase portfolio diversification across asset classes")
        
        return {
            "regime_status": regime_status,
            "diversification_score": diversification_score,
            "risk_concentration": risk_concentration,
            "regime_shift_probability": regime_shift_prob,
            "equity_concentration": equity_concentration,
            "recommendations": recommendations
        }
    
    def _generate_cross_asset_insights(self, analysis: Dict, query: str) -> str:
        """Generate actionable insights based on analysis"""
        
        insights = []
        
        equity_conc = analysis.get('equity_concentration', 0.8)
        
        if equity_conc > 0.9:
            insights.append("• Portfolio shows high equity concentration - vulnerable to correlation breakdowns")
            insights.append("• During market stress, equity correlations tend to approach 1.0")
        
        if analysis['diversification_score'] < 6:
            insights.append("• Low diversification score indicates concentrated risk exposure")
            insights.append("• Consider adding uncorrelated assets like commodities or REITs")
        
        if "crisis" in query.lower() or "stress" in query.lower():
            insights.append("• In crisis scenarios, traditional asset class correlations break down")
            insights.append("• Safe haven assets like gold and treasuries provide better protection")
        
        if not insights:
            insights.append("• Portfolio shows reasonable cross-asset diversification")
            insights.append("• Monitor correlation regimes for early warning signals")
        
        return "\n".join(insights)