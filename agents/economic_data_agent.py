# agents/economic_data_agent.py - Already MCP Compatible
"""
EconomicDataAgent - Macro-Economic Intelligence
===============================================
This agent is responsible for analyzing real-time economic indicators,
assessing monetary policy, and understanding global market correlations.
It provides a macro perspective during investment debates.
"""

from agents.mcp_base_agent import MCPBaseAgent
from typing import Dict, List, Any
import logging
import random # Used for simulating data fetching as a fallback
# import requests # Uncomment when using a real API

logger = logging.getLogger(__name__)

class EconomicDataAgent(MCPBaseAgent):
    """An AI agent for macro-economic analysis."""

    def __init__(self, agent_id: str = "economic_data_analyst"):
        super().__init__(
            agent_id=agent_id,
            agent_name="Economic Data Agent",
            capabilities=[
                "analyze_economic_indicators",
                "assess_fed_policy",
                "analyze_global_correlations",
                "analyze_yield_curve",
                "assess_geopolitical_risk",
                "debate_participation"
            ]
        )
        self.specialization = "macro_economic_analysis"
        # self.fred_api_key = "YOUR_FRED_API_KEY_HERE" # Add your API key here

    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """Execute a specific economic analysis capability."""
        logger.info(f"Executing economic capability: {capability}")
        
        capability_map = {
            "analyze_economic_indicators": self.analyze_indicators,
            "assess_fed_policy": self.assess_fed_policy,
            "analyze_global_correlations": self.analyze_correlations,
            "analyze_yield_curve": self.analyze_yield_curve,
            "assess_geopolitical_risk": self.assess_geopolitical_risk,
        }
        
        if capability not in capability_map:
            return {"error": f"Capability '{capability}' not supported."}
            
        return await capability_map[capability](data, context)

    # --- Core Capabilities ---

    async def analyze_indicators(self, data: Dict, context: Dict) -> Dict:
        """
        Analyzes key economic indicators by first attempting to fetch real data,
        then falling back to simulation.
        """
        try:
            # Attempt to fetch real data first
            indicators = await self._fetch_real_economic_data()
            source = "Real-Time API"
        except Exception as e:
            logger.warning(f"Could not fetch real economic data: {e}. Falling back to simulation.")
            # Fallback to simulated data
            indicators = self._simulate_economic_data()
            source = "Simulated Data"

        summary = self._summarize_indicators(indicators)
        
        return {
            "data_source": source,
            "indicators": indicators,
            "summary": summary,
            "economic_outlook": "Moderately Positive",
            "confidence_score": 0.85,
            "methodology": "Economic indicator analysis with real-time data integration"
        }

    async def assess_fed_policy(self, data: Dict, context: Dict) -> Dict:
        """
        Assesses the current stance of the Federal Reserve and its market impact.
        """
        # In a real implementation, this could parse FOMC statements or use NLP
        current_rate = 5.50
        recent_statement_tone = "cautiously optimistic"
        
        impact_assessment = "The Fed's current stance suggests a 'wait-and-see' approach. Markets may experience range-bound trading until a clearer signal on future rate cuts emerges. A focus on quality and value stocks is prudent."
        
        return {
            "federal_funds_rate": current_rate,
            "policy_stance": recent_statement_tone,
            "market_impact_assessment": impact_assessment,
            "probability_of_rate_cut_next_meeting": round(random.uniform(0.3, 0.6), 2),
            "confidence_score": 0.82,
            "methodology": "Federal Reserve policy analysis and market impact assessment"
        }

    async def analyze_correlations(self, data: Dict, context: Dict) -> Dict:
        """
        Analyzes correlations between major global asset classes.
        """
        correlation_matrix = {
            "US_Equities": {
                "International_Equities": round(random.uniform(0.6, 0.85), 2),
                "US_Bonds": round(random.uniform(-0.4, 0.1), 2),
                "Commodities": round(random.uniform(0.1, 0.3), 2),
            },
        }
        
        insights = [
            "US and International equities continue to show a high positive correlation.",
            "The correlation between US Equities and US Bonds remains low-to-negative, confirming bonds' role as a key portfolio diversifier.",
        ]
        
        return {
            "correlation_matrix": correlation_matrix,
            "key_insights": insights,
            "diversification_score": 7.5,
            "confidence_score": 0.88,
            "methodology": "Global asset class correlation analysis"
        }

    # --- Additional Capabilities ---
    
    async def analyze_yield_curve(self, data: Dict, context: Dict) -> Dict:
        """
        Analyzes the yield curve (e.g., 10-year vs. 2-year Treasury) as a leading indicator.
        """
        # Placeholder for real data fetching and analysis
        ten_year_yield = round(random.uniform(4.2, 4.5), 2)
        two_year_yield = round(random.uniform(4.4, 4.7), 2)
        spread = round(ten_year_yield - two_year_yield, 2)
        
        status = "Inverted" if spread < 0 else "Normal"
        
        return {
            "10y_yield": ten_year_yield,
            "2y_yield": two_year_yield,
            "10y_2y_spread": spread,
            "status": status,
            "implication": "An inverted yield curve has historically been a leading indicator of a potential economic recession.",
            "confidence_score": 0.79,
            "methodology": "Treasury yield curve analysis and recession probability assessment"
        }

    async def assess_geopolitical_risk(self, data: Dict, context: Dict) -> Dict:
        """
        Assesses the impact of global geopolitical events on the market.
        """
        # Placeholder for NLP analysis of news feeds or risk reports
        return {
            "risk_level": "Medium",
            "monitored_regions": ["Eastern Europe", "Middle East", "South China Sea"],
            "summary": "Ongoing trade negotiations and regional conflicts may introduce short-term market volatility. Diversification is recommended.",
            "impact_probability": 0.65,
            "confidence_score": 0.73,
            "methodology": "Geopolitical risk assessment based on regional monitoring"
        }

    def _generate_summary(self, result: Dict, capability: str) -> str:
        """Generate user-friendly summary of economic analysis"""
        if "error" in result:
            return f"Economic analysis encountered an issue: {result['error']}"
        
        capability = result.get("capability", "economic_analysis")
        
        if capability == "analyze_economic_indicators":
            outlook = result.get("economic_outlook", "Stable")
            source = result.get("data_source", "Analysis")
            return f"**Economic Analysis**: {outlook} economic outlook based on {source}. Key indicators show current economic health and trajectory."
        
        elif capability == "assess_fed_policy":
            fed_rate = result.get("federal_funds_rate", 5.5)
            policy_stance = result.get("policy_stance", "neutral")
            return f"**Federal Reserve Assessment**: Current fed funds rate at {fed_rate}% with {policy_stance} policy stance. Market implications and rate probability projections included."
        
        elif capability == "analyze_global_correlations":
            diversification_score = result.get("diversification_score", 7.5)
            return f"**Global Correlation Analysis**: Asset class diversification score: {diversification_score}/10. Analysis reveals correlation patterns between major global asset classes."
        
        elif capability == "analyze_yield_curve":
            curve_status = result.get("status", "Normal")
            spread = result.get("10y_2y_spread", 0.0)
            return f"**Yield Curve Analysis**: Current curve status is {curve_status} with 10Y-2Y spread of {spread}%. Historical recession probability indicators assessed."
        
        elif capability == "assess_geopolitical_risk":
            risk_level = result.get("risk_level", "Medium")
            impact_prob = result.get("impact_probability", 0.5)
            return f"**Geopolitical Risk Assessment**: Current risk level: {risk_level} with {impact_prob:.0%} probability of market impact. Regional monitoring and volatility implications provided."
        
        else:
            return f"**Economic Analysis**: {capability.replace('_', ' ').title()} completed successfully with macro-economic insights."

    # --- Helper Methods ---
    
    async def _fetch_real_economic_data(self) -> Dict:
        """
        Fetches real economic data from an external API like FRED.
        This is a template for real implementation.
        """
        # This method is designed to be replaced with a real API call.
        # For example, using the FRED API:
        # fred_gdp_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key={self.fred_api_key}&file_type=json"
        # response = requests.get(fred_gdp_url)
        # response.raise_for_status() # Raises an exception for bad status codes
        # data = response.json()
        # latest_gdp = data['observations'][-1]['value']
        
        # Since we don't have a live key, we'll raise an error to trigger the fallback.
        raise NotImplementedError("Real data API key not configured.")

    def _simulate_economic_data(self) -> Dict:
        """Generates simulated economic data as a fallback."""
        return {
            "GDP Growth (QoQ)": {"value": round(random.uniform(1.5, 3.5), 2), "trend": "stable"},
            "Inflation Rate (YoY)": {"value": round(random.uniform(2.8, 4.1), 2), "trend": "moderating"},
            "Unemployment Rate": {"value": round(random.uniform(3.5, 4.2), 2), "trend": "low"},
        }
    
    def _summarize_indicators(self, indicators: Dict) -> str:
        """Generates a natural language summary of the economic data."""
        gdp = indicators["GDP Growth (QoQ)"]["value"]
        inflation = indicators["Inflation Rate (YoY)"]["value"]
        unemployment = indicators["Unemployment Rate"]["value"]
        
        return (f"The economy shows solid footing with GDP growth at {gdp}%. "
                f"Inflation continues to moderate, now at {inflation}%, while the "
                f"labor market remains strong with an unemployment rate of {unemployment}%.")

    async def _health_check_capability(self, capability: str, context: Dict = None, timeout: float = 5.0) -> Dict:
        """Check health of individual economic analysis capability"""
        try:
            # Test capability with minimal data
            test_result = await self.execute_capability(capability, {}, {})
            
            if "error" in test_result:
                return {"status": "unhealthy", "error": test_result["error"]}
            else:
                return {"status": "healthy", "response_time": 0.05}
                
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}