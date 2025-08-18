# agents/market_intelligence_agent.py
"""
Simple MarketIntelligenceAgent for Testing
=========================================
Aggressive, growth-focused market intelligence agent compatible with both
MCP and simple testing environments.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging
from enum import Enum

# Try to import MCP base, fallback for testing
try:
    from agents.mcp_base_agent import MCPBaseAgent
    BaseAgentClass = MCPBaseAgent
    HAS_MCP = True
except ImportError:
    class SimpleBaseAgent:
        def __init__(self, agent_id: str, **kwargs):
            self.agent_id = agent_id
            self.capabilities = kwargs.get('capabilities', [])
    BaseAgentClass = SimpleBaseAgent
    HAS_MCP = False

logger = logging.getLogger(__name__)

class DebatePerspective(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    SPECIALIST = "specialist"

class MarketIntelligenceAgent(BaseAgentClass):
    """Aggressive, growth-focused market intelligence agent with debate capabilities"""
    
    def __init__(self, agent_id: str = "market_intelligence"):
        if HAS_MCP:
            super().__init__(
                agent_id=agent_id,
                agent_name="Market Intelligence Agent",
                capabilities=[
                    "market_analysis",
                    "opportunity_identification",
                    "trend_analysis",
                    "sentiment_analysis",
                    "timing_analysis",
                    "debate_participation"
                ]
            )
        else:
            super().__init__(agent_id=agent_id)
        
        # Debate system properties
        self.perspective = DebatePerspective.AGGRESSIVE
        self.specialization = "market_timing_and_opportunity_identification"
        self.debate_strengths = ["market_timing", "opportunity_identification", "trend_analysis", "sentiment_analysis"]
        
        # Market intelligence configuration
        self.market_indicators = {
            "sentiment": ["vix", "put_call_ratio", "fear_greed_index"],
            "technical": ["rsi", "macd", "moving_averages", "momentum"],
            "economic": ["gdp_growth", "unemployment", "inflation", "fed_policy"],
            "sector": ["sector_rotation", "relative_strength"]
        }
        
        # Debate personality configuration
        self.debate_config = {
            "focus": "growth_opportunities",
            "evidence_preference": "forward_looking",
            "bias": "upside_potential",
            "argument_style": "opportunity_driven",
            "challenge_approach": "opportunity_cost_focus",
            "confidence_threshold": 0.5
        }
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """Execute market intelligence capabilities"""
        if not HAS_MCP:
            return {"error": "MCP not available"}
        
        capability_map = {
            "market_analysis": self.analyze_market_conditions,
            "opportunity_identification": self.identify_opportunities,
            "trend_analysis": self.analyze_trends,
            "sentiment_analysis": self.analyze_sentiment,
            "timing_analysis": self.analyze_timing,
            "debate_participation": self.participate_in_debate
        }
        
        if capability not in capability_map:
            raise ValueError(f"Capability {capability} not supported by MarketIntelligence")
        
        return await capability_map[capability](data, context)
    
    # ==========================================
    # DEBATE SYSTEM METHODS
    # ==========================================
    
    async def formulate_debate_position(self, query: str, context: Dict, debate_context: Dict = None) -> Dict:
        """Generate agent's initial position for debate"""
        
        themes = self._extract_debate_themes(query)
        
        # Generate aggressive, opportunity-focused stance
        if "risk" in themes:
            stance = "current market conditions present attractive opportunities despite near-term volatility"
            arguments = [
                "VIX levels indicate moderate fear, creating attractive entry points",
                "Economic fundamentals support risk-taking in quality assets",
                "Market corrections provide opportunities for long-term investors"
            ]
        elif "opportunity" in themes:
            stance = "maximize growth opportunities through strategic positioning"
            arguments = [
                "Market momentum indicators signal continued upward potential",
                "Sector rotation favors growth over defensive positioning",
                "Fed policy environment becoming supportive for risk assets"
            ]
        else:
            stance = "focus on growth opportunities with selective risk management"
            arguments = [
                "Market breadth indicators show healthy participation",
                "Technical analysis supports continued market strength",
                "Economic cycle positioning favors equity exposure"
            ]
        
        # Generate market-focused evidence
        evidence = await self._gather_market_evidence(query, context)
        
        return {
            "stance": stance,
            "key_arguments": arguments,
            "supporting_evidence": evidence,
            "risk_assessment": {
                "primary_risks": ["Short-term volatility", "Geopolitical events", "Sentiment shifts"],
                "mitigation_strategies": ["Quality focus", "Timing optimization", "Selective positioning"]
            },
            "confidence_score": 0.82,
            "perspective_bias": self.debate_config["bias"],
            "argument_style": self.debate_config["argument_style"]
        }
    
    async def respond_to_challenge(self, challenge: str, original_position: Dict, challenge_context: Dict = None) -> Dict:
        """Respond to challenges from other agents"""
        
        challenge_type = self._classify_challenge_type(challenge)
        
        # Aggressive agent responses
        if "risk" in challenge.lower():
            response_arguments = [
                "Risk management is important, but opportunity cost of inaction is significant",
                "Current market resilience suggests defensive positioning may be excessive",
                "Quality growth assets provide both upside potential and relative safety"
            ]
            strategy = "opportunity_cost_focus"
        elif "conservative" in challenge.lower():
            response_arguments = [
                "Conservative approaches often miss significant value creation opportunities",
                "Market timing and quality selection can capture upside while managing risk",
                "Economic fundamentals support a more optimistic outlook"
            ]
            strategy = "growth_emphasis"
        else:
            response_arguments = [
                "Market intelligence suggests current conditions favor growth positioning",
                "Technical and fundamental analysis support the opportunity-focused approach",
                "Forward-looking indicators outweigh historical risk concerns"
            ]
            strategy = "evidence_based_optimism"
        
        return {
            "response_strategy": strategy,
            "counter_arguments": response_arguments,
            "supporting_evidence": original_position.get("supporting_evidence", []),
            "acknowledgments": ["Risk considerations are valid but must be balanced against opportunities"],
            "updated_position": original_position,  # Aggressive agents maintain optimistic positions
            "confidence_change": 0.0,
            "rebuttal_strength": 0.8
        }
    
    async def participate_in_debate(self, data: Dict, context: Dict) -> Dict:
        """Participate in MCP-orchestrated debate"""
        
        debate_request = data.get("debate_request", {})
        query = debate_request.get("query", "")
        role = debate_request.get("role", "participant")
        
        if role == "position":
            return await self.formulate_debate_position(query, context, debate_request)
        elif role == "challenge":
            original_position = debate_request.get("original_position", {})
            challenge = debate_request.get("challenge", "")
            return await self.respond_to_challenge(challenge, original_position, debate_request)
        else:
            return {"error": f"Unknown debate role: {role}"}
    
    def _extract_debate_themes(self, query: str) -> List[str]:
        """Extract themes relevant to market intelligence"""
        query_lower = query.lower()
        themes = []
        
        theme_keywords = {
            "risk": ["risk", "safe", "conservative", "protect", "downside"],
            "opportunity": ["opportunity", "growth", "aggressive", "upside", "timing"],
            "market": ["market", "trend", "momentum", "cycle", "sentiment"],
            "timing": ["timing", "when", "entry", "exit", "now"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    async def _gather_market_evidence(self, query: str, context: Dict) -> List[Dict]:
        """Gather market intelligence evidence"""
        return [
            {
                "type": "market_data",
                "analysis": "Market sentiment indicators show moderate fear levels",
                "data": "VIX: 18.5, Put/Call ratio: 0.85, Fear & Greed index: 42",
                "confidence": 0.85,
                "source": "Real-time market sentiment analysis"
            },
            {
                "type": "economic",
                "analysis": "Economic fundamentals support growth outlook",
                "data": "GDP growth: 2.8%, Unemployment: 3.7%, Core inflation: 2.1%",
                "confidence": 0.90,
                "source": "Economic indicators analysis"
            },
            {
                "type": "technical",
                "analysis": "Technical indicators show bullish momentum",
                "data": "RSI: 65, MACD bullish crossover, 50-day MA support holding",
                "confidence": 0.80,
                "source": "Technical analysis"
            }
        ]
    
    def _classify_challenge_type(self, challenge: str) -> str:
        """Classify the type of challenge being made"""
        challenge_lower = challenge.lower()
        
        if any(word in challenge_lower for word in ["risk", "dangerous", "loss"]):
            return "risk_challenge"
        elif any(word in challenge_lower for word in ["conservative", "cautious", "defensive"]):
            return "conservative_challenge"
        else:
            return "general_disagreement"
    
    # ==========================================
    # MARKET ANALYSIS METHODS
    # ==========================================
    
    async def analyze_market_conditions(self, data: Dict, context: Dict) -> Dict:
        """Analyze current market conditions"""
        return {
            "market_sentiment": "cautiously_optimistic",
            "vix_level": 18.5,
            "trend_direction": "upward",
            "sector_rotation": "growth_favored",
            "confidence": 0.82,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def identify_opportunities(self, data: Dict, context: Dict) -> Dict:
        """Identify market opportunities"""
        return {
            "opportunities": [
                "Technology sector showing renewed strength",
                "International markets offering diversification",
                "Quality growth stocks at attractive valuations"
            ],
            "timing_signals": ["bullish", "momentum_building"],
            "confidence": 0.78,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def analyze_trends(self, data: Dict, context: Dict) -> Dict:
        """Analyze market trends"""
        return {
            "primary_trend": "bullish",
            "trend_strength": "moderate",
            "trend_duration": "6_months",
            "confidence": 0.75,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def analyze_sentiment(self, data: Dict, context: Dict) -> Dict:
        """Analyze market sentiment"""
        return {
            "sentiment_score": 42,  # Fear & Greed index
            "sentiment_level": "neutral_to_positive",
            "key_indicators": {
                "vix": 18.5,
                "put_call_ratio": 0.85
            },
            "confidence": 0.80,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def analyze_timing(self, data: Dict, context: Dict) -> Dict:
        """Analyze market timing signals"""
        return {
            "timing_signal": "favorable",
            "entry_signals": ["momentum_bullish", "sentiment_supportive"],
            "exit_signals": [],
            "confidence": 0.77,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ==========================================
    # COMPATIBILITY METHODS
    # ==========================================
    
    def get_specialization(self) -> str:
        return self.specialization
    
    def get_debate_strengths(self) -> List[str]:
        return self.debate_strengths
    
    def get_specialized_themes(self) -> Dict[str, List[str]]:
        return {
            "market_timing": ["timing", "entry", "exit", "momentum", "trend"],
            "sentiment": ["sentiment", "fear", "greed", "optimism", "pessimism"],
            "opportunities": ["opportunity", "growth", "breakout", "catalyst"],
            "technical": ["technical", "chart", "pattern", "signal", "indicator"]
        }
    
    async def health_check(self) -> Dict:
        """Health check for market intelligence agent"""
        return {
            "status": "healthy",
            "response_time": 0.4,
            "memory_usage": "normal",
            "active_jobs": 0,
            "capabilities": getattr(self, 'capabilities', self.debate_strengths),
            "last_analysis": "market_sentiment",
            "data_feeds": ["market_data", "economic_indicators", "technical_signals"],
            "perspective": self.perspective.value,
            "specialization": self.specialization,
            "debate_ready": True
        }