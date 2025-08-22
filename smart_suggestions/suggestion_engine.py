# smart_suggestions/suggestion_engine.py
"""
Smart Suggestions Engine for Gertie.ai
=====================================
Intelligent suggestion system that analyzes portfolio context, market conditions,
and user behavior to provide contextual recommendations.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import math
from sqlalchemy.orm import Session

# Your existing imports
from db import crud, models
from db.models import PortfolioRiskSnapshot, User, Portfolio, Holding

class SuggestionType(Enum):
    AGENT_QUERY = "agent_query"
    WORKFLOW_TRIGGER = "workflow_trigger"
    DEBATE_TOPIC = "debate_topic"
    FOLLOW_UP = "follow_up"
    CONTEXTUAL = "contextual"

class Urgency(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class SmartSuggestion:
    id: str
    type: SuggestionType
    title: str
    description: str
    query: str
    target_agent: Optional[str] = None
    workflow_type: Optional[str] = None
    confidence: float = 0.5
    urgency: Urgency = Urgency.MEDIUM
    category: str = "General"
    reasoning: str = ""
    expected_outcome: str = ""
    icon: str = "💡"
    color: str = "blue"
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PortfolioContext:
    user_id: str
    portfolio_id: str
    total_value: float
    risk_score: float
    volatility: float
    holdings: List[Dict[str, Any]]
    risk_change_pct: float
    last_analysis: datetime
    performance: Dict[str, float]
    concentration_risk: float
    sector_allocation: Dict[str, float]

@dataclass
class MarketContext:
    vix_level: float
    market_trend: str
    sector_rotation: List[str]
    economic_indicators: Dict[str, float]
    fed_policy_stance: str
    volatility_regime: str

class SmartSuggestionEngine:
    """
    Core engine that generates intelligent suggestions based on:
    1. Portfolio analysis and risk metrics
    2. Market conditions and volatility
    3. User interaction patterns
    4. Recent workflow results
    5. Risk threshold breaches
    """
    
    def __init__(self):
        self.agent_capabilities = {
            "quantitative": ["risk_analysis", "statistical_modeling", "var_calculation"],
            "strategy": ["portfolio_optimization", "asset_allocation", "momentum_analysis"],
            "risk": ["hedging_strategies", "tail_risk", "volatility_analysis"],
            "tax": ["tax_optimization", "harvest_losses", "after_tax_returns"],
            "options": ["options_strategies", "volatility_trading", "income_generation"],
            "screener": ["stock_screening", "factor_analysis", "quality_metrics"],
            "tutor": ["education", "explanation", "concept_clarification"]
        }
        
        # Suggestion templates for different scenarios
        self.suggestion_templates = self._load_suggestion_templates()
        
    def generate_suggestions(
        self, 
        db: Session, 
        user: User, 
        portfolio_context: PortfolioContext,
        market_context: MarketContext,
        recent_interactions: List[Dict] = None
    ) -> List[SmartSuggestion]:
        """Generate personalized suggestions based on all available context"""
        
        suggestions = []
        
        # 1. Portfolio-based suggestions
        portfolio_suggestions = self._generate_portfolio_suggestions(
            portfolio_context, market_context
        )
        suggestions.extend(portfolio_suggestions)
        
        # 2. Risk-based suggestions
        risk_suggestions = self._generate_risk_suggestions(
            db, user, portfolio_context
        )
        suggestions.extend(risk_suggestions)
        
        # 3. Market-based suggestions
        market_suggestions = self._generate_market_suggestions(
            portfolio_context, market_context
        )
        suggestions.extend(market_suggestions)
        
        # 4. Behavioral/interaction-based suggestions
        if recent_interactions:
            behavioral_suggestions = self._generate_behavioral_suggestions(
                recent_interactions, portfolio_context
            )
            suggestions.extend(behavioral_suggestions)
        
        # 5. Time-based suggestions
        time_suggestions = self._generate_time_based_suggestions(
            db, user, portfolio_context
        )
        suggestions.extend(time_suggestions)
        
        # 6. Workflow follow-up suggestions
        followup_suggestions = self._generate_followup_suggestions(
            db, user
        )
        suggestions.extend(followup_suggestions)
        
        # Deduplicate and rank
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        ranked_suggestions = self._rank_suggestions(unique_suggestions, portfolio_context)
        
        return ranked_suggestions[:10]  # Return top 10
    
    def _generate_portfolio_suggestions(
        self, 
        portfolio_context: PortfolioContext,
        market_context: MarketContext
    ) -> List[SmartSuggestion]:
        """Generate suggestions based on portfolio characteristics"""
        
        suggestions = []
        
        # High risk score
        if portfolio_context.risk_score > 80:
            suggestions.append(SmartSuggestion(
                id="high_risk_portfolio",
                type=SuggestionType.AGENT_QUERY,
                title="Portfolio Risk Assessment",
                description=f"Your portfolio risk score is {portfolio_context.risk_score:.0f}. Get personalized risk reduction strategies.",
                query=f"My portfolio has a risk score of {portfolio_context.risk_score:.0f}. What specific steps should I take to reduce risk while maintaining growth potential?",
                target_agent="risk",
                confidence=0.95,
                urgency=Urgency.HIGH,
                category="Risk Management",
                reasoning=f"Risk score of {portfolio_context.risk_score:.0f} exceeds safe thresholds",
                expected_outcome="Specific hedging strategies and portfolio adjustments",
                icon="⚠️",
                color="red",
                metadata={"risk_score": portfolio_context.risk_score}
            ))
        
        # Concentration risk
        if portfolio_context.concentration_risk > 0.15:  # 15% in single position
            top_holding = max(portfolio_context.holdings, key=lambda x: x['weight'])
            suggestions.append(SmartSuggestion(
                id="concentration_risk",
                type=SuggestionType.WORKFLOW_TRIGGER,
                title="Diversification Analysis",
                description=f"{top_holding['ticker']} represents {top_holding['weight']*100:.1f}% of your portfolio.",
                query=f"Analyze concentration risk in my portfolio. {top_holding['ticker']} is {top_holding['weight']*100:.1f}% of holdings. Should I rebalance?",
                workflow_type="comprehensive_analysis",
                confidence=0.88,
                urgency=Urgency.MEDIUM,
                category="Portfolio Balance",
                reasoning="Single position exceeds 15% allocation threshold",
                expected_outcome="Rebalancing recommendations and diversification strategies",
                icon="⚖️",
                color="orange",
                metadata={"concentrated_ticker": top_holding['ticker'], "weight": top_holding['weight']}
            ))
        
        # Poor performance
        if portfolio_context.performance.get('sharpe_ratio', 0) < 0.5:
            suggestions.append(SmartSuggestion(
                id="poor_performance",
                type=SuggestionType.AGENT_QUERY,
                title="Performance Optimization",
                description="Your risk-adjusted returns could be improved.",
                query=f"My portfolio's Sharpe ratio is {portfolio_context.performance.get('sharpe_ratio', 0):.2f}. How can I improve risk-adjusted returns?",
                target_agent="quantitative",
                confidence=0.82,
                urgency=Urgency.MEDIUM,
                category="Performance",
                reasoning="Sharpe ratio below 0.5 indicates poor risk-adjusted returns",
                expected_outcome="Portfolio optimization strategies",
                icon="📈",
                color="blue"
            ))
        
        # Sector imbalance
        tech_allocation = portfolio_context.sector_allocation.get('Technology', 0)
        if tech_allocation > 0.4:  # Over 40% in tech
            suggestions.append(SmartSuggestion(
                id="tech_concentration",
                type=SuggestionType.DEBATE_TOPIC,
                title="Technology Sector Concentration",
                description=f"Technology represents {tech_allocation*100:.1f}% of your portfolio.",
                query=f"My portfolio is {tech_allocation*100:.1f}% allocated to technology stocks. Is this concentration appropriate given current market conditions?",
                confidence=0.78,
                urgency=Urgency.MEDIUM,
                category="Sector Allocation",
                reasoning="Technology allocation exceeds typical diversification guidelines",
                expected_outcome="Multi-agent debate on sector concentration risks and opportunities",
                icon="💻",
                color="purple"
            ))
        
        return suggestions
    
    def _generate_risk_suggestions(
        self, 
        db: Session, 
        user: User, 
        portfolio_context: PortfolioContext
    ) -> List[SmartSuggestion]:
        """Generate risk-based suggestions"""
        
        suggestions = []
        
        # Recent risk change
        if abs(portfolio_context.risk_change_pct) > 15:
            direction = "increased" if portfolio_context.risk_change_pct > 0 else "decreased"
            icon = "📈" if portfolio_context.risk_change_pct > 0 else "📉"
            color = "red" if portfolio_context.risk_change_pct > 0 else "green"
            
            suggestions.append(SmartSuggestion(
                id="risk_change_alert",
                type=SuggestionType.DEBATE_TOPIC,
                title=f"Risk {direction.title()} Significantly",
                description=f"Portfolio risk has {direction} by {abs(portfolio_context.risk_change_pct):.1f}%.",
                query=f"My portfolio risk has {direction} by {abs(portfolio_context.risk_change_pct):.1f}%. What should I do about this change?",
                confidence=0.92,
                urgency=Urgency.HIGH,
                category="Risk Alert",
                reasoning=f"Risk change of {portfolio_context.risk_change_pct:.1f}% exceeds 15% threshold",
                expected_outcome="Multi-agent debate on risk management approach",
                icon=icon,
                color=color,
                metadata={"risk_change_pct": portfolio_context.risk_change_pct}
            ))
        
        # High volatility
        if portfolio_context.volatility > 0.25:  # 25% annualized volatility
            suggestions.append(SmartSuggestion(
                id="high_volatility",
                type=SuggestionType.AGENT_QUERY,
                title="Volatility Management",
                description=f"Portfolio volatility is {portfolio_context.volatility*100:.1f}%, above typical ranges.",
                query=f"My portfolio volatility is {portfolio_context.volatility*100:.1f}%. What strategies can reduce volatility while maintaining returns?",
                target_agent="risk",
                confidence=0.85,
                urgency=Urgency.MEDIUM,
                category="Risk Management",
                reasoning="Portfolio volatility exceeds 25% threshold",
                expected_outcome="Volatility reduction strategies",
                icon="📊",
                color="orange"
            ))
        
        # Check for recent risk threshold breaches
        recent_breaches = crud.get_threshold_breaches(db, str(user.id), days=7)
        if recent_breaches:
            latest_breach = recent_breaches[0]
            suggestions.append(SmartSuggestion(
                id="recent_breach",
                type=SuggestionType.WORKFLOW_TRIGGER,
                title="Risk Threshold Breach Follow-up",
                description="Recent risk threshold breach needs attention.",
                query="Analyze the recent risk threshold breach and recommend immediate actions to bring portfolio back within acceptable risk parameters",
                workflow_type="risk_mitigation",
                confidence=0.90,
                urgency=Urgency.HIGH,
                category="Risk Alert",
                reasoning="Risk threshold breached within last 7 days",
                expected_outcome="Immediate risk mitigation plan",
                icon="🚨",
                color="red",
                metadata={"breach_date": latest_breach.snapshot_date.isoformat()}
            ))
        
        return suggestions
    
    def _generate_market_suggestions(
        self, 
        portfolio_context: PortfolioContext,
        market_context: MarketContext
    ) -> List[SmartSuggestion]:
        """Generate market condition-based suggestions"""
        
        suggestions = []
        
        # High VIX environment
        if market_context.vix_level > 25:
            suggestions.append(SmartSuggestion(
                id="high_vix_hedging",
                type=SuggestionType.AGENT_QUERY,
                title="Volatility Protection",
                description=f"VIX at {market_context.vix_level:.1f} suggests market uncertainty.",
                query=f"VIX is at {market_context.vix_level:.1f}. What hedging strategies should I implement to protect my portfolio?",
                target_agent="risk",
                confidence=0.90,
                urgency=Urgency.HIGH,
                category="Market Protection",
                reasoning="VIX above 25 indicates elevated market stress",
                expected_outcome="Hedging strategies and protective positions",
                icon="🛡️",
                color="red"
            ))
        
        # Sector rotation opportunity
        if market_context.sector_rotation:
            suggestions.append(SmartSuggestion(
                id="sector_rotation",
                type=SuggestionType.WORKFLOW_TRIGGER,
                title="Sector Rotation Strategy",
                description=f"Active rotation detected in {', '.join(market_context.sector_rotation)} sectors.",
                query=f"Analyze current sector rotation trends. Should I adjust my portfolio allocation to capture opportunities in {', '.join(market_context.sector_rotation)}?",
                workflow_type="sector_analysis",
                confidence=0.75,
                urgency=Urgency.MEDIUM,
                category="Market Timing",
                reasoning="Multiple sectors showing rotation signals",
                expected_outcome="Sector allocation recommendations",
                icon="🔄",
                color="green"
            ))
        
        # Federal Reserve policy implications
        if market_context.fed_policy_stance in ["hawkish", "dovish"]:
            policy_impact = "rising rates may pressure growth stocks" if market_context.fed_policy_stance == "hawkish" else "falling rates may benefit growth assets"
            
            suggestions.append(SmartSuggestion(
                id="fed_policy_adjustment",
                type=SuggestionType.DEBATE_TOPIC,
                title="Fed Policy Portfolio Impact",
                description=f"Current {market_context.fed_policy_stance} Fed stance may affect your portfolio.",
                query=f"Given the Federal Reserve's {market_context.fed_policy_stance} policy stance, how should I adjust my portfolio allocation? {policy_impact}",
                confidence=0.80,
                urgency=Urgency.MEDIUM,
                category="Monetary Policy",
                reasoning=f"Fed policy stance is {market_context.fed_policy_stance}",
                expected_outcome="Multi-agent analysis of policy impact on portfolio",
                icon="🏛️",
                color="blue"
            ))
        
        return suggestions
    
    def _generate_behavioral_suggestions(
        self, 
        recent_interactions: List[Dict],
        portfolio_context: PortfolioContext
    ) -> List[SmartSuggestion]:
        """Generate suggestions based on user behavior patterns"""
        
        suggestions = []
        
        # Analyze agent usage patterns
        agent_usage = {}
        for interaction in recent_interactions[-10:]:  # Last 10 interactions
            agent = interaction.get('agent_type', 'unknown')
            agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        # Suggest underutilized agents
        all_agents = set(self.agent_capabilities.keys())
        used_agents = set(agent_usage.keys())
        underutilized = all_agents - used_agents
        
        if underutilized:
            agent_suggestions = {
                "tax": "Analyze tax optimization opportunities in my portfolio",
                "options": "Should I use options strategies for income or protection?",
                "screener": "Find quality stocks that complement my current holdings",
                "tutor": "Explain portfolio concepts I should understand better"
            }
            
            for agent in list(underutilized)[:2]:  # Suggest max 2 underutilized agents
                if agent in agent_suggestions:
                    suggestions.append(SmartSuggestion(
                        id=f"underutilized_{agent}",
                        type=SuggestionType.AGENT_QUERY,
                        title=f"Explore {agent.title()} Analysis",
                        description=f"You haven't consulted our {agent} expert recently.",
                        query=agent_suggestions[agent],
                        target_agent=agent,
                        confidence=0.65,
                        urgency=Urgency.LOW,
                        category="Discovery",
                        reasoning=f"{agent} agent not recently consulted",
                        expected_outcome=f"Insights from {agent} perspective",
                        icon="💡",
                        color="purple"
                    ))
        
        # Suggest workflow if user only uses single agents
        if len(agent_usage) >= 3 and not any('workflow' in str(i) for i in recent_interactions):
            suggestions.append(SmartSuggestion(
                id="try_workflow",
                type=SuggestionType.WORKFLOW_TRIGGER,
                title="Try Multi-Agent Workflow",
                description="Get comprehensive analysis from multiple AI experts working together.",
                query="Perform comprehensive portfolio analysis using multi-agent workflow",
                workflow_type="comprehensive_analysis",
                confidence=0.70,
                urgency=Urgency.LOW,
                category="Discovery",
                reasoning="User frequently uses individual agents but hasn't tried workflow",
                expected_outcome="Comprehensive multi-stage analysis",
                icon="⚙️",
                color="blue"
            ))
        
        return suggestions
    
    def _generate_time_based_suggestions(
        self, 
        db: Session, 
        user: User, 
        portfolio_context: PortfolioContext
    ) -> List[SmartSuggestion]:
        """Generate time-sensitive suggestions"""
        
        suggestions = []
        
        # Stale analysis
        days_since_analysis = (datetime.now() - portfolio_context.last_analysis).days
        if days_since_analysis > 30:
            suggestions.append(SmartSuggestion(
                id="stale_analysis",
                type=SuggestionType.WORKFLOW_TRIGGER,
                title="Portfolio Health Check",
                description=f"Last comprehensive analysis was {days_since_analysis} days ago.",
                query="Perform comprehensive portfolio health check and identify any issues or opportunities",
                workflow_type="health_check",
                confidence=0.70,
                urgency=Urgency.MEDIUM,
                category="Maintenance",
                reasoning=f"Analysis is {days_since_analysis} days old",
                expected_outcome="Updated portfolio assessment",
                icon="🏥",
                color="blue"
            ))
        
        # End of month/quarter suggestions
        now = datetime.now()
        if now.day >= 25:  # Near end of month
            suggestions.append(SmartSuggestion(
                id="month_end_review",
                type=SuggestionType.AGENT_QUERY,
                title="Month-End Portfolio Review",
                description="Consider month-end rebalancing and tax optimization.",
                query="Analyze my portfolio for month-end rebalancing opportunities and tax-loss harvesting potential",
                target_agent="tax",
                confidence=0.60,
                urgency=Urgency.LOW,
                category="Maintenance",
                reasoning="Approaching end of month",
                expected_outcome="Month-end optimization recommendations",
                icon="📅",
                color="green"
            ))
        
        return suggestions
    
    def _generate_followup_suggestions(
        self, 
        db: Session, 
        user: User
    ) -> List[SmartSuggestion]:
        """Generate follow-up suggestions based on recent activities"""
        
        suggestions = []
        
        # Check for recent workflows that might need follow-up
        # This would integrate with your workflow tracking system
        # For now, we'll create a placeholder
        
        suggestions.append(SmartSuggestion(
            id="implement_recommendations",
            type=SuggestionType.FOLLOW_UP,
            title="Implement Recent Recommendations",
            description="You have unimplemented recommendations from recent analysis.",
            query="Help me create an implementation plan for the recommendations from my recent analysis",
            confidence=0.80,
            urgency=Urgency.MEDIUM,
            category="Implementation",
            reasoning="Recent workflow completed with actionable recommendations",
            expected_outcome="Step-by-step implementation guide",
            icon="✅",
            color="green"
        ))
        
        return suggestions
    
    def _deduplicate_suggestions(self, suggestions: List[SmartSuggestion]) -> List[SmartSuggestion]:
        """Remove duplicate suggestions based on ID and similarity"""
        
        seen_ids = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            if suggestion.id not in seen_ids:
                seen_ids.add(suggestion.id)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _rank_suggestions(
        self, 
        suggestions: List[SmartSuggestion],
        portfolio_context: PortfolioContext
    ) -> List[SmartSuggestion]:
        """Rank suggestions by relevance and urgency"""
        
        urgency_weights = {
            Urgency.HIGH: 3,
            Urgency.MEDIUM: 2,
            Urgency.LOW: 1
        }
        
        def calculate_score(suggestion: SmartSuggestion) -> float:
            urgency_score = urgency_weights[suggestion.urgency]
            confidence_score = suggestion.confidence
            
            # Boost risk-related suggestions if portfolio is high risk
            risk_boost = 0.2 if (portfolio_context.risk_score > 75 and 
                               'risk' in suggestion.category.lower()) else 0
            
            return confidence_score * 0.6 + urgency_score * 0.3 + risk_boost
        
        return sorted(suggestions, key=calculate_score, reverse=True)
    
    def _load_suggestion_templates(self) -> Dict[str, Dict]:
        """Load suggestion templates for different scenarios"""
        
        # This could be loaded from a configuration file or database
        return {
            "high_risk": {
                "title": "Portfolio Risk Assessment",
                "category": "Risk Management",
                "icon": "⚠️",
                "color": "red"
            },
            "concentration": {
                "title": "Diversification Analysis", 
                "category": "Portfolio Balance",
                "icon": "⚖️",
                "color": "orange"
            },
            # Add more templates as needed
        }

# =============================================================================
# API INTEGRATION FUNCTIONS
# =============================================================================

def get_portfolio_context(db: Session, user: User) -> Optional[PortfolioContext]:
    """Extract portfolio context from database"""
    
    # Get user's portfolios
    portfolios = crud.get_user_portfolios(db, user.id)
    if not portfolios:
        return None
    
    # Use the first portfolio (or implement portfolio selection logic)
    portfolio = portfolios[0]
    
    # Get latest risk snapshot
    latest_risk = crud.get_latest_risk_snapshot(db, str(user.id), str(portfolio.id))
    
    # Calculate portfolio metrics
    total_value = sum(holding.shares * (holding.purchase_price or 0) for holding in portfolio.holdings)
    
    # Get holdings data
    holdings_data = []
    sector_allocation = {}
    
    for holding in portfolio.holdings:
        if holding.asset:
            weight = (holding.shares * (holding.purchase_price or 0)) / total_value if total_value > 0 else 0
            
            # Simplified sector mapping (you'd want a proper sector lookup)
            sector = _get_sector_for_ticker(holding.asset.ticker)
            
            holdings_data.append({
                'ticker': holding.asset.ticker,
                'weight': weight,
                'sector': sector,
                'value': holding.shares * (holding.purchase_price or 0)
            })
            
            sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
    
    # Calculate concentration risk
    concentration_risk = max([h['weight'] for h in holdings_data]) if holdings_data else 0
    
    # Get risk change percentage
    risk_change_pct = latest_risk.risk_score_change_pct if latest_risk else 0
    
    # Get performance metrics
    performance = {
        'sharpe_ratio': latest_risk.sharpe_ratio if latest_risk else 0,
        'ytd': 0,  # You'd calculate this from historical data
        'max_drawdown': latest_risk.max_drawdown if latest_risk else 0
    }
    
    return PortfolioContext(
        user_id=str(user.id),
        portfolio_id=str(portfolio.id),
        total_value=total_value,
        risk_score=latest_risk.risk_score if latest_risk else 50,
        volatility=latest_risk.volatility if latest_risk else 0.15,
        holdings=holdings_data,
        risk_change_pct=risk_change_pct,
        last_analysis=latest_risk.snapshot_date if latest_risk else datetime.now() - timedelta(days=30),
        performance=performance,
        concentration_risk=concentration_risk,
        sector_allocation=sector_allocation
    )

def get_market_context() -> MarketContext:
    """Get current market context (would integrate with market data APIs)"""
    
    # This would fetch real market data
    # For now, return mock data
    return MarketContext(
        vix_level=22.5,
        market_trend="neutral",
        sector_rotation=["Healthcare", "Technology"],
        economic_indicators={
            "fed_funds_rate": 5.25,
            "inflation_rate": 3.2,
            "unemployment": 3.8
        },
        fed_policy_stance="neutral",
        volatility_regime="normal"
    )

def _get_sector_for_ticker(ticker: str) -> str:
    """Simple sector mapping (you'd use a proper sector classification service)"""
    
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    healthcare_stocks = ['JNJ', 'PFE', 'UNH', 'ABBV']
    financial_stocks = ['JPM', 'BAC', 'WFC', 'GS']
    
    if ticker in tech_stocks:
        return 'Technology'
    elif ticker in healthcare_stocks:
        return 'Healthcare'
    elif ticker in financial_stocks:
        return 'Financial'
    elif ticker in ['SPY', 'QQQ', 'IWM']:
        return 'Diversified'
    elif ticker in ['BND', 'TLT']:
        return 'Bonds'
    else:
        return 'Other'


# =============================================================================
# MAIN API ENDPOINT IMPLEMENTATIONS
# =============================================================================

def generate_smart_suggestions_for_user(
    db: Session, 
    user: User,
    recent_interactions: List[Dict] = None
) -> List[Dict[str, Any]]:
    """Main function to generate smart suggestions for a user"""
    
    # Get portfolio context
    portfolio_context = get_portfolio_context(db, user)
    if not portfolio_context:
        return []
    
    # Get market context
    market_context = get_market_context()
    
    # Initialize suggestion engine
    engine = SmartSuggestionEngine()
    
    # Generate suggestions
    suggestions = engine.generate_suggestions(
        db=db,
        user=user,
        portfolio_context=portfolio_context,
        market_context=market_context,
        recent_interactions=recent_interactions or []
    )
    
    # Convert to API response format
    return [
        {
            **asdict(suggestion),
            'type': suggestion.type.value,
            'urgency': suggestion.urgency.value
        }
        for suggestion in suggestions
    ]