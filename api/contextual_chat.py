# api/contextual_chat.py
"""
Contextual Chat Service
=======================
Advanced contextual chat service that integrates with your multi-agent system
to provide intelligent, portfolio-aware conversations.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import uuid
import asyncio
from pydantic import BaseModel, Field

# Import your existing components
from agents.enhanced_orchestrator import EnhancedFinancialOrchestrator
from core.data_handler import get_market_data_for_portfolio
from db import crud
from api.schemas import User
from sqlalchemy.orm import Session

class ContextualChatRequest(BaseModel):
    """Enhanced chat request with contextual parameters"""
    message: str = Field(..., description="User's chat message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    context_mode: str = Field(default="auto", description="Context mode: 'auto', 'portfolio', 'market', 'minimal'")
    include_market_data: bool = Field(default=True, description="Include real-time market data")
    analysis_depth: str = Field(default="standard", description="Analysis depth: 'quick', 'standard', 'comprehensive'")
    personalization_level: str = Field(default="high", description="Personalization: 'low', 'medium', 'high'")
    preferred_agents: Optional[List[str]] = Field(None, description="Preferred agents for this query")
    max_response_time: int = Field(default=30, description="Maximum response time in seconds")

class SmartSuggestion(BaseModel):
    """Smart suggestion for follow-up actions"""
    suggestion_type: str  # 'question', 'action', 'analysis'
    suggestion_text: str
    priority: int = Field(ge=1, le=10)
    estimated_time: Optional[int] = None  # seconds
    requires_confirmation: bool = False

class ContextualChatResponse(BaseModel):
    """Enhanced chat response with contextual insights"""
    message: str
    conversation_id: str
    confidence_score: float = Field(ge=0, le=1)
    agents_used: List[str]
    context_sources: List[str]
    portfolio_insights: Optional[List[str]] = None
    market_insights: Optional[List[str]] = None
    smart_suggestions: List[SmartSuggestion] = Field(default_factory=list)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str

class ConversationContext:
    """Manages conversation context and history"""
    
    def __init__(self):
        self.conversations = {}  # conversation_id -> context
        self.user_preferences = {}  # user_id -> preferences
        self.context_cache = {}  # Cached context data
    
    def get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get context for a conversation"""
        return self.conversations.get(conversation_id, {
            "messages": [],
            "topics": [],
            "last_portfolio_analysis": None,
            "user_preferences": {},
            "context_embeddings": []
        })
    
    def update_conversation_context(self, conversation_id: str, context_update: Dict[str, Any]):
        """Update conversation context"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = self.get_conversation_context(conversation_id)
        
        # Merge context updates
        for key, value in context_update.items():
            if key == "messages":
                self.conversations[conversation_id]["messages"].extend(value)
                # Keep only last 20 messages for context
                self.conversations[conversation_id]["messages"] = \
                    self.conversations[conversation_id]["messages"][-20:]
            else:
                self.conversations[conversation_id][key] = value

class ContextualChatService:
    """Main contextual chat service"""
    
    def __init__(self):
        self.orchestrator = EnhancedFinancialOrchestrator()
        self.conversation_manager = ConversationContext()
        self.context_analyzers = {
            "portfolio": self._analyze_portfolio_context,
            "market": self._analyze_market_context,
            "risk": self._analyze_risk_context,
            "performance": self._analyze_performance_context
        }
    
    async def process_contextual_chat(
        self, 
        request: ContextualChatRequest, 
        user: User, 
        db: Session
    ) -> ContextualChatResponse:
        """Process a contextual chat request"""
        
        start_time = datetime.now()
        
        try:
            # Generate or get conversation ID
            conversation_id = request.conversation_id or self._generate_conversation_id()
            
            # Build comprehensive context
            context = await self._build_context(request, user, db, conversation_id)
            
            # Determine optimal agents for the query
            selected_agents = await self._select_optimal_agents(request, context)
            
            # Process with multi-agent orchestration
            orchestration_result = await self._orchestrate_response(
                request, context, selected_agents
            )
            
            # Extract insights and suggestions
            insights = self._extract_insights(orchestration_result, context)
            suggestions = self._generate_smart_suggestions(orchestration_result, context)
            
            # Update conversation context
            self._update_conversation_history(
                conversation_id, request.message, orchestration_result, context
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(orchestration_result, context)
            
            # Build response
            response = ContextualChatResponse(
                message=orchestration_result.get("final_response", "I apologize, but I couldn't generate a proper response."),
                conversation_id=conversation_id,
                confidence_score=confidence,
                agents_used=selected_agents,
                context_sources=context.get("sources", []),
                portfolio_insights=insights.get("portfolio", []),
                market_insights=insights.get("market", []),
                smart_suggestions=suggestions,
                response_metadata={
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "context_mode": request.context_mode,
                    "analysis_depth": request.analysis_depth,
                    "personalization_level": request.personalization_level
                },
                timestamp=datetime.now().isoformat()
            )
            
            return response
            
        except Exception as e:
            # Return error response
            return ContextualChatResponse(
                message=f"I encountered an issue processing your request: {str(e)}",
                conversation_id=request.conversation_id or self._generate_conversation_id(),
                confidence_score=0.0,
                agents_used=[],
                context_sources=["error"],
                timestamp=datetime.now().isoformat()
            )
    
    async def _build_context(
        self, 
        request: ContextualChatRequest, 
        user: User, 
        db: Session, 
        conversation_id: str
    ) -> Dict[str, Any]:
        """Build comprehensive context for the chat request"""
        
        context = {
            "user_id": user.id,
            "user_email": user.email,
            "conversation_id": conversation_id,
            "request_timestamp": datetime.now().isoformat(),
            "sources": []
        }
        
        # Get conversation history
        conversation_context = self.conversation_manager.get_conversation_context(conversation_id)
        context["conversation_history"] = conversation_context["messages"]
        context["conversation_topics"] = conversation_context["topics"]
        context["sources"].append("conversation_history")
        
        # Get user's portfolios if context requires it
        if request.context_mode in ["auto", "portfolio"] and request.include_market_data:
            user_portfolios = crud.get_user_portfolios(db=db, user_id=user.id)
            
            if user_portfolios:
                # Use primary portfolio (first one) for context
                primary_portfolio = user_portfolios[0]
                portfolio_data = get_market_data_for_portfolio(primary_portfolio.holdings)
                
                context["portfolio"] = {
                    "id": primary_portfolio.id,
                    "name": primary_portfolio.name,
                    "holdings_count": len(primary_portfolio.holdings),
                    "market_data": portfolio_data
                }
                context["sources"].append("portfolio_data")
        
        # Add market context if requested
        if request.context_mode in ["auto", "market"]:
            market_context = await self._get_market_context()
            context["market"] = market_context
            context["sources"].append("market_data")
        
        # Add user preferences
        if user.preferences:
            context["user_preferences"] = user.preferences
            context["sources"].append("user_preferences")
        
        # Analyze query intent
        context["query_intent"] = self._analyze_query_intent(request.message)
        context["sources"].append("query_analysis")
        
        return context
    
    async def _select_optimal_agents(
        self, 
        request: ContextualChatRequest, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Select optimal agents based on query and context"""
        
        if request.preferred_agents:
            return request.preferred_agents
        
        query_lower = request.message.lower()
        selected_agents = []
        
        # Always include the orchestrator for coordination
        selected_agents.append("EnhancedFinancialOrchestrator")
        
        # Analyze query content to select relevant agents
        if any(word in query_lower for word in ["risk", "volatility", "var", "drawdown", "correlation"]):
            selected_agents.append("QuantitativeAnalystAgent")
        
        if any(word in query_lower for word in ["tax", "harvest", "optimization", "deduction"]):
            selected_agents.append("TaxOptimizationAgent")
        
        if any(word in query_lower for word in ["market", "news", "economic", "sentiment", "outlook"]):
            selected_agents.append("MarketIntelligenceAgent")
        
        if any(word in query_lower for word in ["rebalance", "allocate", "diversify", "strategy"]):
            selected_agents.append("StrategyRebalancingAgent")
        
        if any(word in query_lower for word in ["options", "volatility", "hedge", "protection"]):
            selected_agents.append("OptionsAnalystAgent")
        
        # If no specific agents selected, use general financial analysis
        if len(selected_agents) == 1:  # Only orchestrator
            selected_agents.append("QuantitativeAnalystAgent")
        
        return selected_agents
    
    async def _orchestrate_response(
        self, 
        request: ContextualChatRequest, 
        context: Dict[str, Any], 
        selected_agents: List[str]
    ) -> Dict[str, Any]:
        """Orchestrate multi-agent response"""
        
        try:
            # Use your enhanced orchestrator
            result = await self.orchestrator.process_complex_query(
                query=request.message,
                context=context,
                analysis_depth=request.analysis_depth,
                max_response_time=request.max_response_time
            )
            
            return {
                "success": True,
                "final_response": result.get("summary", "Analysis completed successfully."),
                "agent_results": result.get("agent_results", {}),
                "orchestration_summary": result.get("orchestration_summary", {}),
                "confidence_indicators": result.get("confidence_indicators", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "final_response": f"I encountered an issue while analyzing your request: {str(e)}"
            }
    
    def _analyze_query_intent(self, message: str) -> Dict[str, Any]:
        """Analyze the intent behind the user's query"""
        
        message_lower = message.lower()
        
        intent = {
            "primary_intent": "general_inquiry",
            "confidence": 0.5,
            "entities": [],
            "action_required": False
        }
        
        # Detect primary intents
        if any(word in message_lower for word in ["analyze", "analysis", "review", "check"]):
            intent["primary_intent"] = "analysis_request"
            intent["confidence"] = 0.8
        
        elif any(word in message_lower for word in ["should i", "recommend", "suggest", "advice"]):
            intent["primary_intent"] = "recommendation_request"
            intent["confidence"] = 0.9
            intent["action_required"] = True
        
        elif any(word in message_lower for word in ["buy", "sell", "trade", "execute"]):
            intent["primary_intent"] = "action_request"
            intent["confidence"] = 0.95
            intent["action_required"] = True
        
        elif any(word in message_lower for word in ["explain", "what is", "how does", "why"]):
            intent["primary_intent"] = "explanation_request"
            intent["confidence"] = 0.85
        
        # Extract entities (ticker symbols, numbers, etc.)
        import re
        
        # Find ticker symbols
        tickers = re.findall(r'\b[A-Z]{1,5}\b', message)
        if tickers:
            intent["entities"].extend([{"type": "ticker", "value": ticker} for ticker in tickers])
        
        # Find percentages
        percentages = re.findall(r'\b\d+(?:\.\d+)?%', message)
        if percentages:
            intent["entities"].extend([{"type": "percentage", "value": pct} for pct in percentages])
        
        # Find dollar amounts
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', message)
        if amounts:
            intent["entities"].extend([{"type": "amount", "value": amt} for amt in amounts])
        
        return intent
    
    async def _get_market_context(self) -> Dict[str, Any]:
        """Get current market context"""
        
        # This would integrate with your market data sources
        # For now, return mock context
        return {
            "market_status": "open",
            "major_indices": {
                "SPY": {"price": 445.67, "change_pct": 0.8},
                "QQQ": {"price": 382.45, "change_pct": 1.2},
                "IWM": {"price": 198.34, "change_pct": -0.3}
            },
            "market_sentiment": "bullish",
            "vix": 18.5,
            "recent_news": [
                "Fed signals potential rate pause",
                "Tech earnings beat expectations",
                "Global supply chain improvements"
            ]
        }
    
    def _extract_insights(
        self, 
        orchestration_result: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Extract actionable insights from the orchestration result"""
        
        insights = {
            "portfolio": [],
            "market": [],
            "risk": [],
            "opportunities": []
        }
        
        if not orchestration_result.get("success"):
            return insights
        
        agent_results = orchestration_result.get("agent_results", {})
        
        # Extract portfolio insights
        if "QuantitativeAnalystAgent" in agent_results:
            quant_result = agent_results["QuantitativeAnalystAgent"]
            if quant_result.get("success") and quant_result.get("data"):
                data = quant_result["data"]
                
                # Risk insights
                if "risk_measures" in data:
                    risk_measures = data["risk_measures"]
                    if "95%" in risk_measures:
                        var_95 = risk_measures["95%"].get("var", 0)
                        if abs(var_95) > 0.03:  # 3% VaR threshold
                            insights["risk"].append(f"Portfolio VaR at 95% confidence: {var_95:.1%}")
                
                # Performance insights
                if "performance_stats" in data:
                    perf = data["performance_stats"]
                    annual_return = perf.get("annualized_return_pct", 0)
                    volatility = perf.get("annualized_volatility_pct", 0)
                    
                    if annual_return > 15:
                        insights["portfolio"].append("Strong portfolio performance with above-average returns")
                    elif annual_return < 5:
                        insights["opportunities"].append("Consider strategies to improve portfolio returns")
                    
                    if volatility > 20:
                        insights["risk"].append("Portfolio showing high volatility - consider risk management")
        
        # Extract market insights
        market_context = context.get("market", {})
        if market_context:
            sentiment = market_context.get("market_sentiment", "neutral")
            vix = market_context.get("vix", 20)
            
            if sentiment == "bullish" and vix < 20:
                insights["market"].append("Market conditions favorable for growth strategies")
            elif sentiment == "bearish" or vix > 25:
                insights["market"].append("Elevated market uncertainty - consider defensive positioning")
        
        return insights
    
    def _generate_smart_suggestions(
        self, 
        orchestration_result: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[SmartSuggestion]:
        """Generate smart follow-up suggestions"""
        
        suggestions = []
        
        # Always suggest portfolio review if we have portfolio context
        if "portfolio" in context:
            suggestions.append(SmartSuggestion(
                suggestion_type="analysis",
                suggestion_text="Run a comprehensive portfolio risk analysis",
                priority=7,
                estimated_time=45,
                requires_confirmation=False
            ))
        
        # Suggest rebalancing if allocation seems off
        query_intent = context.get("query_intent", {})
        if "allocation" in query_intent.get("entities", []):
            suggestions.append(SmartSuggestion(
                suggestion_type="action",
                suggestion_text="Generate portfolio rebalancing recommendations",
                priority=8,
                estimated_time=60,
                requires_confirmation=True
            ))
        
        # Suggest tax optimization near year-end
        current_month = datetime.now().month
        if current_month >= 10:  # October onwards
            suggestions.append(SmartSuggestion(
                suggestion_type="analysis",
                suggestion_text="Review tax-loss harvesting opportunities",
                priority=9,
                estimated_time=30,
                requires_confirmation=False
            ))
        
        # Context-based suggestions
        if orchestration_result.get("success"):
            agent_results = orchestration_result.get("agent_results", {})
            
            # If risk analysis was performed, suggest stress testing
            if "QuantitativeAnalystAgent" in agent_results:
                suggestions.append(SmartSuggestion(
                    suggestion_type="question",
                    suggestion_text="How would my portfolio perform in a market correction?",
                    priority=6,
                    estimated_time=90
                ))
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _calculate_confidence_score(
        self, 
        orchestration_result: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the response"""
        
        if not orchestration_result.get("success"):
            return 0.2
        
        confidence_factors = []
        
        # Base confidence from orchestration
        base_confidence = 0.7
        confidence_factors.append(base_confidence)
        
        # Portfolio context availability
        if "portfolio" in context:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Market context availability
        if "market" in context:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.7)
        
        # Agent results quality
        agent_results = orchestration_result.get("agent_results", {})
        successful_agents = sum(1 for result in agent_results.values() if result.get("success"))
        total_agents = len(agent_results)
        
        if total_agents > 0:
            agent_confidence = successful_agents / total_agents
            confidence_factors.append(agent_confidence)
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _update_conversation_history(
        self, 
        conversation_id: str, 
        user_message: str, 
        orchestration_result: Dict[str, Any], 
        context: Dict[str, Any]
    ):
        """Update conversation history and context"""
        
        timestamp = datetime.now().isoformat()
        
        # Add to conversation messages
        new_messages = [
            {
                "role": "user",
                "content": user_message,
                "timestamp": timestamp
            },
            {
                "role": "assistant", 
                "content": orchestration_result.get("final_response", ""),
                "timestamp": timestamp,
                "confidence_score": self._calculate_confidence_score(orchestration_result, context)
            }
        ]
        
        # Update conversation context
        context_update = {
            "messages": new_messages,
            "last_activity": timestamp
        }
        
        # Extract and update topics
        query_intent = context.get("query_intent", {})
        if query_intent.get("primary_intent"):
            if "topics" not in self.conversation_manager.conversations.get(conversation_id, {}):
                context_update["topics"] = []
            else:
                context_update["topics"] = self.conversation_manager.conversations[conversation_id]["topics"]
            
            if query_intent["primary_intent"] not in context_update["topics"]:
                context_update["topics"].append(query_intent["primary_intent"])
        
        self.conversation_manager.update_conversation_context(conversation_id, context_update)
    
    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID"""
        return f"conv_{uuid.uuid4().hex[:12]}"
    
    # Additional context analyzers
    async def _analyze_portfolio_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio-specific context"""
        # Implementation for portfolio context analysis
        pass
    
    async def _analyze_market_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market-specific context"""
        # Implementation for market context analysis
        pass
    
    async def _analyze_risk_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk-specific context"""
        # Implementation for risk context analysis
        pass
    
    async def _analyze_performance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance-specific context"""
        # Implementation for performance context analysis
        pass

# Export for use in other modules
__all__ = [
    "ContextualChatService", 
    "ContextualChatRequest", 
    "ContextualChatResponse",
    "SmartSuggestion"
]