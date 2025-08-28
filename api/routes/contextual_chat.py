# api/routes/contextual_chat.py
"""
Contextual Chat API Routes
==========================
FastAPI routes for the contextual chat service
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from api.contextual_chat import (
    ContextualChatService, 
    ContextualChatRequest, 
    ContextualChatResponse
)
from api.schemas import User
from api.routes.auth import get_current_user
from db.session import get_db

# Initialize the contextual chat service
contextual_chat_service = None

def get_contextual_chat_service():
    """Get the contextual chat service instance"""
    global contextual_chat_service
    if contextual_chat_service is None:
        # Import here to avoid circular imports
        from main import orchestrator
        contextual_chat_service = ContextualChatService(orchestrator=orchestrator)
    return contextual_chat_service

# Router for contextual chat endpoints
router = APIRouter(prefix="/chat", tags=["Contextual Chat"])

@router.post("/message", response_model=ContextualChatResponse)
async def send_chat_message(
    request: ContextualChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    💬 Send Contextual Chat Message
    
    Process a chat message with full context awareness including portfolio data,
    market conditions, and conversation history.
    """
    
    try:
        service = get_contextual_chat_service()
        response = await service.process_contextual_chat(
            request, current_user, db
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat message: {str(e)}"
        )

@router.get("/conversations/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    limit: int = Query(20, ge=1, le=100, description="Number of messages to retrieve"),
    current_user: User = Depends(get_current_user)
):
    """
    📜 Get Conversation History
    
    Retrieve the history of a specific conversation.
    """
    
    try:
        service = get_contextual_chat_service()
        conversation_context = service.conversation_manager.get_conversation_context(
            conversation_id
        )
        
        messages = conversation_context.get("messages", [])
        
        # Return latest messages up to limit
        return {
            "conversation_id": conversation_id,
            "messages": messages[-limit:],
            "total_messages": len(messages),
            "topics": conversation_context.get("topics", []),
            "last_activity": conversation_context.get("last_activity"),
            "has_more": len(messages) > limit
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )

@router.get("/conversations")
async def get_user_conversations(
    limit: int = Query(10, ge=1, le=50, description="Number of conversations to retrieve"),
    include_empty: bool = Query(False, description="Include conversations with no messages"),
    current_user: User = Depends(get_current_user)
):
    """
    📋 Get User's Conversations
    
    Retrieve a list of the user's recent conversations.
    """
    
    try:
        service = get_contextual_chat_service()
        # Filter conversations for this user
        user_conversations = []
        
        for conv_id, context in service.conversation_manager.conversations.items():
            # You would normally check user ownership here
            # For now, we'll return all conversations (implement user filtering in production)
            
            if not include_empty and not context.get("messages"):
                continue
            
            user_conversations.append({
                "conversation_id": conv_id,
                "message_count": len(context.get("messages", [])),
                "topics": context.get("topics", []),
                "last_activity": context.get("last_activity"),
                "preview": context.get("messages", [])[-1].get("content", "")[:100] if context.get("messages") else ""
            })
        
        # Sort by last activity
        user_conversations.sort(
            key=lambda x: x.get("last_activity", ""), 
            reverse=True
        )
        
        return {
            "conversations": user_conversations[:limit],
            "total_count": len(user_conversations),
            "has_more": len(user_conversations) > limit
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversations: {str(e)}"
        )

@router.post("/conversations/{conversation_id}/context")
async def update_conversation_context(
    conversation_id: str,
    context_update: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    🔄 Update Conversation Context
    
    Update the context for a specific conversation.
    """
    
    try:
        service = get_contextual_chat_service()
        service.conversation_manager.update_conversation_context(
            conversation_id, context_update
        )
        
        return {
            "status": "updated",
            "conversation_id": conversation_id,
            "message": "Conversation context updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update conversation context: {str(e)}"
        )

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    🗑️ Delete Conversation
    
    Delete a specific conversation and its history.
    """
    
    try:
        service = get_contextual_chat_service()
        if conversation_id in service.conversation_manager.conversations:
            del service.conversation_manager.conversations[conversation_id]
            
            return {
                "status": "deleted",
                "conversation_id": conversation_id,
                "message": "Conversation deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete conversation: {str(e)}"
        )

@router.post("/quick-analysis")
async def quick_portfolio_analysis(
    query: str = Query(..., description="Quick analysis query"),
    portfolio_id: Optional[int] = Query(None, description="Specific portfolio ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    ⚡ Quick Portfolio Analysis
    
    Get a quick analysis response without creating a full conversation.
    """
    
    try:
        service = get_contextual_chat_service()
        # Create a quick analysis request
        request = ContextualChatRequest(
            message=query,
            context_mode="portfolio",
            analysis_depth="quick",
            max_response_time=15  # 15 seconds for quick analysis
        )
        
        response = await service.process_contextual_chat(
            request, current_user, db
        )
        
        # Return simplified response for quick analysis
        return {
            "message": response.message,
            "confidence_score": response.confidence_score,
            "key_insights": response.portfolio_insights[:3] if response.portfolio_insights else [],
            "quick_suggestions": [
                sugg.suggestion_text for sugg in response.smart_suggestions[:2]
            ],
            "processing_time_ms": response.response_metadata.get("processing_time_ms", 0)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform quick analysis: {str(e)}"
        )

@router.get("/suggestions/templates")
async def get_chat_templates():
    """
    📝 Get Chat Templates
    
    Retrieve pre-defined chat templates for common queries.
    """
    
    templates = {
        "portfolio_analysis": {
            "name": "Portfolio Analysis",
            "description": "Get comprehensive portfolio analysis",
            "template": "Analyze my portfolio's risk profile, performance, and provide optimization recommendations",
            "context_mode": "portfolio",
            "analysis_depth": "comprehensive"
        },
        "risk_assessment": {
            "name": "Risk Assessment",
            "description": "Assess portfolio risk levels",
            "template": "What is my current portfolio risk level and how can I manage it better?",
            "context_mode": "portfolio", 
            "analysis_depth": "standard"
        },
        "market_outlook": {
            "name": "Market Outlook",
            "description": "Get current market analysis",
            "template": "What's the current market outlook and how might it affect my investments?",
            "context_mode": "market",
            "analysis_depth": "standard"
        },
        "tax_optimization": {
            "name": "Tax Optimization",
            "description": "Review tax optimization opportunities",
            "template": "What tax optimization strategies should I consider for my portfolio?",
            "context_mode": "portfolio",
            "analysis_depth": "comprehensive"
        },
        "rebalancing": {
            "name": "Portfolio Rebalancing",
            "description": "Get rebalancing recommendations",
            "template": "Should I rebalance my portfolio and what changes would you recommend?",
            "context_mode": "portfolio",
            "analysis_depth": "standard"
        },
        "sector_analysis": {
            "name": "Sector Analysis",
            "description": "Analyze sector allocation and opportunities",
            "template": "Analyze my sector allocation and identify opportunities or risks",
            "context_mode": "portfolio",
            "analysis_depth": "comprehensive"
        }
    }
    
    return {"templates": templates}

@router.post("/templates/{template_id}")
async def use_chat_template(
    template_id: str,
    custom_parameters: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    🎯 Use Chat Template
    
    Execute a predefined chat template with optional custom parameters.
    """
    
    try:
        service = get_contextual_chat_service()
        # Get template configuration
        templates_response = await get_chat_templates()
        template = templates_response["templates"].get(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        # Create request from template
        message = template["template"]
        
        # Apply custom parameters if provided
        if custom_parameters:
            try:
                message = message.format(**custom_parameters)
            except KeyError as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required parameter: {e}"
                )
        
        request = ContextualChatRequest(
            message=message,
            conversation_id=conversation_id,
            context_mode=template["context_mode"],
            analysis_depth=template["analysis_depth"]
        )
        
        response = await service.process_contextual_chat(
            request, current_user, db
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute template: {str(e)}"
        )

@router.get("/agent-capabilities")
async def get_agent_capabilities():
    """
    🤖 Get Agent Capabilities
    
    Retrieve information about available agents and their capabilities.
    """
    
    try:
        service = get_contextual_chat_service()
        # Get agent capabilities from orchestrator
        orchestrator = service.orchestrator
        
        # This would typically come from your agent registry
        agent_capabilities = {
            "QuantitativeAnalystAgent": {
                "description": "Advanced quantitative analysis and risk modeling",
                "capabilities": [
                    "Portfolio risk analysis",
                    "VaR calculations", 
                    "Performance attribution",
                    "Correlation analysis",
                    "Stress testing"
                ],
                "response_time": "15-30 seconds",
                "confidence_level": "high"
            },
            "TaxOptimizationAgent": {
                "description": "Tax-efficient investment strategies",
                "capabilities": [
                    "Tax-loss harvesting",
                    "Asset location optimization",
                    "Tax-efficient rebalancing",
                    "Year-end tax planning"
                ],
                "response_time": "10-20 seconds",
                "confidence_level": "high"
            },
            "MarketIntelligenceAgent": {
                "description": "Market analysis and economic insights",
                "capabilities": [
                    "Market sentiment analysis",
                    "Economic indicator tracking",
                    "Sector analysis",
                    "News impact assessment"
                ],
                "response_time": "5-15 seconds",
                "confidence_level": "medium"
            },
            "StrategyRebalancingAgent": {
                "description": "Portfolio optimization and rebalancing",
                "capabilities": [
                    "Asset allocation optimization",
                    "Rebalancing strategies",
                    "Risk-adjusted portfolios",
                    "Efficient frontier analysis"
                ],
                "response_time": "20-45 seconds",
                "confidence_level": "high"
            },
            "OptionsAnalystAgent": {
                "description": "Options strategies and volatility analysis",
                "capabilities": [
                    "Options strategy design",
                    "Volatility analysis",
                    "Greeks calculations",
                    "Risk management with options"
                ],
                "response_time": "15-30 seconds",
                "confidence_level": "medium-high"
            }
        }
        
        return {
            "available_agents": agent_capabilities,
            "total_agents": len(agent_capabilities),
            "selection_strategy": "automatic based on query content",
            "max_concurrent_agents": 3
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent capabilities: {str(e)}"
        )

@router.get("/analytics/usage")
async def get_chat_analytics(
    days: int = Query(30, ge=1, le=90, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user)
):
    """
    📊 Get Chat Usage Analytics
    
    Analyze chat usage patterns and effectiveness.
    """
    
    try:
        service = get_contextual_chat_service()
        # Mock analytics data - implement actual tracking in production
        cutoff_date = datetime.now() - timedelta(days=days)
        
        analytics = {
            "period_days": days,
            "total_conversations": 47,
            "total_messages": 312,
            "avg_messages_per_conversation": 6.6,
            "most_common_intents": {
                "analysis_request": 28,
                "recommendation_request": 15,
                "explanation_request": 12,
                "action_request": 8
            },
            "agent_usage": {
                "QuantitativeAnalystAgent": 89,
                "MarketIntelligenceAgent": 56,
                "TaxOptimizationAgent": 34,
                "StrategyRebalancingAgent": 28,
                "OptionsAnalystAgent": 15
            },
            "avg_confidence_score": 0.84,
            "avg_response_time_ms": 2850,
            "user_satisfaction_indicators": {
                "high_confidence_responses": 78,
                "follow_up_questions": 23,
                "template_usage": 12
            },
            "popular_topics": [
                "portfolio risk analysis",
                "market outlook",
                "tax optimization", 
                "rebalancing strategies",
                "sector allocation"
            ],
            "context_effectiveness": {
                "portfolio_context_used": 0.92,
                "market_context_used": 0.76,
                "conversation_history_relevant": 0.68
            }
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve chat analytics: {str(e)}"
        )

@router.post("/feedback")
async def submit_chat_feedback(
    conversation_id: str,
    message_index: int,
    feedback: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    👍 Submit Chat Feedback
    
    Submit feedback on chat responses to improve future interactions.
    """
    
    try:
        service = get_contextual_chat_service()
        # Validate conversation exists
        conversation_context = service.conversation_manager.get_conversation_context(
            conversation_id
        )
        
        if not conversation_context.get("messages"):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if message_index >= len(conversation_context["messages"]):
            raise HTTPException(status_code=400, detail="Message index out of range")
        
        # Store feedback (implement actual storage in production)
        feedback_record = {
            "conversation_id": conversation_id,
            "message_index": message_index,
            "user_id": current_user.id,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        # You would store this in a feedback database table
        print(f"Feedback received: {feedback_record}")
        
        return {
            "status": "received",
            "message": "Thank you for your feedback",
            "feedback_id": f"fb_{uuid.uuid4().hex[:8]}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )

@router.get("/context/portfolio/{portfolio_id}")
async def get_portfolio_context_summary(
    portfolio_id: int,
    include_market_data: bool = Query(True, description="Include current market data"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    📈 Get Portfolio Context Summary
    
    Get a summary of portfolio context that would be used in chat.
    """
    
    try:
        service = get_contextual_chat_service()
        # Verify portfolio ownership
        from db import crud
        user_portfolios = crud.get_user_portfolios(db=db, user_id=current_user.id)
        portfolio = next((p for p in user_portfolios if p.id == portfolio_id), None)
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        context_summary = {
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "holdings_count": len(portfolio.holdings),
            "context_availability": {
                "portfolio_data": True,
                "market_data": include_market_data,
                "historical_data": False,  # Would check actual availability
                "benchmark_data": False
            }
        }
        
        if include_market_data:
            try:
                from core.data_handler import get_market_data_for_portfolio
                market_data = get_market_data_for_portfolio(portfolio.holdings)
                
                context_summary["market_context"] = {
                    "total_value": market_data.get("total_value", 0),
                    "holdings_with_prices": len(market_data.get("holdings_with_values", [])),
                    "data_freshness": "current",
                    "last_updated": datetime.now().isoformat()
                }
            except Exception as e:
                context_summary["market_context"] = {
                    "error": f"Failed to get market data: {str(e)}"
                }
        
        return context_summary
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get portfolio context: {str(e)}"
        )

@router.post("/context/optimize")
async def optimize_context_settings(
    optimization_preferences: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    ⚙️ Optimize Context Settings
    
    Optimize context settings based on user preferences and usage patterns.
    """
    
    try:
        service = get_contextual_chat_service()
        # Store user's context optimization preferences
        service.conversation_manager.user_preferences[current_user.id] = {
            "context_optimization": optimization_preferences,
            "updated_at": datetime.now().isoformat()
        }
        
        # Suggest optimal settings based on preferences
        suggested_settings = {
            "context_mode": "auto",  # Default
            "analysis_depth": "standard",
            "personalization_level": "high",
            "max_response_time": 30
        }
        
        # Adjust based on preferences
        if optimization_preferences.get("prefer_speed"):
            suggested_settings["analysis_depth"] = "quick"
            suggested_settings["max_response_time"] = 15
        
        if optimization_preferences.get("prefer_accuracy"):
            suggested_settings["analysis_depth"] = "comprehensive"
            suggested_settings["max_response_time"] = 60
        
        if optimization_preferences.get("minimal_context"):
            suggested_settings["context_mode"] = "minimal"
            suggested_settings["personalization_level"] = "low"
        
        return {
            "status": "optimized",
            "suggested_settings": suggested_settings,
            "optimization_applied": True,
            "message": "Context settings optimized based on your preferences"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize context settings: {str(e)}"
        )

@router.get("/health")
async def chat_service_health():
    """
    🏥 Chat Service Health Check
    
    Check the health status of the contextual chat service.
    """
    
    try:
        service = get_contextual_chat_service()
        # Check orchestrator health
        orchestrator_healthy = hasattr(service.orchestrator, 'agents') or hasattr(service.orchestrator, 'roster')
        
        # Check conversation manager
        conversation_manager_healthy = service.conversation_manager is not None
        
        # Check active conversations
        active_conversations = len(service.conversation_manager.conversations)
        
        health_status = {
            "status": "healthy" if (orchestrator_healthy and conversation_manager_healthy) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "orchestrator": "healthy" if orchestrator_healthy else "unhealthy",
                "conversation_manager": "healthy" if conversation_manager_healthy else "unhealthy",
                "context_analyzers": "healthy"  # Would check actual analyzers
            },
            "metrics": {
                "active_conversations": active_conversations,
                "cached_contexts": len(service.conversation_manager.context_cache),
                "user_preferences": len(service.conversation_manager.user_preferences)
            },
            "capabilities": {
                "multi_agent_orchestration": orchestrator_healthy,
                "portfolio_context": True,
                "market_context": True,
                "conversation_history": True,
                "smart_suggestions": True
            }
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Export router for main application
__all__ = ["router", "get_contextual_chat_service"]