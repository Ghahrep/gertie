# api/routes/smart_suggestions.py
"""
Smart Suggestions API Endpoints
===============================
RESTful API endpoints for the intelligent suggestion system
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

# Your existing imports
from db.session import get_db
from api.schemas import User
from api.routes.auth import get_current_user
from smart_suggestions.suggestion_engine import (
    generate_smart_suggestions_for_user,
    SmartSuggestionEngine,
    get_portfolio_context,
    get_market_context
)

router = APIRouter()

# =============================================================================
# SMART SUGGESTIONS ENDPOINTS
# =============================================================================

@router.get("/suggestions", response_model=List[Dict[str, Any]], tags=["Smart Suggestions"])
async def get_smart_suggestions(
    include_context: bool = Query(True, description="Include portfolio and market context in response"),
    limit: int = Query(10, ge=1, le=20, description="Maximum number of suggestions to return"),
    category_filter: Optional[str] = Query(None, description="Filter suggestions by category"),
    urgency_filter: Optional[str] = Query(None, description="Filter by urgency: low, medium, high"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get personalized smart suggestions for the current user
    
    Returns intelligent recommendations based on:
    - Portfolio risk metrics and composition
    - Market conditions and volatility  
    - User interaction patterns
    - Recent analysis results
    """
    
    try:
        # Get recent user interactions from session/logs (simplified for demo)
        recent_interactions = _get_recent_user_interactions(db, current_user.id)
        
        # Generate suggestions
        suggestions = generate_smart_suggestions_for_user(
            db=db,
            user=current_user,
            recent_interactions=recent_interactions
        )
        
        # Apply filters
        if category_filter:
            suggestions = [s for s in suggestions if s['category'].lower() == category_filter.lower()]
        
        if urgency_filter:
            suggestions = [s for s in suggestions if s['urgency'] == urgency_filter.lower()]
        
        # Limit results
        suggestions = suggestions[:limit]
        
        # Add context if requested
        response = {
            "suggestions": suggestions,
            "total_count": len(suggestions),
            "generated_at": datetime.now().isoformat()
        }
        
        if include_context:
            portfolio_context = get_portfolio_context(db, current_user)
            market_context = get_market_context()
            
            response["context"] = {
                "portfolio": {
                    "risk_score": portfolio_context.risk_score if portfolio_context else None,
                    "total_value": portfolio_context.total_value if portfolio_context else None,
                    "risk_change_pct": portfolio_context.risk_change_pct if portfolio_context else None
                },
                "market": {
                    "vix_level": market_context.vix_level,
                    "market_trend": market_context.market_trend,
                    "volatility_regime": market_context.volatility_regime
                }
            }
        
        return response
        
    except Exception as e:
        print(f"❌ Error generating suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {str(e)}")

@router.post("/suggestions/execute", tags=["Smart Suggestions"])
async def execute_suggestion(
    suggestion_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Execute a smart suggestion by routing to the appropriate system
    
    Handles different suggestion types:
    - agent_query: Route to specific agent
    - workflow_trigger: Start multi-agent workflow
    - debate_topic: Initiate agent debate
    - follow_up: Execute follow-up action
    """
    
    try:
        suggestion_type = suggestion_data.get('type')
        suggestion_id = suggestion_data.get('id')
        query = suggestion_data.get('query')
        
        # Log suggestion execution
        _log_suggestion_execution(db, current_user.id, suggestion_data)
        
        if suggestion_type == 'agent_query':
            # Route to specific agent
            target_agent = suggestion_data.get('target_agent')
            
            # You would integrate with your existing agent routing here
            # For now, return a mock response
            return {
                "success": True,
                "action": "agent_query",
                "target_agent": target_agent,
                "query": query,
                "redirect_url": f"/chat?agent={target_agent}&query={query}",
                "message": f"Routing to {target_agent} agent for analysis"
            }
            
        elif suggestion_type == 'workflow_trigger':
            # Start workflow
            workflow_type = suggestion_data.get('workflow_type', 'comprehensive_analysis')
            
            # Integrate with your existing workflow system
            return {
                "success": True,
                "action": "workflow_trigger",
                "workflow_type": workflow_type,
                "query": query,
                "redirect_url": f"/workflow?query={query}",
                "message": f"Starting {workflow_type} workflow"
            }
            
        elif suggestion_type == 'debate_topic':
            # Start agent debate
            return {
                "success": True,
                "action": "debate_topic",
                "query": query,
                "redirect_url": f"/debate?topic={query}",
                "message": "Starting multi-agent debate on this topic"
            }
            
        elif suggestion_type == 'follow_up':
            # Execute follow-up action
            metadata = suggestion_data.get('metadata', {})
            
            return {
                "success": True,
                "action": "follow_up",
                "query": query,
                "metadata": metadata,
                "message": "Follow-up action executed"
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown suggestion type: {suggestion_type}")
            
    except Exception as e:
        print(f"❌ Error executing suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute suggestion: {str(e)}")

@router.get("/suggestions/categories", tags=["Smart Suggestions"])
async def get_suggestion_categories(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get available suggestion categories for filtering"""
    
    categories = [
        {"id": "risk_management", "name": "Risk Management", "icon": "⚠️"},
        {"id": "portfolio_balance", "name": "Portfolio Balance", "icon": "⚖️"},
        {"id": "performance", "name": "Performance", "icon": "📈"},
        {"id": "market_protection", "name": "Market Protection", "icon": "🛡️"},
        {"id": "market_timing", "name": "Market Timing", "icon": "🔄"},
        {"id": "sector_allocation", "name": "Sector Allocation", "icon": "💻"},
        {"id": "discovery", "name": "Discovery", "icon": "💡"},
        {"id": "maintenance", "name": "Maintenance", "icon": "🏥"},
        {"id": "implementation", "name": "Implementation", "icon": "✅"},
        {"id": "risk_alert", "name": "Risk Alert", "icon": "🚨"}
    ]
    
    return {"categories": categories}

@router.get("/suggestions/stats", tags=["Smart Suggestions"])
async def get_suggestion_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get statistics about suggestion generation and execution"""
    
    try:
        # Get suggestion execution history
        execution_history = _get_suggestion_execution_history(db, current_user.id, days)
        
        # Calculate statistics
        total_suggestions = len(execution_history)
        executed_suggestions = len([h for h in execution_history if h.get('executed')])
        execution_rate = (executed_suggestions / total_suggestions * 100) if total_suggestions > 0 else 0
        
        # Breakdown by type
        type_breakdown = {}
        category_breakdown = {}
        
        for item in execution_history:
            suggestion_type = item.get('type', 'unknown')
            category = item.get('category', 'unknown')
            
            type_breakdown[suggestion_type] = type_breakdown.get(suggestion_type, 0) + 1
            category_breakdown[category] = category_breakdown.get(category, 0) + 1
        
        return {
            "period_days": days,
            "total_suggestions_generated": total_suggestions,
            "suggestions_executed": executed_suggestions,
            "execution_rate_pct": round(execution_rate, 1),
            "type_breakdown": type_breakdown,
            "category_breakdown": category_breakdown,
            "most_popular_category": max(category_breakdown.items(), key=lambda x: x[1])[0] if category_breakdown else None,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting suggestion stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/suggestions/feedback", tags=["Smart Suggestions"])
async def submit_suggestion_feedback(
    feedback_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit feedback on suggestion quality and relevance
    
    Helps improve the suggestion engine over time
    """
    
    try:
        suggestion_id = feedback_data.get('suggestion_id')
        rating = feedback_data.get('rating')  # 1-5 scale
        helpful = feedback_data.get('helpful')  # boolean
        comments = feedback_data.get('comments', '')
        
        if not suggestion_id or rating is None:
            raise HTTPException(status_code=400, detail="suggestion_id and rating are required")
        
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="rating must be between 1 and 5")
        
        # Store feedback (you'd implement proper feedback storage)
        feedback_record = {
            "user_id": current_user.id,
            "suggestion_id": suggestion_id,
            "rating": rating,
            "helpful": helpful,
            "comments": comments,
            "submitted_at": datetime.now().isoformat()
        }
        
        _store_suggestion_feedback(db, feedback_record)
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "feedback_id": f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    except Exception as e:
        print(f"❌ Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.get("/suggestions/context", tags=["Smart Suggestions"])
async def get_suggestion_context(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed context information used for suggestion generation
    
    Useful for debugging and understanding why certain suggestions were made
    """
    
    try:
        # Get portfolio context
        portfolio_context = get_portfolio_context(db, current_user)
        market_context = get_market_context()
        recent_interactions = _get_recent_user_interactions(db, current_user.id)
        
        return {
            "portfolio_context": {
                "total_value": portfolio_context.total_value if portfolio_context else None,
                "risk_score": portfolio_context.risk_score if portfolio_context else None,
                "volatility": portfolio_context.volatility if portfolio_context else None,
                "concentration_risk": portfolio_context.concentration_risk if portfolio_context else None,
                "holdings_count": len(portfolio_context.holdings) if portfolio_context else 0,
                "sector_allocation": portfolio_context.sector_allocation if portfolio_context else {},
                "risk_change_pct": portfolio_context.risk_change_pct if portfolio_context else None,
                "last_analysis_days_ago": (datetime.now() - portfolio_context.last_analysis).days if portfolio_context else None
            },
            "market_context": {
                "vix_level": market_context.vix_level,
                "market_trend": market_context.market_trend,
                "sector_rotation": market_context.sector_rotation,
                "fed_policy_stance": market_context.fed_policy_stance,
                "volatility_regime": market_context.volatility_regime
            },
            "behavioral_context": {
                "recent_interactions_count": len(recent_interactions),
                "agents_used_recently": list(set(i.get('agent_type') for i in recent_interactions)),
                "last_interaction_date": max([i.get('timestamp') for i in recent_interactions]) if recent_interactions else None
            },
            "suggestion_triggers": _identify_active_triggers(portfolio_context, market_context),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")

# =============================================================================
# INTEGRATION WITH EXISTING CHAT/WORKFLOW SYSTEMS
# =============================================================================

@router.post("/suggestions/chat/contextual", tags=["Smart Suggestions"])
async def get_contextual_chat_suggestions(
    chat_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get contextual suggestions based on current chat conversation
    
    Integrates with your existing chat system to provide relevant follow-ups
    """
    
    try:
        current_agent = chat_data.get('current_agent')
        conversation_history = chat_data.get('conversation_history', [])
        last_response = chat_data.get('last_response', {})
        
        # Generate contextual suggestions based on chat context
        contextual_suggestions = _generate_contextual_chat_suggestions(
            db, current_user, current_agent, conversation_history, last_response
        )
        
        return {
            "suggestions": contextual_suggestions,
            "context_agent": current_agent,
            "conversation_length": len(conversation_history)
        }
        
    except Exception as e:
        print(f"❌ Error generating contextual suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate contextual suggestions: {str(e)}")

@router.get("/suggestions/dashboard/widgets", tags=["Smart Suggestions"])
async def get_dashboard_widget_suggestions(
    widget_context: Optional[str] = Query(None, description="Current dashboard widget context"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get suggestions for dashboard widgets and metrics cards
    
    Provides "Ask AI" button suggestions based on what's displayed
    """
    
    try:
        # Generate widget-specific suggestions
        widget_suggestions = _generate_dashboard_widget_suggestions(
            db, current_user, widget_context
        )
        
        return {
            "widget_suggestions": widget_suggestions,
            "widget_context": widget_context
        }
        
    except Exception as e:
        print(f"❌ Error generating widget suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate widget suggestions: {str(e)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_recent_user_interactions(db: Session, user_id: int, days: int = 7) -> List[Dict]:
    """Get recent user interactions for behavioral analysis"""
    
    # This would query your interaction logs
    # For now, return mock data
    return [
        {
            "timestamp": datetime.now() - timedelta(days=1),
            "agent_type": "quantitative",
            "query": "analyze portfolio risk",
            "executed": True
        },
        {
            "timestamp": datetime.now() - timedelta(days=2),
            "agent_type": "strategy", 
            "query": "rebalancing recommendations",
            "executed": True
        }
    ]

def _log_suggestion_execution(db: Session, user_id: int, suggestion_data: Dict):
    """Log suggestion execution for analytics"""
    
    # Store in your logging system
    print(f"📝 Logging suggestion execution for user {user_id}: {suggestion_data.get('id')}")

def _get_suggestion_execution_history(db: Session, user_id: int, days: int) -> List[Dict]:
    """Get suggestion execution history for statistics"""
    
    # Query your logs for suggestion history
    # For now, return mock data
    return [
        {
            "suggestion_id": "high_risk_portfolio",
            "type": "agent_query",
            "category": "Risk Management",
            "executed": True,
            "timestamp": datetime.now() - timedelta(days=1)
        }
    ]

def _store_suggestion_feedback(db: Session, feedback_record: Dict):
    """Store user feedback on suggestions"""
    
    # Store in your feedback system
    print(f"💬 Storing feedback: {feedback_record}")

def _identify_active_triggers(portfolio_context, market_context) -> List[str]:
    """Identify which conditions are triggering suggestions"""
    
    triggers = []
    
    if portfolio_context and portfolio_context.risk_score > 80:
        triggers.append("High portfolio risk score")
    
    if market_context and market_context.vix_level > 25:
        triggers.append("Elevated market volatility (VIX)")
    
    if portfolio_context and portfolio_context.concentration_risk > 0.15:
        triggers.append("Portfolio concentration risk")
    
    return triggers

def _generate_contextual_chat_suggestions(
    db: Session, 
    user: User, 
    current_agent: str,
    conversation_history: List[Dict],
    last_response: Dict
) -> List[Dict]:
    """Generate contextual suggestions based on chat conversation"""
    
    suggestions = []
    
    # If talking to quantitative agent, suggest follow-up with risk agent
    if current_agent == "quantitative" and "risk" in str(last_response).lower():
        suggestions.append({
            "id": "followup_risk_agent",
            "title": "Explore Risk Management",
            "description": "Get specific hedging strategies from our risk expert",
            "query": "What specific hedging strategies should I implement based on this risk analysis?",
            "target_agent": "risk",
            "type": "agent_query"
        })
    
    # If discussing portfolio optimization, suggest workflow
    if "optimization" in str(conversation_history).lower():
        suggestions.append({
            "id": "comprehensive_optimization",
            "title": "Comprehensive Optimization Workflow",
            "description": "Get multi-agent analysis for complete portfolio optimization",
            "query": "Perform comprehensive portfolio optimization analysis",
            "type": "workflow_trigger"
        })
    
    return suggestions

def _generate_dashboard_widget_suggestions(
    db: Session,
    user: User, 
    widget_context: Optional[str]
) -> List[Dict]:
    """Generate suggestions for dashboard widgets"""
    
    suggestions = []
    
    if widget_context == "risk_metrics":
        suggestions.extend([
            {
                "id": "analyze_risk_metrics",
                "title": "Analyze Risk Metrics",
                "description": "Get detailed analysis of your risk metrics",
                "query": "Analyze my current risk metrics and explain what they mean for my portfolio",
                "target_agent": "quantitative"
            },
            {
                "id": "reduce_portfolio_risk",
                "title": "Reduce Portfolio Risk", 
                "description": "Get strategies to reduce portfolio risk",
                "query": "What specific steps can I take to reduce my portfolio risk?",
                "target_agent": "risk"
            }
        ])
    
    elif widget_context == "performance":
        suggestions.extend([
            {
                "id": "improve_performance",
                "title": "Improve Performance",
                "description": "Get recommendations to enhance returns",
                "query": "How can I improve my portfolio's risk-adjusted performance?",
                "target_agent": "strategy"
            }
        ])
    
    return suggestions

# =============================================================================
# MAIN.PY INTEGRATION
# =============================================================================

# Add this to your main.py file:
"""
# Import the smart suggestions router
from api.routes import smart_suggestions

# Include the router
app.include_router(smart_suggestions.router, prefix="/api/v1", tags=["smart_suggestions"])

# Add WebSocket endpoint for real-time suggestions
@app.websocket("/ws/suggestions/{user_id}")
async def suggestion_websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    
    try:
        while True:
            # Check for new suggestions periodically
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Generate suggestions (you'd implement proper user auth here)
            # suggestions = generate_smart_suggestions_for_user(...)
            
            # Send to client if new suggestions available
            # await websocket.send_json({"type": "suggestions_update", "data": suggestions})
            
    except WebSocketDisconnect:
        print(f"Suggestions WebSocket disconnected for user {user_id}")
"""