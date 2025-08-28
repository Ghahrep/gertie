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
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

router = APIRouter()

class SuggestionResponse(BaseModel):
    id: str
    type: str
    title: str
    description: str
    query: str
    target_agent: Optional[str] = None
    workflow_type: Optional[str] = None
    confidence: float
    urgency: str
    category: str
    reasoning: str
    expected_outcome: str
    icon: str
    color: str
    metadata: Optional[Dict[str, Any]] = None

class ContextResponse(BaseModel):
    # Define the structure for your context if you want stronger typing
    portfolio: Dict[str, Any]
    market: Dict[str, Any]
    ab_testing: Optional[Dict[str, Any]] = None

class SuggestionsApiResponse(BaseModel):
    suggestions: List[SuggestionResponse]
    total_count: int
    generated_at: datetime
    context: Optional[ContextResponse] = None

# =============================================================================
# SMART SUGGESTIONS ENDPOINTS
# =============================================================================

@router.get("/suggestions", response_model=SuggestionsApiResponse, tags=["Smart Suggestions"])
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
        from smart_suggestions.realtime_context_service import get_realtime_context_service
        from smart_suggestions.ab_testing_service import get_ab_testing_service, apply_ab_testing_to_suggestions
        
        # --- FIX 3: Fetch context ONCE to avoid redundant database calls ---
        portfolio_context = get_portfolio_context(db, current_user)
        recent_interactions = _get_recent_user_interactions(db, current_user.id)
        
        # NOTE: Assumes generate_smart_suggestions_for_user is updated to accept portfolio_context
        # to prevent its own internal fetch, as recommended in the review.
        all_suggestions = generate_smart_suggestions_for_user(
            db=db,
            user=current_user,
            recent_interactions=recent_interactions,
            portfolio_context=portfolio_context
        )
        
        # Apply A/B testing variants (check for active tests)
        # (This logic can be further optimized, but is functionally okay for now)
        ab_service = get_ab_testing_service()
        if ab_service.active_tests:
            user_active_tests = [
                test_id for test_id in ab_service.active_tests.keys()
                if ab_service.assign_user_to_variant(test_id, str(current_user.id))
            ]
            
            if user_active_tests:
                from smart_suggestions.suggestion_engine import SmartSuggestion, SuggestionType, Urgency
                from dataclasses import asdict
                
                suggestion_objects = [
                    SmartSuggestion(**{**s, 'type': SuggestionType(s['type']), 'urgency': Urgency(s['urgency'])})
                    for s in all_suggestions
                ]
                
                enhanced_suggestions = apply_ab_testing_to_suggestions(
                    suggestion_objects, str(current_user.id), user_active_tests
                )
                
                # Convert back to dict format
                all_suggestions = [
                    {**asdict(s), 'type': s.type.value, 'urgency': s.urgency.value}
                    for s in enhanced_suggestions
                ]

        # --- FIX 2: Apply filters BEFORE limiting the results for correctness ---
        filtered_suggestions = all_suggestions
        if category_filter:
            filtered_suggestions = [s for s in filtered_suggestions if s['category'].lower() == category_filter.lower()]
        
        if urgency_filter:
            filtered_suggestions = [s for s in filtered_suggestions if s['urgency'] == urgency_filter.lower()]
        
        # Apply the final limit AFTER filtering
        final_suggestions = filtered_suggestions[:limit]
        
        # Build the response object that will be validated by the response_model
        response_data = {
            "suggestions": final_suggestions,
            "total_count": len(final_suggestions),
            "generated_at": datetime.now()
        }
        
        # Add context if requested, REUSING the context object from the start
        if include_context:
            market_context = get_market_context()
            
            response_data["context"] = {
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
            
            if ab_service.active_tests:
                response_data["context"]["ab_testing"] = {
                    "active_tests": len(ab_service.active_tests),
                    "user_assignments": ab_service.user_assignments.get(str(current_user.id), {})
                }
        
        # Send real-time context update (non-blocking)
        try:
            realtime_service = get_realtime_context_service()
            await realtime_service.update_dashboard_state(str(current_user.id), {
                "current_view": category_filter or "general",
                "visible_metrics": {"suggestion_request": True},
                "active_widgets": ["suggestions"],
                "time_range": "current"
            })
        except Exception as context_error:
            print(f"Warning: Real-time context update failed: {context_error}")
            
        return response_data
        
    except Exception as e:
        print(f"Error generating suggestions: {e}")
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

@router.get("/admin/ab-tests", tags=["A/B Testing"])
async def list_ab_tests(current_user: User = Depends(get_current_user)):
    """List all A/B tests"""
    try:
        from smart_suggestions.ab_testing_service import get_ab_testing_service
        ab_service = get_ab_testing_service()
        
        active_tests = {test_id: ab_service.get_test_analytics(test_id) 
                       for test_id in ab_service.active_tests.keys()}
        completed_tests = {test_id: ab_service.get_test_analytics(test_id) 
                          for test_id in ab_service.completed_tests.keys()}
        
        return {
            "active_tests": active_tests,
            "completed_tests": completed_tests,
            "system_analytics": ab_service.get_global_analytics_dashboard()
        }
    except Exception as e:
        return {"error": str(e), "active_tests": {}, "completed_tests": {}}

@router.post("/admin/ab-tests", tags=["A/B Testing"])
async def create_ab_test(
    test_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Create new A/B test"""
    try:
        from smart_suggestions.ab_testing_service import get_ab_testing_service
        ab_service = get_ab_testing_service()
        
        test = ab_service.create_suggestion_variant_test(
            test_name=test_config["name"],
            description=test_config["description"],
            variants=test_config["variants"],
            duration_days=test_config.get("duration_days", 14)
        )
        
        return {"status": "success", "test_id": test.test_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}

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