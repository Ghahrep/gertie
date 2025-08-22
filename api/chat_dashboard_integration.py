# api/chat_dashboard_integration.py
"""
Priority 3.3: Chat-Dashboard Integration
========================================
Seamless integration between chat and dashboard with:
1. Floating chat widget with dashboard context
2. Real-time dashboard updates from chat actions
3. Dashboard-to-chat flow for deep analysis
4. Synchronized state management
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import uuid
import asyncio

# Import your existing components
from db.session import get_db
from db import crud
from api.routes.auth import get_current_user
from api.schemas import User
from core.data_handler import get_market_data_for_portfolio
from api.contextual_chat import ContextualChatService, ContextualChatRequest

# Enhanced models for dashboard integration
from pydantic import BaseModel, Field

class DashboardChatRequest(BaseModel):
    """Chat request with dashboard context"""
    message: str = Field(..., description="Chat message")
    dashboard_context: Dict[str, Any] = Field(default_factory=dict, description="Current dashboard state")
    widget_mode: str = Field(default="floating", description="Widget mode: 'floating', 'panel', 'fullscreen'")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    analysis_trigger: Optional[str] = Field(None, description="What triggered this analysis from dashboard")

class DashboardUpdate(BaseModel):
    """Update to be applied to dashboard"""
    update_type: str  # 'metric', 'chart', 'widget', 'notification', 'highlight'
    target_component: str  # Component ID to update
    update_data: Dict[str, Any]  # Update payload
    priority: int = Field(default=5, ge=1, le=10)  # Update priority
    animation: Optional[str] = Field(None, description="Animation type for update")

class ChatDashboardResponse(BaseModel):
    """Response with both chat and dashboard updates"""
    chat_response: str
    conversation_id: str
    dashboard_updates: List[DashboardUpdate]
    real_time_data: Optional[Dict[str, Any]] = None
    suggested_actions: List[Dict[str, Any]] = Field(default_factory=list)
    widget_state: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float

class DashboardWidget(BaseModel):
    """Configuration for dashboard chat widget"""
    widget_id: str
    position: str  # 'bottom-right', 'bottom-left', 'side-panel'
    size: str  # 'compact', 'standard', 'expanded'
    auto_hide: bool = True
    context_awareness: bool = True
    real_time_updates: bool = True

class ChatDashboardIntegration:
    """Core service for chat-dashboard integration"""
    
    def __init__(self):
        self.contextual_chat = ContextualChatService()
        self.active_widgets = {}  # user_id -> widget_config
        self.dashboard_states = {}  # user_id -> dashboard_state
        self.websocket_connections = {}  # user_id -> websocket
        
        # Dashboard component mappings
        self.component_mappings = {
            "portfolio_value": "portfolio-summary-card",
            "risk_metrics": "risk-analysis-widget", 
            "holdings_table": "holdings-data-table",
            "performance_chart": "portfolio-performance-chart",
            "allocation_pie": "asset-allocation-pie",
            "recent_transactions": "transaction-history-widget"
        }
        
        # Analysis triggers from dashboard
        self.dashboard_triggers = {
            "portfolio_summary_click": "Analyze my overall portfolio performance and risk profile",
            "holding_click": "Analyze this specific holding: {ticker}",
            "risk_metric_alert": "My {metric} is showing {value} - what should I do?",
            "allocation_concern": "My asset allocation shows {concern} - how can I optimize it?",
            "performance_anomaly": "I notice {anomaly} in my performance - can you explain?",
            "transaction_question": "Help me understand this recent transaction: {transaction}"
        }
    
    async def process_dashboard_chat(
        self, 
        request: DashboardChatRequest, 
        user: User, 
        db: Session
    ) -> ChatDashboardResponse:
        """Process chat with full dashboard integration"""
        
        # Update dashboard state
        self.dashboard_states[user.id] = request.dashboard_context
        
        # Enhance chat request with dashboard context
        enhanced_chat_request = ContextualChatRequest(
            message=request.message,
            conversation_id=request.conversation_id,
            context_mode="auto",
            include_market_data=True,
            analysis_depth="standard",
            personalization_level="high"
        )
        
        # Process with contextual chat service
        chat_response = await self.contextual_chat.process_contextual_chat(
            enhanced_chat_request, user, db
        )
        
        # Generate dashboard updates based on chat response
        dashboard_updates = await self._generate_dashboard_updates(
            chat_response, request.dashboard_context, user.id
        )
        
        # Get real-time data if needed
        real_time_data = await self._get_real_time_dashboard_data(user, db)
        
        # Generate suggested actions
        suggested_actions = self._generate_dashboard_actions(
            chat_response, request.dashboard_context
        )
        
        # Update widget state
        widget_state = self._update_widget_state(
            user.id, request.widget_mode, chat_response
        )
        
        response = ChatDashboardResponse(
            chat_response=chat_response.message,
            conversation_id=chat_response.conversation_id,
            dashboard_updates=dashboard_updates,
            real_time_data=real_time_data,
            suggested_actions=suggested_actions,
            widget_state=widget_state,
            confidence_score=chat_response.confidence_score
        )
        
        # Send real-time updates via WebSocket if connected
        await self._send_websocket_update(user.id, response)
        
        return response
    
    async def _generate_dashboard_updates(
        self, 
        chat_response, 
        dashboard_context: Dict, 
        user_id: int
    ) -> List[DashboardUpdate]:
        """Generate dashboard updates based on chat analysis"""
        
        updates = []
        
        # Analyze chat response for update triggers
        response_text = chat_response.message.lower()
        
        # Portfolio value updates
        if "portfolio" in response_text and "value" in response_text:
            updates.append(DashboardUpdate(
                update_type="highlight",
                target_component=self.component_mappings["portfolio_value"],
                update_data={
                    "highlight_color": "#4F46E5",
                    "highlight_duration": 3000,
                    "reason": "Portfolio value mentioned in analysis"
                },
                priority=7,
                animation="pulse"
            ))
        
        # Risk metric updates
        if any(word in response_text for word in ["risk", "volatility", "var", "drawdown"]):
            updates.append(DashboardUpdate(
                update_type="metric",
                target_component=self.component_mappings["risk_metrics"],
                update_data={
                    "new_metrics": {
                        "risk_alert": True,
                        "analysis_available": True,
                        "last_analysis": datetime.now().isoformat()
                    },
                    "badge_text": "Analyzed",
                    "badge_color": "blue"
                },
                priority=8,
                animation="slide-in"
            ))
        
        # Holdings table updates
        if "stock" in response_text or "holding" in response_text:
            # Extract any ticker symbols mentioned
            import re
            tickers = re.findall(r'\b[A-Z]{2,5}\b', chat_response.message)
            
            if tickers:
                updates.append(DashboardUpdate(
                    update_type="widget",
                    target_component=self.component_mappings["holdings_table"],
                    update_data={
                        "highlight_tickers": tickers,
                        "analysis_context": "Mentioned in AI analysis",
                        "show_details": True
                    },
                    priority=9,
                    animation="highlight-row"
                ))
        
        # Performance chart updates
        if "performance" in response_text or "return" in response_text:
            updates.append(DashboardUpdate(
                update_type="chart",
                target_component=self.component_mappings["performance_chart"],
                update_data={
                    "add_annotation": {
                        "text": "AI Analysis Point",
                        "timestamp": datetime.now().isoformat(),
                        "type": "analysis"
                    },
                    "highlight_period": "recent"
                },
                priority=6,
                animation="fade-in"
            ))
        
        # Allocation updates
        if any(word in response_text for word in ["allocation", "diversify", "balance", "rebalance"]):
            updates.append(DashboardUpdate(
                update_type="chart",
                target_component=self.component_mappings["allocation_pie"],
                update_data={
                    "show_recommendation_overlay": True,
                    "recommendation_source": "AI Analysis",
                    "highlight_sectors": True
                },
                priority=7,
                animation="rotate-in"
            ))
        
        # Notification updates
        if chat_response.confidence_score > 0.8:
            updates.append(DashboardUpdate(
                update_type="notification",
                target_component="notification-center",
                update_data={
                    "notification": {
                        "id": str(uuid.uuid4()),
                        "type": "info",
                        "title": "AI Analysis Complete",
                        "message": "High-confidence analysis available in chat",
                        "action_button": "View Analysis",
                        "action_data": {"conversation_id": chat_response.conversation_id}
                    }
                },
                priority=5
            ))
        
        return updates
    
    async def _get_real_time_dashboard_data(self, user: User, db: Session) -> Dict[str, Any]:
        """Get real-time data for dashboard updates"""
        
        # Get fresh portfolio data
        user_portfolios = crud.get_user_portfolios(db=db, user_id=user.id)
        
        if not user_portfolios:
            return {}
        
        primary_portfolio = user_portfolios[0]
        market_data = get_market_data_for_portfolio(primary_portfolio.holdings)
        
        # Calculate real-time metrics
        real_time_data = {
            "portfolio_value": market_data.get("total_value", 0),
            "day_change": self._calculate_day_change(market_data),
            "last_updated": datetime.now().isoformat(),
            "market_status": "open",  # Would check real market hours
            "holdings_count": len(market_data.get("holdings_with_values", [])),
            "top_movers": self._identify_top_movers(market_data.get("holdings_with_values", []))
        }
        
        return real_time_data
    
    def _calculate_day_change(self, market_data: Dict) -> Dict[str, float]:
        """Calculate day change for portfolio"""
        # Simplified calculation - would use real price data
        total_value = market_data.get("total_value", 0)
        estimated_change = total_value * 0.012  # Mock 1.2% gain
        
        return {
            "absolute": estimated_change,
            "percentage": (estimated_change / total_value * 100) if total_value > 0 else 0
        }
    
    def _identify_top_movers(self, holdings: List[Dict]) -> List[Dict]:
        """Identify top moving holdings"""
        # Simplified - would use real price change data
        top_movers = []
        
        for holding in holdings[:5]:  # Top 5 holdings
            # Mock price movement
            import random
            change_pct = random.uniform(-3.0, 4.0)
            
            top_movers.append({
                "ticker": holding.get("ticker", ""),
                "change_pct": change_pct,
                "direction": "up" if change_pct > 0 else "down"
            })
        
        # Sort by absolute change
        top_movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        
        return top_movers[:3]  # Return top 3 movers
    
    def _generate_dashboard_actions(
        self, 
        chat_response, 
        dashboard_context: Dict
    ) -> List[Dict[str, Any]]:
        """Generate suggested actions for dashboard"""
        
        actions = []
        
        # Based on chat smart suggestions
        for suggestion in chat_response.smart_suggestions:
            if suggestion.suggestion_type == "action":
                actions.append({
                    "id": str(uuid.uuid4()),
                    "type": "chat_suggestion",
                    "title": suggestion.suggestion_text,
                    "priority": suggestion.priority,
                    "action": "open_chat_with_query",
                    "action_data": {"query": suggestion.suggestion_text}
                })
        
        # Based on portfolio insights
        if chat_response.portfolio_insights:
            for insight in chat_response.portfolio_insights[:2]:  # Top 2 insights
                actions.append({
                    "id": str(uuid.uuid4()),
                    "type": "portfolio_insight",
                    "title": f"Address: {insight[:50]}...",
                    "priority": 6,
                    "action": "open_analysis_panel",
                    "action_data": {"insight": insight}
                })
        
        return actions
    
    def _update_widget_state(
        self, 
        user_id: int, 
        widget_mode: str, 
        chat_response
    ) -> Dict[str, Any]:
        """Update chat widget state"""
        
        widget_state = {
            "mode": widget_mode,
            "conversation_active": True,
            "last_response_time": datetime.now().isoformat(),
            "confidence_indicator": "high" if chat_response.confidence_score > 0.8 else "medium",
            "suggestions_available": len(chat_response.smart_suggestions) > 0,
            "context_rich": chat_response.portfolio_insights is not None,
            "can_auto_execute": False  # Require user confirmation for trades
        }
        
        return widget_state
    
    async def _send_websocket_update(self, user_id: int, response: ChatDashboardResponse):
        """Send real-time updates via WebSocket"""
        
        if user_id in self.websocket_connections:
            websocket = self.websocket_connections[user_id]
            
            try:
                update_message = {
                    "type": "dashboard_update",
                    "timestamp": datetime.now().isoformat(),
                    "updates": [update.dict() for update in response.dashboard_updates],
                    "real_time_data": response.real_time_data,
                    "suggested_actions": response.suggested_actions
                }
                
                await websocket.send_text(json.dumps(update_message))
                
            except Exception as e:
                print(f"WebSocket send error for user {user_id}: {e}")
                # Remove disconnected websocket
                del self.websocket_connections[user_id]

# Initialize the integration service
chat_dashboard_service = ChatDashboardIntegration()

# Router for chat-dashboard integration endpoints
router = APIRouter(prefix="/chat-dashboard", tags=["Chat Dashboard Integration"])

@router.post("/chat", response_model=ChatDashboardResponse)
async def process_dashboard_chat(
    request: DashboardChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    🎯 Process Chat with Dashboard Integration
    
    Handles chat messages with full dashboard context awareness and generates
    real-time dashboard updates based on AI analysis.
    """
    
    try:
        response = await chat_dashboard_service.process_dashboard_chat(
            request, current_user, db
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process dashboard chat: {str(e)}"
        )

@router.post("/widget/configure")
async def configure_chat_widget(
    widget_config: DashboardWidget,
    current_user: User = Depends(get_current_user)
):
    """
    ⚙️ Configure Chat Widget Settings
    
    Configure the floating chat widget for the current user's dashboard.
    """
    
    try:
        # Store widget configuration
        chat_dashboard_service.active_widgets[current_user.id] = widget_config.dict()
        
        return {
            "status": "configured",
            "widget_id": widget_config.widget_id,
            "message": "Chat widget configured successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure widget: {str(e)}"
        )

@router.get("/widget/state")
async def get_widget_state(
    current_user: User = Depends(get_current_user)
):
    """
    📊 Get Current Widget State
    
    Retrieve the current state of the chat widget for the dashboard.
    """
    
    try:
        widget_config = chat_dashboard_service.active_widgets.get(
            current_user.id, 
            {
                "widget_id": f"widget_{current_user.id}",
                "position": "bottom-right",
                "size": "standard",
                "auto_hide": True,
                "context_awareness": True,
                "real_time_updates": True
            }
        )
        
        return {
            "widget_config": widget_config,
            "is_active": current_user.id in chat_dashboard_service.active_widgets,
            "has_websocket": current_user.id in chat_dashboard_service.websocket_connections
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get widget state: {str(e)}"
        )

@router.post("/trigger/{trigger_type}")
async def trigger_dashboard_analysis(
    trigger_type: str,
    trigger_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    🚀 Trigger Analysis from Dashboard
    
    Trigger contextual analysis based on dashboard interactions.
    """
    
    try:
        # Get trigger query template
        trigger_query = chat_dashboard_service.dashboard_triggers.get(trigger_type)
        
        if not trigger_query:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown trigger type: {trigger_type}"
            )
        
        # Format query with trigger data
        formatted_query = trigger_query.format(**trigger_data)
        
        # Create dashboard chat request
        chat_request = DashboardChatRequest(
            message=formatted_query,
            dashboard_context=trigger_data,
            widget_mode="panel",
            analysis_trigger=trigger_type
        )
        
        # Process the triggered analysis
        response = await chat_dashboard_service.process_dashboard_chat(
            chat_request, current_user, db
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger analysis: {str(e)}"
        )

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    current_user: User = Depends(get_current_user)
):
    """
    🌐 WebSocket for Real-Time Dashboard Updates
    
    Establishes WebSocket connection for real-time dashboard updates.
    """
    
    await websocket.accept()
    
    # Store WebSocket connection
    chat_dashboard_service.websocket_connections[current_user.id] = websocket
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "user_id": current_user.id,
            "timestamp": datetime.now().isoformat(),
            "features_enabled": [
                "real_time_updates",
                "dashboard_sync",
                "contextual_chat"
            ]
        }))
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages or timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client requests
                try:
                    client_data = json.loads(message)
                    
                    if client_data.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                    elif client_data.get("type") == "dashboard_state_update":
                        # Update stored dashboard state
                        chat_dashboard_service.dashboard_states[current_user.id] = client_data.get("state", {})
                        
                        await websocket.send_text(json.dumps({
                            "type": "state_acknowledged",
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                except json.JSONDecodeError:
                    # Ignore invalid JSON
                    pass
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        # Clean up connection
        if current_user.id in chat_dashboard_service.websocket_connections:
            del chat_dashboard_service.websocket_connections[current_user.id]
    
    except Exception as e:
        await websocket.close(code=1000, reason=f"Error: {str(e)}")
        if current_user.id in chat_dashboard_service.websocket_connections:
            del chat_dashboard_service.websocket_connections[current_user.id]

@router.get("/analytics/engagement")
async def get_chat_dashboard_analytics(
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """
    📈 Get Chat-Dashboard Engagement Analytics
    
    Analyze how users interact with the chat-dashboard integration.
    """
    
    try:
        # Mock analytics data - implement actual tracking
        analytics = {
            "period_days": days,
            "total_interactions": 156,
            "dashboard_triggers": {
                "portfolio_summary_click": 45,
                "holding_click": 32,
                "risk_metric_alert": 28,
                "allocation_concern": 24,
                "performance_anomaly": 15,
                "transaction_question": 12
            },
            "response_satisfaction": {
                "high_confidence": 78,  # responses with >0.8 confidence
                "medium_confidence": 52,
                "low_confidence": 26
            },
            "widget_usage": {
                "floating_mode": 89,
                "panel_mode": 45,
                "fullscreen_mode": 22
            },
            "avg_response_time_ms": 2340,
            "dashboard_updates_generated": 234,
            "user_engagement_score": 8.7
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics: {str(e)}"
        )

# Export for main application
__all__ = ["router", "chat_dashboard_service", "ChatDashboardIntegration"]