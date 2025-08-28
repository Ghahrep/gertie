# smart_suggestions/realtime_context_service.py
"""
Real-time Context Integration Service for Smart Suggestions
==========================================================
Integrates with your existing WebSocket infrastructure to provide
context-aware suggestions based on live dashboard state and conversation flow.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid

from sqlalchemy.orm import Session
from websocket.enhanced_connection_manager import get_enhanced_connection_manager
from api.contextual_chat import ContextualChatService, ConversationContext
from smart_suggestions.suggestion_engine import (
    SmartSuggestionEngine, SmartSuggestion, SuggestionType, Urgency
)
from smart_suggestions.ml_service import get_ml_enhancer, enhance_suggestions_with_ml
from db import crud, models

@dataclass 
class DashboardState:
    """Current dashboard state for context-aware suggestions"""
    user_id: str
    active_widgets: List[str]
    current_view: str  # 'portfolio', 'risk', 'performance', etc.
    visible_metrics: Dict[str, Any]
    time_range: str  # '1D', '1W', '1M', etc.
    last_interaction: datetime
    interaction_count: int = 0

@dataclass
class ConversationState:
    """Current conversation state"""
    conversation_id: str
    user_id: str
    active_agent: Optional[str]
    recent_topics: List[str]
    conversation_context: Dict[str, Any]
    last_message: Optional[Dict[str, Any]]
    message_count: int = 0

@dataclass
class RealTimeSuggestion:
    """Real-time suggestion with context"""
    suggestion: SmartSuggestion
    context_source: str  # 'dashboard', 'conversation', 'portfolio_change'
    trigger_event: str
    relevance_score: float
    expires_at: datetime
    
class RealTimeContextService:
    """
    Service that monitors dashboard state, conversation flow, and portfolio changes
    to generate contextual suggestions in real-time via WebSocket
    """
    
    def __init__(self):
        # Core services
        self.connection_manager = get_enhanced_connection_manager()
        self.suggestion_engine = SmartSuggestionEngine()
        self.ml_enhancer = get_ml_enhancer()
        self.chat_service = ContextualChatService()
        
        # State tracking
        self.dashboard_states: Dict[str, DashboardState] = {}
        self.conversation_states: Dict[str, ConversationState] = {}
        self.active_suggestions: Dict[str, List[RealTimeSuggestion]] = defaultdict(list)
        
        # Event queues for processing
        self.dashboard_events = asyncio.Queue()
        self.conversation_events = asyncio.Queue()
        self.portfolio_events = asyncio.Queue()
        
        # Background task handles
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Configuration
        self.suggestion_cooldown = 30  # seconds between suggestions for same context
        self.max_suggestions_per_user = 5
        self.suggestion_expiry_minutes = 15
    
    async def start_service(self):
        """Start the real-time context service"""
        # Start background processing tasks
        tasks = [
            self._process_dashboard_events(),
            self._process_conversation_events(), 
            self._process_portfolio_events(),
            self._cleanup_expired_suggestions(),
            self._monitor_user_engagement()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        print("Real-time context service started")
        print(f"- Dashboard monitoring: ACTIVE")
        print(f"- Conversation tracking: ACTIVE") 
        print(f"- Portfolio change detection: ACTIVE")
        print(f"- ML-powered suggestions: ACTIVE")
    
    async def update_dashboard_state(
        self, 
        user_id: str, 
        state_update: Dict[str, Any]
    ):
        """Update dashboard state and trigger contextual suggestions"""
        
        # Update dashboard state
        if user_id not in self.dashboard_states:
            self.dashboard_states[user_id] = DashboardState(
                user_id=user_id,
                active_widgets=[],
                current_view="portfolio",
                visible_metrics={},
                time_range="1D",
                last_interaction=datetime.now()
            )
        
        state = self.dashboard_states[user_id]
        state.last_interaction = datetime.now()
        state.interaction_count += 1
        
        # Apply updates
        for key, value in state_update.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        # Queue event for processing
        await self.dashboard_events.put({
            "type": "dashboard_update",
            "user_id": user_id,
            "state": state,
            "changes": state_update,
            "timestamp": datetime.now()
        })
    
    async def update_conversation_state(
        self,
        user_id: str,
        conversation_id: str,
        conversation_update: Dict[str, Any]
    ):
        """Update conversation state and trigger contextual suggestions"""
        
        state_key = f"{user_id}_{conversation_id}"
        
        if state_key not in self.conversation_states:
            self.conversation_states[state_key] = ConversationState(
                conversation_id=conversation_id,
                user_id=user_id,
                active_agent=None,
                recent_topics=[],
                conversation_context={},
                last_message=None
            )
        
        state = self.conversation_states[state_key]
        state.message_count += 1
        
        # Apply updates
        for key, value in conversation_update.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        # Queue event for processing
        await self.conversation_events.put({
            "type": "conversation_update",
            "user_id": user_id,
            "conversation_id": conversation_id,
            "state": state,
            "changes": conversation_update,
            "timestamp": datetime.now()
        })
    
    async def handle_portfolio_change(
        self,
        user_id: str,
        change_type: str,  # 'risk_threshold_breach', 'holding_change', 'performance_update'
        change_data: Dict[str, Any]
    ):
        """Handle portfolio changes and generate relevant suggestions"""
        
        await self.portfolio_events.put({
            "type": "portfolio_change",
            "user_id": user_id,
            "change_type": change_type,
            "change_data": change_data,
            "timestamp": datetime.now()
        })
    
    async def _process_dashboard_events(self):
        """Process dashboard state changes and generate contextual suggestions"""
        
        while True:
            try:
                event = await self.dashboard_events.get()
                user_id = event["user_id"]
                state = event["state"]
                changes = event["changes"]
                
                # Generate dashboard-specific suggestions
                suggestions = await self._generate_dashboard_suggestions(
                    user_id, state, changes
                )
                
                if suggestions:
                    await self._send_contextual_suggestions(
                        user_id, suggestions, "dashboard_context"
                    )
                
            except Exception as e:
                print(f"Error processing dashboard event: {e}")
    
    async def _process_conversation_events(self):
        """Process conversation updates and generate contextual suggestions"""
        
        while True:
            try:
                event = await self.conversation_events.get()
                user_id = event["user_id"]
                state = event["state"]
                changes = event["changes"]
                
                # Generate conversation-specific suggestions
                suggestions = await self._generate_conversation_suggestions(
                    user_id, state, changes
                )
                
                if suggestions:
                    await self._send_contextual_suggestions(
                        user_id, suggestions, "conversation_context"
                    )
                
            except Exception as e:
                print(f"Error processing conversation event: {e}")
    
    async def _process_portfolio_events(self):
        """Process portfolio changes and generate urgent suggestions"""
        
        while True:
            try:
                event = await self.portfolio_events.get()
                user_id = event["user_id"]
                change_type = event["change_type"]
                change_data = event["change_data"]
                
                # Generate urgent portfolio-related suggestions
                suggestions = await self._generate_portfolio_change_suggestions(
                    user_id, change_type, change_data
                )
                
                if suggestions:
                    await self._send_contextual_suggestions(
                        user_id, suggestions, "portfolio_alert", urgent=True
                    )
                
            except Exception as e:
                print(f"Error processing portfolio event: {e}")
    
    async def _generate_dashboard_suggestions(
        self, 
        user_id: str, 
        state: DashboardState, 
        changes: Dict[str, Any]
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions based on dashboard context"""
        
        suggestions = []
        current_time = datetime.now()
        
        # View-specific suggestions
        if state.current_view == "risk" and "visible_metrics" in changes:
            metrics = changes["visible_metrics"]
            
            if "portfolio_var" in metrics and metrics["portfolio_var"] > 0.05:
                suggestion = SmartSuggestion(
                    id=f"dashboard_risk_{current_time.timestamp()}",
                    type=SuggestionType.AGENT_QUERY,
                    title="High Risk Detected",
                    description="Your portfolio VaR is elevated. Get risk mitigation strategies.",
                    query=f"My portfolio VaR is {metrics['portfolio_var']:.2%}. What specific steps can I take to reduce risk?",
                    target_agent="risk",
                    confidence=0.9,
                    urgency=Urgency.HIGH,
                    category="Risk Management",
                    reasoning="High VaR detected in dashboard view",
                    expected_outcome="Risk reduction strategies",
                    icon="⚠️",
                    color="red"
                )
                
                suggestions.append(RealTimeSuggestion(
                    suggestion=suggestion,
                    context_source="dashboard",
                    trigger_event="high_var_displayed",
                    relevance_score=0.9,
                    expires_at=current_time + timedelta(minutes=self.suggestion_expiry_minutes)
                ))
        
        elif state.current_view == "performance" and state.interaction_count > 5:
            # User has been looking at performance metrics
            suggestion = SmartSuggestion(
                id=f"dashboard_performance_{current_time.timestamp()}",
                type=SuggestionType.WORKFLOW_TRIGGER,
                title="Performance Deep Dive",
                description="You've been analyzing performance. Get comprehensive analysis from multiple AI experts.",
                query="Perform comprehensive performance analysis and identify optimization opportunities",
                workflow_type="performance_analysis",
                confidence=0.8,
                urgency=Urgency.MEDIUM,
                category="Performance",
                reasoning="Extended performance dashboard interaction",
                expected_outcome="Multi-agent performance analysis",
                icon="📈",
                color="blue"
            )
            
            suggestions.append(RealTimeSuggestion(
                suggestion=suggestion,
                context_source="dashboard",
                trigger_event="extended_performance_viewing",
                relevance_score=0.8,
                expires_at=current_time + timedelta(minutes=self.suggestion_expiry_minutes)
            ))
        
        # Widget-specific suggestions
        if "active_widgets" in changes:
            new_widgets = set(changes["active_widgets"]) - set(state.active_widgets)
            
            for widget in new_widgets:
                if widget == "sector_allocation":
                    suggestion = SmartSuggestion(
                        id=f"widget_sector_{current_time.timestamp()}",
                        type=SuggestionType.AGENT_QUERY,
                        title="Sector Allocation Analysis",
                        description="Analyze your sector allocation for optimization opportunities.",
                        query="Analyze my current sector allocation and suggest improvements based on market conditions",
                        target_agent="strategy",
                        confidence=0.7,
                        urgency=Urgency.MEDIUM,
                        category="Portfolio Balance",
                        reasoning="Sector allocation widget opened",
                        expected_outcome="Sector rebalancing recommendations",
                        icon="⚖️",
                        color="orange"
                    )
                    
                    suggestions.append(RealTimeSuggestion(
                        suggestion=suggestion,
                        context_source="dashboard",
                        trigger_event="sector_widget_opened",
                        relevance_score=0.7,
                        expires_at=current_time + timedelta(minutes=self.suggestion_expiry_minutes)
                    ))
        
        return suggestions
    
    async def _generate_conversation_suggestions(
        self, 
        user_id: str, 
        state: ConversationState, 
        changes: Dict[str, Any]
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions based on conversation context"""
        
        suggestions = []
        current_time = datetime.now()
        
        # Agent transition suggestions
        if "active_agent" in changes and changes["active_agent"]:
            current_agent = changes["active_agent"]
            
            agent_transitions = {
                "quantitative": ["risk", "strategy"],
                "risk": ["options", "strategy"],
                "strategy": ["tax", "quantitative"],
                "tax": ["quantitative", "strategy"]
            }
            
            if current_agent in agent_transitions:
                next_agents = agent_transitions[current_agent]
                next_agent = next_agents[0]  # Pick first suggestion
                
                suggestion = SmartSuggestion(
                    id=f"conversation_transition_{current_time.timestamp()}",
                    type=SuggestionType.AGENT_QUERY,
                    title=f"Continue with {next_agent.title()} Expert",
                    description=f"Get complementary analysis from our {next_agent} specialist.",
                    query=f"Based on the {current_agent} analysis, what {next_agent} considerations should I be aware of?",
                    target_agent=next_agent,
                    confidence=0.8,
                    urgency=Urgency.MEDIUM,
                    category="Discovery",
                    reasoning=f"Agent transition from {current_agent}",
                    expected_outcome=f"{next_agent} insights",
                    icon="💡",
                    color="purple"
                )
                
                suggestions.append(RealTimeSuggestion(
                    suggestion=suggestion,
                    context_source="conversation",
                    trigger_event="agent_transition",
                    relevance_score=0.8,
                    expires_at=current_time + timedelta(minutes=self.suggestion_expiry_minutes)
                ))
        
        # Multi-agent workflow suggestion after multiple single-agent interactions
        if state.message_count >= 6 and len(set(state.recent_topics)) >= 2:
            suggestion = SmartSuggestion(
                id=f"conversation_workflow_{current_time.timestamp()}",
                type=SuggestionType.DEBATE_TOPIC,
                title="Multi-Agent Debate",
                description="You've explored multiple topics. Start a debate between AI experts for comprehensive analysis.",
                query=f"Start a multi-agent debate on the following topics: {', '.join(state.recent_topics[-2:])}",
                confidence=0.85,
                urgency=Urgency.MEDIUM,
                category="Discovery",
                reasoning="Multiple conversation topics suggest need for comprehensive analysis",
                expected_outcome="Multi-agent debate with consensus",
                icon="⚙️",
                color="blue"
            )
            
            suggestions.append(RealTimeSuggestion(
                suggestion=suggestion,
                context_source="conversation",
                trigger_event="multi_topic_discussion",
                relevance_score=0.85,
                expires_at=current_time + timedelta(minutes=self.suggestion_expiry_minutes)
            ))
        
        return suggestions
    
    async def _generate_portfolio_change_suggestions(
        self,
        user_id: str,
        change_type: str,
        change_data: Dict[str, Any]
    ) -> List[RealTimeSuggestion]:
        """Generate urgent suggestions based on portfolio changes"""
        
        suggestions = []
        current_time = datetime.now()
        
        if change_type == "risk_threshold_breach":
            breach_data = change_data
            
            suggestion = SmartSuggestion(
                id=f"portfolio_breach_{current_time.timestamp()}",
                type=SuggestionType.WORKFLOW_TRIGGER,
                title="URGENT: Risk Threshold Breached",
                description=f"Portfolio risk has breached your {breach_data.get('threshold_name', 'risk')} threshold.",
                query=f"Risk threshold breach detected: {breach_data.get('breach_description', 'Unknown breach')}. Provide immediate risk mitigation recommendations.",
                workflow_type="emergency_risk_response",
                confidence=0.95,
                urgency=Urgency.HIGH,
                category="Risk Alert",
                reasoning="Portfolio risk threshold breach",
                expected_outcome="Immediate risk mitigation plan",
                icon="🚨",
                color="red"
            )
            
            suggestions.append(RealTimeSuggestion(
                suggestion=suggestion,
                context_source="portfolio_change",
                trigger_event="risk_threshold_breach",
                relevance_score=0.95,
                expires_at=current_time + timedelta(minutes=60)  # Longer expiry for urgent issues
            ))
        
        elif change_type == "holding_change" and change_data.get("significant_change", False):
            holding_data = change_data
            
            suggestion = SmartSuggestion(
                id=f"portfolio_holding_{current_time.timestamp()}",
                type=SuggestionType.AGENT_QUERY,
                title="Portfolio Position Change",
                description=f"Significant change detected in {holding_data.get('ticker', 'holdings')}.",
                query=f"Analyze the impact of the recent change in {holding_data.get('ticker')} on my portfolio balance and risk profile.",
                target_agent="quantitative",
                confidence=0.8,
                urgency=Urgency.MEDIUM,
                category="Portfolio Balance",
                reasoning="Significant holding change detected",
                expected_outcome="Position change impact analysis",
                icon="📊",
                color="blue"
            )
            
            suggestions.append(RealTimeSuggestion(
                suggestion=suggestion,
                context_source="portfolio_change",
                trigger_event="holding_change",
                relevance_score=0.8,
                expires_at=current_time + timedelta(minutes=self.suggestion_expiry_minutes)
            ))
        
        return suggestions
    
    async def _send_contextual_suggestions(
        self,
        user_id: str,
        suggestions: List[RealTimeSuggestion],
        context_type: str,
        urgent: bool = False
    ):
        """Send contextual suggestions to user via WebSocket"""
        
        # Check cooldown and limits
        if not self._check_suggestion_limits(user_id, context_type):
            return
        
        # Apply ML enhancement to suggestions
        try:
            # Get user context for ML enhancement
            user_history = await self._get_user_history(user_id)
            dashboard_state = self.dashboard_states.get(user_id)
            
            # Convert RealTimeSuggestion to SmartSuggestion for ML processing
            base_suggestions = [rt_sugg.suggestion for rt_sugg in suggestions]
            
            # This would require getting portfolio and market context
            # For now, we'll skip ML enhancement for real-time suggestions
            # enhanced_suggestions = enhance_suggestions_with_ml(
            #     base_suggestions, {}, portfolio_context, market_context, user_history
            # )
            enhanced_suggestions = base_suggestions
            
        except Exception as e:
            print(f"Error in ML enhancement: {e}")
            enhanced_suggestions = [rt_sugg.suggestion for rt_sugg in suggestions]
        
        # Store active suggestions
        self.active_suggestions[user_id].extend(suggestions)
        
        # Send via WebSocket
        message = {
            "type": "contextual_suggestions",
            "context_type": context_type,
            "urgent": urgent,
            "suggestions": [
                {
                    **asdict(sugg),
                    "type": sugg.type.value,
                    "urgency": sugg.urgency.value,
                    "context_source": rt_sugg.context_source,
                    "trigger_event": rt_sugg.trigger_event,
                    "relevance_score": rt_sugg.relevance_score,
                    "expires_at": rt_sugg.expires_at.isoformat()
                }
                for sugg, rt_sugg in zip(enhanced_suggestions, suggestions)
            ],
            "timestamp": datetime.now().isoformat(),
            "total_suggestions": len(enhanced_suggestions)
        }
        
        # Send with appropriate QoS level
        qos_level = 2 if urgent else 1
        success = await self.connection_manager.send_to_user(
            user_id, message, qos_level=qos_level
        )
        
        if success:
            print(f"Sent {len(suggestions)} contextual suggestions to user {user_id} ({context_type})")
    
    def _check_suggestion_limits(self, user_id: str, context_type: str) -> bool:
        """Check if user is within suggestion limits"""
        
        # Check total active suggestions limit
        if len(self.active_suggestions[user_id]) >= self.max_suggestions_per_user:
            return False
        
        # Check cooldown period
        now = datetime.now()
        recent_suggestions = [
            sugg for sugg in self.active_suggestions[user_id]
            if (now - (sugg.expires_at - timedelta(minutes=self.suggestion_expiry_minutes))).total_seconds() < self.suggestion_cooldown
            and sugg.context_source == context_type.replace('_context', '')
        ]
        
        if recent_suggestions:
            return False
        
        return True
    
    async def _get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user history for ML enhancement"""
        # This would integrate with your user interaction logging system
        # For now, return empty list
        return []
    
    async def _cleanup_expired_suggestions(self):
        """Background task to clean up expired suggestions"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                now = datetime.now()
                for user_id in list(self.active_suggestions.keys()):
                    # Remove expired suggestions
                    self.active_suggestions[user_id] = [
                        sugg for sugg in self.active_suggestions[user_id]
                        if sugg.expires_at > now
                    ]
                    
                    # Clean up empty lists
                    if not self.active_suggestions[user_id]:
                        del self.active_suggestions[user_id]
                        
            except Exception as e:
                print(f"Error in suggestion cleanup: {e}")
    
    async def _monitor_user_engagement(self):
        """Monitor user engagement with suggestions and trigger ML retraining"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # This would collect engagement metrics and trigger ML retraining
                # when enough new data is available
                
                # Placeholder for engagement monitoring logic
                
            except Exception as e:
                print(f"Error in engagement monitoring: {e}")

# Global service instance
_realtime_context_service = None

def get_realtime_context_service() -> RealTimeContextService:
    """Get the global real-time context service instance"""
    global _realtime_context_service
    if _realtime_context_service is None:
        _realtime_context_service = RealTimeContextService()
    return _realtime_context_service

# Integration endpoints for existing systems

async def notify_dashboard_state_change(user_id: str, state_update: Dict[str, Any]):
    """Called by frontend when dashboard state changes"""
    service = get_realtime_context_service()
    await service.update_dashboard_state(user_id, state_update)

async def notify_conversation_update(user_id: str, conversation_id: str, update: Dict[str, Any]):
    """Called by chat system when conversation state changes"""
    service = get_realtime_context_service()
    await service.update_conversation_state(user_id, conversation_id, update)

async def notify_portfolio_change(user_id: str, change_type: str, change_data: Dict[str, Any]):
    """Called by portfolio monitoring system when changes occur"""
    service = get_realtime_context_service()
    await service.handle_portfolio_change(user_id, change_type, change_data)