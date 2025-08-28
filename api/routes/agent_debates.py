# api/routes/agent_debates.py
"""
Multi-Agent Debate API Endpoints
===============================
Revolutionary API endpoints for initiating and managing multi-agent investment debates.
The world's first autonomous AI investment advisor debate system.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
import asyncio
from enum import Enum

# Import our revolutionary debate system
from mcp.debate_engine import DebateEngine
from mcp.consensus_builder import ConsensusBuilder
from api.routes.auth import get_current_user
from services.mcp_client import get_mcp_client
from smart_suggestions.suggestion_engine import get_portfolio_context
from api.schemas import BaseResponse

router = APIRouter(prefix="/debates", tags=["Multi-Agent Debates"])

# Request/Response Models
class DebateUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class AgentDebateRequest(BaseModel):
    """Request to initiate multi-agent debate"""
    query: str = Field(..., description="Investment question for agent debate", min_length=10)
    preferred_agents: Optional[List[str]] = Field(
        None, 
        description="Preferred agents (auto-selected if not specified)",
        example=["quantitative_analyst", "market_intelligence", "tax_strategist"]
    )
    debate_rounds: int = Field(3, ge=1, le=5, description="Number of debate rounds")
    require_unanimous_consensus: bool = Field(False, description="Require unanimous agreement")
    include_minority_report: bool = Field(True, description="Include minority opinions in results")
    urgency_level: DebateUrgency = Field(DebateUrgency.MEDIUM, description="Debate urgency level")
    max_debate_duration: int = Field(900, ge=60, le=1800, description="Max debate duration in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Should I reduce portfolio risk given current market volatility?",
                "preferred_agents": ["quantitative_analyst", "market_intelligence"],
                "debate_rounds": 3,
                "require_unanimous_consensus": False,
                "include_minority_report": True,
                "urgency_level": "medium",
                "max_debate_duration": 600
            }
        }

class AgentDebateResponse(BaseModel):
    """Response from initiated debate"""
    debate_id: str
    status: str
    estimated_completion_time: datetime
    participants: List[str]
    debate_config: Dict[str, Any]
    websocket_url: str

class DebateStatusResponse(BaseModel):
    """Current status of ongoing debate"""
    debate_id: str
    status: str  # "active", "completed", "error"
    current_stage: str
    rounds_completed: int
    rounds_total: int
    participants: List[str]
    consensus_ready: bool
    confidence_score: float
    estimated_time_remaining: Optional[int] = None
    
class DebateResultsResponse(BaseModel):
    """Complete debate results"""
    debate_id: str
    query: str
    participants: Dict[str, Dict]  # agent_id -> agent_info
    rounds: List[Dict]
    final_consensus: Dict
    confidence_score: float
    duration_seconds: float
    debate_summary: str
    implementation_guidance: Dict
    minority_opinions: List[Dict]
    
class DebateRoundUpdate(BaseModel):
    """Real-time debate round update"""
    debate_id: str
    round_number: int
    stage: str
    update_type: str  # "position", "challenge", "response", "consensus"
    agent_id: Optional[str] = None
    content: Dict
    timestamp: datetime

# WebSocket Connection Manager
class DebateConnectionManager:
    """Manages WebSocket connections for live debate streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, debate_id: str):
        """Connect client to debate stream"""
        await websocket.accept()
        if debate_id not in self.active_connections:
            self.active_connections[debate_id] = []
        self.active_connections[debate_id].append(websocket)
        
    def disconnect(self, websocket: WebSocket, debate_id: str):
        """Disconnect client from debate stream"""
        if debate_id in self.active_connections:
            self.active_connections[debate_id].remove(websocket)
            if not self.active_connections[debate_id]:
                del self.active_connections[debate_id]
                
    async def broadcast_to_debate(self, debate_id: str, message: Dict):
        """Broadcast update to all clients watching debate"""
        if debate_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[debate_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.active_connections[debate_id].remove(conn)

# Global connection manager
debate_manager = DebateConnectionManager()

# Main API Endpoints

@router.post("/initiate", response_model=AgentDebateResponse)
async def initiate_agent_debate(
    portfolio_id: str,
    request: AgentDebateRequest,
    background_tasks: BackgroundTasks,
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    🎭 Initiate Revolutionary Multi-Agent Investment Debate
    
    This endpoint starts the world's first autonomous AI investment debate where
    specialized agents collaborate through structured discussion to reach optimal decisions.
    
    **Revolutionary Features:**
    - Multiple AI agents with distinct investment perspectives
    - Structured debate rounds with evidence-based arguments
    - Intelligent consensus building with minority opinion preservation
    - Real-time streaming of debate progress
    """
    
    try:
        # Get portfolio context for the debate
        portfolio_context = await get_portfolio_context(portfolio_id, current_user.id)
        
        # Validate and select agents
        if request.preferred_agents:
            available_agents = await mcp_client.get_available_agents()
            invalid_agents = [agent for agent in request.preferred_agents 
                            if agent not in available_agents]
            if invalid_agents:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid agents: {invalid_agents}. Available: {list(available_agents.keys())}"
                )
            selected_agents = request.preferred_agents
        else:
            # Auto-select optimal agents based on query
            selected_agents = await mcp_client.select_optimal_agents(
                request.query, portfolio_context
            )
        
        # Prepare debate parameters
        debate_params = {
            "rounds": request.debate_rounds,
            "require_unanimous": request.require_unanimous_consensus,
            "include_minorities": request.include_minority_report,
            "urgency": request.urgency_level.value,
            "max_duration": request.max_debate_duration,
            "portfolio_id": portfolio_id,
            "user_id": current_user.id
        }
        
        # Initiate debate through MCP
        debate_id = await mcp_client.debate_engine.initiate_debate(
            query=request.query,
            agents=selected_agents,
            portfolio_context=portfolio_context,
            debate_params=debate_params
        )
        
        # Calculate estimated completion time
        estimated_duration = request.debate_rounds * 120  # 2 minutes per round estimate
        estimated_completion = datetime.now().timestamp() + estimated_duration
        
        # Set up background monitoring
        background_tasks.add_task(monitor_debate_progress, debate_id)
        
        return AgentDebateResponse(
            debate_id=debate_id,
            status="initiated",
            estimated_completion_time=datetime.fromtimestamp(estimated_completion),
            participants=selected_agents,
            debate_config=debate_params,
            websocket_url=f"/ws/debates/{debate_id}/stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate debate: {str(e)}")

@router.get("/{debate_id}/status", response_model=DebateStatusResponse)
async def get_debate_status(
    debate_id: str,
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    📊 Get Current Status of Multi-Agent Debate
    
    Monitor the progress of an ongoing debate between AI agents.
    """
    
    try:
        status = await mcp_client.debate_engine.get_debate_status(debate_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        # Calculate estimated time remaining
        time_remaining = None
        if status["status"] == "active":
            rounds_remaining = status.get("rounds_total", 3) - status["rounds_completed"]
            time_remaining = rounds_remaining * 120  # 2 minutes per round
        
        return DebateStatusResponse(
            debate_id=debate_id,
            status=status["status"],
            current_stage=status["current_stage"],
            rounds_completed=status["rounds_completed"],
            rounds_total=status.get("rounds_total", 3),
            participants=status["participants"],
            consensus_ready=status["consensus_ready"],
            confidence_score=status["confidence_score"],
            estimated_time_remaining=time_remaining
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get debate status: {str(e)}")

@router.get("/{debate_id}/results", response_model=DebateResultsResponse)
async def get_debate_results(
    debate_id: str,
    include_detailed_rounds: bool = True,
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    🏆 Get Complete Multi-Agent Debate Results
    
    Retrieve the final results of a completed debate, including consensus,
    minority opinions, and implementation guidance.
    """
    
    try:
        results = await mcp_client.debate_engine.get_debate_results(debate_id)
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        # Filter rounds if detailed view not requested
        rounds_data = results["rounds"] if include_detailed_rounds else []
        
        return DebateResultsResponse(
            debate_id=debate_id,
            query=results["query"],
            participants=results["participants"],
            rounds=rounds_data,
            final_consensus=results["final_consensus"],
            confidence_score=results["confidence_score"],
            duration_seconds=results["duration"],
            debate_summary=results["debate_summary"],
            implementation_guidance=results["final_consensus"].get("implementation_guidance", {}),
            minority_opinions=results["final_consensus"].get("minority_opinions", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get debate results: {str(e)}")

@router.post("/{debate_id}/intervention")
async def add_user_intervention(
    debate_id: str,
    intervention: Dict[str, Any],
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    👤 Add User Intervention to Ongoing Debate
    
    Allow users to inject additional context or constraints into an active debate.
    """
    
    try:
        # Validate debate is active
        status = await mcp_client.debate_engine.get_debate_status(debate_id)
        if status.get("status") != "active":
            raise HTTPException(status_code=400, detail="Cannot intervene in inactive debate")
        
        # Add intervention to debate context
        result = await mcp_client.debate_engine.add_user_intervention(
            debate_id, intervention, current_user.id
        )
        
        # Broadcast intervention to connected clients
        await debate_manager.broadcast_to_debate(debate_id, {
            "type": "user_intervention",
            "intervention": intervention,
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "intervention_added", "result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add intervention: {str(e)}")

@router.get("/active", response_model=List[DebateStatusResponse])
async def get_active_debates(
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    📋 Get All Active Debates for User
    
    Retrieve status of all ongoing debates for the current user.
    """
    
    try:
        active_debates = await mcp_client.debate_engine.get_user_active_debates(current_user.id)
        
        debates_status = []
        for debate_id in active_debates:
            status = await mcp_client.debate_engine.get_debate_status(debate_id)
            if "error" not in status:
                debates_status.append(DebateStatusResponse(
                    debate_id=debate_id,
                    status=status["status"],
                    current_stage=status["current_stage"],
                    rounds_completed=status["rounds_completed"],
                    rounds_total=status.get("rounds_total", 3),
                    participants=status["participants"],
                    consensus_ready=status["consensus_ready"],
                    confidence_score=status["confidence_score"]
                ))
        
        return debates_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active debates: {str(e)}")

@router.delete("/{debate_id}")
async def cancel_debate(
    debate_id: str,
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    ❌ Cancel Ongoing Debate
    
    Cancel an active debate and return partial results if any.
    """
    
    try:
        result = await mcp_client.debate_engine.cancel_debate(debate_id, current_user.id)
        
        # Notify connected clients
        await debate_manager.broadcast_to_debate(debate_id, {
            "type": "debate_cancelled",
            "reason": "user_request",
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "cancelled", "partial_results": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel debate: {str(e)}")

# WebSocket Endpoint for Real-Time Debate Streaming

@router.websocket("/{debate_id}/stream")
async def stream_debate_progress(websocket: WebSocket, debate_id: str):
    """
    🌐 Real-Time Debate Progress Streaming
    
    WebSocket endpoint for watching debates unfold in real-time.
    Streams agent positions, challenges, responses, and consensus building.
    """
    
    await debate_manager.connect(websocket, debate_id)
    
    try:
        # Send initial debate status
        mcp_client = websocket.scope.get("mcp_client")  # Injected by middleware
        if mcp_client:
            initial_status = await mcp_client.debate_engine.get_debate_status(debate_id)
            await websocket.send_text(json.dumps({
                "type": "initial_status",
                "data": initial_status,
                "timestamp": datetime.now().isoformat()
            }))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message or timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client requests (ping, status requests, etc.)
                try:
                    client_request = json.loads(message)
                    if client_request.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    elif client_request.get("type") == "request_status":
                        if mcp_client:
                            status = await mcp_client.debate_engine.get_debate_status(debate_id)
                            await websocket.send_text(json.dumps({
                                "type": "status_update",
                                "data": status,
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
        debate_manager.disconnect(websocket, debate_id)
    except Exception as e:
        await websocket.close(code=1000, reason=f"Error: {str(e)}")
        debate_manager.disconnect(websocket, debate_id)

# Agent Performance and Analytics Endpoints

@router.get("/analytics/agent-performance")
async def get_agent_performance_analytics(
    timeframe_days: int = 30,
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    📈 Get Agent Performance Analytics
    
    Analyze how well different agents perform in debates and consensus building.
    """
    
    try:
        analytics = await mcp_client.get_agent_performance_analytics(
            user_id=current_user.id,
            timeframe_days=timeframe_days
        )
        
        return {
            "timeframe_days": timeframe_days,
            "agent_metrics": analytics["agent_metrics"],
            "consensus_quality": analytics["consensus_quality"],
            "debate_efficiency": analytics["debate_efficiency"],
            "recommendation_accuracy": analytics["recommendation_accuracy"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.get("/analytics/debate-quality")
async def get_debate_quality_metrics(
    debate_id: Optional[str] = None,
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    🎯 Get Debate Quality Metrics
    
    Analyze the quality of debates and consensus building process.
    """
    
    try:
        if debate_id:
            # Get metrics for specific debate
            metrics = await mcp_client.get_debate_quality_metrics(debate_id)
        else:
            # Get aggregate metrics for user's debates
            metrics = await mcp_client.get_user_debate_quality_metrics(current_user.id)
        
        return {
            "quality_score": metrics["quality_score"],
            "dimensions": metrics["dimensions"],
            "recommendations": metrics["recommendations"],
            "trends": metrics.get("trends", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality metrics: {str(e)}")

# Debate Templates and Presets

@router.get("/templates")
async def get_debate_templates():
    """
    📝 Get Debate Templates
    
    Retrieve pre-configured debate templates for common investment scenarios.
    """
    
    templates = {
        "portfolio_risk_assessment": {
            "name": "Portfolio Risk Assessment",
            "description": "Comprehensive risk analysis with multiple agent perspectives",
            "query_template": "Analyze the risk level of my portfolio and recommend adjustments",
            "recommended_agents": ["quantitative_analyst", "market_intelligence", "options_analyst"],
            "debate_rounds": 3,
            "urgency": "medium"
        },
        "rebalancing_decision": {
            "name": "Portfolio Rebalancing Decision",
            "description": "Multi-agent analysis for optimal portfolio rebalancing",
            "query_template": "Should I rebalance my portfolio and what changes should I make?",
            "recommended_agents": ["quantitative_analyst", "tax_strategist", "market_intelligence"],
            "debate_rounds": 3,
            "urgency": "low"
        },
        "market_timing_strategy": {
            "name": "Market Timing Strategy",
            "description": "Debate on market entry/exit timing with economic considerations",
            "query_template": "Given current market conditions, should I increase or decrease my equity exposure?",
            "recommended_agents": ["market_intelligence", "economic_data", "quantitative_analyst"],
            "debate_rounds": 4,
            "urgency": "high"
        },
        "tax_optimization_review": {
            "name": "Tax Optimization Review",
            "description": "Comprehensive tax strategy debate with multiple perspectives",
            "query_template": "What tax optimization strategies should I implement before year-end?",
            "recommended_agents": ["tax_strategist", "quantitative_analyst", "options_analyst"],
            "debate_rounds": 3,
            "urgency": "medium"
        },
        "options_strategy_selection": {
            "name": "Options Strategy Selection",
            "description": "Multi-agent analysis for optimal options strategies",
            "query_template": "What options strategies should I implement to enhance my portfolio?",
            "recommended_agents": ["options_analyst", "quantitative_analyst", "market_intelligence"],
            "debate_rounds": 3,
            "urgency": "medium"
        }
    }
    
    return {"templates": templates}

@router.post("/from-template/{template_id}")
async def initiate_debate_from_template(
    template_id: str,
    portfolio_id: str,
    background_tasks: BackgroundTasks,  # Move this up
    custom_query: Optional[str] = None,  # Optional parameters after required ones
    mcp_client = Depends(get_mcp_client),
    current_user = Depends(get_current_user)
):
    """
    🎭 Initiate Debate from Template
    
    Start a debate using a pre-configured template for common scenarios.
    """
    
    try:
        # Get template configuration
        templates = await get_debate_templates()
        template = templates["templates"].get(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        # Create debate request from template
        query = custom_query or template["query_template"]
        
        request = AgentDebateRequest(
            query=query,
            preferred_agents=template["recommended_agents"],
            debate_rounds=template["debate_rounds"],
            urgency_level=DebateUrgency(template["urgency"])
        )
        
        # Initiate debate
        return await initiate_agent_debate(
            portfolio_id=portfolio_id,
            request=request,
            background_tasks=background_tasks,
            mcp_client=mcp_client,
            current_user=current_user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate template debate: {str(e)}")

# Background Tasks

async def monitor_debate_progress(debate_id: str):
    """Background task to monitor debate progress and broadcast updates"""
    
    try:
        # This would be implemented to monitor MCP debate progress
        # and broadcast updates to connected WebSocket clients
        
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            # Get debate status (would use actual MCP client)
            # status = await mcp_client.debate_engine.get_debate_status(debate_id)
            
            # If debate completed or error, break
            # if status.get("status") in ["completed", "error"]:
            #     break
            
            # Broadcast updates to connected clients
            # await debate_manager.broadcast_to_debate(debate_id, {
            #     "type": "progress_update",
            #     "data": status,
            #     "timestamp": datetime.now().isoformat()
            # })
            
            break  # Placeholder - remove when implementing actual monitoring
            
    except Exception as e:
        # Log error and notify clients
        await debate_manager.broadcast_to_debate(debate_id, {
            "type": "error",
            "message": f"Monitoring error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })

# Additional Schema Models for Complex Responses

class AgentPerformanceMetrics(BaseModel):
    """Agent performance metrics"""
    agent_id: str
    debates_participated: int
    consensus_contributions: int
    arguments_accepted: float
    average_confidence: float
    accuracy_score: float
    specialization_effectiveness: float

class ConsensusQualityMetrics(BaseModel):
    """Consensus quality assessment"""
    agreement_level: float
    evidence_strength: float
    confidence_consistency: float
    minority_handling: float
    overall_quality: str

class DebateEfficiencyMetrics(BaseModel):
    """Debate process efficiency metrics"""
    average_duration: float
    rounds_to_consensus: float
    challenge_resolution_rate: float
    participant_engagement: float

# Utility Functions

def validate_debate_query(query: str) -> bool:
    """Validate that query is suitable for debate"""
    
    # Check minimum length
    if len(query.split()) < 5:
        return False
    
    # Check for investment-related keywords
    investment_keywords = [
        "portfolio", "invest", "risk", "return", "allocate", "buy", "sell",
        "strategy", "market", "stock", "bond", "option", "tax", "rebalance"
    ]
    
    query_lower = query.lower()
    has_investment_context = any(keyword in query_lower for keyword in investment_keywords)
    
    return has_investment_context

def estimate_debate_complexity(query: str, portfolio_context: Dict) -> str:
    """Estimate debate complexity level"""
    
    complexity_factors = 0
    
    # Query length factor
    if len(query.split()) > 20:
        complexity_factors += 1
    
    # Portfolio size factor
    holdings_count = len(portfolio_context.get("holdings", []))
    if holdings_count > 10:
        complexity_factors += 1
    
    # Multi-topic query
    topics = ["risk", "tax", "option", "timing", "allocation"]
    topic_count = sum(1 for topic in topics if topic in query.lower())
    if topic_count > 2:
        complexity_factors += 1
    
    # Urgency indicators
    urgent_words = ["urgent", "immediate", "crisis", "emergency"]
    if any(word in query.lower() for word in urgent_words):
        complexity_factors += 1
    
    if complexity_factors >= 3:
        return "high"
    elif complexity_factors >= 1:
        return "medium"
    else:
        return "low"

# Export router for main application
__all__ = ["router", "DebateConnectionManager", "debate_manager"]