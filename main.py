# main.py - Enhanced with Workflow API Support (CORRECTED VERSION)
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from datetime import datetime
from urllib.parse import unquote

# Import your enhanced orchestrator
from agents.orchestrator import FinancialOrchestrator
from api.routes import users, auth, portfolios
from db.session import get_db
from db import crud
from api.schemas import User
from core.data_handler import get_market_data_for_portfolio
from api.routes import alerts

# --- API Application Setup ---
app = FastAPI(
    title="Gertie.ai - Multi-Agent Financial Platform",
    description="An API for interacting with a team of specialized financial AI agents with workflow capabilities.",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "null"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# === WebSocket Integration for Real-Time Notifications ===
from fastapi import WebSocket, WebSocketDisconnect, Query
from datetime import datetime, timezone
import json

# WebSocket components (graceful fallback if not available)
try:
    from websocket.connection_manager import get_connection_manager, start_websocket_heartbeat
    WEBSOCKET_AVAILABLE = True
    print("✅ WebSocket components loaded successfully")
except ImportError:
    print("⚠️ WebSocket components not found - will be disabled until components are added")
    WEBSOCKET_AVAILABLE = False

# Initialize WebSocket manager
if WEBSOCKET_AVAILABLE:
    manager = get_connection_manager()
else:
    manager = None



# Include the routers
app.include_router(users.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(portfolios.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])

# Create a single, long-lived instance of the enhanced orchestrator
orchestrator = FinancialOrchestrator()

# --- API Request & Response Models ---
class ChatRequest(BaseModel):
    query: str
    workflow_enabled: Optional[bool] = True

class WorkflowStatusRequest(BaseModel):
    session_id: str

class WorkflowResponse(BaseModel):
    success: bool
    workflow_triggered: bool
    session_id: Optional[str] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    progress_percentage: Optional[int] = None
    step_description: Optional[str] = None
    partial_results: Optional[Dict] = None
    data: Dict[str, Any]

class DebateRequest(BaseModel):
    query: str
    debate_mode: str = "full_panel"
    participants: Optional[List[str]] = None

class DebateResponse(BaseModel):
    success: bool
    debate_id: str
    agents: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    consensus: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None

# --- Formatting Functions ---
def format_agent_response_for_frontend(result, agent_type):
    """Format sophisticated agent responses for frontend consumption"""
    
    agent_info = {
        "quantitative": {"name": "Dr. Sarah Chen", "specialization": "Quantitative Analysis & Risk Metrics"},
        "strategy": {"name": "Marcus Rodriguez", "specialization": "Investment Strategy & Portfolio Optimization"}, 
        "risk": {"name": "Elena Volkov", "specialization": "Risk Management & Hedging"},
        "rebalancing": {"name": "Alex Thompson", "specialization": "Portfolio Rebalancing"},
        "regime": {"name": "Dr. James Liu", "specialization": "Market Regime Analysis"},
        "behavioral": {"name": "Dr. Maya Patel", "specialization": "Behavioral Finance"},
        "scenario": {"name": "Robert Kim", "specialization": "Scenario Analysis & Stress Testing"},
        "tutor": {"name": "Professor Williams", "specialization": "Financial Education"},
        "screener": {"name": "Alex Morgan", "specialization": "Security Screening & Selection"}
    }
    
    agent = agent_info.get(agent_type, {"name": "Financial Advisor", "specialization": "General Analysis"})
    
    if not result or not result.get("success", True):
        error_msg = result.get("error", "Analysis could not be completed") if result else "No response from agent"
        
        suggestions = {
            "strategy": "Try asking: 'what should I do with my portfolio?' or 'recommend investment strategy'",
            "screener": "Try asking: 'find stocks for my portfolio' or 'recommend quality stocks'",
            "quantitative": "Try asking: 'analyze portfolio risk' or 'calculate risk metrics'"
        }
        
        return {
            "agent": agent["name"],
            "agent_type": agent_type,
            "specialization": agent["specialization"],
            "message": f"I encountered an issue: {error_msg}. {suggestions.get(agent_type, 'Please rephrase your question.')}",
            "metrics": {},
            "recommendations": ["Try rephrasing your question", "Use specific financial terms", "Ask about portfolio analysis"],
            "confidence": 0.1,
            "timestamp": datetime.now().isoformat(),
            "error": True
        }
    
    # Handle workflow responses
    if result.get("workflow_type") == "multi_stage_analysis":
        return format_workflow_response_for_frontend(result)
    
    # Extract metrics from sophisticated agent response
    metrics = {}
    data = result.get("data", {}) if result else {}
    
    # Extract performance and risk metrics
    if "risk_adjusted_ratios" in data:
        ratios = data["risk_adjusted_ratios"]
        if "sharpe_ratio" in ratios:
            metrics["Sharpe Ratio"] = f"{ratios['sharpe_ratio']:.3f}"
        if "sortino_ratio" in ratios:
            metrics["Sortino Ratio"] = f"{ratios['sortino_ratio']:.3f}"
    
    if "performance_stats" in data:
        perf = data["performance_stats"]
        if "annualized_return_pct" in perf:
            metrics["Annual Return"] = f"{perf['annualized_return_pct']:.1f}%"
        if "annualized_volatility_pct" in perf:
            metrics["Volatility"] = f"{perf['annualized_volatility_pct']:.1f}%"
    
    # Extract SecurityScreener recommendations
    if result.get("agent_used") == "SecurityScreener" and "recommendations" in result:
        recommendations = []
        for rec in result["recommendations"][:3]:
            ticker = rec.get("ticker", "")
            score = rec.get("overall_score", 0)
            recommendations.append(f"Buy {ticker} (Score: {score:.2f})")
    else:
        recommendations = [
            "Review analysis recommendations carefully",
            "Consider implementing suggested changes gradually",
            "Monitor portfolio metrics regularly"
        ]
    
    return {
        "agent": agent["name"],
        "agent_type": agent_type,
        "specialization": agent["specialization"],
        "message": result.get("summary", "Analysis completed successfully.") if result else "Analysis in progress...",
        "metrics": metrics,
        "recommendations": recommendations,
        "confidence": 0.92 if result.get("success", True) else 0.3,
        "timestamp": datetime.now().isoformat(),
        "full_data": data
    }

def format_workflow_response_for_frontend(result: Dict) -> Dict:
    """Format multi-stage workflow responses for frontend"""
    
    workflow_status = result.get("workflow_status", {})
    detailed_results = result.get("detailed_results", {})
    
    # Extract key metrics from each stage
    metrics = {}
    
    # From screening stage
    if "screening" in detailed_results:
        screening = detailed_results["screening"]
        if "recommendations" in screening:
            metrics["Stocks Analyzed"] = str(screening.get("universe_screened", "500+"))
            metrics["Top Recommendations"] = str(len(screening["recommendations"]))
    
    # From analysis stage  
    if "analysis" in detailed_results:
        analysis = detailed_results["analysis"]
        if "portfolio_characteristics" in analysis:
            chars = analysis["portfolio_characteristics"]
            if "concentration_risk" in chars:
                metrics["Concentration Risk"] = chars["concentration_risk"].title()
    
    # Create comprehensive recommendations
    recommendations = result.get("recommendations", [])
    if "implementation_plan" in result:
        plan_steps = result["implementation_plan"]
        for step in plan_steps[:2]:  # First 2 steps
            recommendations.append(f"Step {step['step']}: {step['action']}")
    
    return {
        "agent": "AI Investment Team",
        "agent_type": "workflow",
        "specialization": "Multi-Stage Portfolio Analysis",
        "message": result.get("summary", "Comprehensive analysis completed."),
        "metrics": metrics,
        "recommendations": recommendations,
        "confidence": result.get("confidence_score", 0.92) * 100,
        "timestamp": datetime.now().isoformat(),
        "workflow_data": {
            "session_id": workflow_status.get("session_id"),
            "steps_completed": workflow_status.get("steps_completed", []),
            "progress": workflow_status.get("progress", {}),
            "detailed_results": detailed_results
        },
        "full_data": detailed_results
    }

def get_workflow_step_description(step: int) -> str:
    """Get human-readable description for workflow step"""
    descriptions = {
        1: "Formulating investment strategy...",
        2: "Screening securities for opportunities...", 
        3: "Performing quantitative risk analysis...",
        4: "Synthesizing recommendations..."
    }
    return descriptions.get(step, "Processing...")

# --- API Endpoints ---

@app.post("/api/v1/chat", tags=["Agent Interaction"])
def chat_with_agent_team(
    request: ChatRequest,
    agent_type: Optional[str] = None,
    db: Session = Depends(get_db), 
    current_user: User = Depends(auth.get_current_user)
):
    """Enhanced chat with workflow support and agent-specific routing"""
    
    try:
        agent_mapping = {
            "quantitative": "QuantitativeAnalystAgent",
            "dr_sarah": "QuantitativeAnalystAgent", 
            "strategy": "StrategyArchitectAgent",
            "marcus": "StrategyArchitectAgent",
            "risk": "HedgingStrategistAgent", 
            "elena": "HedgingStrategistAgent",
            "rebalancing": "StrategyRebalancingAgent",
            "backtesting": "StrategyBacktesterAgent",
            "regime": "RegimeForecastingAgent",
            "behavioral": "BehavioralFinanceAgent",
            "scenario": "ScenarioSimulationAgent",
            "tutor": "FinancialTutorAgent",
            "screener": "SecurityScreenerAgent"
        }
        
        # If specific agent requested, route directly
        if agent_type and agent_type in agent_mapping:
            target_agent_name = agent_mapping[agent_type]
            
            # Verify agent exists and has run method
            if target_agent_name not in orchestrator.roster:
                return {
                    "error": f"Agent {target_agent_name} not found in orchestrator roster",
                    "available_agents": list(orchestrator.roster.keys())
                }
            
            agent_instance = orchestrator.roster[target_agent_name]
            if not hasattr(agent_instance, 'run'):
                return {
                    "error": f"Agent {target_agent_name} does not have run() method",
                    "agent_type": type(agent_instance).__name__
                }
            
            user_portfolios = crud.get_user_portfolios(db=db, user_id=current_user.id)
            portfolio_context = {}
            if user_portfolios:
                portfolio_context = get_market_data_for_portfolio(user_portfolios[0].holdings)
            
            # Enhance query for specific agents
            enhanced_query = request.query
            if agent_type == "strategy" and not any(word in request.query.lower() for word in ["rebalance", "momentum", "mean-reversion"]):
                tickers = [h.asset.ticker for h in user_portfolios[0].holdings if h.asset] if user_portfolios else []
                if tickers:
                    enhanced_query = f"recommend portfolio strategy for my holdings: {', '.join(tickers[:5])}"
                else:
                    enhanced_query = "recommend portfolio optimization strategy"
            elif agent_type == "screener":
                if not any(word in request.query.lower() for word in ["find", "screen", "recommend"]):
                    enhanced_query = f"find quality stocks that complement my portfolio: {request.query}"
            
            result = agent_instance.run(enhanced_query, context=portfolio_context)
            return format_agent_response_for_frontend(result, agent_type)
        
        # Use enhanced orchestrator with workflow support
        result_data = orchestrator.route_query(
            user_query=request.query, 
            db_session=db, 
            current_user=current_user
        )
        
        # Check if this was a workflow response
        if result_data.get("workflow_type") == "multi_stage_analysis":
            return format_workflow_response_for_frontend(result_data)
        
        return {"data": result_data}
        
    except Exception as e:
        print(f"💥 Chat endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": f"Chat failed: {str(e)}",
            "success": False
        }

@app.get("/api/v1/workflow/status/{session_id}", tags=["Workflow Management"])
def get_workflow_status(
    session_id: str,
    current_user: User = Depends(auth.get_current_user)
):
    """Get current status of a workflow session"""
    
    status = orchestrator.get_workflow_status(session_id)
    
    if not status:
        return {
            "success": False,
            "error": "Workflow session not found"
        }
    
    return {
        "success": True,
        "workflow_status": status,
        "step_description": get_workflow_step_description(status.get("progress", {}).get("current_step", 1))
    }

@app.post("/api/v1/workflow/start", tags=["Workflow Management"])
def start_workflow_manually(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user)
):
    """Start multi-stage workflow with enhanced error handling"""
    
    try:
        print(f"🚀 Starting workflow for user {current_user.id}")
        print(f"📝 Query: {request.query}")
        
        # Use the enhanced orchestrator's workflow capabilities
        result = orchestrator.route_query(
            user_query=request.query,
            db_session=db,
            current_user=current_user
        )
        
        print(f"📊 Orchestrator result success: {result.get('success', True)}")
        
        # Check if this triggered a workflow
        if result.get("workflow_type") == "multi_stage_analysis":
            workflow_status = result.get("workflow_status", {})
            
            return WorkflowResponse(
                success=True,
                workflow_triggered=True,
                session_id=workflow_status.get("session_id"),
                current_step=workflow_status.get("progress", {}).get("current_step", 1),
                total_steps=workflow_status.get("progress", {}).get("total_steps", 4),
                progress_percentage=workflow_status.get("progress", {}).get("percentage", 25),
                step_description=get_workflow_step_description(workflow_status.get("progress", {}).get("current_step", 1)),
                data=result
            )
        else:
            # Single agent response - not a workflow
            return WorkflowResponse(
                success=True,
                workflow_triggered=False,
                data=result
            )
            
    except Exception as e:
        print(f"💥 Workflow start error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return WorkflowResponse(
            success=False,
            workflow_triggered=False,
            data={"error": f"Workflow failed to start: {str(e)}"}
        )

@app.get("/api/v1/test/orchestrator", tags=["Testing"])
def test_orchestrator():
    """Test endpoint to verify orchestrator is working"""
    try:
        # Test that all agents are properly initialized
        agent_status = {}
        for agent_name, agent_instance in orchestrator.roster.items():
            try:
                # Test that agent has required methods
                has_run = hasattr(agent_instance, 'run')
                has_name = hasattr(agent_instance, 'name')
                has_purpose = hasattr(agent_instance, 'purpose')
                
                agent_status[agent_name] = {
                    "initialized": True,
                    "has_run_method": has_run,
                    "has_name": has_name,
                    "has_purpose": has_purpose,
                    "name": getattr(agent_instance, 'name', 'Unknown'),
                    "purpose": getattr(agent_instance, 'purpose', 'Unknown')
                }
            except Exception as e:
                agent_status[agent_name] = {
                    "initialized": False,
                    "error": str(e)
                }
        
        return {
            "success": True,
            "orchestrator_status": "healthy",
            "total_agents": len(orchestrator.roster),
            "active_workflows": len(orchestrator.active_workflows) if hasattr(orchestrator, 'active_workflows') else 0,
            "agents": agent_status
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "orchestrator_status": "error"
        }

@app.post("/api/v1/follow-up", tags=["Agent Interaction"])
def execute_follow_up_analysis(
    original_result: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user)
):
    """Execute follow-up analysis (e.g., SecurityScreener → QuantitativeAnalyst)"""
    
    try:
        result = orchestrator.execute_follow_up_analysis(
            original_result=original_result,
            db_session=db,
            current_user=current_user
        )
        
        return {"data": result}
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Follow-up analysis failed: {str(e)}"
        }

@app.post("/api/v1/debate", response_model=DebateResponse, tags=["AI Debate"])
def start_ai_debate(
    request: DebateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user)
):
    """Start a multi-agent debate with enhanced SecurityScreener participation"""
    
    user_portfolios = crud.get_user_portfolios(db=db, user_id=current_user.id)
    portfolio_context = {}
    if user_portfolios:
        portfolio_context = get_market_data_for_portfolio(user_portfolios[0].holdings)
    
    # Enhanced core agents including SecurityScreener
    core_agents = [
        "QuantitativeAnalystAgent",
        "StrategyArchitectAgent", 
        "StrategyRebalancingAgent",
        "SecurityScreenerAgent"
    ]
    
    query_lower = request.query.lower()
    
    if any(word in query_lower for word in ["hedge", "protect", "risk"]):
        core_agents.append("HedgingStrategistAgent")
    if any(word in query_lower for word in ["regime", "forecast", "economic"]):
        core_agents.append("RegimeForecastingAgent")
    
    debate_results = []
    
    for agent_name in core_agents:
        if agent_name in orchestrator.roster:
            agent = orchestrator.roster[agent_name]
            
            # Enhanced query for SecurityScreener in debates
            enhanced_query = request.query
            if agent_name == "SecurityScreenerAgent":
                enhanced_query = f"Based on this investment question: '{request.query}', provide specific stock recommendations with detailed rationale."
            
            result = agent.run(enhanced_query, context=portfolio_context)
            
            debate_results.append({
                "agent_id": agent_name.lower().replace("agent", ""),
                "agent_name": agent.name,
                "specialization": agent.purpose,
                "position": result.get("summary", "Analysis in progress..."),
                "confidence": result.get("confidence", 0.6),
                "supporting_evidence": result.get("data", {}),
                "recommendations": result.get("recommendations", [])
            })
    
    # Enhanced consensus with SecurityScreener input
    screener_result = next((r for r in debate_results if r["agent_id"] == "securityscreener"), None)
    specific_recommendations = []
    
    if screener_result and screener_result.get("supporting_evidence"):
        evidence = screener_result["supporting_evidence"]
        if "recommendations" in evidence:
            recs = evidence["recommendations"]
            specific_recommendations = [f"Buy {r.get('ticker', 'N/A')} - {r.get('rationale', 'Quality opportunity')}" for r in recs[:3]]
    
    consensus = {
        "agreement_level": 87.0,
        "final_recommendation": "Implement systematic portfolio enhancement with factor-based security selection",
        "supporting_arguments": [
            "Multi-factor analysis validates security selection approach",
            "Risk metrics support defensive positioning with quality stocks",
            "Quantitative analysis confirms portfolio optimization potential",
            "Security screening provides actionable implementation path"
        ],
        "key_risks": [
            "Market volatility during implementation period",
            "Factor rotation risk in current environment",
            "Execution timing considerations"
        ]
    }
    
    return DebateResponse(
        success=True,
        debate_id=f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        agents=[{
            "id": r["agent_id"],
            "name": r["agent_name"], 
            "specialization": r["specialization"],
            "confidence": r["confidence"]
        } for r in debate_results],
        messages=[{
            "agent_id": r["agent_id"],
            "agent_name": r["agent_name"],
            "content": r["position"],
            "phase": "position",
            "timestamp": datetime.now().isoformat(),
            "confidence": r["confidence"]
        } for r in debate_results],
        consensus=consensus,
        recommendations=specific_recommendations if specific_recommendations else [
            "Consider systematic portfolio rebalancing",
            "Implement factor-based security selection", 
            "Monitor risk metrics during transition"
        ]
    )

@app.get("/api/v1/debate/agents", tags=["AI Debate"])
def get_available_debate_agents():
    """Get list of available agents for debate participation"""
    
    agents_info = []
    for agent_name, agent_instance in orchestrator.roster.items():
        agents_info.append({
            "id": agent_name.lower().replace("agent", ""),
            "name": agent_instance.name,
            "specialization": agent_instance.purpose,
            "available": True
        })
    
    return {"agents": agents_info}

@app.get("/")
def read_root():
    return {"message": "Welcome to Gertie.ai API v2.0 with Enhanced Multi-Agent Workflows"}



# === WebSocket Endpoints for Real-Time Notifications ===

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, token: str = None):
    # Decode URL-encoded user_id
    decoded_user_id = unquote(user_id)
    
    # Verify JWT token (if needed)
    # if not verify_token(token):
    #     await websocket.close(code=4001)
    #     return
        
    await manager.connect(websocket, decoded_user_id)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Optional: Handle incoming messages
    except WebSocketDisconnect:
        await manager.disconnect(websocket, decoded_user_id)

@app.get("/api/v1/websocket/stats", tags=["WebSocket"])
async def get_websocket_stats():
    """Get current WebSocket connection statistics"""
    
    if not WEBSOCKET_AVAILABLE or not manager:
        return {
            "status": "unavailable",
            "message": "WebSocket service not available",
            "websocket_enabled": False
        }
    
    try:
        stats = manager.get_connection_stats()
        return {
            "status": "success",
            "websocket_enabled": True,
            "stats": stats
        }
    except Exception as e:
        print(f"❌ Error getting WebSocket stats: {e}")
        return {
            "status": "error",
            "message": "Failed to get connection statistics",
            "error": str(e)
        }

@app.post("/api/v1/websocket/test-notification/{user_id}", tags=["WebSocket"])
async def send_test_notification(user_id: str):
    """Send a test risk alert notification to a specific user"""
    
    if not WEBSOCKET_AVAILABLE or not manager:
        return {
            "status": "unavailable",
            "message": "WebSocket service not available"
        }
    
    # Create test risk alert data
    test_alert_data = {
        "portfolio_id": "test_portfolio_websocket",
        "portfolio_name": "Test Portfolio (WebSocket)",
        "risk_score": 78.5,
        "risk_change_pct": 23.7,
        "volatility": 0.31,
        "threshold_breached": True,
        "severity": "high",
        "alert_id": f"test_alert_{datetime.now().strftime('%H%M%S')}"
    }
    
    try:
        success = await manager.send_risk_alert(user_id, test_alert_data)
        
        return {
            "status": "success" if success else "no_connection",
            "message": f"Test notification {'sent successfully' if success else 'failed - no active connections'} to user {user_id}",
            "alert_data": test_alert_data,
            "user_id": user_id
        }
        
    except Exception as e:
        print(f"❌ Error sending test notification: {e}")
        return {
            "status": "error",
            "message": f"Failed to send test notification: {str(e)}"
        }

@app.post("/api/v1/websocket/broadcast", tags=["WebSocket"])
async def broadcast_system_message(message_data: dict):
    """Broadcast a system message to all connected users"""
    
    if not WEBSOCKET_AVAILABLE or not manager:
        return {
            "status": "unavailable", 
            "message": "WebSocket service not available"
        }
    
    try:
        count = await manager.broadcast_system_message(message_data)
        
        return {
            "status": "success",
            "message": f"System message broadcast to {count} users",
            "broadcast_count": count
        }
        
    except Exception as e:
        print(f"❌ Error broadcasting message: {e}")
        return {
            "status": "error",
            "message": f"Failed to broadcast: {str(e)}"
        }

# === Helper Functions for Risk Attribution Integration ===

async def send_risk_alert_notification(user_id: str, risk_data: dict, workflow_id: str = None):
    """
    Send risk alert notification via WebSocket
    Call this from your risk attribution service when thresholds are breached
    
    Args:
        user_id: User ID to notify
        risk_data: Risk calculation results from your risk attribution system
        workflow_id: Optional workflow session ID for linking to analysis
    
    Returns:
        bool: True if notification sent successfully
    """
    
    if not WEBSOCKET_AVAILABLE or not manager:
        print(f"⚠️ WebSocket not available - cannot send risk alert to user {user_id}")
        return False
    
    # Format alert data for WebSocket transmission
    alert_data = {
        "portfolio_id": risk_data.get('portfolio_id'),
        "portfolio_name": risk_data.get('portfolio_name', 'Portfolio'),
        "risk_score": risk_data.get('risk_score'),
        "risk_change_pct": risk_data.get('risk_score_change_pct'),
        "volatility": risk_data.get('volatility'),
        "threshold_breached": risk_data.get('is_threshold_breach', False),
        "severity": "high" if risk_data.get('risk_score', 0) > 80 else "medium",
        "workflow_id": workflow_id,
        "alert_id": risk_data.get('snapshot_id'),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        success = await manager.send_risk_alert(user_id, alert_data)
        print(f"📤 Risk alert notification {'sent' if success else 'failed'} to user {user_id}")
        return success
        
    except Exception as e:
        print(f"❌ Failed to send risk alert to user {user_id}: {e}")
        return False

async def send_workflow_update_notification(user_id: str, workflow_data: dict):
    """
    Send workflow progress update via WebSocket
    Call this from your workflow orchestrator to notify users of analysis progress
    
    Args:
        user_id: User ID to notify
        workflow_data: Workflow status and progress information
    
    Returns:
        bool: True if notification sent successfully
    """
    
    if not WEBSOCKET_AVAILABLE or not manager:
        print(f"⚠️ WebSocket not available - cannot send workflow update to user {user_id}")
        return False
    
    try:
        success = await manager.send_workflow_update(user_id, workflow_data)
        print(f"📤 Workflow update {'sent' if success else 'failed'} to user {user_id}")
        return success
        
    except Exception as e:
        print(f"❌ Failed to send workflow update to user {user_id}: {e}")
        return False

# === Enhanced Startup Event ===

@app.on_event("startup")
async def startup_websocket_services():
    """Initialize WebSocket services on application startup"""
    
    print("🚀 Initializing WebSocket services...")
    
    if WEBSOCKET_AVAILABLE:
        try:
            # Start the WebSocket heartbeat task
            start_websocket_heartbeat()
            print("✅ WebSocket services initialized successfully")
            print("🔌 Real-time notifications enabled")
            
        except Exception as e:
            print(f"❌ Failed to initialize WebSocket services: {e}")
            print("⚠️ Continuing without real-time notifications")
    else:
        print("⚠️ WebSocket components not available")
        print("📋 To enable real-time notifications:")
        print("   1. Create websocket/ directory")
        print("   2. Add connection_manager.py")
        print("   3. Restart the application")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)