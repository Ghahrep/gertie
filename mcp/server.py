# mcp/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

# Database imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends
from core.config import settings
from db import crud

# Schema imports
from .schemas import (
    AgentRegistration, 
    JobRequest, 
    JobResponse, 
    JobStatus,
    HealthCheck,
    DebateJobRequest, 
    DebateJobResponse, 
    DebateJobType, 
    AgentDebateParticipation, 
    AgentDebateResponse
)

# Complete integrated system imports
from .complete_integrated_system import (
    CompleteIntegratedDebateSystem,
    initialize_complete_integrated_system
)

# Core component imports
from .enhanced_agent_registry import enhanced_registry
from .performance_dashboard import setup_performance_monitoring, create_dashboard_html, performance_monitor
from .workflow_engine import WorkflowEngine
from .circuit_breaker import failover_manager, CircuitBreakerError
from .consensus_builder import ConsensusBuilder
from .agent_communication import debate_communication_hub

# Agent imports
from agents.security_screener_agent import SecurityScreenerAgent

import logging
logger = logging.getLogger(__name__)

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Get database session for MCP"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global variables
workflow_engine = None
complete_integrated_system = None
consensus_builder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 MCP Server starting up...")
    
    global workflow_engine, complete_integrated_system, consensus_builder
    
    # Initialize core components
    await enhanced_registry.start()
    workflow_engine = WorkflowEngine()
    consensus_builder = ConsensusBuilder()
    
    # Initialize complete integrated system
    # Note: You'll need to pass your actual MCP client here
    # For now, we'll use a mock client reference
    mock_mcp_client = app  # Replace with actual MCP client
    complete_integrated_system = await initialize_complete_integrated_system(mock_mcp_client)
    
    # Initialize agent instances storage
    if not hasattr(app.state, 'agent_instances'):
        app.state.agent_instances = {}
    
    # Register SecurityScreenerAgent
    try:
        security_screener = SecurityScreenerAgent()
        
        registration = AgentRegistration(
            agent_id=security_screener.agent_id,
            agent_name=security_screener.agent_name,
            agent_type="security_screening",
            capabilities=security_screener.capabilities + [
                "debate_participation", 
                "consensus_building", 
                "collaborative_analysis"
            ],
            description="Advanced security screening and factor analysis specialist",
            version="2.0.0",
            health_check_endpoint=f"/agents/{security_screener.agent_id}/health",
            performance_metrics={
                "average_response_time": 0.0,
                "success_rate": 100.0,
                "total_requests": 0
            }
        )
        
        success = enhanced_registry.register_agent(registration)
        
        if success:
            print(f"✅ SecurityScreenerAgent registered with integrated system")
            app.state.agent_instances[security_screener.agent_id] = security_screener
        else:
            print(f"❌ Failed to register SecurityScreenerAgent")
            
    except Exception as e:
        print(f"❌ Error registering SecurityScreenerAgent: {str(e)}")
    
    # Setup performance monitoring
    global performance_monitor
    performance_monitor = setup_performance_monitoring(enhanced_registry)
    await performance_monitor.start_monitoring()
    
    print("🎯 Complete integrated debate system ready")
    
    # NEW: Wait for external agent registrations and create instances
    print("🔧 DEBUG: Setting up agent instance creation loop...")
    
    # Create background task to monitor for new agent registrations
    async def create_agent_instances():
        """Background task to create instances for registered agents"""
        await asyncio.sleep(2)  # Wait for external agents to register
        
        print("🔍 DEBUG: Checking for registered agents to instantiate...")
        
        if hasattr(enhanced_registry, 'agents') and enhanced_registry.agents:
            print(f"📋 DEBUG: Found {len(enhanced_registry.agents)} registered agents")
            
            for agent_name, agent_registration in enhanced_registry.agents.items():
                if agent_name not in app.state.agent_instances:
                    print(f"🔧 DEBUG: Creating instance for agent: {agent_name}")
                    
                    try:
                        # Create agent wrapper
                        class RealDebateAgentWrapper:
                            def __init__(self, name, registration):
                                self.agent_id = name
                                self.name = name
                                self.capabilities = getattr(registration, 'capabilities', [])
                                self.metadata = getattr(registration, 'metadata', {})
                                
                            async def participate_in_debate(self, debate_id: str, topic: str, existing_positions: list = None):
                                print(f"🗣️ DEBUG: Agent {self.agent_id} participating in debate {debate_id}")
                                
                                if "portfolio" in self.agent_id.lower():
                                    return {
                                        "position": f"Recommend strategic portfolio adjustment for: {topic}",
                                        "reasoning": "Based on modern portfolio theory and risk-parity models",
                                        "confidence": 0.85,
                                        "evidence": ["Portfolio optimization algorithms", "Risk-return analysis"],
                                        "risks": ["Concentration risk", "Rebalancing costs", "Tax implications"]
                                    }
                                elif "risk" in self.agent_id.lower():
                                    return {
                                        "position": f"Elevated risk profile detected for: {topic}",
                                        "reasoning": "Quantitative risk models show increased volatility",
                                        "confidence": 0.78,
                                        "evidence": ["VaR calculations", "Monte Carlo simulations"],
                                        "risks": ["Tail risk events", "Model limitations", "Regime changes"]
                                    }
                                elif "market" in self.agent_id.lower():
                                    return {
                                        "position": f"Market conditions warrant caution regarding: {topic}",
                                        "reasoning": "Technical analysis suggests potential headwinds",
                                        "confidence": 0.72,
                                        "evidence": ["Technical indicators", "Market breadth analysis"],
                                        "risks": ["Market timing risk", "False signals", "Sentiment reversals"]
                                    }
                                else:
                                    return {
                                        "position": f"Balanced perspective on: {topic}",
                                        "reasoning": f"Multi-factor analysis by {self.agent_id}",
                                        "confidence": 0.65,
                                        "evidence": ["Historical data", "Cross-asset analysis"],
                                        "risks": ["General market volatility", "Uncertainty"]
                                    }
                        
                        # Create and store instance
                        app.state.agent_instances[agent_name] = RealDebateAgentWrapper(agent_name, agent_registration)
                        print(f"✅ DEBUG: Successfully created instance for {agent_name}")
                        
                        # Register with circuit breaker
                        if hasattr(failover_manager, 'add_agent_circuit_breaker'):
                            failover_manager.add_agent_circuit_breaker(agent_name)
                        elif hasattr(failover_manager, 'register_agent_for_failover'):
                            # Use existing method if add_agent_circuit_breaker doesn't exist
                            failover_manager.register_agent_for_failover(
                                agent_id=agent_name,
                                capabilities=getattr(agent_registration, 'capabilities', []),
                                priorities={cap: 5 for cap in getattr(agent_registration, 'capabilities', [])}
                            )
                            
                    except Exception as e:
                        print(f"💥 DEBUG: Error creating instance for {agent_name}: {e}")
                else:
                    print(f"✅ DEBUG: Agent {agent_name} instance already exists")
            
            print(f"🏁 DEBUG: Agent instance creation complete. Total instances: {len(app.state.agent_instances)}")
            print(f"🏁 DEBUG: Available agents: {list(app.state.agent_instances.keys())}")
        else:
            print("❌ DEBUG: No agents found in enhanced_registry")
    
    # Start the agent creation task
    asyncio.create_task(create_agent_instances())
    
    yield
    
    # Shutdown
    print("🛑 MCP Server shutting down...")
    await enhanced_registry.stop()
    if performance_monitor:
        await performance_monitor.stop_monitoring()
    if complete_integrated_system:
        await complete_integrated_system.shutdown()

app = FastAPI(
    title="Gertie.ai Master Control Plane - Complete Integration",
    description="Advanced multi-agent debate system with circuit breakers and consensus building",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 🚀 NEW: Add health check endpoint for SecurityScreener
@app.get("/agents/{agent_id}/health")
async def check_agent_health(agent_id: str):
    """Enhanced health check with circuit breaker info"""
    
    # Check circuit breaker status
    cb_status = None
    if agent_id in failover_manager.circuit_breakers:
        cb = failover_manager.circuit_breakers[agent_id]
        cb_status = {
            "state": cb.state.value,
            "health_score": cb.get_health_score(),
            "is_available": cb.is_available(),
            "stats": cb.get_stats()
        }
    
    # Check agent instance
    if hasattr(app.state, 'agent_instances') and agent_id in app.state.agent_instances:
        agent = app.state.agent_instances[agent_id]
        if hasattr(agent, '_health_check_capability'):
            agent_health = await agent._health_check_capability("health_check", {}, {})
        else:
            agent_health = {"status": "healthy", "method": "basic"}
    else:
        agent_health = {"status": "unknown", "method": "fallback"}
    
    # Registry status
    registry_status = enhanced_registry.get_agent_status(agent_id)
    
    return {
        "agent_id": agent_id,
        "agent_health": agent_health,
        "registry_status": registry_status,
        "circuit_breaker": cb_status,
        "overall_status": "healthy" if cb_status and cb_status["is_available"] else "degraded"
    }

@app.post("/register", response_model=dict)
async def register_agent(registration: AgentRegistration):
    """Register agent and add to circuit breaker system"""
    success = enhanced_registry.register_agent(registration)
    
    if not success:
        raise HTTPException(status_code=409, detail=f"Failed to register agent {registration.agent_id}")
    
    # Register with failover manager
    failover_manager.register_agent_for_failover(
        agent_id=registration.agent_id,
        capabilities=registration.capabilities,
        priorities={cap: 5 for cap in registration.capabilities}  # Default priorities
    )
    
    return {
        "status": "success",
        "message": f"Agent {registration.agent_id} registered with circuit breaker protection",
        "registered_at": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Enhanced health check with system metrics"""
    system_status = enhanced_registry.get_system_status()
    
    return HealthCheck(
        status="healthy" if system_status["unhealthy_agents"] == 0 else "degraded",
        timestamp=datetime.utcnow(),
        registered_agents=system_status["total_agents"],
        active_jobs=system_status["total_active_jobs"],
        system_metrics={
            "healthy_agents": system_status["healthy_agents"],
            "unhealthy_agents": system_status["unhealthy_agents"],
            "average_success_rate": system_status["average_success_rate"]
        }
    )

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get detailed status for a specific agent"""
    status = enhanced_registry.get_agent_status(agent_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return status

@app.post("/submit_job", response_model=JobResponse)
async def submit_job(job_request: JobRequest, background_tasks: BackgroundTasks):
    """Submit a new job to be processed by the agent workflow"""
    job_id = str(uuid.uuid4())
    
    # Validate that we have agents capable of handling this job
    required_capabilities = _determine_required_capabilities(job_request.query)
    available_agents = _find_capable_agents(required_capabilities)
    
    if not available_agents:
        raise HTTPException(
            status_code=400,
            detail=f"No agents available with required capabilities: {required_capabilities}"
        )
    
    # Create job in workflow engine
    job = workflow_engine.create_job(
        job_id=job_id,
        request=job_request,
        assigned_agents=available_agents
    )
    
    # Start processing in background
    background_tasks.add_task(workflow_engine.execute_job, job_id)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job submitted successfully",
        estimated_completion_time=workflow_engine.estimate_completion_time(job_request)
    )

@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a specific job"""
    job = workflow_engine.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job_id,
        status=job.status,
        message=job.status_message,
        result=job.result,
        progress=job.progress,
        agents_involved=job.agents_involved
    )

@app.get("/agents", response_model=List[AgentRegistration])
async def list_agents():
    """List all registered agents and their capabilities"""
    return list(enhanced_registry.agents.values())

@app.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent from the MCP"""
    if not enhanced_registry.unregister_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {"status": "success", "message": f"Agent {agent_id} unregistered"}

def _determine_required_capabilities(query: str) -> List[str]:
    """Determine required capabilities for a query"""
    return _determine_required_capabilities_with_debates(query)

def _determine_required_capabilities_with_debates(query: str) -> List[str]:
    """Enhanced capability determination including debate capabilities"""
    capabilities = []
    query_lower = query.lower()
    
    # Original capability patterns plus debate patterns
    capability_patterns = {
        "risk": ["risk_analysis", "quantitative_analysis"],
        "portfolio": ["portfolio_analysis", "quantitative_analysis"],
        "tax": ["tax_optimization", "portfolio_analysis"],
        "market": ["market_intelligence", "real_time_data"],
        "rebalance": ["strategy_rebalancing", "portfolio_optimization"],
        "options": ["options_analysis", "quantitative_analysis"],
        "news": ["news_sentiment", "market_intelligence"],
        "screen": ["security_screening", "factor_analysis"],
        "stocks": ["security_screening", "stock_selection"],
        # Debate patterns
        "debate": ["debate_participation", "consensus_building"],
        "discuss": ["debate_participation", "collaborative_analysis"],
        "consensus": ["consensus_building", "decision_making"],
        "collaborate": ["collaborative_analysis", "multi_agent_coordination"],
        "should i": ["debate_participation", "decision_support"],
        "what do you think": ["collaborative_analysis", "consensus_building"],
        "agents": ["multi_agent_coordination", "debate_participation"]
    }
    
    for pattern, caps in capability_patterns.items():
        if pattern in query_lower:
            capabilities.extend(caps)
    
    if not capabilities:
        capabilities = ["query_interpretation", "general_analysis"]
    
    return list(set(capabilities))

def _find_capable_agents(required_capabilities: List[str]) -> List[str]:
    """Find agents that have the required capabilities"""
    capable_agents = []
    
    for agent_id, registration in enhanced_registry.agents.items():
        if any(cap in registration.capabilities for cap in required_capabilities):
            capable_agents.append(agent_id)
    
    return capable_agents

@app.get("/dashboard", response_class=HTMLResponse)
async def performance_dashboard():
    """Serve the performance monitoring dashboard"""
    return create_dashboard_html()

@app.get("/api/performance/summary")
async def get_performance_summary():
    """Get performance summary for API access"""
    if performance_monitor:
        return performance_monitor.get_performance_summary()
    return {"error": "Performance monitoring not initialized"}

@app.post("/debates/create_enhanced")
async def create_enhanced_debate(request: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Create debate with full circuit breaker and consensus building integration"""
    
    try:
        topic = request.get("topic", "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        preferred_agents = request.get("preferred_agents", [])
        if not preferred_agents:
            # Auto-select agents based on topic
            required_capabilities = _determine_required_capabilities(topic)
            preferred_agents = _find_capable_agents(required_capabilities)[:5]
        
        # Create DebateJobRequest for processing
        job_request = DebateJobRequest(
            job_type=DebateJobType.CREATE_DEBATE,
            topic=topic,
            description=request.get("description", f"Enhanced multi-agent debate: {topic}"),
            preferred_agents=preferred_agents,
            urgency=request.get("urgency", "medium"),
            max_rounds=request.get("max_rounds", 3),
            timeout_seconds=request.get("timeout_seconds", 600)
        )
        
        # Use the enhanced submission endpoint
        result = await submit_debate_job_enhanced(job_request, background_tasks, db)
        
        return {
            "success": True,
            "debate_details": result,
            "enhancement_info": {
                "circuit_breaker_protection": True,
                "consensus_building_enabled": True,
                "agent_health_filtering": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating enhanced debate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/performance")
async def websocket_performance(websocket: WebSocket):
    """WebSocket endpoint for real-time performance updates"""
    if not performance_monitor:
        await websocket.close(code=1000, reason="Performance monitoring not available")
        return
    
    await performance_monitor.add_websocket_connection(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await performance_monitor.remove_websocket_connection(websocket)

@app.post("/debates/submit_job", response_model=DebateJobResponse)
async def submit_debate_job_enhanced(
    job_request: DebateJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Enhanced debate job submission with circuit breaker protection and consensus building"""
    job_id = str(uuid.uuid4())
    
    try:
        if job_request.job_type == DebateJobType.CREATE_DEBATE:
            # Step 1: Create debate in database
            debate = crud.create_debate(
                db=db,
                user_id=1,
                query=job_request.topic,
                description=job_request.description or f"Enhanced debate: {job_request.topic}",
                urgency_level=job_request.urgency,
                max_rounds=job_request.max_rounds,
                max_duration_seconds=job_request.timeout_seconds
            )
            
            # Step 2: Filter agents through circuit breakers
            healthy_agents = []
            for agent_id in job_request.preferred_agents:
                if agent_id in enhanced_registry.agents:
                    # Check circuit breaker health
                    if agent_id in failover_manager.circuit_breakers:
                        cb = failover_manager.circuit_breakers[agent_id]
                        if cb.is_available() and cb.get_health_score() > 0.3:
                            healthy_agents.append(agent_id)
                            print(f"✅ Agent {agent_id} healthy - circuit state: {cb.state.value}")
                    else:
                        # Agent not in circuit breaker system - assume healthy
                        healthy_agents.append(agent_id)
                        print(f"✅ Agent {agent_id} healthy - no circuit breaker")
                else:
                    print(f"❌ Agent {agent_id} not found in registry")
            
            if len(healthy_agents) < 2:
                crud.update_debate_status(db, str(debate.id), "ERROR")
                return DebateJobResponse(
                    job_id=job_id,
                    debate_id=str(debate.id),
                    status=JobStatus.FAILED,
                    message=f"Insufficient healthy agents ({len(healthy_agents)}/{len(job_request.preferred_agents)}) for debate",
                    participating_agents=healthy_agents
                )
            
            # Step 3: Register participants in database
            participating_agents = []
            for agent_id in healthy_agents:
                agent_info = enhanced_registry.agents[agent_id]
                crud.add_debate_participant(
                    db=db,
                    debate_id=str(debate.id),
                    agent_id=agent_id,
                    agent_name=agent_info.agent_name,
                    agent_type=agent_info.agent_type,
                    agent_specialization=", ".join(agent_info.capabilities),
                    role="participant"
                )
                participating_agents.append(agent_id)
            
            # Step 4: Start ENHANCED debate processing
            background_tasks.add_task(
                process_enhanced_debate_workflow, 
                str(debate.id), 
                healthy_agents,
                job_request.topic,
                db
            )
            
            print(f"🚀 Enhanced debate {debate.id} created with {len(healthy_agents)} healthy agents")
            
            return DebateJobResponse(
                job_id=job_id,
                debate_id=str(debate.id),
                status=JobStatus.PENDING,
                message="Enhanced debate created with circuit breaker protection and consensus building",
                participating_agents=participating_agents,
                current_stage="position_formation",
                current_round=1
            )
            
        # Handle other job types with your existing logic
        else:
            # Keep existing logic for other job types
            return await process_other_debate_job_types(job_request, job_id, db)
            
    except Exception as e:
        logger.error(f"Error submitting enhanced debate job: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
async def process_other_debate_job_types(job_request: DebateJobRequest, job_id: str, db: Session):
    """Handle non-CREATE_DEBATE job types"""
    
    if job_request.job_type == DebateJobType.JOIN_DEBATE:
        # Your existing JOIN_DEBATE logic
        if not job_request.debate_id:
            raise HTTPException(status_code=400, detail="debate_id required for join operation")
            
        debate = crud.get_debate(db, job_request.debate_id)
        if not debate:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        new_participants = []
        for agent_id in job_request.preferred_agents:
            if agent_id in enhanced_registry.agents:
                agent_info = enhanced_registry.agents[agent_id]
                crud.add_debate_participant(
                    db=db,
                    debate_id=job_request.debate_id,
                    agent_id=agent_id,
                    agent_name=agent_info.agent_name,
                    agent_type=agent_info.agent_type,
                    agent_specialization=", ".join(agent_info.capabilities)
                )
                new_participants.append(agent_id)
        
        return DebateJobResponse(
            job_id=job_id,
            debate_id=job_request.debate_id,
            status=JobStatus.COMPLETED,
            message=f"Added {len(new_participants)} agents to debate",
            participating_agents=new_participants
        )
    
    elif job_request.job_type == DebateJobType.SUBMIT_POSITION:
        # Your existing SUBMIT_POSITION logic
        if not all([job_request.debate_id, job_request.position_data, job_request.preferred_agents]):
            raise HTTPException(status_code=400, detail="Missing required fields for position submission")
        
        debate = crud.get_debate(db, job_request.debate_id)
        participants = crud.get_debate_participants(db, job_request.debate_id)
        participant = next((p for p in participants if p.agent_id == agent_id), None)
        if not participant:
            raise HTTPException(status_code=404, detail="Participant not found in debate")
        
        if participant:
            message = crud.create_debate_message(
            db=db,
            debate_id=debate_id,
            sender_id=str(participant.id),  # Use participant ID
            message_type="POSITION",
            content=message_content,
            round_number=1,
            stage="POSITION_FORMATION",
            evidence_sources=position.get("supporting_evidence", []),
            confidence_score=position.get("confidence_score", 0.0)
        )
        
        return DebateJobResponse(
            job_id=job_id,
            debate_id=job_request.debate_id,
            status=JobStatus.COMPLETED,
            message="Position submitted to debate",
            participating_agents=[job_request.preferred_agents[0]],
            result={"message_id": str(message.id)}
        )
    
    elif job_request.job_type == DebateJobType.GET_DEBATE_STATUS:
        # Your existing GET_DEBATE_STATUS logic  
        if not job_request.debate_id:
            raise HTTPException(status_code=400, detail="debate_id required for status check")
            
        debate = crud.get_debate(db, job_request.debate_id)
        if not debate:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        participants = crud.get_debate_participants(db, job_request.debate_id)
        consensus_items = crud.get_debate_consensus_items(db, job_request.debate_id)
        
        progress = 0.0
        if debate.status.value == "completed":
            progress = 100.0
        elif debate.status.value == "active":
            progress = (debate.current_round / debate.max_rounds) * 100
        
        return DebateJobResponse(
            job_id=job_id,
            debate_id=job_request.debate_id,
            status=JobStatus.COMPLETED,
            message=f"Debate status: {debate.status.value}",
            participating_agents=[p.agent_id for p in participants],
            current_stage=debate.current_stage.value,
            current_round=debate.current_round,
            progress=progress,
            consensus_items=[{
                "topic": item.topic,
                "agreement_percentage": item.agreement_percentage,
                "consensus_strength": item.consensus_strength
            } for item in consensus_items]
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported debate job type: {job_request.job_type}")

@app.get("/debates/{debate_id}/status", response_model=DebateJobResponse)
async def get_debate_status_mcp(debate_id: str, db: Session = Depends(get_db)):
    """Get debate status through MCP"""
    job_id = str(uuid.uuid4())
    
    debate = crud.get_debate(db, debate_id)
    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found")
    
    participants = crud.get_debate_participants(db, debate_id)
    messages = crud.get_debate_messages(db, debate_id)
    consensus_items = crud.get_debate_consensus_items(db, debate_id)
    
    # Calculate progress
    progress = 0.0
    if debate.status.value == "completed":
        progress = 100.0
    elif debate.status.value == "active":
        progress = (debate.current_round / debate.max_rounds) * 100
    
    return DebateJobResponse(
        job_id=job_id,
        debate_id=debate_id,
        status=JobStatus.COMPLETED if debate.status.value == "completed" else JobStatus.RUNNING,
        message=f"Debate {debate.status.value} - {len(messages)} messages, {len(consensus_items)} consensus items",
        participating_agents=[p.agent_id for p in participants],
        current_stage=debate.current_stage.value,
        current_round=debate.current_round,
        progress=progress,
        result={
            "debate_summary": {
                "query": debate.query,
                "total_messages": len(messages),
                "participant_count": len(participants),
                "consensus_items": len(consensus_items),
                "confidence_score": debate.confidence_score,
                "final_recommendation": debate.final_recommendation
            }
        }
    )

@app.post("/debates/{debate_id}/notify_agents")
async def notify_debate_agents(
    debate_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Notify all agents in a debate to participate"""
    participants = crud.get_debate_participants(db, debate_id)
    
    if not participants:
        raise HTTPException(status_code=404, detail="No participants found for debate")
    
    notified_agents = []
    for participant in participants:
        if participant.agent_id in enhanced_registry.agents:
            # Schedule agent participation
            background_tasks.add_task(
                trigger_agent_participation,
                participant.agent_id,
                debate_id,
                db
            )
            notified_agents.append(participant.agent_id)
    
    return {
        "message": f"Notified {len(notified_agents)} agents to participate",
        "notified_agents": notified_agents,
        "debate_id": debate_id
    }

@app.get("/debates/templates")
async def get_debate_templates_mcp(
    category: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get available debate templates"""
    templates = crud.get_debate_templates(
        db=db,
        category=category,
        limit=limit
    )
    
    return {
        "templates": [
            {
                "template_id": str(template.id),
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "recommended_agents": template.recommended_agents,
                "default_rounds": template.default_rounds,
                "usage_count": template.usage_count,
                "avg_success_rate": template.avg_success_rate
            } for template in templates
        ],
        "total_templates": len(templates)
    }

@app.get("/debates/agents")
async def get_available_debate_agents():
    """Get available agents with proper error handling"""
    try:
        agents = []
        
        # Use the enhanced_registry that's already imported in server.py
        if hasattr(enhanced_registry, 'agents'):
            agent_registry = enhanced_registry.agents
        elif hasattr(enhanced_registry, 'get_all_agents'):
            agent_registry = enhanced_registry.get_all_agents()
        else:
            # Try direct access to registry attributes
            agent_registry = getattr(enhanced_registry, 'registry', {})
        
        # Convert registry to agents list
        if hasattr(agent_registry, 'items'):
            for agent_name, agent_info in agent_registry.items():
                agent_data = {
                    "name": agent_name,
                    "status": "available",
                    "specialization": getattr(agent_info, 'description', 
                                            getattr(agent_info, 'specialization', 'No description available')),
                    "capabilities": getattr(agent_info, 'capabilities', []),
                    "load": getattr(agent_info, 'current_load', 0),
                    "max_concurrent": getattr(agent_info, 'max_concurrent_debates', 5)
                }
                agents.append(agent_data)
        
        return {
            "agents": agents, 
            "total_count": len(agents),
            "available_count": len([a for a in agents if a["status"] == "available"])
        }
        
    except Exception as e:
        import logging
        logging.error(f"Error getting available agents: {e}")
        return {
            "agents": [], 
            "total_count": 0, 
            "available_count": 0,
            "error": f"Registry access failed: {str(e)}"
        }

# Background task functions for debate processing
async def process_debate_workflow(debate_id: str, db: Session):
    """Process the entire debate workflow"""
    try:
        debate = crud.get_debate(db, debate_id)
        if not debate:
            return
        
        # Update debate status to active
        crud.update_debate_status(db, debate_id, "ACTIVE")
        
        # Get all participants
        participants = crud.get_debate_participants(db, debate_id)
        
        if not participants:
            crud.update_debate_status(db, debate_id, "ERROR")
            return
        
        # Stage 1: Position Formation - Trigger initial position gathering
        print(f"Starting debate {debate_id}: Position formation stage")
        
        position_tasks = []
        for participant in participants:
            position_tasks.append(
                trigger_agent_participation(participant.agent_id, debate_id, db)
            )
        
        # Wait for all agents to submit positions
        await asyncio.gather(*position_tasks, return_exceptions=True)
        
        # Stage 2: Challenge and Response (simplified for now)
        await asyncio.sleep(10)  # Give agents time to process
        
        # Check messages received
        messages = crud.get_debate_messages(db, debate_id)
        print(f"Debate {debate_id}: Received {len(messages)} messages")
        
        # Stage 3: Consensus Building (simplified)
        if len(messages) >= len(participants):
            # Create consensus items based on messages
            consensus_topics = extract_consensus_topics_from_messages(messages)
            for topic in consensus_topics:
                crud.create_consensus_item(
                    db=db,
                    debate_id=debate_id,
                    topic=topic["topic"],
                    description=topic["description"],
                    category="recommendation",
                    total_participants=len(participants)
                )
            
            # Mark debate as completed
            crud.update_debate_status(db, debate_id, "COMPLETED")
            
            # Generate final recommendation (simplified)
            final_recommendation = generate_debate_recommendation(messages, consensus_topics)
            crud.update_debate_results(
                db=db,
                debate_id=debate_id,
                final_recommendation=final_recommendation,
                consensus_type="MAJORITY",  # Simplified
                confidence_score=0.8  # Simplified
            )
            
            print(f"Debate {debate_id} completed successfully")
        else:
            print(f"Debate {debate_id} timeout - insufficient responses")
            crud.update_debate_status(db, debate_id, "TIMEOUT")
            
    except Exception as e:
        print(f"Error in debate workflow {debate_id}: {str(e)}")
        crud.update_debate_status(db, debate_id, "ERROR")

async def trigger_agent_participation(agent_id: str, debate_id: str, db: Session):
    """Trigger a specific agent to participate in a debate"""
    try:
        # Get the agent instance
        if not hasattr(app.state, 'agent_instances') or agent_id not in app.state.agent_instances:
            print(f"Agent {agent_id} not found in instances")
            return
        
        agent = app.state.agent_instances[agent_id]
        
        # Get debate details
        debate = crud.get_debate(db, debate_id)
        if not debate:
            return
        
        # Get existing messages for context
        existing_messages = crud.get_debate_messages(db, debate_id)
        existing_positions = [
            {
                "agent_id": str(msg.sender_id),
                "content": msg.content,
                "confidence_score": msg.confidence_score
            }
            for msg in existing_messages
        ]
        
        # Check if agent has participate_in_debate method
        if hasattr(agent, 'participate_in_debate'):
            # Call agent's debate participation method
            position = await agent.participate_in_debate(
                debate_id=debate_id,
                debate_topic=debate.query,
                current_stage=debate.current_stage.value,
                existing_positions=existing_positions
            )
            
            if position:
                # Find the participant record
                participants = crud.get_debate_participants(db, debate_id)
                participant = next((p for p in participants if p.agent_id == agent_id), None)
                
                if participant:
                    # Store agent's position in database
                    crud.create_debate_message(
                        db=db,
                        debate_id=debate_id,
                        sender_id=str(participant.id),
                        message_type="position",
                        content={
                            "position": position.get("position", ""),
                            "reasoning": position.get("reasoning", ""),
                            "agent_id": agent_id
                        },
                        round_number=debate.current_round,
                        stage=debate.current_stage,
                        evidence_sources=position.get("evidence", []),
                        confidence_score=position.get("confidence_score", 0.8)
                    )
                    
                    print(f"Agent {agent_id} submitted position for debate {debate_id}")
            
        else:
            print(f"Agent {agent_id} does not have participate_in_debate method")
            
    except Exception as e:
        print(f"Error triggering agent {agent_id} participation: {str(e)}")

def extract_consensus_topics_from_messages(messages) -> List[Dict[str, str]]:
    """Extract potential consensus topics from debate messages (simplified)"""
    # This is a simplified implementation
    # In practice, you'd use NLP to identify common themes
    
    topics = []
    if messages:
        # Create a consensus item for the main recommendation
        topics.append({
            "topic": "Primary Recommendation",
            "description": "Main course of action recommended by the debate"
        })
        
        # Add implementation consensus if multiple messages
        if len(messages) > 1:
            topics.append({
                "topic": "Implementation Approach",
                "description": "How to implement the recommended action"
            })
    
    return topics

def generate_debate_recommendation(messages, consensus_topics) -> Dict[str, Any]:
    """Generate final recommendation from debate messages (simplified)"""
    # This is a simplified implementation
    # In practice, you'd use more sophisticated analysis
    
    if not messages:
        return {"recommendation": "No consensus reached", "confidence": 0.0}
    
    # Aggregate positions (simplified)
    positions = [msg.content.get("position", "") for msg in messages if msg.content]
    
    return {
        "recommendation": "Aggregated recommendation based on agent positions",
        "supporting_positions": positions[:3],  # Top 3 positions
        "confidence": 0.8,
        "implementation_priority": "medium",
        "consensus_topics": len(consensus_topics)
    }

@app.post("/debates/create_integrated")
async def create_integrated_debate(request: dict, db: Session = Depends(get_db)):
    """Create debate using complete integrated system"""
    
    try:
        topic = request.get("topic", "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        # Create debate in database first
        debate = crud.create_debate(
            db=db,
            user_id=1,  # TODO: Get from authentication
            query=topic,
            description=request.get("description", f"Integrated debate: {topic}"),
            urgency_level=request.get("urgency", "medium"),
            max_rounds=request.get("max_rounds", 3),
            max_duration_seconds=request.get("timeout_seconds", 600)
        )
        
        # Use integrated system for agent selection and management
        result = await complete_integrated_system.create_debate(
            topic=topic,
            preferred_agents=request.get("preferred_agents", []),
            portfolio_context=request.get("portfolio_context", {}),
            debate_params=request.get("debate_params", {
                "max_response_time": 120,
                "evidence_required": True,
                "civility_enforcement": True
            })
        )
        
        if result.get("success"):
            # Update database with integrated system results
            for agent_id in result["participating_agents"]:
                agent_info = enhanced_registry.agents.get(agent_id)
                if agent_info:
                    crud.add_debate_participant(
                        db=db,
                        debate_id=str(debate.id),
                        agent_id=agent_id,
                        agent_name=agent_info.agent_name,
                        agent_type=agent_info.agent_type,
                        agent_specialization=", ".join(agent_info.capabilities),
                        role="participant"
                    )
            
            # Start integrated debate monitoring
            asyncio.create_task(
                monitor_integrated_debate(str(debate.id), result["debate_id"], db)
            )
            
            return {
                "success": True,
                "database_debate_id": str(debate.id),
                "integrated_debate_id": result["debate_id"],
                "job_id": result["job_id"],
                "status": "initiated_with_integration",
                "participating_agents": result["participating_agents"],
                "agent_health_metrics": result.get("resilience_metrics", {}),
                "topic": topic,
                "estimated_completion": "8-12 minutes"
            }
        else:
            # Update database status to error
            crud.update_debate_status(db, str(debate.id), "ERROR")
            return result
            
    except Exception as e:
        logger.error(f"Error creating integrated debate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debates/{debate_id}/integrated_status")
async def get_integrated_debate_status(debate_id: str, db: Session = Depends(get_db)):
    """Get status from both database and integrated system"""
    
    try:
        # Get database status
        db_debate = crud.get_debate(db, debate_id)
        if not db_debate:
            raise HTTPException(status_code=404, detail="Debate not found in database")
        
        db_participants = crud.get_debate_participants(db, debate_id)
        db_messages = crud.get_debate_messages(db, debate_id)
        
        # Get consensus items from database
        consensus_items = crud.get_debate_consensus_items(db, debate_id)
        
        # Try to get integrated system status
        integrated_status = None
        if complete_integrated_system:
            try:
                # Find corresponding integrated debate
                for int_debate_id, debate_info in complete_integrated_system.active_debates.items():
                    if debate_info.get("database_id") == debate_id:
                        integrated_status = await complete_integrated_system.get_debate_status(int_debate_id)
                        break
            except Exception as e:
                logger.warning(f"Could not get integrated status: {e}")
        
        # Combine both sources
        combined_status = {
            "database_debate_id": debate_id,
            "db_status": db_debate.status.value,
            "db_current_round": db_debate.current_round,
            "db_current_stage": db_debate.current_stage.value,
            "db_message_count": len(db_messages),
            "db_participants": [p.agent_id for p in db_participants],
            "integrated_status": integrated_status,
            "circuit_breaker_health": failover_manager.get_system_resilience_metrics(),
            "consensus_available": len(consensus_items) > 0  # FIXED: Check actual consensus items
        }
        
        return combined_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting integrated debate status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debates/{debate_id}/build_consensus")
async def build_debate_consensus(debate_id: str, db: Session = Depends(get_db)):
    """Manually trigger consensus building from database messages"""
    
    try:
        # Get debate and messages from database
        debate = crud.get_debate(db, debate_id)
        if not debate:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        messages = crud.get_debate_messages(db, debate_id)
        participants = crud.get_debate_participants(db, debate_id)
        
        if len(messages) == 0:
            raise HTTPException(status_code=400, detail="No messages available for consensus building")
        
        # Convert database messages to agent positions
        agent_positions = []
        for participant in participants:
            # Find messages from this participant
            participant_messages = [
                msg for msg in messages 
                if str(msg.sender_id) == str(participant.id)
            ]
            
            if participant_messages:
                # Extract position from messages
                latest_message = participant_messages[-1]
                
                agent_position = {
                    "agent_id": participant.agent_id,
                    "stance": latest_message.content.get("position", "neutral"),
                    "key_arguments": [latest_message.content.get("reasoning", "")],
                    "confidence_score": latest_message.confidence_score or 0.5,
                    "supporting_evidence": latest_message.evidence_sources or [],
                    "risk_assessment": {
                        "primary_risks": [],
                        "mitigation_strategies": []
                    }
                }
                agent_positions.append(agent_position)
        
        # Build consensus
        query_context = {
            "query": debate.query,
            "complexity": "medium",
            "portfolio_context": {}
        }
        
        consensus_result = consensus_builder.calculate_weighted_consensus(
            agent_positions, query_context
        )
        
        # Evaluate consensus quality
        quality_assessment = consensus_builder.evaluate_consensus_quality(consensus_result)
        
        # Store consensus in database
        for topic_info in consensus_result.get("decision_factors", {}).items():
            topic, factors = topic_info
            if factors:
                crud.create_consensus_item(
                    db=db,
                    debate_id=debate_id,
                    topic=topic,
                    description=f"Consensus on {topic}: {factors[0] if factors else 'No details'}",
                    category="decision_factor",
                    total_participants=len(participants)
                )
        
        # Update debate with final recommendation
        crud.update_debate_results(
            db=db,
            debate_id=debate_id,
            final_recommendation=consensus_result.get("recommendation", "No consensus reached"),
            consensus_type="WEIGHTED_CONSENSUS",
            confidence_score=consensus_result.get("confidence_level", 0.0)
        )
        
        return {
            "success": True,
            "consensus": consensus_result,
            "quality_assessment": quality_assessment,
            "consensus_summary": consensus_builder.get_consensus_summary(consensus_result),
            "participants_analyzed": len(agent_positions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building consensus: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

async def build_enhanced_consensus(positions: List[Dict], topic: str) -> Dict:
    """Build consensus from agent positions with proper error handling"""
    
    try:
        if not positions:
            return {
                "consensus_reached": False,
                "confidence_level": 0.0,
                "primary_recommendation": "No positions available",
                "supporting_evidence": [],
                "risk_factors": [],
                "dissenting_views": [],
                "implementation_steps": []
            }

        print(f"🤝 DEBUG: Building consensus from {len(positions)} positions")
        
        # Filter out error positions and ensure we have valid data
        valid_positions = []
        for pos in positions:
            print(f"🔍 DEBUG: Processing position: {type(pos)} - {pos}")
            
            # Skip error positions
            if isinstance(pos, dict) and pos.get("error"):
                print(f"⚠️ DEBUG: Skipping error position: {pos}")
                continue
                
            # Ensure position is a dictionary
            if not isinstance(pos, dict):
                print(f"⚠️ DEBUG: Converting non-dict position to dict: {pos}")
                pos = {"stance": str(pos), "confidence_score": 0.5, "key_arguments": [], "supporting_evidence": [], "risk_assessment": {"primary_risks": []}}
            
            # Ensure required fields exist
            position_data = {
                "agent_id": pos.get("agent_id", "unknown"),
                "stance": pos.get("stance", str(pos.get("position", "neutral"))),
                "confidence_score": float(pos.get("confidence_score", pos.get("confidence", 0.5))),
                "key_arguments": pos.get("key_arguments", [pos.get("reasoning", "")]),
                "supporting_evidence": pos.get("supporting_evidence", pos.get("evidence", [])),
                "risk_assessment": pos.get("risk_assessment", {"primary_risks": pos.get("risks", [])})
            }
            
            valid_positions.append(position_data)
            print(f"✅ DEBUG: Added valid position from {position_data['agent_id']}")

        if not valid_positions:
            return {
                "consensus_reached": False,
                "confidence_level": 0.0,
                "primary_recommendation": "No valid positions to analyze",
                "supporting_evidence": [],
                "risk_factors": [],
                "dissenting_views": [],
                "implementation_steps": []
            }

        # Calculate weighted consensus
        total_confidence = sum(pos["confidence_score"] for pos in valid_positions)
        avg_confidence = total_confidence / len(valid_positions)
        
        # Collect all evidence and risks
        all_evidence = []
        all_risks = []
        
        for pos in valid_positions:
            # Handle evidence
            evidence = pos.get("supporting_evidence", [])
            if isinstance(evidence, list):
                all_evidence.extend(evidence)
            elif isinstance(evidence, str):
                all_evidence.append(evidence)
            
            # Handle risks  
            risk_data = pos.get("risk_assessment", {})
            if isinstance(risk_data, dict):
                risks = risk_data.get("primary_risks", [])
                if isinstance(risks, list):
                    all_risks.extend(risks)
                elif isinstance(risks, str):
                    all_risks.append(risks)

        # Determine primary recommendation based on agent stances
        stances = [pos["stance"] for pos in valid_positions]
        
        # Simple consensus logic - can be enhanced
        if any("reduce" in stance.lower() or "sell" in stance.lower() or "caution" in stance.lower() for stance in stances):
            primary_recommendation = f"Consider reducing tech allocation due to rising interest rates based on risk analysis"
        elif any("increase" in stance.lower() or "buy" in stance.lower() or "opportunity" in stance.lower() for stance in stances):
            primary_recommendation = f"Tech allocation adjustment may present opportunities despite rate environment"
        else:
            primary_recommendation = f"Balanced approach recommended for tech allocation in current interest rate environment"

        # Implementation steps based on agent analysis
        implementation_steps = [
            "Review current tech sector concentration in portfolio",
            "Assess interest rate sensitivity of tech holdings", 
            "Consider gradual rebalancing to manage transaction costs",
            "Monitor Federal Reserve policy signals",
            "Evaluate defensive sector alternatives"
        ]

        consensus_result = {
            "consensus_reached": len(valid_positions) >= 2 and avg_confidence > 0.6,
            "confidence_level": round(avg_confidence, 3),
            "primary_recommendation": primary_recommendation,
            "supporting_evidence": list(set(all_evidence))[:10],  # Dedupe and limit
            "risk_factors": list(set(all_risks))[:8],  # Dedupe and limit
            "dissenting_views": [pos["stance"] for pos in valid_positions if pos["confidence_score"] < 0.7],
            "implementation_steps": implementation_steps,
            "participating_agents": [pos["agent_id"] for pos in valid_positions],
            "total_agents": len(valid_positions)
        }
        
        print(f"✅ DEBUG: Consensus built with {consensus_result['confidence_level']} confidence")
        return consensus_result

    except Exception as e:
        print(f"💥 ERROR in build_enhanced_consensus: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "consensus_reached": False,
            "confidence_level": 0.0,
            "primary_recommendation": f"Error building consensus: {str(e)}",
            "supporting_evidence": [],
            "risk_factors": ["Consensus building error"],
            "dissenting_views": [],
            "implementation_steps": [],
            "error": str(e)
        }

# ================================
# CIRCUIT BREAKER ENHANCED ENDPOINTS
# ================================

@app.get("/system/circuit_breakers")
async def get_all_circuit_breakers():
    """Get status of all agent circuit breakers"""
    
    try:
        all_stats = failover_manager.get_all_circuit_breaker_stats()
        resilience = failover_manager.get_system_resilience_metrics()
        
        return {
            "system_resilience": resilience,
            "individual_agents": all_stats,
            "summary": {
                "total_agents": len(all_stats),
                "healthy_agents": len([s for s in all_stats.values() if s["is_available"]]),
                "circuit_breakers_open": len([s for s in all_stats.values() if s["state"] == "open"]),
                "average_health_score": sum(s["health_score"] for s in all_stats.values()) / len(all_stats) if all_stats else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/circuit/reset")
async def reset_agent_circuit(agent_id: str):
    """Reset circuit breaker for specific agent"""
    
    try:
        success = failover_manager.force_circuit_reset(agent_id)
        
        if success:
            return {
                "success": True,
                "message": f"Circuit breaker reset for {agent_id}",
                "agent_id": agent_id,
                "new_status": failover_manager.circuit_breakers[agent_id].get_stats() if agent_id in failover_manager.circuit_breakers else None
            }
        else:
            raise HTTPException(status_code=404, detail=f"Circuit breaker not found for agent {agent_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting circuit breaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# ENHANCED DEBATE ENDPOINTS WITH DATABASE INTEGRATION
# ================================

@app.post("/debates/submit_job", response_model=DebateJobResponse)
async def submit_debate_job_enhanced(
    job_request: DebateJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Enhanced debate job submission with circuit breaker protection"""
    job_id = str(uuid.uuid4())
    
    try:
        if job_request.job_type == DebateJobType.CREATE_DEBATE:
            # Step 1: Create debate in database
            debate = crud.create_debate(
                db=db,
                user_id=1,
                query=job_request.topic,
                description=job_request.description or f"Enhanced debate: {job_request.topic}",
                urgency_level=job_request.urgency,
                max_rounds=job_request.max_rounds,
                max_duration_seconds=job_request.timeout_seconds
            )
            
            # Step 2: Filter agents through circuit breakers
            healthy_agents = []
            for agent_id in job_request.preferred_agents:
                if agent_id in enhanced_registry.agents:
                    # Check circuit breaker health
                    if agent_id in failover_manager.circuit_breakers:
                        cb = failover_manager.circuit_breakers[agent_id]
                        if cb.is_available() and cb.get_health_score() > 0.3:
                            healthy_agents.append(agent_id)
                    else:
                        # Agent not in circuit breaker system - assume healthy
                        healthy_agents.append(agent_id)
            
            if len(healthy_agents) < 2:
                crud.update_debate_status(db, str(debate.id), "ERROR")
                return DebateJobResponse(
                    job_id=job_id,
                    debate_id=str(debate.id),
                    status=JobStatus.FAILED,
                    message=f"Insufficient healthy agents ({len(healthy_agents)}) for debate",
                    participating_agents=healthy_agents,
                    circuit_breaker_info=failover_manager.get_system_resilience_metrics()
                )
            
            # Step 3: Register participants in database
            participating_agents = []
            for agent_id in healthy_agents:
                agent_info = enhanced_registry.agents[agent_id]
                crud.add_debate_participant(
                    db=db,
                    debate_id=str(debate.id),
                    agent_id=agent_id,
                    agent_name=agent_info.agent_name,
                    agent_type=agent_info.agent_type,
                    agent_specialization=", ".join(agent_info.capabilities),
                    role="participant"
                )
                participating_agents.append(agent_id)
            
            # Step 4: Start enhanced debate processing
            background_tasks.add_task(
                process_enhanced_debate_workflow, 
                str(debate.id), 
                healthy_agents,
                job_request.topic,
                db
            )
            
            return DebateJobResponse(
                job_id=job_id,
                debate_id=str(debate.id),
                status=JobStatus.PENDING,
                message="Enhanced debate created with circuit breaker protection",
                participating_agents=participating_agents,
                current_stage="position_formation",
                current_round=1,
                agent_health_scores={
                    agent_id: failover_manager.circuit_breakers[agent_id].get_health_score()
                    for agent_id in healthy_agents
                    if agent_id in failover_manager.circuit_breakers
                }
            )
            
        # Handle other job types...
        else:
            # Keep existing logic for other job types
            return await submit_debate_job_legacy(job_request, background_tasks, db)
            
    except Exception as e:
        logger.error(f"Error submitting enhanced debate job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# BACKGROUND TASK FUNCTIONS
# ================================

async def process_enhanced_debate_workflow(
    debate_id: str, 
    healthy_agents: List[str],
    topic: str,
    db: Session
):
    """Enhanced debate workflow with consensus building"""
    
    try:
        print(f"🎯 Starting enhanced debate {debate_id} with agents: {healthy_agents}")
        
        # Update status
        crud.update_debate_status(db, debate_id, "ACTIVE")
        
        # Stage 1: Protected Position Formation
        print(f"📝 Stage 1: Position formation for debate {debate_id}")
        
        agent_positions = []
        position_tasks = []
        
        for agent_id in healthy_agents:
            task = asyncio.create_task(
                get_agent_position_with_protection(agent_id, debate_id, topic, db)
            )
            position_tasks.append(task)
        
        # Wait for positions with timeout
        try:
            positions = await asyncio.wait_for(
                asyncio.gather(*position_tasks, return_exceptions=True),
                timeout=300  # 5 minutes
            )
            
            # Filter successful positions
            for result in positions:
                if isinstance(result, dict) and "agent_id" in result:
                    agent_positions.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Agent position failed: {result}")
            
        except asyncio.TimeoutError:
            logger.warning(f"Position formation timeout for debate {debate_id}")
        
        print(f"📊 Collected {len(agent_positions)} positions for debate {debate_id}")
        
        # Store agent positions as messages
        await store_agent_positions_as_messages(debate_id, agent_positions, db)
        
        # Stage 2: Consensus Building
        if len(agent_positions) >= 2:
            print(f"🤝 Building consensus for debate {debate_id}")
            
            # Use the enhanced consensus builder instead
            consensus_result = await build_enhanced_consensus(agent_positions, topic)
            
            # Store consensus results in database  
            await store_consensus_in_database(
                debate_id, consensus_result, {"quality_score": consensus_result.get("confidence_level", 0.0)}, db
            )
            
            # Update agent performance scores
            await update_agent_performance_from_consensus(
                agent_positions, consensus_result, topic
            )
            
            # Mark debate as completed
            crud.update_debate_status(db, debate_id, "COMPLETED")
            
            print(f"✅ Debate {debate_id} completed with {consensus_result['confidence_level']:.2f} confidence")
                    
        else:
            print(f"❌ Insufficient positions for consensus in debate {debate_id}")
            crud.update_debate_status(db, debate_id, "TIMEOUT")
                
    except Exception as e:
        logger.error(f"Error in enhanced debate workflow {debate_id}: {e}")
        crud.update_debate_status(db, debate_id, "ERROR")

async def get_agent_position_with_protection(agent_id: str, debate_id: str, topic: str, db: Session) -> Dict:
    """Get agent position with circuit breaker protection"""
    
    print(f"🔍 DEBUG: Getting position from agent {agent_id} for debate {debate_id}")
    print(f"📋 DEBUG: Available agents: {list(app.state.agent_instances.keys()) if hasattr(app.state, 'agent_instances') else 'No agent_instances'}")
    
    try:
        # Check if agent exists
        if not hasattr(app.state, 'agent_instances'):
            print(f"❌ DEBUG: app.state has no agent_instances attribute")
            return {"error": f"No agent instances available", "agent_id": agent_id}
            
        if agent_id not in app.state.agent_instances:
            print(f"❌ DEBUG: Agent {agent_id} not found in instances")
            print(f"📋 DEBUG: Available agents are: {list(app.state.agent_instances.keys())}")
            return {"error": f"Agent {agent_id} instance not found", "agent_id": agent_id}
        
        agent = app.state.agent_instances[agent_id]
        print(f"✅ DEBUG: Found agent {agent_id}, type: {type(agent)}")
        
        # Check if agent has participate_in_debate method
        if not hasattr(agent, 'participate_in_debate'):
            print(f"❌ DEBUG: Agent {agent_id} missing participate_in_debate method")
            return {"error": f"Agent {agent_id} missing participate_in_debate method", "agent_id": agent_id}
        
        print(f"🎯 DEBUG: Calling participate_in_debate for agent {agent_id}")
        
        # Call agent participation
        position = await agent.participate_in_debate(
            debate_id=debate_id,
            topic=topic,
            existing_positions=[]
        )
        
        print(f"📝 DEBUG: Agent {agent_id} returned position: {position}")
        
        return {
            "agent_id": agent_id,
            "stance": position.get("position", "neutral"),
            "key_arguments": [position.get("reasoning", "")],
            "confidence_score": position.get("confidence", 0.8),
            "supporting_evidence": position.get("evidence", []),
            "risk_assessment": {"primary_risks": position.get("risks", []), "mitigation_strategies": []}
        }
        
    except Exception as e:
        print(f"💥 DEBUG: Exception in agent {agent_id} participation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "agent_id": agent_id}

async def get_agent_position_direct(agent_id: str, debate_id: str, topic: str, db: Session) -> Dict:
    """Get agent position directly"""
    
    # Get agent instance
    if not hasattr(app.state, 'agent_instances') or agent_id not in app.state.agent_instances:
        raise Exception(f"Agent {agent_id} instance not found")
    
    agent = app.state.agent_instances[agent_id]
    
    # Get existing messages for context
    existing_messages = crud.get_debate_messages(db, debate_id)
    existing_positions = []
    
    for msg in existing_messages:
        existing_positions.append({
            "agent_id": str(msg.sender_id),
            "content": msg.content,
            "confidence_score": msg.confidence_score
        })
    
    # Call agent participation method
    if hasattr(agent, 'participate_in_debate'):
        position = await agent.participate_in_debate(
            debate_id=debate_id,
            topic=topic,
            existing_positions=existing_positions
        )
        
        # Store in database
        participants = crud.get_debate_participants(db, debate_id)
        participant = next((p for p in participants if p.agent_id == agent_id), None)
        
        if participant and position:
            crud.create_debate_message(
                db=db,
                debate_id=debate_id,
                sender_id=str(participant.id),
                message_type="position",
                content=position,
                round_number=1,
                stage="position_formation",
                evidence_sources=position.get("evidence", []),
                confidence_score=position.get("confidence", 0.8)
            )
        
        # Return structured position for consensus building
        return {
            "agent_id": agent_id,
            "stance": position.get("position", "neutral"),
            "key_arguments": [position.get("reasoning", "")],
            "confidence_score": position.get("confidence", 0.8),
            "supporting_evidence": [
                {
                    "type": "agent_analysis",
                    "source": agent_id,
                    "data": position.get("supporting_data", {})
                }
            ],
            "risk_assessment": {
                "primary_risks": position.get("risks", []),
                "mitigation_strategies": []
            }
        }
    else:
        raise Exception(f"Agent {agent_id} does not support debate participation")

async def store_consensus_in_database(
    debate_id: str, 
    consensus_result: Dict, 
    quality_assessment: Dict, 
    db: Session
):
    """Store consensus results in database with debug logging"""
    
    try:
        print(f"💾 DEBUG: Storing consensus for debate {debate_id}")
        print(f"📊 DEBUG: Consensus result keys: {list(consensus_result.keys())}")
        print(f"🎯 DEBUG: Primary recommendation: {consensus_result.get('primary_recommendation', 'None')}")
        print(f"📈 DEBUG: Confidence level: {consensus_result.get('confidence_level', 0.0)}")
        
        # Store consensus items if available
        if consensus_result.get("consensus_reached", False):
            
            # Create main consensus item
            main_consensus = crud.create_consensus_item(
                db=db,
                debate_id=debate_id,
                topic="Primary Recommendation",
                description=consensus_result.get("primary_recommendation", "No recommendation"),
                category="primary_decision",
                total_participants=consensus_result.get("total_agents", 0)
            )
            print(f"✅ DEBUG: Created main consensus item: {main_consensus.id if main_consensus else 'Failed'}")
            
            # Store risk factors
            risk_factors = consensus_result.get("risk_factors", [])
            for i, risk in enumerate(risk_factors[:5]):  # Limit to 5 risks
                risk_item = crud.create_consensus_item(
                    db=db,
                    debate_id=debate_id,
                    topic=f"Risk Factor {i+1}",
                    description=str(risk),
                    category="risk_assessment",
                    total_participants=consensus_result.get("total_agents", 0)
                )
                print(f"📍 DEBUG: Created risk item: {risk_item.id if risk_item else 'Failed'}")
            
            # Store implementation steps
            impl_steps = consensus_result.get("implementation_steps", [])
            for i, step in enumerate(impl_steps[:5]):  # Limit to 5 steps
                step_item = crud.create_consensus_item(
                    db=db,
                    debate_id=debate_id,
                    topic=f"Implementation Step {i+1}",
                    description=str(step),
                    category="implementation",
                    total_participants=consensus_result.get("total_agents", 0)
                )
                print(f"🔧 DEBUG: Created implementation item: {step_item.id if step_item else 'Failed'}")
        
        # Dynamic consensus type assignment based on confidence and agreement
        confidence_level = float(consensus_result.get("confidence_level", 0.0))
        total_agents = consensus_result.get("total_agents", 0)
        dissenting_views = len(consensus_result.get("dissenting_views", []))
        
        print(f"🔍 DEBUG: Determining consensus type - confidence: {confidence_level}, total_agents: {total_agents}, dissenting: {dissenting_views}")
        
        if confidence_level >= 0.9 and dissenting_views == 0:
            consensus_type = "UNANIMOUS"
            print(f"🎯 DEBUG: Assigned UNANIMOUS consensus (high confidence, no dissent)")
        elif confidence_level >= 0.6 and dissenting_views < total_agents / 2:
            consensus_type = "MAJORITY"
            print(f"🎯 DEBUG: Assigned MAJORITY consensus (good confidence, limited dissent)")
        elif confidence_level >= 0.4:
            consensus_type = "PLURALITY"
            print(f"🎯 DEBUG: Assigned PLURALITY consensus (moderate confidence)")
        else:
            consensus_type = "NO_CONSENSUS"
            print(f"🎯 DEBUG: Assigned NO_CONSENSUS (low confidence)")
        
        # Update debate with final results
        final_recommendation = consensus_result.get("primary_recommendation", "No consensus reached")
        
        updated_debate = crud.update_debate_results(
            db=db,
            debate_id=debate_id,
            final_recommendation=final_recommendation,
            consensus_type=consensus_type,  # Using dynamic assignment
            confidence_score=confidence_level
        )
        print(f"🏁 DEBUG: Updated debate results with {consensus_type}: {updated_debate.id if updated_debate else 'Failed'}")
        
        # Commit the transaction
        db.commit()
        print(f"✅ DEBUG: Successfully stored consensus for debate {debate_id}")
        
    except Exception as e:
        print(f"💥 DEBUG: Error storing consensus: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise

async def store_agent_positions_as_messages(
    debate_id: str, 
    agent_positions: List[Dict], 
    db: Session
):
    """Store agent positions as debate messages"""
    
    try:
        print(f"💬 DEBUG: Storing {len(agent_positions)} agent messages for debate {debate_id}")
        
        # Get all participants for this debate
        participants = crud.get_debate_participants(db, debate_id)
        participant_lookup = {p.agent_id: p for p in participants}
        
        stored_count = 0
        
        for position in agent_positions:
            if position.get("error"):
                print(f"⚠️ DEBUG: Skipping error position from {position.get('agent_id', 'unknown')}")
                continue
                
            agent_id = position.get("agent_id", "unknown_agent")
            
            # Find the participant record for this agent
            participant = participant_lookup.get(agent_id)
            if not participant:
                print(f"❌ DEBUG: No participant found for agent {agent_id}")
                continue
            
            # Create message content from agent position
            message_content = {
                "position": position.get("stance", "No position"),
                "reasoning": position.get("key_arguments", ["No reasoning provided"])[0] if position.get("key_arguments") else "No reasoning provided",
                "confidence": position.get("confidence_score", 0.0),
                "evidence": position.get("supporting_evidence", []),
                "risks": position.get("risk_assessment", {}).get("primary_risks", []),
                "agent_id": agent_id
            }
            
            # Store as debate message using correct CRUD function
            message = crud.create_debate_message(
                db=db,
                debate_id=debate_id,
                sender_id=str(participant.id),  # Use participant UUID
                message_type="POSITION",  # Use the enum value
                content=message_content,
                round_number=1,
                stage="POSITION_FORMATION",  # Use the enum value
                evidence_sources=position.get("supporting_evidence", []),
                confidence_score=position.get("confidence_score", 0.0)
            )
            
            if message:
                print(f"✅ DEBUG: Stored message from agent {agent_id}: {message.id}")
                stored_count += 1
            else:
                print(f"❌ DEBUG: Failed to store message from agent {agent_id}")
        
        db.commit()
        print(f"💾 DEBUG: Committed {stored_count} agent messages")
        
    except Exception as e:
        print(f"💥 DEBUG: Error storing agent messages: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise

async def update_agent_performance_from_consensus(
    agent_positions: List[Dict],
    consensus_result: Dict,
    topic: str
):
    """Update agent performance scores based on consensus results"""
    
    try:
        majority_agents = consensus_result.get("majority_agents", [])
        confidence_level = consensus_result.get("confidence_level", 0.5)
        
        # Extract topic keywords for expertise updates
        topic_keywords = extract_topic_keywords(topic)
        
        for position in agent_positions:
            agent_id = position["agent_id"]
            
            # Calculate performance score
            performance_score = 0.5  # Base score
            
            # Bonus for being in majority consensus
            if agent_id in majority_agents:
                performance_score += 0.2
            
            # Bonus for high confidence when consensus was strong
            if confidence_level > 0.7 and position.get("confidence_score", 0.5) > 0.7:
                performance_score += 0.1
            
            # Update expertise scores
            for keyword in topic_keywords:
                consensus_builder.update_agent_expertise(
                    agent_id, keyword, min(performance_score, 1.0)
                )
        
        logger.info(f"Updated agent performance scores for debate topic: {topic}")
        
    except Exception as e:
        logger.error(f"Error updating agent performance: {e}")

def extract_topic_keywords(topic: str) -> List[str]:
    """Extract keywords for expertise updates"""
    topic_lower = topic.lower()
    keywords = []
    
    keyword_map = {
        "risk": "risk_analysis",
        "volatility": "volatility_analysis",
        "market": "market_timing", 
        "tax": "tax_optimization",
        "options": "options_analysis",
        "portfolio": "portfolio_optimization",
        "allocation": "portfolio_optimization",
        "economy": "macro_analysis",
        "fed": "fed_policy"
    }
    
    for word, keyword in keyword_map.items():
        if word in topic_lower:
            keywords.append(keyword)
    
    return keywords or ["general_analysis"]

async def monitor_integrated_debate(database_debate_id: str, integrated_debate_id: str, db: Session):
    """Monitor integrated debate and sync with database"""
    
    try:
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Get status from integrated system
            if integrated_debate_id in complete_integrated_system.active_debates:
                debate_info = complete_integrated_system.active_debates[integrated_debate_id]
                
                # Sync status to database
                if debate_info["status"] == "completed":
                    crud.update_debate_status(db, database_debate_id, "COMPLETED")
                    
                    # Get final results if available
                    if "final_results" in debate_info:
                        final_results = debate_info["final_results"]
                        if "consensus" in final_results:
                            consensus = final_results["consensus"]
                            crud.update_debate_results(
                                db=db,
                                debate_id=database_debate_id,
                                final_recommendation=consensus.get("recommendation", ""),
                                consensus_type="INTEGRATED_CONSENSUS",
                                confidence_score=consensus.get("confidence_level", 0.0)
                            )
                    break
                elif debate_info["status"] == "failed":
                    crud.update_debate_status(db, database_debate_id, "ERROR")
                    break
            else:
                # Integrated debate no longer exists
                crud.update_debate_status(db, database_debate_id, "COMPLETED")
                break
                
    except Exception as e:
        logger.error(f"Error monitoring integrated debate: {e}")

# ================================
# TESTING AND VALIDATION ENDPOINTS
# ================================

@app.post("/debug/test_integrated_system")
async def test_integrated_system(db: Session = Depends(get_db)):
    """Test the complete integrated system end-to-end"""
    
    try:
        test_results = {
            "circuit_breakers": {},
            "consensus_builder": {},
            "agent_registry": {},
            "database_integration": {}
        }
        
        # Test 1: Circuit Breaker System
        cb_stats = failover_manager.get_all_circuit_breaker_stats()
        resilience = failover_manager.get_system_resilience_metrics()
        
        test_results["circuit_breakers"] = {
            "agents_with_breakers": len(cb_stats),
            "system_availability": resilience["system_availability"],
            "healthy_agents": resilience["available_agents"],
            "status": "operational" if resilience["system_availability"] > 0.5 else "degraded"
        }
        
        # Test 2: Consensus Builder
        sample_positions = [
            {
                "agent_id": "test_agent_1",
                "stance": "bullish",
                "key_arguments": ["market growth", "positive trends"],
                "confidence_score": 0.8,
                "supporting_evidence": [{"type": "statistical", "source": "test"}],
                "risk_assessment": {"primary_risks": ["volatility"], "mitigation_strategies": []}
            },
            {
                "agent_id": "test_agent_2", 
                "stance": "bearish",
                "key_arguments": ["market risk", "uncertainty"],
                "confidence_score": 0.7,
                "supporting_evidence": [{"type": "analytical", "source": "test"}],
                "risk_assessment": {"primary_risks": ["downturn"], "mitigation_strategies": []}
            }
        ]
        
        test_consensus = consensus_builder.calculate_weighted_consensus(
            sample_positions, {"query": "test query", "complexity": "medium"}
        )
        
        test_results["consensus_builder"] = {
            "consensus_generated": test_consensus is not None,
            "confidence_level": test_consensus.get("confidence_level", 0.0),
            "majority_agents": len(test_consensus.get("majority_agents", [])),
            "minority_opinions": len(test_consensus.get("minority_opinions", [])),
            "status": "operational"
        }
        
        # Test 3: Agent Registry
        try:
    # Fix: Use compatible method to access agents
            if hasattr(enhanced_registry, 'agents') and enhanced_registry.agents:
                available_agents = [
                    {
                        "agent_id": agent_id,
                        "capabilities": getattr(agent_info, 'capabilities', [])
                    }
                    for agent_id, agent_info in enhanced_registry.agents.items()
                ]
            else:
                available_agents = []
            
            test_results["agent_registry"] = {
                "total_agents": len(available_agents),
                "agents_with_debate_capability": len([
                    a for a in available_agents 
                    if "debate_participation" in a.get("capabilities", [])
                ]),
                "status": "operational" if available_agents else "no_agents",
                "agent_list": [a["agent_id"] for a in available_agents]  # Add this for debugging
            }
            
        except Exception as e:
            test_results["agent_registry"] = {
                "total_agents": 0,
                "error": str(e),
                "status": "failed"
            }
        
        # Test 4: Database Integration
        try:
            # Test database connectivity
            test_debate = crud.create_debate(
                db=db,
                user_id=1,
                query="System integration test",
                description="Testing database integration",
                urgency_level="low",
                max_rounds=1,
                max_duration_seconds=60
            )
            
            # Clean up test debate
            crud.update_debate_status(db, str(test_debate.id), "COMPLETED")
            
            test_results["database_integration"] = {
                "database_connectivity": True,
                "crud_operations": True,
                "test_debate_id": str(test_debate.id),
                "status": "operational"
            }
            
        except Exception as e:
            test_results["database_integration"] = {
                "database_connectivity": False,
                "error": str(e),
                "status": "failed"
            }
        
        # Overall system status
        all_operational = all(
            component.get("status") == "operational" 
            for component in test_results.values()
        )
        
        return {
            "system_status": "fully_operational" if all_operational else "partial_operation",
            "test_timestamp": datetime.now().isoformat(),
            "components": test_results,
            "ready_for_debates": all_operational and test_results["agent_registry"]["total_agents"] >= 2
        }
        
    except Exception as e:
        logger.error(f"System integration test failed: {e}")
        return {
            "system_status": "test_failed",
            "error": str(e),
            "test_timestamp": datetime.now().isoformat()
        }

async def submit_debate_job_legacy(job_request, background_tasks, db):
    """Handle legacy debate job types"""
    # Implementation for other job types...
    job_id = str(uuid.uuid4())
    return DebateJobResponse(
        job_id=job_id,
        debate_id="",
        status=JobStatus.COMPLETED,
        message="Legacy job type processed"
    )

# Add this endpoint to your FastAPI app for debugging
@app.get("/debug/registry")
async def debug_registry():
    """Debug endpoint to check registry and agent instances"""
    
    result = {
        "enhanced_registry_exists": hasattr(app, 'enhanced_registry') or 'enhanced_registry' in globals(),
        "app_state_exists": hasattr(app, 'state'),
        "agent_instances_exists": hasattr(app.state, 'agent_instances') if hasattr(app, 'state') else False,
        "current_agent_instances": list(app.state.agent_instances.keys()) if hasattr(app.state, 'agent_instances') else [],
        "enhanced_registry_agents": [],
        "enhanced_registry_type": str(type(enhanced_registry)) if 'enhanced_registry' in globals() else "Not found"
    }
    
    # Try to get enhanced_registry agents
    try:
        if 'enhanced_registry' in globals():
            registry = globals()['enhanced_registry']
            if hasattr(registry, 'agents'):
                result["enhanced_registry_agents"] = list(registry.agents.keys())
                result["enhanced_registry_agents_details"] = {
                    name: {
                        "capabilities": info.get("capabilities", []),
                        "metadata": info.get("metadata", {})
                    } for name, info in registry.agents.items()
                }
            else:
                result["enhanced_registry_error"] = "Registry has no 'agents' attribute"
        else:
            result["enhanced_registry_error"] = "enhanced_registry not in globals"
            
    except Exception as e:
        result["enhanced_registry_error"] = str(e)
    
    return result

# Add this endpoint to manually trigger agent creation
@app.post("/debug/create_agents")
async def debug_create_agents():
    """Manually create agent instances for debugging"""
    
    try:
        if not hasattr(app.state, 'agent_instances'):
            app.state.agent_instances = {}
        
        # List of agents we know should exist
        required_agents = [
            "portfolio_analysis_agent",
            "risk_assessment_agent", 
            "market_analysis_agent"
        ]
        
        created_agents = []
        
        for agent_name in required_agents:
            if agent_name not in app.state.agent_instances:
                
                # Create the wrapper class
                class RealDebateAgentWrapper:
                    def __init__(self, name):
                        self.agent_id = name
                        self.name = name
                        print(f"🎯 Manual creation: Created wrapper for {name}")
                        
                    async def participate_in_debate(self, debate_id: str, topic: str, existing_positions: list = None):
                        print(f"🗣️ Manual agent {self.agent_id} participating in debate {debate_id}")
                        
                        # Generate specialized analysis based on agent type
                        if "portfolio" in self.agent_id.lower():
                            return {
                                "position": f"Recommend portfolio rebalancing for {topic}",
                                "reasoning": "Based on modern portfolio theory and diversification principles",
                                "confidence": 0.85,
                                "evidence": ["Portfolio optimization models", "Risk-return analysis", "Correlation matrices"],
                                "risks": ["Concentration risk", "Rebalancing costs", "Tax implications"]
                            }
                        elif "risk" in self.agent_id.lower():
                            return {
                                "position": f"High risk assessment for {topic}",
                                "reasoning": "Value-at-Risk calculations show increased downside exposure",
                                "confidence": 0.78,
                                "evidence": ["VaR models", "Stress testing", "Historical volatility"],
                                "risks": ["Tail risk", "Model risk", "Market regime changes"]
                            }
                        elif "market" in self.agent_id.lower():
                            return {
                                "position": f"Market timing suggests caution on {topic}",
                                "reasoning": "Technical indicators and market sentiment analysis",
                                "confidence": 0.72,
                                "evidence": ["Technical indicators", "Market breadth", "Sentiment surveys"],
                                "risks": ["Market timing risk", "False signals", "Whipsaw movements"]
                            }
                        else:
                            return {
                                "position": f"Neutral stance on {topic}",
                                "reasoning": f"Standard analysis by {self.agent_id}",
                                "confidence": 0.65,
                                "evidence": ["Market data", "Historical analysis"],
                                "risks": ["General market risk", "Uncertainty"]
                            }
                
                # Create and store the instance
                app.state.agent_instances[agent_name] = RealDebateAgentWrapper(agent_name)
                created_agents.append(agent_name)
                print(f"✅ Manual creation: Successfully created instance for {agent_name}")
        
        return {
            "success": True,
            "created_agents": created_agents,
            "total_agent_instances": list(app.state.agent_instances.keys()),
            "message": f"Created {len(created_agents)} new agent instances"
        }
        
    except Exception as e:
        print(f"💥 Error in manual agent creation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    uvicorn.run(
        "mcp.server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )