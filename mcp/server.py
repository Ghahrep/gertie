# mcp/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

from .workflow_engine import WorkflowEngine
from .schemas import (
    AgentRegistration, 
    JobRequest, 
    JobResponse, 
    JobStatus,
    HealthCheck
)

# Global registry to hold agent information
agent_registry: Dict[str, AgentRegistration] = {}
workflow_engine = WorkflowEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 MCP Server starting up...")
    yield
    # Shutdown
    print("🛑 MCP Server shutting down...")

app = FastAPI(
    title="Gertie.ai Master Control Plane",
    description="Centralized orchestration server for AI agents",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register", response_model=dict)
async def register_agent(registration: AgentRegistration):
    """Register a new agent with the MCP"""
    agent_id = registration.agent_id
    
    if agent_id in agent_registry:
        raise HTTPException(
            status_code=409, 
            detail=f"Agent {agent_id} already registered"
        )
    
    agent_registry[agent_id] = registration
    print(f"✅ Registered agent: {agent_id} with capabilities: {registration.capabilities}")
    
    return {
        "status": "success",
        "message": f"Agent {agent_id} registered successfully",
        "registered_at": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint for the MCP"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        registered_agents=len(agent_registry),
        active_jobs=workflow_engine.get_active_job_count()
    )

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
    return list(agent_registry.values())

@app.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent from the MCP"""
    if agent_id not in agent_registry:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    del agent_registry[agent_id]
    return {"status": "success", "message": f"Agent {agent_id} unregistered"}

def _determine_required_capabilities(query: str) -> List[str]:
    """Analyze query to determine what agent capabilities are needed"""
    capabilities = []
    query_lower = query.lower()
    
    # Map query patterns to required capabilities
    capability_patterns = {
        "risk": ["risk_analysis", "quantitative_analysis"],
        "portfolio": ["portfolio_analysis", "quantitative_analysis"],
        "tax": ["tax_optimization", "portfolio_analysis"],
        "market": ["market_intelligence", "real_time_data"],
        "rebalance": ["strategy_rebalancing", "portfolio_optimization"],
        "options": ["options_analysis", "quantitative_analysis"],
        "news": ["news_sentiment", "market_intelligence"]
    }
    
    for pattern, caps in capability_patterns.items():
        if pattern in query_lower:
            capabilities.extend(caps)
    
    # Default capabilities if no specific patterns found
    if not capabilities:
        capabilities = ["query_interpretation", "general_analysis"]
    
    return list(set(capabilities))  # Remove duplicates

def _find_capable_agents(required_capabilities: List[str]) -> List[str]:
    """Find agents that have the required capabilities"""
    capable_agents = []
    
    for agent_id, registration in agent_registry.items():
        if any(cap in registration.capabilities for cap in required_capabilities):
            capable_agents.append(agent_id)
    
    return capable_agents

if __name__ == "__main__":
    uvicorn.run(
        "mcp.server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )