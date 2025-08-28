# Create agents/mcp_bridge.py

import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn
import httpx

from mcp.schemas import AgentRegistration, AgentJobRequest, AgentJobResponse, JobStatus
from agents.direct_agents import (
    PortfolioAnalysisAgent,
    RiskAssessmentAgent, 
    StockScreeningAgent,
    MarketAnalysisAgent
)

class AgentMCPBridge:
    """Bridge that allows direct agents to work with MCP server"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8001"):
        self.mcp_server_url = mcp_server_url
        self.agents = self._initialize_agents()
        self.registrations = self._create_registrations()
        
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all direct agents"""
        return {
            "portfolio_analysis": PortfolioAnalysisAgent(),
            "risk_assessment": RiskAssessmentAgent(),
            "stock_screening": StockScreeningAgent(), 
            "market_analysis": MarketAnalysisAgent()
        }
    
    def _create_registrations(self) -> List[AgentRegistration]:
        """Create MCP registrations for each agent"""
        registrations = []
        
        # Portfolio Analysis Agent
        registrations.append(AgentRegistration(
            agent_id="portfolio_analysis_agent",
            agent_name="Portfolio Analysis Agent",
            agent_type="PortfolioAnalysisAgent",
            capabilities=["portfolio_analysis", "portfolio_data_fetch"],
            endpoint_url="http://localhost:8002/agent/portfolio",
            max_concurrent_jobs=3
        ))
        
        # Risk Assessment Agent
        registrations.append(AgentRegistration(
            agent_id="risk_assessment_agent", 
            agent_name="Risk Assessment Agent",
            agent_type="RiskAssessmentAgent",
            capabilities=["risk_analysis", "quantitative_analysis"],
            endpoint_url="http://localhost:8002/agent/risk",
            max_concurrent_jobs=3
        ))
        
        # Stock Screening Agent
        registrations.append(AgentRegistration(
            agent_id="stock_screening_agent",
            agent_name="Stock Screening Agent", 
            agent_type="StockScreeningAgent",
            capabilities=["market_data_fetch", "general_analysis"],
            endpoint_url="http://localhost:8002/agent/screening",
            max_concurrent_jobs=5
        ))
        
        # Market Analysis Agent
        registrations.append(AgentRegistration(
            agent_id="market_analysis_agent",
            agent_name="Market Analysis Agent",
            agent_type="MarketAnalysisAgent", 
            capabilities=["market_intelligence", "real_time_data"],
            endpoint_url="http://localhost:8002/agent/market",
            max_concurrent_jobs=4
        ))
        
        return registrations
    
    async def register_all_agents(self):
        """Register all agents with the MCP server"""
        async with httpx.AsyncClient() as client:
            for registration in self.registrations:
                try:
                    response = await client.post(
                        f"{self.mcp_server_url}/register",
                        json=registration.model_dump()
                    )
                    response.raise_for_status()
                    print(f"Registered agent: {registration.agent_id}")
                except Exception as e:
                    print(f"Failed to register {registration.agent_id}: {str(e)}")
    
    async def execute_agent_job(self, job_request: AgentJobRequest) -> AgentJobResponse:
        """Execute a job request using the appropriate direct agent"""
        
        # Map capability to agent
        agent_map = {
            "portfolio_analysis": "portfolio_analysis",
            "portfolio_data_fetch": "portfolio_analysis", 
            "risk_analysis": "risk_assessment",
            "quantitative_analysis": "risk_assessment",
            "market_data_fetch": "stock_screening",
            "general_analysis": "stock_screening",
            "market_intelligence": "market_analysis",
            "real_time_data": "market_analysis"
        }
        
        agent_key = agent_map.get(job_request.capability)
        if not agent_key or agent_key not in self.agents:
            return AgentJobResponse(
                step_id=job_request.step_id,
                status=JobStatus.FAILED,
                error_message=f"No agent available for capability: {job_request.capability}"
            )
        
        agent = self.agents[agent_key]
        
        try:
            # Extract query from input data
            query = job_request.input_data.get("query", "")
            if not query:
                # Try to construct query from context
                context = job_request.context
                query = f"Analyze using {job_request.capability}"
            
            # Execute agent
            start_time = asyncio.get_event_loop().time()
            result = await agent.execute(query)
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            return AgentJobResponse(
                step_id=job_request.step_id,
                status=JobStatus.COMPLETED,
                result=result,
                confidence_score=result.get("confidence", 0.8),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            return AgentJobResponse(
                step_id=job_request.step_id,
                status=JobStatus.FAILED,
                error_message=str(e)
            )

# FastAPI app to serve as agent endpoints
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    bridge = AgentMCPBridge()
    app.state.bridge = bridge
    await bridge.register_all_agents()
    print("Agent MCP Bridge started and agents registered")
    yield
    # Shutdown
    print("Agent MCP Bridge shutting down")

app = FastAPI(
    title="Agent MCP Bridge",
    description="Bridge between direct agents and MCP server",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/agent/{agent_type}")
async def handle_agent_request(agent_type: str, job_request: AgentJobRequest):
    """Handle MCP job requests for agents"""
    bridge = app.state.bridge
    response = await bridge.execute_agent_job(job_request)
    return response.model_dump()

@app.get("/health")
async def health_check():
    """Health check for agent bridge"""
    return {
        "status": "healthy",
        "registered_agents": len(app.state.bridge.registrations),
        "timestamp": asyncio.get_event_loop().time()
    }

if __name__ == "__main__":
    uvicorn.run(
        "agents.mcp_bridge:app",
        host="0.0.0.0", 
        port=8002,
        reload=True
    )