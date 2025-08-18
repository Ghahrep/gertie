# agents/mcp_base_agent.py
import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import time

from mcp.schemas import (
    AgentRegistration, 
    AgentJobRequest, 
    AgentJobResponse, 
    JobStatus,
    AgentCapability
)

logger = logging.getLogger(__name__)

class MCPBaseAgent(ABC):
    """
    Base class for all MCP-compatible agents.
    Agents are stateless workers that register with MCP and respond to job requests.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        agent_name: str, 
        capabilities: List[str],
        mcp_url: str = "http://localhost:8001",
        max_concurrent_jobs: int = 5
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.capabilities = capabilities
        self.mcp_url = mcp_url
        self.max_concurrent_jobs = max_concurrent_jobs
        self.is_registered = False
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Initialize HTTP session for MCP communication
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Start the agent and register with MCP"""
        self.session = aiohttp.ClientSession()
        await self.register_with_mcp()
        await self.start_job_listener()
        
    async def stop(self):
        """Stop the agent and cleanup resources"""
        if self.session:
            await self.session.close()
        
        # Cancel all active jobs
        for job_id, task in self.active_jobs.items():
            task.cancel()
            logger.info(f"Cancelled job {job_id}")
        
        await self.unregister_from_mcp()
        
    async def register_with_mcp(self):
        """Register this agent with the Master Control Plane"""
        registration = AgentRegistration(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            agent_type=self.__class__.__name__,
            capabilities=self.capabilities,
            endpoint_url=f"http://localhost:{self.get_port()}/agent",
            max_concurrent_jobs=self.max_concurrent_jobs,
            response_time_sla=self.get_response_time_sla(),
            metadata=self.get_agent_metadata()
        )
        
        try:
            async with self.session.post(
                f"{self.mcp_url}/register",
                json=registration.dict()
            ) as response:
                if response.status == 200:
                    self.is_registered = True
                    result = await response.json()
                    logger.info(f"✅ Agent {self.agent_id} registered successfully: {result['message']}")
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Failed to register agent {self.agent_id}: {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ Error registering agent {self.agent_id}: {str(e)}")
    
    async def unregister_from_mcp(self):
        """Unregister this agent from the MCP"""
        if not self.is_registered:
            return
            
        try:
            async with self.session.delete(
                f"{self.mcp_url}/agents/{self.agent_id}"
            ) as response:
                if response.status == 200:
                    self.is_registered = False
                    logger.info(f"✅ Agent {self.agent_id} unregistered successfully")
                    
        except Exception as e:
            logger.error(f"❌ Error unregistering agent {self.agent_id}: {str(e)}")
    
    async def start_job_listener(self):
        """Start listening for job requests from MCP"""
        # In a real implementation, this would set up a webhook endpoint or polling mechanism
        # For now, we'll simulate with a polling mechanism
        logger.info(f"🎧 Agent {self.agent_id} started listening for jobs")
        
        while self.is_registered:
            try:
                # Poll for new jobs (in production, use webhooks or message queue)
                await self.check_for_new_jobs()
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                logger.error(f"Error in job listener for {self.agent_id}: {str(e)}")
                await asyncio.sleep(5)  # Back off on error
    
    async def check_for_new_jobs(self):
        """Check for new jobs assigned to this agent"""
        # This is a simplified polling mechanism
        # In production, use webhooks, message queues, or WebSocket connections
        pass  # MCP will call handle_job_request directly
    
    async def handle_job_request(self, job_request: AgentJobRequest) -> AgentJobResponse:
        """Handle a job request from the MCP"""
        start_time = time.time()
        step_id = job_request.step_id
        
        try:
            logger.info(f"🔄 Processing job step {step_id} with capability {job_request.capability}")
            
            # Check if we can handle this capability
            if job_request.capability not in self.capabilities:
                return AgentJobResponse(
                    step_id=step_id,
                    status=JobStatus.FAILED,
                    error_message=f"Agent {self.agent_id} does not support capability {job_request.capability}"
                )
            
            # Fetch any missing data autonomously
            enriched_data = await self.autonomous_data_fetch(
                job_request.input_data.get("query", ""), 
                job_request.context
            )
            
            # Merge enriched data with input data
            combined_data = {**job_request.input_data, **enriched_data}
            
            # Execute the specific capability
            result = await self.execute_capability(
                job_request.capability, 
                combined_data, 
                job_request.context
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return AgentJobResponse(
                step_id=step_id,
                status=JobStatus.COMPLETED,
                result=result,
                confidence_score=self.calculate_confidence_score(result),
                execution_time_ms=execution_time_ms,
                metadata={
                    "agent_id": self.agent_id,
                    "capability_version": self.get_capability_version(job_request.capability),
                    "data_sources_used": self.get_data_sources_used(combined_data)
                }
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"❌ Error processing step {step_id}: {str(e)}")
            
            return AgentJobResponse(
                step_id=step_id,
                status=JobStatus.FAILED,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"agent_id": self.agent_id}
            )
    
    async def autonomous_data_fetch(self, query: str, context: Dict) -> Dict:
        """Agents autonomously fetch missing data based on query"""
        enriched_data = {}
        query_lower = query.lower()
        
        # Fetch portfolio data if needed and not already provided
        if ("portfolio" in query_lower or "holdings" in query_lower) and not context.get("portfolio_data"):
            logger.info(f"🔍 Autonomously fetching portfolio data for query: {query}")
            portfolio_data = await self.fetch_portfolio_data(query, context)
            if portfolio_data:
                enriched_data["portfolio_data"] = portfolio_data
        
        # Fetch market data if needed and not already provided
        if ("market" in query_lower or "price" in query_lower) and not context.get("market_data"):
            logger.info(f"📈 Autonomously fetching market data for query: {query}")
            market_data = await self.fetch_market_data(query, context)
            if market_data:
                enriched_data["market_data"] = market_data
        
        # Fetch economic data if needed
        if ("economic" in query_lower or "fed" in query_lower) and not context.get("economic_data"):
            logger.info(f"🏛️ Autonomously fetching economic data for query: {query}")
            economic_data = await self.fetch_economic_data(query, context)
            if economic_data:
                enriched_data["economic_data"] = economic_data
        
        # Fetch news data if needed
        if ("news" in query_lower or "sentiment" in query_lower) and not context.get("news_data"):
            logger.info(f"📰 Autonomously fetching news data for query: {query}")
            news_data = await self.fetch_news_data(query, context)
            if news_data:
                enriched_data["news_data"] = news_data
        
        return enriched_data
    
    async def fetch_portfolio_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Fetch portfolio data from the portfolio service"""
        portfolio_id = context.get("portfolio_id")
        if not portfolio_id:
            return None
        
        try:
            # Mock portfolio data fetch - replace with actual API call
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "portfolio_id": portfolio_id,
                "holdings": [
                    {"symbol": "AAPL", "shares": 100, "current_price": 150.25},
                    {"symbol": "GOOGL", "shares": 50, "current_price": 2800.50},
                    {"symbol": "TSLA", "shares": 25, "current_price": 900.75}
                ],
                "total_value": 305518.75,
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {str(e)}")
            return None
    
    async def fetch_market_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Fetch real-time market data"""
        try:
            # Mock market data fetch - replace with actual API call
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "market_indices": {
                    "SPY": {"price": 445.20, "change": 2.15, "change_pct": 0.48},
                    "QQQ": {"price": 375.80, "change": -1.25, "change_pct": -0.33},
                    "IWM": {"price": 195.45, "change": 0.85, "change_pct": 0.44}
                },
                "volatility": {
                    "VIX": 18.25
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None
    
    async def fetch_economic_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Fetch economic indicators and data"""
        try:
            # Mock economic data fetch - replace with actual API call
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "indicators": {
                    "fed_funds_rate": 5.25,
                    "inflation_rate": 3.2,
                    "unemployment_rate": 3.7,
                    "gdp_growth": 2.1
                },
                "treasury_yields": {
                    "1Y": 4.95,
                    "10Y": 4.25,
                    "30Y": 4.35
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching economic data: {str(e)}")
            return None
    
    async def fetch_news_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Fetch relevant news and sentiment data"""
        try:
            # Mock news data fetch - replace with actual API call
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "headlines": [
                    {
                        "title": "Fed Signals Potential Rate Cut in Q2",
                        "sentiment": "positive",
                        "relevance": 0.85,
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    {
                        "title": "Tech Earnings Beat Expectations",
                        "sentiment": "positive", 
                        "relevance": 0.92,
                        "timestamp": "2024-01-15T09:15:00Z"
                    }
                ],
                "overall_sentiment": 0.72,
                "market_sentiment": "optimistic"
            }
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return None
    
    @abstractmethod
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """Execute a specific capability - must be implemented by subclasses"""
        pass
    
    def calculate_confidence_score(self, result: Dict) -> float:
        """Calculate confidence score for the result"""
        # Base implementation - can be overridden by subclasses
        if not result:
            return 0.0
        
        # Simple confidence calculation based on data completeness
        data_completeness = len([v for v in result.values() if v is not None]) / len(result)
        return min(0.95, 0.7 + (data_completeness * 0.25))
    
    def get_capability_version(self, capability: str) -> str:
        """Get version of the capability implementation"""
        return "1.0.0"  # Override in subclasses
    
    def get_data_sources_used(self, data: Dict) -> List[str]:
        """Get list of data sources used in the analysis"""
        sources = []
        if "portfolio_data" in data:
            sources.append("portfolio_service")
        if "market_data" in data:
            sources.append("market_data_api")
        if "economic_data" in data:
            sources.append("economic_indicators_api")
        if "news_data" in data:
            sources.append("news_sentiment_api")
        return sources
    
    def get_port(self) -> int:
        """Get the port this agent should listen on"""
        # Generate port based on agent_id hash for consistency
        return 8000 + (hash(self.agent_id) % 1000)
    
    def get_response_time_sla(self) -> int:
        """Get the response time SLA for this agent in seconds"""
        return 30  # Override in subclasses if needed
    
    def get_agent_metadata(self) -> Dict[str, Any]:
        """Get additional metadata about this agent"""
        return {
            "version": "1.0.0",
            "startup_time": datetime.utcnow().isoformat(),
            "supported_data_sources": self.get_supported_data_sources()
        }
    
    def get_supported_data_sources(self) -> List[str]:
        """Get list of data sources this agent can work with"""
        return ["portfolio_service", "market_data_api"]  # Override in subclasses