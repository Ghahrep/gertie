# services/mcp_client.py
import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from mcp.schemas import JobRequest, JobResponse, JobStatus, HealthCheck

logger = logging.getLogger(__name__)

class MCPClientError(Exception):
    """Custom exception for MCP client errors"""
    pass

class MCPClient:
    """
    Client for communicating with the Master Control Plane (MCP) server.
    Handles job submission, status checking, and MCP health monitoring.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._connection_pool_limit = 100
        self._connection_pool_limit_per_host = 30
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def start(self):
        """Initialize the HTTP session"""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=self._connection_pool_limit,
                limit_per_host=self._connection_pool_limit_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            logger.info(f"MCP Client initialized with base URL: {self.base_url}")
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("MCP Client session closed")
    
    async def health_check(self) -> HealthCheck:
        """Check the health status of the MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return HealthCheck(**data)
                else:
                    error_text = await response.text()
                    raise MCPClientError(f"Health check failed: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            raise MCPClientError(f"Failed to connect to MCP: {str(e)}")
    
    async def submit_job(self, job_request: JobRequest) -> JobResponse:
        """Submit a new job to the MCP for processing"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/submit_job",
                json=job_request.dict()
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return JobResponse(**data)
                else:
                    error_text = await response.text()
                    raise MCPClientError(f"Job submission failed: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            raise MCPClientError(f"Failed to submit job to MCP: {str(e)}")
    
    async def get_job_status(self, job_id: str) -> Optional[JobResponse]:
        """Get the current status of a job"""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.base_url}/job/{job_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    return JobResponse(**data)
                elif response.status == 404:
                    return None
                else:
                    error_text = await response.text()
                    raise MCPClientError(f"Failed to get job status: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            raise MCPClientError(f"Failed to get job status from MCP: {str(e)}")
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        await self._ensure_session()
        
        try:
            async with self.session.delete(f"{self.base_url}/job/{job_id}/cancel") as response:
                if response.status == 200:
                    return True
                elif response.status == 404:
                    logger.warning(f"Job {job_id} not found for cancellation")
                    return False
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to cancel job {job_id}: {response.status} - {error_text}")
                    return False
                    
        except aiohttp.ClientError as e:
            logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            return False
    
    async def list_agents(self) -> list:
        """Get list of all registered agents"""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.base_url}/agents") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise MCPClientError(f"Failed to list agents: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            raise MCPClientError(f"Failed to list agents from MCP: {str(e)}")
    
    async def wait_for_job_completion(
        self, 
        job_id: str, 
        poll_interval: float = 1.0, 
        max_wait_time: float = 300.0
    ) -> JobResponse:
        """
        Wait for a job to complete, polling for status updates.
        
        Args:
            job_id: The job ID to wait for
            poll_interval: How often to check status (seconds)
            max_wait_time: Maximum time to wait (seconds)
            
        Returns:
            JobResponse: Final job response when completed
            
        Raises:
            MCPClientError: If job fails or times out
        """
        start_time = datetime.utcnow()
        
        while True:
            job_response = await self.get_job_status(job_id)
            
            if not job_response:
                raise MCPClientError(f"Job {job_id} not found")
            
            # Check if job is in a terminal state
            if job_response.status == JobStatus.COMPLETED:
                logger.info(f"Job {job_id} completed successfully")
                return job_response
            
            elif job_response.status == JobStatus.FAILED:
                error_msg = job_response.error_details or "Unknown error"
                raise MCPClientError(f"Job {job_id} failed: {error_msg}")
            
            elif job_response.status == JobStatus.CANCELLED:
                raise MCPClientError(f"Job {job_id} was cancelled")
            
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_wait_time:
                raise MCPClientError(f"Job {job_id} timed out after {max_wait_time} seconds")
            
            # Log progress if available
            if job_response.progress is not None:
                logger.debug(f"Job {job_id} progress: {job_response.progress:.1f}%")
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def submit_and_wait(
        self, 
        job_request: JobRequest, 
        poll_interval: float = 1.0, 
        max_wait_time: float = 300.0
    ) -> JobResponse:
        """
        Submit a job and wait for its completion.
        
        Args:
            job_request: The job request to submit
            poll_interval: How often to check status (seconds)
            max_wait_time: Maximum time to wait (seconds)
            
        Returns:
            JobResponse: Final job response when completed
        """
        # Submit the job
        job_response = await self.submit_job(job_request)
        
        if job_response.status == JobStatus.FAILED:
            raise MCPClientError(f"Job failed to start: {job_response.message}")
        
        # Wait for completion
        return await self.wait_for_job_completion(
            job_response.job_id, 
            poll_interval, 
            max_wait_time
        )
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics from MCP"""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise MCPClientError(f"Failed to get metrics: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            raise MCPClientError(f"Failed to get metrics from MCP: {str(e)}")
    
    async def _ensure_session(self):
        """Ensure the HTTP session is initialized"""
        if self.session is None:
            await self.start()
    
    async def ping(self) -> bool:
        """Simple ping to check if MCP is responding"""
        try:
            health = await self.health_check()
            return health.status == "healthy"
        except:
            return False

# Global MCP client instance
_mcp_client_instance: Optional[MCPClient] = None

async def get_mcp_client() -> MCPClient:
    """Get or create the global MCP client instance"""
    global _mcp_client_instance
    
    if _mcp_client_instance is None:
        _mcp_client_instance = MCPClient()
        await _mcp_client_instance.start()
    
    return _mcp_client_instance

async def close_mcp_client():
    """Close the global MCP client instance"""
    global _mcp_client_instance
    
    if _mcp_client_instance:
        await _mcp_client_instance.close()
        _mcp_client_instance = None

# Context manager for temporary MCP client
class TemporaryMCPClient:
    """Context manager for creating temporary MCP client instances"""
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.client: Optional[MCPClient] = None
    
    async def __aenter__(self) -> MCPClient:
        self.client = MCPClient(self.base_url, self.timeout)
        await self.client.start()
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()