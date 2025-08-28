# Create workflows/mcp_integration.py

import httpx
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from mcp.schemas import JobRequest, JobResponse, JobStatus

class MCPServerClient:
    """Client to communicate with your existing MCP server"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8001"):
        self.base_url = mcp_server_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def submit_job(self, job_request: JobRequest) -> JobResponse:
        """Submit job to MCP server"""
        try:
            response = await self.client.post(
                f"{self.base_url}/submit_job",
                json=job_request.model_dump()
            )
            response.raise_for_status()
            data = response.json()
            return JobResponse(**data)
        except Exception as e:
            raise Exception(f"Failed to submit job to MCP server: {str(e)}")
    
    async def get_job_status(self, job_id: str) -> JobResponse:
        """Get job status from MCP server"""
        try:
            response = await self.client.get(f"{self.base_url}/job/{job_id}")
            response.raise_for_status()
            data = response.json()
            return JobResponse(**data)
        except Exception as e:
            raise Exception(f"Failed to get job status from MCP server: {str(e)}")
    
    async def get_health(self) -> Dict[str, Any]:
        """Get MCP server health status"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get MCP server health: {str(e)}")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()