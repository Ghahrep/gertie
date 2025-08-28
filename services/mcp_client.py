# services/mcp_client.py
import aiohttp
import asyncio
import logging
import json
import time
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from mcp.schemas import JobRequest, JobResponse, JobStatus, HealthCheck

logger = logging.getLogger(__name__)

class MCPClientError(Exception):
    """Custom exception for MCP client errors"""
    pass

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

@dataclass
class ClientMetrics:
    """Track client performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    avg_response_time: float = 0.0
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    last_request_time: Optional[datetime] = None

class MCPClient:
    """
    Enhanced Client for communicating with the Master Control Plane (MCP) server.
    Handles job submission, status checking, MCP health monitoring with enhanced
    error handling, caching, and performance monitoring.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._connection_pool_limit = 100
        self._connection_pool_limit_per_host = 30
        
        # Enhanced features
        self.metrics = ClientMetrics()
        self._response_cache: Dict[str, Dict] = {}
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60.0
        self._cache_ttl = 300  # 5 minutes
        self._max_retries = 3
        self._retry_delay_base = 1.0
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def start(self):
        """Initialize the HTTP session with enhanced configuration"""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=self._connection_pool_limit,
                limit_per_host=self._connection_pool_limit_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "EnhancedMCPClient/1.0"
                }
            )
            
            logger.info(f"Enhanced MCP Client initialized with base URL: {self.base_url}")
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("MCP Client session closed")
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests"""
        if self.metrics.circuit_breaker_state == CircuitBreakerState.CLOSED:
            return True
        elif self.metrics.circuit_breaker_state == CircuitBreakerState.OPEN:
            if (self._circuit_breaker_last_failure and 
                datetime.now() - self._circuit_breaker_last_failure > 
                timedelta(seconds=self._circuit_breaker_timeout)):
                self.metrics.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        elif self.metrics.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            return True
        return False
    
    def _record_success(self):
        """Record successful request"""
        self.metrics.successful_requests += 1
        self._circuit_breaker_failures = 0
        
        if self.metrics.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self.metrics.circuit_breaker_state = CircuitBreakerState.CLOSED
            logger.info("Circuit breaker closed after successful request")
    
    def _record_failure(self):
        """Record failed request"""
        self.metrics.failed_requests += 1
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.now()
        
        if (self._circuit_breaker_failures >= self._circuit_breaker_threshold and
            self.metrics.circuit_breaker_state != CircuitBreakerState.OPEN):
            self.metrics.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.error(f"Circuit breaker opened after {self._circuit_breaker_failures} failures")
    
    def _generate_cache_key(self, method: str, url: str, data: Optional[Dict] = None) -> str:
        """Generate cache key for request"""
        key_data = f"{method}:{url}:{json.dumps(data, sort_keys=True) if data else ''}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if available"""
        if cache_key not in self._response_cache:
            return None
        
        cached = self._response_cache[cache_key]
        if datetime.now() - cached['timestamp'] > timedelta(seconds=self._cache_ttl):
            del self._response_cache[cache_key]
            return None
        
        self.metrics.cache_hits += 1
        return cached['data']
    
    def _cache_response(self, cache_key: str, data: Dict):
        """Cache successful response"""
        # Simple cache size management
        if len(self._response_cache) > 1000:
            # Remove oldest 10% of entries
            sorted_entries = sorted(
                self._response_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            for i in range(len(sorted_entries) // 10):
                del self._response_cache[sorted_entries[i][0]]
        
        self._response_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def _make_request_with_retry(self, method: str, url: str, 
                                     data: Optional[Dict] = None, 
                                     cacheable: bool = False) -> Dict:
        """Make HTTP request with retry logic and enhanced error handling"""
        
        if not self._check_circuit_breaker():
            raise MCPClientError("Circuit breaker is OPEN - MCP server unavailable")
        
        # Check cache first
        if cacheable:
            cache_key = self._generate_cache_key(method, url, data)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        await self._ensure_session()
        self.metrics.total_requests += 1
        self.metrics.last_request_time = datetime.now()
        
        last_exception = None
        start_time = time.time()
        
        for attempt in range(self._max_retries + 1):
            try:
                kwargs = {}
                if data is not None:
                    kwargs['json'] = data
                
                async with self.session.request(method, url, **kwargs) as response:
                    response_time = time.time() - start_time
                    
                    # Update average response time
                    if self.metrics.avg_response_time == 0:
                        self.metrics.avg_response_time = response_time
                    else:
                        self.metrics.avg_response_time = (
                            0.8 * self.metrics.avg_response_time + 0.2 * response_time
                        )
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        raise MCPClientError(f"HTTP {response.status}: {error_text}")
                    
                    response_data = await response.json()
                    
                    # Cache successful GET requests
                    if cacheable and method.upper() == 'GET':
                        self._cache_response(cache_key, response_data)
                    
                    self._record_success()
                    logger.debug(f"MCP request successful: {method} {url} ({response_time:.3f}s)")
                    return response_data
                    
            except asyncio.TimeoutError as e:
                last_exception = MCPClientError(f"Request timed out after {self.timeout}s")
                logger.warning(f"MCP request timeout: {method} {url} (attempt {attempt + 1})")
                
            except aiohttp.ClientError as e:
                last_exception = MCPClientError(f"Network error: {str(e)}")
                logger.warning(f"MCP network error: {method} {url} (attempt {attempt + 1}): {e}")
                
            except Exception as e:
                last_exception = MCPClientError(f"Unexpected error: {str(e)}")
                logger.error(f"MCP unexpected error: {method} {url} (attempt {attempt + 1}): {e}")
            
            # Retry with exponential backoff
            if attempt < self._max_retries:
                delay = self._retry_delay_base * (2 ** attempt)
                logger.info(f"Retrying MCP request in {delay}s (attempt {attempt + 2})")
                await asyncio.sleep(delay)
        
        # All retries failed
        self._record_failure()
        logger.error(f"MCP request failed after {self._max_retries + 1} attempts: {method} {url}")
        raise last_exception
    
    async def health_check(self) -> HealthCheck:
        """Check the health status of the MCP server with caching"""
        try:
            data = await self._make_request_with_retry(
                "GET", 
                f"{self.base_url}/health",
                cacheable=True
            )
            return HealthCheck(**data)
        except MCPClientError:
            raise
        except Exception as e:
            raise MCPClientError(f"Health check failed: {str(e)}")
    
    async def submit_job(self, job_request: JobRequest) -> JobResponse:
        """Submit a new job to the MCP for processing with retry logic"""
        try:
            data = await self._make_request_with_retry(
                "POST",
                f"{self.base_url}/submit_job",
                data=job_request.model_dump()
            )
            return JobResponse(**data)
        except MCPClientError:
            raise
        except Exception as e:
            raise MCPClientError(f"Job submission failed: {str(e)}")
    
    async def get_job_status(self, job_id: str) -> Optional[JobResponse]:
        """Get the current status of a job with caching"""
        try:
            data = await self._make_request_with_retry(
                "GET",
                f"{self.base_url}/job/{job_id}",
                cacheable=True
            )
            return JobResponse(**data)
        except MCPClientError as e:
            if "404" in str(e):
                return None
            raise
        except Exception as e:
            raise MCPClientError(f"Failed to get job status: {str(e)}")
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job with enhanced error handling"""
        try:
            await self._make_request_with_retry(
                "DELETE",
                f"{self.base_url}/job/{job_id}/cancel"
            )
            return True
        except MCPClientError as e:
            if "404" in str(e):
                logger.warning(f"Job {job_id} not found for cancellation")
                return False
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def list_agents(self) -> list:
        """Get list of all registered agents with caching"""
        try:
            return await self._make_request_with_retry(
                "GET",
                f"{self.base_url}/agents",
                cacheable=True
            )
        except MCPClientError:
            raise
        except Exception as e:
            raise MCPClientError(f"Failed to list agents: {str(e)}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from MCP server"""
        try:
            return await self._make_request_with_retry(
                "GET",
                f"{self.base_url}/api/performance/summary",
                cacheable=True
            )
        except MCPClientError:
            raise
        except Exception as e:
            raise MCPClientError(f"Failed to get performance summary: {str(e)}")
    
    async def wait_for_job_completion(
        self, 
        job_id: str, 
        poll_interval: float = 1.0, 
        max_wait_time: float = 300.0
    ) -> JobResponse:
        """
        Wait for a job to complete, polling for status updates with enhanced monitoring.
        """
        start_time = datetime.utcnow()
        last_progress = None
        
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
            
            # Log progress if available and changed
            if (job_response.progress is not None and 
                job_response.progress != last_progress):
                logger.info(f"Job {job_id} progress: {job_response.progress:.1f}%")
                last_progress = job_response.progress
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def submit_and_wait(
        self, 
        job_request: JobRequest, 
        poll_interval: float = 1.0, 
        max_wait_time: float = 300.0
    ) -> JobResponse:
        """Submit a job and wait for its completion with enhanced monitoring."""
        # Submit the job
        job_response = await self.submit_job(job_request)
        
        if job_response.status == JobStatus.FAILED:
            raise MCPClientError(f"Job failed to start: {job_response.message}")
        
        logger.info(f"Job {job_response.job_id} submitted, waiting for completion...")
        
        # Wait for completion
        return await self.wait_for_job_completion(
            job_response.job_id, 
            poll_interval, 
            max_wait_time
        )
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics from MCP with caching"""
        try:
            return await self._make_request_with_retry(
                "GET",
                f"{self.base_url}/metrics",
                cacheable=True
            )
        except MCPClientError:
            raise
        except Exception as e:
            raise MCPClientError(f"Failed to get metrics: {str(e)}")
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        success_rate = (
            self.metrics.successful_requests / max(self.metrics.total_requests, 1)
        )
        cache_hit_rate = (
            self.metrics.cache_hits / max(self.metrics.total_requests, 1)
        )
        
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": success_rate,
            "cache_hits": self.metrics.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_response_time_ms": self.metrics.avg_response_time * 1000,
            "circuit_breaker_state": self.metrics.circuit_breaker_state.value,
            "cached_responses": len(self._response_cache),
            "last_request": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None
        }
    
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

# Global MCP client instance (preserving your existing pattern)
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

# Context manager for temporary MCP client (preserving your existing pattern)
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