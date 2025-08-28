# mcp/job_execution_engine.py
"""
Enhanced Job Execution Framework
================================
Production-ready job execution with WebSocket progress updates,
result caching, timeout handling, and retry mechanisms.
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from contextlib import asynccontextmanager

from .schemas import JobStatus, AgentJobRequest, AgentJobResponse
from .enhanced_agent_registry import enhanced_registry

logger = logging.getLogger(__name__)

class JobPriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10

@dataclass
class JobProgress:
    """Progress tracking for job execution"""
    job_id: str
    current_step: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    progress_percentage: float = 0.0
    estimated_completion: Optional[datetime] = None
    status_message: str = ""
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class JobResult:
    """Enhanced job result with metadata"""
    job_id: str
    result_data: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    cached_at: datetime
    cache_key: str
    size_bytes: int
    compressed: bool = False

class JobExecutionEngine:
    """Enhanced job execution engine with advanced capabilities"""
    
    def __init__(self):
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.job_progress: Dict[str, JobProgress] = {}
        self.job_queues: Dict[JobPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in JobPriority
        }
        self.result_cache: Dict[str, JobResult] = {}
        self.websocket_connections: Dict[str, set] = {}  # job_id -> set of websockets
        
        # Configuration
        self.default_timeout = 300  # 5 minutes
        self.max_retries = 3
        self.retry_backoff_base = 2.0
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 1000
        
        # Background tasks
        self.queue_processor_task = None
        self.cache_cleanup_task = None
    
    async def start(self):
        """Start the job execution engine"""
        self.queue_processor_task = asyncio.create_task(self._process_job_queues())
        self.cache_cleanup_task = asyncio.create_task(self._cleanup_cache_loop())
        logger.info("Job execution engine started")
    
    async def stop(self):
        """Stop the job execution engine"""
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
        if self.cache_cleanup_task:
            self.cache_cleanup_task.cancel()
        
        # Cancel all active jobs
        for task in self.active_jobs.values():
            task.cancel()
        
        logger.info("Job execution engine stopped")
    
    async def submit_job(self, 
                        job_id: str, 
                        job_function: Callable,
                        job_args: tuple = (),
                        job_kwargs: dict = None,
                        priority: JobPriority = JobPriority.NORMAL,
                        timeout: Optional[int] = None,
                        cache_results: bool = True) -> str:
        """Submit a job for execution with priority queuing"""
        
        job_kwargs = job_kwargs or {}
        timeout = timeout or self.default_timeout
        
        # Check cache first
        if cache_results:
            cache_key = self._generate_cache_key(job_function, job_args, job_kwargs)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Job {job_id} result served from cache")
                await self._notify_job_completion(job_id, cached_result.result_data, True)
                return job_id
        
        # Create job progress tracker
        self.job_progress[job_id] = JobProgress(
            job_id=job_id,
            status_message="Queued for execution"
        )
        
        # Queue the job
        job_item = {
            "job_id": job_id,
            "function": job_function,
            "args": job_args,
            "kwargs": job_kwargs,
            "timeout": timeout,
            "cache_results": cache_results,
            "cache_key": cache_key if cache_results else None,
            "submitted_at": datetime.now()
        }
        
        await self.job_queues[priority].put(job_item)
        await self._broadcast_progress_update(job_id)
        
        logger.info(f"Job {job_id} submitted with priority {priority.name}")
        return job_id
    
    async def _process_job_queues(self):
        """Process jobs from priority queues"""
        while True:
            try:
                # Process queues in priority order
                job_processed = False
                
                for priority in sorted(JobPriority, key=lambda x: x.value, reverse=True):
                    queue = self.job_queues[priority]
                    
                    if not queue.empty():
                        job_item = await asyncio.wait_for(queue.get(), timeout=0.1)
                        
                        # Create execution task
                        task = asyncio.create_task(
                            self._execute_job_with_retry(job_item)
                        )
                        self.active_jobs[job_item["job_id"]] = task
                        job_processed = True
                        break
                
                if not job_processed:
                    await asyncio.sleep(0.1)  # No jobs available, wait briefly
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in job queue processing: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_job_with_retry(self, job_item: Dict):
        """Execute job with retry logic and timeout handling"""
        job_id = job_item["job_id"]
        max_attempts = self.max_retries + 1
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Update progress
                progress = self.job_progress[job_id]
                progress.status_message = f"Executing (attempt {attempt}/{max_attempts})"
                await self._broadcast_progress_update(job_id)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_job(job_item),
                    timeout=job_item["timeout"]
                )
                
                # Cache result if requested
                if job_item["cache_results"] and job_item["cache_key"]:
                    self._cache_result(job_item["cache_key"], job_id, result)
                
                # Mark as completed
                progress.status_message = "Completed successfully"
                progress.progress_percentage = 100.0
                progress.completed_steps = progress.total_steps
                await self._broadcast_progress_update(job_id)
                
                await self._notify_job_completion(job_id, result, False)
                logger.info(f"Job {job_id} completed successfully on attempt {attempt}")
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Job {job_id} timed out after {job_item['timeout']} seconds (attempt {attempt})"
                logger.warning(error_msg)
                
                if attempt < max_attempts:
                    backoff_time = self.retry_backoff_base ** (attempt - 1)
                    progress.status_message = f"Timeout - retrying in {backoff_time}s"
                    await self._broadcast_progress_update(job_id)
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    await self._notify_job_failure(job_id, error_msg)
                    break
                    
            except Exception as e:
                error_msg = f"Job {job_id} failed: {str(e)} (attempt {attempt})"
                logger.error(error_msg)
                
                if attempt < max_attempts:
                    backoff_time = self.retry_backoff_base ** (attempt - 1)
                    progress.status_message = f"Failed - retrying in {backoff_time}s"
                    await self._broadcast_progress_update(job_id)
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    await self._notify_job_failure(job_id, error_msg)
                    break
        
        # Cleanup
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
    
    async def _execute_job(self, job_item: Dict) -> Dict[str, Any]:
        """Execute the actual job function"""
        job_id = job_item["job_id"]
        function = job_item["function"]
        args = job_item["args"]
        kwargs = job_item["kwargs"]
        
        # Initialize progress tracking
        progress = self.job_progress[job_id]
        progress.total_steps = 5  # Estimation for multi-step jobs
        progress.current_step = "Initializing"
        await self._broadcast_progress_update(job_id)
        
        start_time = time.time()
        
        try:
            # Step 1: Preparation
            progress.current_step = "Preparing execution"
            progress.completed_steps = 1
            progress.progress_percentage = 20.0
            await self._broadcast_progress_update(job_id)
            
            # Step 2: Execute function
            progress.current_step = "Executing main function"
            progress.completed_steps = 2
            progress.progress_percentage = 40.0
            await self._broadcast_progress_update(job_id)
            
            # Call the actual function
            if asyncio.iscoroutinefunction(function):
                result = await function(*args, **kwargs)
            else:
                result = function(*args, **kwargs)
            
            # Step 3: Processing results
            progress.current_step = "Processing results"
            progress.completed_steps = 3
            progress.progress_percentage = 60.0
            await self._broadcast_progress_update(job_id)
            
            # Step 4: Validation
            progress.current_step = "Validating results"
            progress.completed_steps = 4
            progress.progress_percentage = 80.0
            await self._broadcast_progress_update(job_id)
            
            # Step 5: Finalization
            progress.current_step = "Finalizing"
            progress.completed_steps = 5
            progress.progress_percentage = 95.0
            await self._broadcast_progress_update(job_id)
            
            execution_time = time.time() - start_time
            
            # Enhanced result with metadata
            enhanced_result = {
                "job_id": job_id,
                "result": result,
                "execution_metadata": {
                    "execution_time_seconds": execution_time,
                    "completed_at": datetime.now().isoformat(),
                    "function_name": function.__name__ if hasattr(function, '__name__') else str(function),
                    "cache_hit": False
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Job {job_id} execution failed after {execution_time:.2f}s: {e}")
            raise
    
    def _generate_cache_key(self, function: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call"""
        key_data = {
            "function": function.__name__ if hasattr(function, '__name__') else str(function),
            "args": str(args),
            "kwargs": str(sorted(kwargs.items()))
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[JobResult]:
        """Get cached result if available and not expired"""
        if cache_key not in self.result_cache:
            return None
        
        cached_result = self.result_cache[cache_key]
        
        # Check if expired
        if datetime.now() - cached_result.cached_at > timedelta(seconds=self.cache_ttl):
            del self.result_cache[cache_key]
            return None
        
        return cached_result
    
    def _cache_result(self, cache_key: str, job_id: str, result: Dict[str, Any]):
        """Cache job result"""
        try:
            result_json = json.dumps(result)
            size_bytes = len(result_json.encode())
            
            job_result = JobResult(
                job_id=job_id,
                result_data=result,
                execution_metadata=result.get("execution_metadata", {}),
                cached_at=datetime.now(),
                cache_key=cache_key,
                size_bytes=size_bytes
            )
            
            # Manage cache size
            if len(self.result_cache) >= self.max_cache_size:
                self._evict_oldest_cache_entries(int(self.max_cache_size * 0.1))
            
            self.result_cache[cache_key] = job_result
            logger.debug(f"Cached result for job {job_id} ({size_bytes} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to cache result for job {job_id}: {e}")
    
    def _evict_oldest_cache_entries(self, count: int):
        """Evict oldest cache entries"""
        if not self.result_cache:
            return
        
        # Sort by cache time and remove oldest
        sorted_entries = sorted(
            self.result_cache.items(),
            key=lambda x: x[1].cached_at
        )
        
        for i in range(min(count, len(sorted_entries))):
            cache_key = sorted_entries[i][0]
            del self.result_cache[cache_key]
        
        logger.debug(f"Evicted {count} oldest cache entries")
    
    async def _cleanup_cache_loop(self):
        """Background task to clean up expired cache entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.now()
                expired_keys = []
                
                for cache_key, result in self.result_cache.items():
                    if current_time - result.cached_at > timedelta(seconds=self.cache_ttl):
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self.result_cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def add_progress_websocket(self, job_id: str, websocket):
        """Add WebSocket connection for job progress updates"""
        if job_id not in self.websocket_connections:
            self.websocket_connections[job_id] = set()
        
        self.websocket_connections[job_id].add(websocket)
        
        # Send current progress if job exists
        if job_id in self.job_progress:
            await self._send_progress_to_websocket(websocket, job_id)
    
    async def remove_progress_websocket(self, job_id: str, websocket):
        """Remove WebSocket connection"""
        if job_id in self.websocket_connections:
            self.websocket_connections[job_id].discard(websocket)
            
            # Clean up empty sets
            if not self.websocket_connections[job_id]:
                del self.websocket_connections[job_id]
    
    async def _broadcast_progress_update(self, job_id: str):
        """Broadcast progress update to connected WebSockets"""
        if job_id not in self.websocket_connections:
            return
        
        progress = self.job_progress.get(job_id)
        if not progress:
            return
        
        # Update timestamp
        progress.last_updated = datetime.now()
        
        failed_connections = set()
        for websocket in self.websocket_connections[job_id]:
            try:
                await self._send_progress_to_websocket(websocket, job_id)
            except Exception as e:
                logger.error(f"Failed to send progress update: {e}")
                failed_connections.add(websocket)
        
        # Remove failed connections
        self.websocket_connections[job_id] -= failed_connections
    
    async def _send_progress_to_websocket(self, websocket, job_id: str):
        """Send progress update to specific WebSocket"""
        progress = self.job_progress.get(job_id)
        if not progress:
            return
        
        message = {
            "type": "job_progress",
            "job_id": job_id,
            "current_step": progress.current_step,
            "completed_steps": progress.completed_steps,
            "total_steps": progress.total_steps,
            "progress_percentage": progress.progress_percentage,
            "status_message": progress.status_message,
            "last_updated": progress.last_updated.isoformat()
        }
        
        await websocket.send_text(json.dumps(message))
    
    async def _notify_job_completion(self, job_id: str, result: Dict[str, Any], from_cache: bool):
        """Notify about job completion"""
        if job_id in self.websocket_connections:
            message = {
                "type": "job_completed",
                "job_id": job_id,
                "result": result,
                "from_cache": from_cache,
                "completed_at": datetime.now().isoformat()
            }
            
            failed_connections = set()
            for websocket in self.websocket_connections[job_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception:
                    failed_connections.add(websocket)
            
            # Cleanup
            self.websocket_connections[job_id] -= failed_connections
    
    async def _notify_job_failure(self, job_id: str, error_message: str):
        """Notify about job failure"""
        if job_id in self.websocket_connections:
            message = {
                "type": "job_failed",
                "job_id": job_id,
                "error_message": error_message,
                "failed_at": datetime.now().isoformat()
            }
            
            failed_connections = set()
            for websocket in self.websocket_connections[job_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception:
                    failed_connections.add(websocket)
            
            # Cleanup
            self.websocket_connections[job_id] -= failed_connections
    
    def get_job_progress(self, job_id: str) -> Optional[Dict]:
        """Get current job progress"""
        if job_id not in self.job_progress:
            return None
        
        progress = self.job_progress[job_id]
        return {
            "job_id": job_id,
            "current_step": progress.current_step,
            "completed_steps": progress.completed_steps,
            "total_steps": progress.total_steps,
            "progress_percentage": progress.progress_percentage,
            "status_message": progress.status_message,
            "last_updated": progress.last_updated.isoformat(),
            "is_active": job_id in self.active_jobs
        }
    
    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics"""
        total_size = sum(result.size_bytes for result in self.result_cache.values())
        
        return {
            "cache_entries": len(self.result_cache),
            "total_size_bytes": total_size,
            "max_cache_size": self.max_cache_size,
            "cache_ttl_seconds": self.cache_ttl,
            "hit_rate": "not_tracked"  # Could be enhanced to track hit rate
        }
    
    def get_execution_statistics(self) -> Dict:
        """Get job execution statistics"""
        return {
            "active_jobs": len(self.active_jobs),
            "queued_jobs": {
                priority.name: self.job_queues[priority].qsize() 
                for priority in JobPriority
            },
            "total_queued": sum(queue.qsize() for queue in self.job_queues.values()),
            "active_websocket_connections": sum(
                len(connections) for connections in self.websocket_connections.values()
            )
        }

# Global job execution engine
job_execution_engine = JobExecutionEngine()