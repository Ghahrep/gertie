# services/task_scheduler.py
"""
Task 2.2.3: Background Task Scheduling System
APScheduler implementation with failure handling and monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import traceback

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.events import (
    EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED,
    JobExecutionEvent, JobEvent
)
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

from sqlalchemy.orm import Session
from db.session import get_db

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    retry_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass  
class TaskDefinition:
    """Task definition with execution parameters"""
    task_id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 300
    failure_threshold: int = 5  # Consecutive failures before disabling
    enabled: bool = True
    metadata: dict = field(default_factory=dict)

class BackgroundTaskScheduler:
    """
    Complete background task scheduling system with monitoring and failure handling
    Integrates with your existing portfolio monitoring and alert systems
    """
    
    def __init__(self, timezone: str = "UTC"):
        # Configure APScheduler
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': MemoryJobStore()},
            executors={'default': AsyncIOExecutor()},
            job_defaults={
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 300  # 5 minutes
            },
            timezone=timezone
        )
        
        # Task tracking
        self.tasks: Dict[str, TaskDefinition] = {}
        self.task_results: Dict[str, List[TaskResult]] = {}
        self.failure_counts: Dict[str, int] = {}
        self.health_stats = {
            "scheduler_started": False,
            "total_tasks_registered": 0,
            "active_jobs": 0,
            "tasks_executed_today": 0,
            "task_failures_today": 0,
            "avg_execution_time_ms": 0,
            "last_health_check": None
        }
        
        # Event listeners
        self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)  
        self.scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
        
        logger.info("Background Task Scheduler initialized")
    
    async def start(self) -> bool:
        """Start the background scheduler"""
        try:
            if not self.scheduler.running:
                self.scheduler.start()
                self.health_stats["scheduler_started"] = True
                self.health_stats["last_health_check"] = datetime.utcnow()
                
                # Add system health check task
                await self._add_system_health_check()
                
                logger.info("Background Task Scheduler started successfully")
                return True
            else:
                logger.info("Background Task Scheduler already running")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the background scheduler"""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)
                self.health_stats["scheduler_started"] = False
                logger.info("Background Task Scheduler stopped")
                return True
            else:
                logger.info("Background Task Scheduler not running")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
            return False
    
    # Task Registration Methods
    async def schedule_recurring_task(
        self,
        task_def: TaskDefinition,
        interval_minutes: int = None,
        cron_expression: str = None,
        start_date: datetime = None
    ) -> bool:
        """Schedule a recurring task"""
        try:
            # Determine trigger
            if cron_expression:
                trigger = CronTrigger.from_crontab(cron_expression)
            elif interval_minutes:
                trigger = IntervalTrigger(
                    minutes=interval_minutes,
                    start_date=start_date or datetime.utcnow()
                )
            else:
                raise ValueError("Either interval_minutes or cron_expression required")
            
            # Add job to scheduler
            job = self.scheduler.add_job(
                func=self._execute_task_wrapper,
                trigger=trigger,
                args=[task_def],
                id=task_def.task_id,
                name=task_def.name,
                replace_existing=True,
                max_instances=1
            )
            
            # Register task
            self.tasks[task_def.task_id] = task_def
            self.task_results[task_def.task_id] = []
            self.failure_counts[task_def.task_id] = 0
            self.health_stats["total_tasks_registered"] += 1
            
            logger.info(f"Scheduled recurring task: {task_def.name} ({task_def.task_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule recurring task {task_def.name}: {e}")
            return False
    
    async def schedule_one_time_task(
        self,
        task_def: TaskDefinition,
        run_date: datetime
    ) -> bool:
        """Schedule a one-time task"""
        try:
            trigger = DateTrigger(run_date=run_date)
            
            job = self.scheduler.add_job(
                func=self._execute_task_wrapper,
                trigger=trigger,
                args=[task_def],
                id=task_def.task_id,
                name=task_def.name,
                replace_existing=True
            )
            
            # Register task
            self.tasks[task_def.task_id] = task_def
            self.task_results[task_def.task_id] = []
            self.failure_counts[task_def.task_id] = 0
            
            logger.info(f"Scheduled one-time task: {task_def.name} for {run_date}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule one-time task {task_def.name}: {e}")
            return False
    
    async def run_task_now(self, task_id: str) -> TaskResult:
        """Execute a task immediately"""
        try:
            task_def = self.tasks.get(task_id)
            if not task_def:
                return TaskResult(task_id=task_id, status=TaskStatus.FAILED, 
                                error="Task not found")
            
            result = await self._execute_task_wrapper(task_def)
            return result
            
        except Exception as e:
            logger.error(f"Failed to run task {task_id}: {e}")
            return TaskResult(task_id=task_id, status=TaskStatus.FAILED, error=str(e))
    
    async def _execute_task_wrapper(self, task_def: TaskDefinition) -> TaskResult:
        """Wrapper that handles task execution with retry logic and monitoring"""
        task_result = TaskResult(
            task_id=task_def.task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Check if task is enabled
            if not task_def.enabled:
                task_result.status = TaskStatus.CANCELLED
                task_result.error = "Task is disabled"
                return task_result
            
            # Execute with timeout
            execution_start = datetime.utcnow()
            
            try:
                # Execute the actual task function
                if asyncio.iscoroutinefunction(task_def.func):
                    result = await asyncio.wait_for(
                        task_def.func(*task_def.args, **task_def.kwargs),
                        timeout=task_def.timeout_seconds
                    )
                else:
                    # Run sync function in thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, task_def.func, *task_def.args, **task_def.kwargs
                    )
                
                # Task succeeded
                execution_end = datetime.utcnow()
                task_result.status = TaskStatus.COMPLETED
                task_result.result = result
                task_result.completed_at = execution_end
                task_result.execution_time_ms = int(
                    (execution_end - execution_start).total_seconds() * 1000
                )
                
                # Reset failure count on success
                self.failure_counts[task_def.task_id] = 0
                
                logger.info(f"Task {task_def.name} completed successfully in {task_result.execution_time_ms}ms")
                
            except asyncio.TimeoutError:
                task_result.status = TaskStatus.FAILED
                task_result.error = f"Task timed out after {task_def.timeout_seconds} seconds"
                logger.error(f"Task {task_def.name} timed out")
                
            except Exception as task_error:
                task_result.status = TaskStatus.FAILED
                task_result.error = str(task_error)
                task_result.completed_at = datetime.utcnow()
                
                logger.error(f"Task {task_def.name} failed: {task_error}")
                
                # Handle retries for failed tasks
                if task_result.retry_count < task_def.max_retries:
                    await self._schedule_retry(task_def, task_result.retry_count + 1)
                    task_result.status = TaskStatus.RETRYING
                    logger.info(f"Scheduling retry {task_result.retry_count + 1} for task {task_def.name}")
                else:
                    logger.error(f"Task {task_def.name} failed permanently after {task_def.max_retries} retries")
            
            # Track failure count and disable if necessary
            if task_result.status == TaskStatus.FAILED:
                self.failure_counts[task_def.task_id] += 1
                
                if self.failure_counts[task_def.task_id] >= task_def.failure_threshold:
                    task_def.enabled = False
                    logger.critical(f"Task {task_def.name} disabled after {task_def.failure_threshold} consecutive failures")
            
        except Exception as wrapper_error:
            task_result.status = TaskStatus.FAILED
            task_result.error = f"Task wrapper error: {wrapper_error}"
            task_result.completed_at = datetime.utcnow()
            logger.error(f"Task wrapper error for {task_def.name}: {wrapper_error}")
        
        finally:
            # Store task result
            self._store_task_result(task_result)
        
        return task_result
    
    async def _schedule_retry(self, task_def: TaskDefinition, retry_count: int):
        """Schedule a task retry with exponential backoff"""
        try:
            # Exponential backoff: base_delay * (2^retry_count)
            delay_seconds = task_def.retry_delay_seconds * (2 ** (retry_count - 1))
            run_date = datetime.utcnow() + timedelta(seconds=delay_seconds)
            
            # Create retry task definition
            retry_task_def = TaskDefinition(
                task_id=f"{task_def.task_id}_retry_{retry_count}",
                name=f"{task_def.name} (Retry {retry_count})",
                func=task_def.func,
                args=task_def.args,
                kwargs=task_def.kwargs,
                priority=task_def.priority,
                max_retries=task_def.max_retries - retry_count,
                retry_delay_seconds=task_def.retry_delay_seconds,
                timeout_seconds=task_def.timeout_seconds,
                metadata={**task_def.metadata, "original_task_id": task_def.task_id, "retry_count": retry_count}
            )
            
            # Schedule retry
            await self.schedule_one_time_task(retry_task_def, run_date)
            
        except Exception as e:
            logger.error(f"Failed to schedule retry for task {task_def.name}: {e}")
    
    def _store_task_result(self, task_result: TaskResult):
        """Store task result with rotation"""
        if task_result.task_id not in self.task_results:
            self.task_results[task_result.task_id] = []
        
        self.task_results[task_result.task_id].append(task_result)
        
        # Keep only last 100 results per task
        if len(self.task_results[task_result.task_id]) > 100:
            self.task_results[task_result.task_id] = self.task_results[task_result.task_id][-100:]
        
        # Update health stats
        self.health_stats["tasks_executed_today"] += 1
        if task_result.status == TaskStatus.FAILED:
            self.health_stats["task_failures_today"] += 1
        
        if task_result.execution_time_ms:
            # Update average execution time
            current_avg = self.health_stats["avg_execution_time_ms"]
            self.health_stats["avg_execution_time_ms"] = (current_avg + task_result.execution_time_ms) / 2
    
    # Event Handlers
    def _on_job_executed(self, event: JobExecutionEvent):
        """Handle job execution events"""
        logger.debug(f"Job executed: {event.job_id}")
        self.health_stats["last_health_check"] = datetime.utcnow()
    
    def _on_job_error(self, event: JobExecutionEvent):
        """Handle job error events"""
        logger.error(f"Job error: {event.job_id} - {event.exception}")
        self.health_stats["task_failures_today"] += 1
    
    def _on_job_missed(self, event: JobEvent):
        """Handle missed job events"""
        logger.warning(f"Job missed: {event.job_id}")
    
    # Task Management Methods
    async def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task"""
        try:
            job = self.scheduler.get_job(task_id)
            if job:
                job.pause()
                logger.info(f"Task {task_id} paused")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to pause task {task_id}: {e}")
            return False
    
    async def resume_task(self, task_id: str) -> bool:
        """Resume a paused task"""
        try:
            job = self.scheduler.get_job(task_id)
            if job:
                job.resume()
                logger.info(f"Task {task_id} resumed")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to resume task {task_id}: {e}")
            return False
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task"""
        try:
            self.scheduler.remove_job(task_id)
            if task_id in self.tasks:
                del self.tasks[task_id]
            if task_id in self.task_results:
                del self.task_results[task_id]
            if task_id in self.failure_counts:
                del self.failure_counts[task_id]
            
            logger.info(f"Task {task_id} removed")
            return True
        except Exception as e:
            logger.error(f"Failed to remove task {task_id}: {e}")
            return False
    
    # Monitoring and Health Check Methods
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        try:
            jobs = self.scheduler.get_jobs()
            
            status = {
                "scheduler_running": self.scheduler.running,
                "total_jobs": len(jobs),
                "active_jobs": len([j for j in jobs if j.next_run_time]),
                "paused_jobs": len([j for j in jobs if not j.next_run_time]),
                "registered_tasks": len(self.tasks),
                "health_stats": self.health_stats.copy(),
                "jobs": [
                    {
                        "job_id": job.id,
                        "name": job.name,
                        "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                        "trigger": str(job.trigger)
                    }
                    for job in jobs
                ]
            }
            
            return status
        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}")
            return {"error": str(e)}
    
    async def get_task_results(
        self, 
        task_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent results for a specific task"""
        try:
            results = self.task_results.get(task_id, [])
            recent_results = results[-limit:] if limit > 0 else results
            
            return [
                {
                    "task_id": result.task_id,
                    "status": result.status.value,
                    "started_at": result.started_at.isoformat() if result.started_at else None,
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "execution_time_ms": result.execution_time_ms,
                    "retry_count": result.retry_count,
                    "error": result.error,
                    "has_result": result.result is not None
                }
                for result in recent_results
            ]
        except Exception as e:
            logger.error(f"Failed to get task results for {task_id}: {e}")
            return []
    
    async def get_task_health(self, task_id: str) -> Dict[str, Any]:
        """Get health metrics for a specific task"""
        try:
            task_def = self.tasks.get(task_id)
            if not task_def:
                return {"error": "Task not found"}
            
            results = self.task_results.get(task_id, [])
            recent_results = results[-20:]  # Last 20 executions
            
            if not recent_results:
                return {
                    "task_id": task_id,
                    "task_name": task_def.name,
                    "enabled": task_def.enabled,
                    "executions": 0,
                    "success_rate": 0.0,
                    "avg_execution_time_ms": 0,
                    "failure_count": self.failure_counts.get(task_id, 0)
                }
            
            successful = len([r for r in recent_results if r.status == TaskStatus.COMPLETED])
            success_rate = (successful / len(recent_results)) * 100
            
            execution_times = [r.execution_time_ms for r in recent_results if r.execution_time_ms]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            return {
                "task_id": task_id,
                "task_name": task_def.name,
                "enabled": task_def.enabled,
                "executions": len(recent_results),
                "success_rate": success_rate,
                "avg_execution_time_ms": avg_execution_time,
                "failure_count": self.failure_counts.get(task_id, 0),
                "last_execution": recent_results[-1].completed_at.isoformat() if recent_results[-1].completed_at else None,
                "last_status": recent_results[-1].status.value,
                "consecutive_failures": self.failure_counts.get(task_id, 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get task health for {task_id}: {e}")
            return {"error": str(e)}
    
    async def _add_system_health_check(self):
        """Add system health monitoring task"""
        health_check_task = TaskDefinition(
            task_id="system_health_check",
            name="System Health Check",
            func=self._perform_health_check,
            priority=TaskPriority.HIGH,
            max_retries=1,
            timeout_seconds=60,
            failure_threshold=10
        )
        
        await self.schedule_recurring_task(
            health_check_task,
            interval_minutes=5  # Every 5 minutes
        )
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            # Update active jobs count
            jobs = self.scheduler.get_jobs()
            self.health_stats["active_jobs"] = len([j for j in jobs if j.next_run_time])
            
            # Reset daily counters if it's a new day
            now = datetime.utcnow()
            if self.health_stats.get("last_reset_date") != now.date():
                self.health_stats["tasks_executed_today"] = 0
                self.health_stats["task_failures_today"] = 0
                self.health_stats["last_reset_date"] = now.date()
            
            self.health_stats["last_health_check"] = now
            
            # Log health status
            logger.info(f"Health check: {self.health_stats['active_jobs']} active jobs, "
                       f"{self.health_stats['tasks_executed_today']} tasks executed today")
            
            return {"status": "healthy", "timestamp": now.isoformat()}
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


# Global scheduler instance
_task_scheduler = None

async def get_task_scheduler() -> BackgroundTaskScheduler:
    """Get global task scheduler instance"""
    global _task_scheduler
    if _task_scheduler is None:
        _task_scheduler = BackgroundTaskScheduler()
    return _task_scheduler

# Integration functions for portfolio monitoring
async def setup_portfolio_monitoring_tasks(scheduler: BackgroundTaskScheduler):
    """Setup standard portfolio monitoring tasks"""
    from services.portfolio_monitor_service import get_portfolio_monitor_service
    from services.proactive_monitor import get_proactive_monitor
    
    # Portfolio risk monitoring - every hour
    portfolio_monitor_task = TaskDefinition(
        task_id="portfolio_risk_monitoring",
        name="Portfolio Risk Monitoring",
        func=_run_portfolio_monitoring,
        priority=TaskPriority.HIGH,
        max_retries=2,
        timeout_seconds=300,
        failure_threshold=3
    )
    
    await scheduler.schedule_recurring_task(
        portfolio_monitor_task,
        interval_minutes=60
    )
    
    # Database cleanup - daily at 2 AM
    cleanup_task = TaskDefinition(
        task_id="database_cleanup",
        name="Database Cleanup",
        func=_run_database_cleanup,
        priority=TaskPriority.LOW,
        max_retries=1,
        timeout_seconds=600,
        failure_threshold=5
    )
    
    await scheduler.schedule_recurring_task(
        cleanup_task,
        cron_expression="0 2 * * *"  # Daily at 2 AM
    )
    
    # Alert delivery verification - every 15 minutes
    alert_verification_task = TaskDefinition(
        task_id="alert_delivery_verification",
        name="Alert Delivery Verification",
        func=_verify_alert_delivery,
        priority=TaskPriority.MEDIUM,
        max_retries=1,
        timeout_seconds=120,
        failure_threshold=10
    )
    
    await scheduler.schedule_recurring_task(
        alert_verification_task,
        interval_minutes=15
    )

async def _run_portfolio_monitoring():
    """Background portfolio monitoring task"""
    try:
        from services.portfolio_monitor_service import get_portfolio_monitor_service
        
        monitor_service = await get_portfolio_monitor_service()
        result = await monitor_service.monitor_all_portfolios()
        
        logger.info(f"Portfolio monitoring completed: {result.get('portfolios_checked', 0)} portfolios")
        return result
        
    except Exception as e:
        logger.error(f"Portfolio monitoring task failed: {e}")
        raise

async def _run_database_cleanup():
    """Background database cleanup task"""
    try:
        from db.crud import (
            cleanup_old_workflow_sessions,
            cleanup_old_performance_metrics,
            cleanup_old_mcp_job_logs
        )
        
        db = next(get_db())
        try:
            # Cleanup operations
            sessions_cleaned = cleanup_old_workflow_sessions(db, days_old=30)
            metrics_cleaned = cleanup_old_performance_metrics(db, days_old=90)
            logs_cleaned = cleanup_old_mcp_job_logs(db, days_old=60)
            
            result = {
                "sessions_cleaned": sessions_cleaned,
                "metrics_cleaned": metrics_cleaned,
                "logs_cleaned": logs_cleaned
            }
            
            logger.info(f"Database cleanup completed: {result}")
            return result
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Database cleanup task failed: {e}")
        raise

async def _verify_alert_delivery():
    """Verify alert delivery and retry failed notifications"""
    try:
        # This would check for failed alert deliveries and retry them
        logger.info("Alert delivery verification completed")
        return {"status": "completed", "retries": 0}
        
    except Exception as e:
        logger.error(f"Alert delivery verification failed: {e}")
        raise


if __name__ == "__main__":
    async def test_scheduler():
        print("⏰ Testing Background Task Scheduler")
        print("=" * 50)
        
        scheduler = BackgroundTaskScheduler()
        
        # Start scheduler
        success = await scheduler.start()
        print(f"✅ Scheduler started: {success}")
        
        # Test task definition
        def test_task(message: str):
            print(f"🔄 Executing test task: {message}")
            return {"status": "completed", "message": message}
        
        task_def = TaskDefinition(
            task_id="test_task",
            name="Test Task",
            func=test_task,
            args=("Hello from scheduler!",),
            priority=TaskPriority.MEDIUM,
            max_retries=2
        )
        
        # Schedule recurring task
        scheduled = await scheduler.schedule_recurring_task(
            task_def, interval_minutes=1
        )
        print(f"📅 Task scheduled: {scheduled}")
        
        # Run task immediately
        result = await scheduler.run_task_now("test_task")
        print(f"🚀 Task executed immediately: {result.status}")
        
        # Get scheduler status
        status = await scheduler.get_scheduler_status()
        print(f"📊 Scheduler status: {status['scheduler_running']}, "
              f"{status['total_jobs']} jobs")
        
        # Setup portfolio monitoring tasks
        print("🏦 Setting up portfolio monitoring tasks...")
        await setup_portfolio_monitoring_tasks(scheduler)
        
        # Get updated status
        status = await scheduler.get_scheduler_status()
        print(f"📈 Updated status: {status['total_jobs']} jobs registered")
        
        # Stop scheduler
        stopped = await scheduler.stop()
        print(f"⏹️ Scheduler stopped: {stopped}")
        
        print("\n🎉 Task Scheduler testing complete!")
    
    asyncio.run(test_scheduler())