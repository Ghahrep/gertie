# mcp/workflow_engine.py
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from dataclasses import dataclass, field

from .schemas import JobRequest, JobStatus

logger = logging.getLogger(__name__)

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    step_id: str
    agent_id: str
    capability: str
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: JobStatus = JobStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class Job:
    """Represents a complete job with multiple workflow steps"""
    job_id: str
    request: JobRequest
    status: JobStatus = JobStatus.PENDING
    status_message: str = "Job created"
    result: Optional[Dict[str, Any]] = None
    progress: float = 0.0
    agents_involved: List[str] = field(default_factory=list)
    workflow_steps: List[WorkflowStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class WorkflowEngine:
    """Core engine for managing and executing multi-step analysis workflows"""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
    def create_job(self, job_id: str, request: JobRequest, assigned_agents: List[str]) -> Job:
        """Create a new job with workflow steps"""
        job = Job(job_id=job_id, request=request, agents_involved=assigned_agents)
        
        # Generate workflow steps based on the request
        workflow_steps = self._generate_workflow_steps(request, assigned_agents)
        job.workflow_steps = workflow_steps
        
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} with {len(workflow_steps)} steps")
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID"""
        return self.jobs.get(job_id)
    
    def get_active_job_count(self) -> int:
        """Get the number of currently active jobs"""
        return len(self.active_jobs)
    
    async def execute_job(self, job_id: str) -> None:
        """Execute a job workflow asynchronously"""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.status_message = "Executing workflow steps"
            
            # Create asyncio task for this job
            task = asyncio.create_task(self._execute_workflow_steps(job))
            self.active_jobs[job_id] = task
            
            # Wait for completion
            await task
            
        except Exception as e:
            logger.error(f"Error executing job {job_id}: {str(e)}")
            job.status = JobStatus.FAILED
            job.status_message = f"Job failed: {str(e)}"
        finally:
            job.completed_at = datetime.utcnow()
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _execute_workflow_steps(self, job: Job) -> None:
        """Execute the workflow steps for a job"""
        total_steps = len(job.workflow_steps)
        completed_steps = 0
        
        # Group steps by dependency level
        step_levels = self._organize_steps_by_dependencies(job.workflow_steps)
        
        for level, steps in step_levels.items():
            logger.info(f"Executing workflow level {level} with {len(steps)} steps")
            
            # Execute all steps at this level in parallel
            tasks = []
            for step in steps:
                task = asyncio.create_task(self._execute_step(step))
                tasks.append(task)
            
            # Wait for all steps at this level to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update progress
            for i, result in enumerate(results):
                step = steps[i]
                if isinstance(result, Exception):
                    logger.error(f"Step {step.step_id} failed: {str(result)}")
                    step.status = JobStatus.FAILED
                    step.error_message = str(result)
                else:
                    step.status = JobStatus.COMPLETED
                    step.result = result
                    completed_steps += 1
                
                step.completed_at = datetime.utcnow()
            
            # Update job progress
            job.progress = (completed_steps / total_steps) * 100
            
            # Check if any critical steps failed
            failed_steps = [s for s in steps if s.status == JobStatus.FAILED]
            if failed_steps and self._has_critical_failures(failed_steps):
                job.status = JobStatus.FAILED
                job.status_message = f"Critical workflow steps failed: {[s.step_id for s in failed_steps]}"
                return
        
        # Consolidate results
        job.result = self._consolidate_workflow_results(job.workflow_steps)
        job.status = JobStatus.COMPLETED
        job.status_message = "Workflow completed successfully"
        logger.info(f"Job {job.job_id} completed successfully")
    
    async def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step.status = JobStatus.RUNNING
        step.started_at = datetime.utcnow()
        
        # Simulate agent communication and processing
        # In a real implementation, this would communicate with actual agents
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Mock result based on capability
        mock_result = self._generate_mock_result(step.capability, step.input_data)
        
        logger.info(f"Step {step.step_id} ({step.capability}) completed")
        return mock_result
    
    def _generate_workflow_steps(self, request: JobRequest, assigned_agents: List[str]) -> List[WorkflowStep]:
        """Generate workflow steps based on the request and available agents"""
        steps = []
        query_lower = request.query.lower()
        
        # Step 1: Data collection (always first)
        if "portfolio" in query_lower or "holdings" in query_lower:
            steps.append(WorkflowStep(
                step_id="data_collection_portfolio",
                agent_id=assigned_agents[0] if assigned_agents else "default",
                capability="portfolio_data_fetch",
                input_data={"query": request.query, "portfolio_id": request.context.get("portfolio_id")}
            ))
        
        if "market" in query_lower or "economic" in query_lower:
            steps.append(WorkflowStep(
                step_id="data_collection_market",
                agent_id=assigned_agents[0] if assigned_agents else "default",
                capability="market_data_fetch",
                input_data={"query": request.query}
            ))
        
        # Step 2: Analysis steps (depend on data collection)
        data_deps = [s.step_id for s in steps]  # All data collection steps
        
        if "risk" in query_lower:
            steps.append(WorkflowStep(
                step_id="risk_analysis",
                agent_id=self._find_agent_with_capability(assigned_agents, "risk_analysis"),
                capability="risk_analysis",
                input_data={"query": request.query},
                dependencies=data_deps
            ))
        
        if "tax" in query_lower:
            steps.append(WorkflowStep(
                step_id="tax_optimization",
                agent_id=self._find_agent_with_capability(assigned_agents, "tax_optimization"),
                capability="tax_optimization",
                input_data={"query": request.query},
                dependencies=data_deps
            ))
        
        if "rebalance" in query_lower:
            steps.append(WorkflowStep(
                step_id="rebalancing_analysis",
                agent_id=self._find_agent_with_capability(assigned_agents, "strategy_rebalancing"),
                capability="strategy_rebalancing",
                input_data={"query": request.query},
                dependencies=data_deps
            ))
        
        # Step 3: Synthesis (depends on all analysis steps)
        analysis_deps = [s.step_id for s in steps if s.step_id not in data_deps]
        if analysis_deps:
            steps.append(WorkflowStep(
                step_id="result_synthesis",
                agent_id=assigned_agents[0] if assigned_agents else "default",
                capability="result_synthesis",
                input_data={"query": request.query},
                dependencies=analysis_deps
            ))
        
        return steps
    
    def _organize_steps_by_dependencies(self, steps: List[WorkflowStep]) -> Dict[int, List[WorkflowStep]]:
        """Organize workflow steps into dependency levels for parallel execution"""
        step_levels: Dict[int, List[WorkflowStep]] = {}
        step_map = {step.step_id: step for step in steps}
        
        def get_step_level(step: WorkflowStep, visited: set) -> int:
            if step.step_id in visited:
                return 0  # Circular dependency, put at level 0
            
            if not step.dependencies:
                return 0
            
            visited.add(step.step_id)
            max_dep_level = -1
            
            for dep_id in step.dependencies:
                if dep_id in step_map:
                    dep_level = get_step_level(step_map[dep_id], visited.copy())
                    max_dep_level = max(max_dep_level, dep_level)
            
            return max_dep_level + 1
        
        # Assign each step to a level
        for step in steps:
            level = get_step_level(step, set())
            if level not in step_levels:
                step_levels[level] = []
            step_levels[level].append(step)
        
        return step_levels
    
    def _find_agent_with_capability(self, agents: List[str], capability: str) -> str:
        """Find the best agent for a specific capability"""
        # In a real implementation, this would check agent registrations
        return agents[0] if agents else "default_agent"
    
    def _has_critical_failures(self, failed_steps: List[WorkflowStep]) -> bool:
        """Determine if failed steps are critical to the workflow"""
        critical_capabilities = ["portfolio_data_fetch", "result_synthesis"]
        return any(step.capability in critical_capabilities for step in failed_steps)
    
    def _consolidate_workflow_results(self, steps: List[WorkflowStep]) -> Dict[str, Any]:
        """Consolidate results from all workflow steps"""
        consolidated = {
            "workflow_summary": {
                "total_steps": len(steps),
                "successful_steps": len([s for s in steps if s.status == JobStatus.COMPLETED]),
                "failed_steps": len([s for s in steps if s.status == JobStatus.FAILED])
            },
            "results": {},
            "recommendations": [],
            "confidence_score": 0.85  # Mock confidence score
        }
        
        for step in steps:
            if step.status == JobStatus.COMPLETED and step.result:
                consolidated["results"][step.capability] = step.result
        
        # Generate summary recommendations based on results
        if "risk_analysis" in consolidated["results"]:
            consolidated["recommendations"].append({
                "type": "risk_management",
                "priority": "high",
                "description": "Consider reducing portfolio volatility by 15%"
            })
        
        if "tax_optimization" in consolidated["results"]:
            consolidated["recommendations"].append({
                "type": "tax_strategy",
                "priority": "medium",
                "description": "Implement tax-loss harvesting to reduce tax burden"
            })
        
        return consolidated
    
    def _generate_mock_result(self, capability: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock results for testing (replace with real agent communication)"""
        base_result = {
            "capability": capability,
            "timestamp": datetime.utcnow().isoformat(),
            "input_query": input_data.get("query", ""),
        }
        
        if capability == "risk_analysis":
            base_result.update({
                "var_95": 0.024,
                "max_drawdown": 0.187,
                "sharpe_ratio": 1.34,
                "risk_grade": "Moderate"
            })
        elif capability == "tax_optimization":
            base_result.update({
                "potential_tax_savings": 3420.50,
                "recommended_harvesting": ["TSLA: -$1,200", "NVDA: -$2,220"],
                "optimal_asset_location": "Move bonds to 401k, growth stocks to Roth"
            })
        elif capability == "strategy_rebalancing":
            base_result.update({
                "rebalancing_needed": True,
                "target_allocations": {"stocks": 0.70, "bonds": 0.25, "cash": 0.05},
                "trades_required": [
                    {"symbol": "VTI", "action": "sell", "shares": 50},
                    {"symbol": "BND", "action": "buy", "shares": 100}
                ]
            })
        
        return base_result
    
    def estimate_completion_time(self, request: JobRequest) -> Optional[str]:
        """Estimate when a job will complete based on complexity"""
        query_complexity = len(request.query.split()) + len(request.context)
        
        if query_complexity < 10:
            estimated_seconds = 30
        elif query_complexity < 25:
            estimated_seconds = 90
        else:
            estimated_seconds = 180
        
        completion_time = datetime.utcnow() + timedelta(seconds=estimated_seconds)
        return completion_time.isoformat()