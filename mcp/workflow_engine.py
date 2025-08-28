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
    
    def __init__(self, agent_registry=None, app_state=None):
        self.jobs: Dict[str, Job] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.agent_registry = agent_registry  # Reference to enhanced_registry
        self.app_state = app_state  # Reference to FastAPI app.state for MCP agents
        
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
                task = asyncio.create_task(self._execute_step(step, job.request.context))
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
    
    async def _execute_step(self, step: WorkflowStep, job_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step with real agent calls"""
        step.status = JobStatus.RUNNING
        step.started_at = datetime.utcnow()
        
        try:
            # Check if this is an MCP agent (stored in app.state)
            if self.app_state and hasattr(self.app_state, 'agent_instances') and step.agent_id in self.app_state.agent_instances:
                return await self._execute_mcp_agent_step(step, job_context)
            
            # Check if this is a registry agent (HTTP endpoint)
            elif self.agent_registry and step.agent_id in self.agent_registry.agents:
                return await self._execute_registry_agent_step(step, job_context)
            
            # Fallback to mock for unknown agents
            else:
                logger.warning(f"Unknown agent {step.agent_id}, using mock result")
                await asyncio.sleep(0.1)  # Brief delay
                return self._generate_mock_result(step.capability, step.input_data)
        
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {str(e)}")
            raise e
    
    async def _execute_mcp_agent_step(self, step: WorkflowStep, job_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step using an MCP agent instance"""
        agent = self.app_state.agent_instances[step.agent_id]
        
        # Prepare data and context for the agent
        data = {
            "query": step.input_data.get("query", ""),
            **step.input_data
        }
        context = {
            **job_context,
            "workflow_step_id": step.step_id
        }
        
        logger.info(f"Executing MCP agent {step.agent_id} capability {step.capability}")
        
        # Execute the capability
        result = await agent.execute_capability(step.capability, data, context)
        
        logger.info(f"MCP agent {step.agent_id} completed capability {step.capability}")
        return result
    
    async def _execute_registry_agent_step(self, step: WorkflowStep, job_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step using a registry agent (HTTP endpoint)"""
        import httpx
        
        agent_info = self.agent_registry.agents[step.agent_id]
        endpoint_url = agent_info.endpoint_url
        
        if not endpoint_url:
            logger.warning(f"No endpoint URL for agent {step.agent_id}, using mock result")
            return self._generate_mock_result(step.capability, step.input_data)
        
        # Prepare request payload
        payload = {
            "capability": step.capability,
            "data": step.input_data,
            "context": job_context
        }
        
        logger.info(f"Calling registry agent {step.agent_id} at {endpoint_url}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint_url,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Agent {step.agent_id} returned {response.status_code}")
                    return self._generate_mock_result(step.capability, step.input_data)
        
        except Exception as e:
            logger.error(f"Error calling agent {step.agent_id}: {str(e)}")
            return self._generate_mock_result(step.capability, step.input_data)
    
    def _generate_workflow_steps(self, request: JobRequest, assigned_agents: List[str]) -> List[WorkflowStep]:
        """Generate workflow steps based on the request and available agents"""
        steps = []
        query_lower = request.query.lower()
        
        # NEW: Enhanced step generation for SecurityScreener capabilities
        if any(pattern in query_lower for pattern in ["screen", "stocks", "quality", "value", "factor"]):
            # Direct screening step - no dependencies needed
            security_screener = self._find_agent_with_capability(assigned_agents, "security_screening")
            if security_screener:
                steps.append(WorkflowStep(
                    step_id="security_screening",
                    agent_id=security_screener,
                    capability="security_screening",
                    input_data={"query": request.query}
                ))
        
        # Factor analysis step
        if "factor" in query_lower or "quality" in query_lower or "value" in query_lower:
            factor_agent = self._find_agent_with_capability(assigned_agents, "factor_analysis")
            if factor_agent:
                steps.append(WorkflowStep(
                    step_id="factor_analysis",
                    agent_id=factor_agent,
                    capability="factor_analysis",
                    input_data={"query": request.query}
                ))
        
        # Portfolio complement analysis
        if "complement" in query_lower and ("portfolio" in query_lower or request.context.get("portfolio_id")):
            complement_agent = self._find_agent_with_capability(assigned_agents, "portfolio_complement_analysis")
            if complement_agent:
                steps.append(WorkflowStep(
                    step_id="portfolio_complement",
                    agent_id=complement_agent,
                    capability="portfolio_complement_analysis",
                    input_data={"query": request.query}
                ))
        
        # Legacy workflow steps (keep existing logic)
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
        
        # If no specific steps generated, create a default analysis step
        if not steps:
            default_agent = assigned_agents[0] if assigned_agents else "default"
            steps.append(WorkflowStep(
                step_id="general_analysis",
                agent_id=default_agent,
                capability="general_analysis",
                input_data={"query": request.query}
            ))
        
        logger.info(f"Generated {len(steps)} workflow steps: {[s.step_id for s in steps]}")
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
    
    def _find_agent_with_capability(self, agents: List[str], capability: str) -> Optional[str]:
        """Find the best agent for a specific capability"""
        # Check MCP agents first
        if self.app_state and hasattr(self.app_state, 'agent_instances'):
            for agent_id in agents:
                if agent_id in self.app_state.agent_instances:
                    agent = self.app_state.agent_instances[agent_id]
                    if hasattr(agent, 'capabilities') and capability in agent.capabilities:
                        return agent_id
        
        # Check registry agents
        if self.agent_registry:
            for agent_id in agents:
                if agent_id in self.agent_registry.agents:
                    agent_info = self.agent_registry.agents[agent_id]
                    if capability in agent_info.capabilities:
                        return agent_id
        
        # Return first agent as fallback
        return agents[0] if agents else None
    
    def _has_critical_failures(self, failed_steps: List[WorkflowStep]) -> bool:
        """Determine if failed steps are critical to the workflow"""
        critical_capabilities = ["portfolio_data_fetch", "result_synthesis", "security_screening"]
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
            "confidence_score": 0.85
        }
        
        for step in steps:
            if step.status == JobStatus.COMPLETED and step.result:
                consolidated["results"][step.capability] = step.result
                
                # Extract recommendations from MCP agent results
                if isinstance(step.result, dict):
                    if "recommendations" in step.result:
                        consolidated["recommendations"].extend(step.result["recommendations"])
                    elif "screening_results" in step.result:
                        screening_results = step.result["screening_results"]
                        if isinstance(screening_results, dict) and "recommendations" in screening_results:
                            consolidated["recommendations"].extend(screening_results["recommendations"])
        
        return consolidated
    
    def _generate_mock_result(self, capability: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock results for testing (replace with real agent communication)"""
        base_result = {
            "capability": capability,
            "timestamp": datetime.utcnow().isoformat(),
            "input_query": input_data.get("query", ""),
        }
        
        if capability == "security_screening":
            base_result.update({
                "screening_type": "factor_based",
                "universe_screened": 500,
                "recommendations": [
                    {"ticker": "AAPL", "score": 0.87, "rationale": "High quality, reasonable valuation"},
                    {"ticker": "MSFT", "score": 0.84, "rationale": "Strong fundamentals, growth potential"},
                    {"ticker": "GOOGL", "score": 0.82, "rationale": "Quality business, attractive value metrics"}
                ]
            })
        elif capability == "factor_analysis":
            base_result.update({
                "factors_analyzed": ["quality", "value", "growth"],
                "top_securities": [
                    {"ticker": "JPM", "quality_score": 0.91, "value_score": 0.78},
                    {"ticker": "BRK.B", "quality_score": 0.89, "value_score": 0.85}
                ]
            })
        elif capability == "risk_analysis":
            base_result.update({
                "var_95": 0.024,
                "max_drawdown": 0.187,
                "sharpe_ratio": 1.34,
                "risk_grade": "Moderate"
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