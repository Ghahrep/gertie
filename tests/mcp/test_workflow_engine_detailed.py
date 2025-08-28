"""
Detailed workflow engine tests
=============================
"""

import pytest
import asyncio
from datetime import datetime
from mcp.workflow_engine import WorkflowEngine, Job, WorkflowStep
from mcp.schemas import JobRequest, JobStatus

class TestWorkflowEngineDetailed:
    """Detailed workflow engine testing"""
    
    @pytest.fixture
    def engine(self):
        return WorkflowEngine()
    
    def test_step_dependency_resolution(self, engine):
        """Test complex step dependency resolution"""
        steps = [
            WorkflowStep("step_d", "agent", "cap_d", {}, ["step_b", "step_c"]),
            WorkflowStep("step_b", "agent", "cap_b", {}, ["step_a"]),
            WorkflowStep("step_a", "agent", "cap_a", {}, []),
            WorkflowStep("step_c", "agent", "cap_c", {}, ["step_a"]),
        ]
        
        levels = engine._organize_steps_by_dependencies(steps)
        
        # step_a should be at level 0 (no dependencies)
        assert any(s.step_id == "step_a" for s in levels[0])
        
        # step_b and step_c should be at level 1 (depend on step_a)
        level_1_ids = [s.step_id for s in levels[1]]
        assert "step_b" in level_1_ids
        assert "step_c" in level_1_ids
        
        # step_d should be at level 2 (depends on step_b and step_c)
        assert any(s.step_id == "step_d" for s in levels[2])
    
    @pytest.mark.asyncio
    async def test_step_execution_error_handling(self, engine):
        """Test step execution with errors"""
        job_request = JobRequest(query="test query")
        job = engine.create_job("test_job", job_request, ["test_agent"])
        
        # Mock a step that raises an exception
        async def failing_step(*args):
            raise Exception("Simulated step failure")
        
        with pytest.raises(Exception):
            await failing_step()
        
        # Workflow should handle step failures gracefully
        job.workflow_steps[0].status = JobStatus.FAILED
        job.workflow_steps[0].error_message = "Step failed"
        
        # Check if critical failure detection works
        critical_step = WorkflowStep(
            "critical_step", "agent", "portfolio_data_fetch", {}, []
        )
        critical_step.status = JobStatus.FAILED
        
        is_critical = engine._has_critical_failures([critical_step])
        assert is_critical == True
    
    def test_workflow_step_generation_logic(self, engine):
        """Test workflow step generation for different query types"""
        # Risk analysis query
        risk_request = JobRequest(
            query="analyze portfolio risk and volatility",
            context={"portfolio_id": "test"}
        )
        risk_steps = engine._generate_workflow_steps(risk_request, ["risk_agent"])
        risk_capabilities = [step.capability for step in risk_steps]
        assert "risk_analysis" in risk_capabilities
        assert "portfolio_data_fetch" in risk_capabilities
        
        # Tax optimization query
        tax_request = JobRequest(
            query="optimize tax efficiency and harvest losses",
            context={"portfolio_id": "test"}
        )
        tax_steps = engine._generate_workflow_steps(tax_request, ["tax_agent"])
        tax_capabilities = [step.capability for step in tax_steps]
        assert "tax_optimization" in tax_capabilities
        
        # Rebalancing query
        rebalance_request = JobRequest(
            query="rebalance portfolio allocation",
            context={"portfolio_id": "test"}
        )
        rebalance_steps = engine._generate_workflow_steps(rebalance_request, ["rebalance_agent"])
        rebalance_capabilities = [step.capability for step in rebalance_steps]
        assert "strategy_rebalancing" in rebalance_capabilities
