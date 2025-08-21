"""
Performance benchmark tests
==========================
"""

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from mcp.workflow_engine import WorkflowEngine
from mcp.consensus_builder import ConsensusBuilder
from mcp.schemas import JobRequest

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.performance
    def test_workflow_engine_throughput(self):
        """Test workflow engine throughput under load"""
        engine = WorkflowEngine()
        num_jobs = 100
        
        start_time = time.time()
        
        # Create many jobs rapidly
        job_ids = []
        for i in range(num_jobs):
            job_request = JobRequest(
                query=f"performance test job {i}",
                context={"job_number": i}
            )
            job = engine.create_job(f"perf_job_{i}", job_request, ["test_agent"])
            job_ids.append(job.job_id)
        
        creation_time = time.time() - start_time
        
        # Should create 100 jobs in under 1 second
        assert creation_time < 1.0
        assert len(engine.jobs) == num_jobs
        
        print(f"Created {num_jobs} jobs in {creation_time:.3f} seconds")
        print(f"Throughput: {num_jobs/creation_time:.1f} jobs/second")
    
    @pytest.mark.performance
    def test_consensus_algorithm_scalability(self):
        """Test consensus algorithm performance with many agents"""
        consensus_builder = ConsensusBuilder()
        
        # Generate many agent positions
        num_agents = 200
        positions = []
        
        for i in range(num_agents):
            position = {
                "agent_id": f"agent_{i}",
                "stance": f"position_{i % 5}",  # 5 different stances
                "key_arguments": [f"argument_{i}_1", f"argument_{i}_2"],
                "supporting_evidence": [
                    {"type": "analytical", "confidence": 0.7 + (i % 3) * 0.1}
                ],
                "confidence_score": 0.6 + (i % 4) * 0.1,
                "risk_assessment": {"primary_risks": [f"risk_{i % 10}"]}
            }
            positions.append(position)
        
        start_time = time.time()
        
        consensus = consensus_builder.calculate_weighted_consensus(
            positions, {"query": "large scale consensus test"}
        )
        
        computation_time = time.time() - start_time
        
        # Should handle 200 agents in under 5 seconds
        assert computation_time < 5.0
        assert "recommendation" in consensus
        assert consensus["confidence_level"] > 0
        
        print(f"Consensus with {num_agents} agents computed in {computation_time:.3f} seconds")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """Test concurrent workflow execution performance"""
        engine = WorkflowEngine()
        num_concurrent = 20
        
        async def execute_mock_workflow(job_id):
            job_request = JobRequest(
                query=f"concurrent test {job_id}",
                context={"concurrent_job": True}
            )
            
            job = engine.create_job(job_id, job_request, ["test_agent"])
            
            # Mock workflow execution
            await asyncio.sleep(0.1)  # Simulate work
            job.status = "completed"
            return job
        
        start_time = time.time()
        
        # Execute workflows concurrently
        tasks = [
            execute_mock_workflow(f"concurrent_{i}")
            for i in range(num_concurrent)
        ]
        
        results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        # Should execute 20 concurrent workflows quickly
        assert execution_time < 2.0  # Should benefit from concurrency
        assert len(results) == num_concurrent
        assert all(job.status == "completed" for job in results)
        
        print(f"Executed {num_concurrent} concurrent workflows in {execution_time:.3f} seconds")
    
    @pytest.mark.performance
    def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = WorkflowEngine()
        consensus_builder = ConsensusBuilder()
        
        # Simulate heavy usage
        for cycle in range(10):
            # Create and process jobs
            for i in range(20):
                job_request = JobRequest(
                    query=f"memory test cycle {cycle} job {i}",
                    context={"large_data": "x" * 1000}  # Some data
                )
                job = engine.create_job(f"mem_test_{cycle}_{i}", job_request, ["test_agent"])
            
            # Build consensus with many positions
            positions = [
                {
                    "agent_id": f"agent_{i}",
                    "stance": "test_stance",
                    "key_arguments": ["arg1", "arg2"],
                    "supporting_evidence": [{"type": "test", "confidence": 0.8}],
                    "confidence_score": 0.8,
                    "risk_assessment": {"primary_risks": ["test_risk"]}
                }
                for i in range(50)
            ]
            consensus_builder.calculate_weighted_consensus(positions, {"query": "memory test"})
            
            # Clean up jobs periodically (simulate garbage collection)
            if cycle % 5 == 0:
                engine.jobs.clear()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (< 100MB for this test)
        assert memory_growth < 100
        
        print(f"Memory growth: {memory_growth:.1f} MB")
        print(f"Initial: {initial_memory:.1f} MB, Final: {final_memory:.1f} MB")