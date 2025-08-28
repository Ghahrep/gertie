# test_components.py
"""
Test individual MCP components without running servers.
This tests the core logic of workflows and schemas.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_schemas():
    """Test MCP schemas validation"""
    print("🔍 Testing MCP Schemas...")
    
    try:
        from mcp.schemas import (
            AgentRegistration, 
            JobRequest, 
            JobResponse, 
            JobStatus,
            HealthCheck
        )
        
        # Test AgentRegistration
        agent_reg = AgentRegistration(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type="TestAgent", 
            capabilities=["test_capability"]
        )
        print(f"  ✅ AgentRegistration schema works")
        
        # Test JobRequest
        job_req = JobRequest(
            query="Test analysis query",
            context={"test": True}
        )
        print(f"  ✅ JobRequest schema works")
        
        # Test JobResponse
        job_resp = JobResponse(
            job_id="test_job_123",
            status=JobStatus.PENDING,
            message="Job created"
        )
        print(f"  ✅ JobResponse schema works")
        
        # Test HealthCheck
        health = HealthCheck(
            status="healthy",
            timestamp=datetime.utcnow(),
            registered_agents=0,
            active_jobs=0
        )
        print(f"  ✅ HealthCheck schema works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Schema test failed: {e}")
        return False

async def test_workflow_engine():
    """Test workflow engine logic"""
    print("\n🔍 Testing Workflow Engine...")
    
    try:
        from mcp.workflow_engine import WorkflowEngine
        from mcp.schemas import JobRequest
        
        # Create workflow engine
        engine = WorkflowEngine()
        print(f"  ✅ WorkflowEngine created")
        
        # Create test job
        job_request = JobRequest(
            query="Test portfolio risk analysis",
            context={"portfolio_id": "test_123"}
        )
        
        job = engine.create_job(
            job_id="test_job_456",
            request=job_request,
            assigned_agents=["test_agent"]
        )
        print(f"  ✅ Job created with {len(job.workflow_steps)} workflow steps")
        
        # Test job retrieval
        retrieved_job = engine.get_job("test_job_456")
        if retrieved_job and retrieved_job.job_id == "test_job_456":
            print(f"  ✅ Job retrieval works")
        else:
            raise Exception("Job retrieval failed")
        
        # Test estimation
        estimated_time = engine.estimate_completion_time(job_request)
        if estimated_time:
            print(f"  ✅ Completion time estimation works: {estimated_time}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Workflow engine test failed: {e}")
        return False

async def test_mcp_base_agent():
    """Test base agent functionality"""
    print("\n🔍 Testing MCP Base Agent...")
    
    try:
        from agents.minimal_test_agent import MinimalTestAgent
        from mcp.schemas import AgentJobRequest
        
        # Create minimal test agent
        agent = MinimalTestAgent()
        print(f"  ✅ Minimal test agent created")
        
        # Test capability execution
        result = await agent.execute_capability(
            "test_analysis", 
            {"query": "test query"}, 
            {"test_context": True}
        )
        if result.get("analysis_type") == "test_analysis":
            print(f"  ✅ Capability execution works")
        
        # Test autonomous data fetch
        enriched_data = await agent.autonomous_data_fetch(
            "portfolio analysis query",
            {"portfolio_id": "test_portfolio"}
        )
        print(f"  ✅ Autonomous data fetch works: {len(enriched_data)} data sources")
        
        # Test job request handling
        job_request = AgentJobRequest(
            job_id="test_job",
            step_id="test_step",
            capability="test_analysis",
            input_data={"query": "test query"}
        )
        
        job_response = await agent.handle_job_request(job_request)
        if job_response.step_id == "test_step":
            print(f"  ✅ Job request handling works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MCP base agent test failed: {e}")
        return False

async def test_mcp_client():
    """Test MCP client functionality"""
    print("\n🔍 Testing MCP Client...")
    
    try:
        from services.mcp_client import MCPClient, TemporaryMCPClient
        
        # Test client creation
        client = MCPClient(base_url="http://localhost:8001")
        print(f"  ✅ MCP Client created")
        
        # Test temporary client context manager
        async with TemporaryMCPClient() as temp_client:
            print(f"  ✅ Temporary MCP Client context manager works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MCP client test failed: {e}")
        return False

async def main():
    """Run all component tests"""
    print("🧪 MCP Component Tests")
    print("Testing individual components without server dependencies")
    print("=" * 60)
    
    test_results = []
    
    # Run individual component tests
    test_results.append(await test_schemas())
    test_results.append(await test_workflow_engine())
    test_results.append(await test_mcp_base_agent())
    test_results.append(await test_mcp_client())
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 COMPONENT TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print(f"\n🎉 All component tests passed!")
        print(f"💡 Next steps:")
        print(f"  1. Run: python start_mcp_server.py")
        print(f"  2. In another terminal: python test_mcp_foundation.py")
    else:
        print(f"\n⚠️ Some component tests failed.")
        print(f"🔧 Please fix the issues before running the full MCP server test.")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())