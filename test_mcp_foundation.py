# test_mcp_foundation.py
"""
Test script for MCP Foundation (Tasks 1.1 - 1.3)
Tests the MCP server, workflow engine, agent registration, and API integration.
"""

import asyncio
import aiohttp
import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
MCP_BASE_URL = "http://localhost:8001"
API_BASE_URL = "http://localhost:8000"  # Your main API server

class MCPFoundationTester:
    """Test suite for MCP foundation components"""
    
    def __init__(self):
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def run_all_tests(self):
        """Run all MCP foundation tests"""
        print("🚀 Starting MCP Foundation Tests")
        print("=" * 50)
        
        # Test 1.1: MCP Server Foundation
        await self.test_mcp_server_health()
        await self.test_mcp_endpoints()
        
        # Test 1.2: Agent Registration (Mock)
        await self.test_agent_registration()
        
        # Test 1.3: API Integration
        await self.test_api_integration()
        
        # Test Workflow Engine
        await self.test_workflow_execution()
        
        # Print test summary
        self.print_test_summary()
        
    async def test_mcp_server_health(self):
        """Test MCP server health endpoint"""
        test_name = "MCP Server Health Check"
        try:
            print(f"\n🔍 Testing: {test_name}")
            
            async with self.session.get(f"{MCP_BASE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"  ✅ Health check passed")
                    print(f"  📊 Status: {health_data.get('status')}")
                    print(f"  🤖 Registered agents: {health_data.get('registered_agents', 0)}")
                    print(f"  ⚡ Active jobs: {health_data.get('active_jobs', 0)}")
                    
                    self.test_results.append({
                        "test": test_name,
                        "status": "PASS",
                        "details": health_data
                    })
                else:
                    raise Exception(f"Health check failed with status {response.status}")
                    
        except Exception as e:
            print(f"  ❌ {test_name} failed: {str(e)}")
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
    
    async def test_mcp_endpoints(self):
        """Test MCP server endpoints"""
        test_name = "MCP Server Endpoints"
        try:
            print(f"\n🔍 Testing: {test_name}")
            
            # Test agents list endpoint
            async with self.session.get(f"{MCP_BASE_URL}/agents") as response:
                if response.status == 200:
                    agents = await response.json()
                    print(f"  ✅ Agents endpoint working")
                    print(f"  📋 Current agents: {len(agents)}")
                else:
                    raise Exception(f"Agents endpoint failed with status {response.status}")
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS",
                "details": {"agents_count": len(agents)}
            })
            
        except Exception as e:
            print(f"  ❌ {test_name} failed: {str(e)}")
            self.test_results.append({
                "test": test_name,
                "status": "FAIL", 
                "error": str(e)
            })
    
    async def test_agent_registration(self):
        """Test agent registration with MCP"""
        test_name = "Agent Registration"
        try:
            print(f"\n🔍 Testing: {test_name}")
            
            # Mock agent registration data
            test_agent = {
                "agent_id": "test_agent_001",
                "agent_name": "Test Agent",
                "agent_type": "TestAgent",
                "capabilities": ["test_capability", "mock_analysis"],
                "endpoint_url": "http://localhost:8002/agent",
                "max_concurrent_jobs": 3,
                "response_time_sla": 30,
                "metadata": {
                    "version": "1.0.0",
                    "test_mode": True
                }
            }
            
            # Register the test agent
            async with self.session.post(
                f"{MCP_BASE_URL}/register",
                json=test_agent
            ) as response:
                if response.status == 200:
                    registration_result = await response.json()
                    print(f"  ✅ Agent registration successful")
                    print(f"  🤖 Registered: {test_agent['agent_id']}")
                    print(f"  📝 Message: {registration_result.get('message')}")
                    
                    # Verify agent appears in agents list
                    async with self.session.get(f"{MCP_BASE_URL}/agents") as list_response:
                        if list_response.status == 200:
                            agents = await list_response.json()
                            agent_ids = [agent.get('agent_id') for agent in agents]
                            
                            if test_agent['agent_id'] in agent_ids:
                                print(f"  ✅ Agent found in registry")
                            else:
                                raise Exception("Agent not found in registry after registration")
                        else:
                            raise Exception("Failed to verify agent registration")
                    
                    # Clean up - unregister test agent
                    async with self.session.delete(
                        f"{MCP_BASE_URL}/agents/{test_agent['agent_id']}"
                    ) as delete_response:
                        if delete_response.status == 200:
                            print(f"  🧹 Test agent cleaned up")
                        
                    self.test_results.append({
                        "test": test_name,
                        "status": "PASS",
                        "details": registration_result
                    })
                else:
                    error_text = await response.text()
                    raise Exception(f"Registration failed: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"  ❌ {test_name} failed: {str(e)}")
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
    
    async def test_workflow_execution(self):
        """Test workflow engine job creation and execution"""
        test_name = "Workflow Engine"
        try:
            print(f"\n🔍 Testing: {test_name}")
            
            # First, register a test agent that can handle our test job
            test_agent = {
                "agent_id": "workflow_test_agent",
                "agent_name": "Workflow Test Agent",
                "agent_type": "TestAgent",
                "capabilities": ["test_analysis", "portfolio_analysis"],
                "endpoint_url": "http://localhost:8002/agent",
                "max_concurrent_jobs": 2,
                "response_time_sla": 30,
                "metadata": {"test_mode": True}
            }
            
            # Register the test agent
            async with self.session.post(
                f"{MCP_BASE_URL}/register",
                json=test_agent
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to register test agent: {response.status} - {error_text}")
                
                print(f"  ✅ Test agent registered for workflow testing")
            
            # Create a test job request that the agent can handle
            job_request = {
                "query": "Test portfolio analysis",
                "context": {
                    "portfolio_id": "test_portfolio_123",
                    "test_mode": True
                },
                "priority": 5,
                "timeout_seconds": 60,
                "required_capabilities": ["test_analysis"]
            }
            
            # Submit job to MCP
            async with self.session.post(
                f"{MCP_BASE_URL}/submit_job",
                json=job_request
            ) as response:
                if response.status == 200:
                    job_response = await response.json()
                    job_id = job_response.get('job_id')
                    
                    print(f"  ✅ Job submitted successfully")
                    print(f"  🆔 Job ID: {job_id}")
                    print(f"  📊 Status: {job_response.get('status')}")
                    
                    # Wait a bit and check job status
                    await asyncio.sleep(2)  # Give more time for processing
                    
                    async with self.session.get(
                        f"{MCP_BASE_URL}/job/{job_id}"
                    ) as status_response:
                        if status_response.status == 200:
                            status_data = await status_response.json()
                            print(f"  📈 Job status: {status_data.get('status')}")
                            print(f"  ⏱️ Progress: {status_data.get('progress', 0):.1f}%")
                            
                            # Clean up - unregister test agent
                            async with self.session.delete(
                                f"{MCP_BASE_URL}/agents/workflow_test_agent"
                            ) as delete_response:
                                if delete_response.status == 200:
                                    print(f"  🧹 Test agent cleaned up")
                            
                            self.test_results.append({
                                "test": test_name,
                                "status": "PASS",
                                "details": {
                                    "job_id": job_id,
                                    "job_status": status_data.get('status'),
                                    "final_progress": status_data.get('progress', 0)
                                }
                            })
                        else:
                            raise Exception(f"Failed to get job status: {status_response.status}")
                else:
                    error_text = await response.text()
                    raise Exception(f"Job submission failed: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"  ❌ {test_name} failed: {str(e)}")
            
            # Try to clean up test agent even if test failed
            try:
                async with self.session.delete(
                    f"{MCP_BASE_URL}/agents/workflow_test_agent"
                ) as delete_response:
                    pass  # Ignore cleanup errors
            except:
                pass
                
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
    
    async def test_api_integration(self):
        """Test API integration with MCP (if main API server is running)"""
        test_name = "API Integration"
        try:
            print(f"\n🔍 Testing: {test_name}")
            
            # Test if main API server is running
            try:
                async with self.session.get(f"{API_BASE_URL}/health") as response:
                    api_running = response.status == 200
            except:
                api_running = False
            
            if not api_running:
                print(f"  ⚠️ Main API server not running at {API_BASE_URL}")
                print(f"  ℹ️ Skipping API integration test")
                self.test_results.append({
                    "test": test_name,
                    "status": "SKIP",
                    "reason": "Main API server not running"
                })
                return
            
            # Test AI analysis endpoint (mock request)
            test_request = {
                "query": "Test AI analysis integration",
                "analysis_depth": "standard",
                "include_recommendations": True
            }
            
            # Note: This would require authentication in real implementation
            async with self.session.post(
                f"{API_BASE_URL}/ai-analysis/portfolios/test_portfolio/ai-analysis",
                json=test_request
            ) as response:
                if response.status in [200, 401, 403]:  # 401/403 expected without auth
                    print(f"  ✅ API endpoint accessible")
                    if response.status == 200:
                        result = await response.json()
                        print(f"  🆔 Job ID: {result.get('job_id')}")
                    else:
                        print(f"  🔐 Authentication required (expected)")
                    
                    self.test_results.append({
                        "test": test_name,
                        "status": "PASS",
                        "details": {"endpoint_accessible": True}
                    })
                else:
                    raise Exception(f"API endpoint failed: {response.status}")
                    
        except Exception as e:
            print(f"  ❌ {test_name} failed: {str(e)}")
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r["status"] == "SKIP"])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️ Skipped: {skipped_tests}")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  • {result['test']}: {result.get('error', 'Unknown error')}")
        
        if passed_tests == total_tests - skipped_tests:
            print(f"\n🎉 All available tests passed! MCP foundation is working correctly.")
        else:
            print(f"\n⚠️ Some tests failed. Please check the errors above.")
        
        print("=" * 50)

async def main():
    """Main test runner"""
    print("🧪 MCP Foundation Test Suite")
    print("Testing Tasks 1.1, 1.2, and 1.3")
    
    try:
        async with MCPFoundationTester() as tester:
            await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test runner failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if we can import the MCP modules
    try:
        from mcp.schemas import JobRequest, JobResponse, AgentRegistration
        print("✅ MCP schemas imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MCP modules: {e}")
        print("Make sure you've created the mcp/ directory and files")
        sys.exit(1)
    
    # Run the tests
    asyncio.run(main())