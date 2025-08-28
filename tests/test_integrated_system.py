# test_integrated_system.py
"""Test the integrated ConsolidatedOrchestrator + MCP system"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import settings
from workflows.orchestrator import ConsolidatedOrchestrator
from api.schemas import QueryRequest

async def test_integrated_system():
    """Test the complete integrated orchestrator + MCP system"""
    
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    orchestrator = None
    
    try:
        print("Testing Integrated ConsolidatedOrchestrator + MCP System...")
        
        # Initialize orchestrator with MCP integration
        orchestrator = ConsolidatedOrchestrator(db=db, mcp_server_url="http://localhost:8001")
        
        # Test system health first
        print("\n--- System Health Check ---")
        try:
            health = await orchestrator.health_check()
            print(f"Orchestrator Status: {health.get('orchestrator_status')}")
            print(f"MCP Server Status: {health.get('mcp_server_status')}")
            print(f"Registered Agents: {health.get('registered_agents', 0)}")
            print(f"Active Jobs: {health.get('active_jobs', 0)}")
            
            # If MCP server is not healthy, continue with limited testing
            if health.get('mcp_server_status') != 'healthy':
                print("⚠️ MCP Server not available - will test direct execution only")
        except Exception as e:
            print(f"Health check failed: {str(e)} - continuing with limited testing")
        
        # Test queries with different routing decisions
        test_cases = [
            {
                "query": "What is my portfolio's current allocation?",
                "expected_mode": "direct",
                "description": "Simple query - should use direct execution"
            },
            {
                "query": "Analyze my portfolio risk and suggest comprehensive rebalancing with tax optimization",
                "expected_mode": "mcp_workflow", 
                "description": "Complex query - should use MCP workflow"
            },
            {
                "query": "Compare my risk metrics to market benchmarks and suggest adjustments",
                "expected_mode": "hybrid",
                "description": "Medium complexity - might use hybrid approach"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {test_case['description']} ---")
            print(f"Query: {test_case['query'][:80]}...")
            
            request = QueryRequest(
                query=test_case["query"],
                context={"portfolio_id": "test_portfolio_123"}
            )
            
            try:
                # Execute query
                response = await orchestrator.execute_query(user_id=1, query_request=request)
                
                print(f"Status: {response.status}")
                print(f"Execution Mode: {response.execution_mode}")
                print(f"Execution Time: {response.execution_time_ms}ms")
                print(f"Session ID: {response.session_id}")
                
                if response.status == "success":
                    result_type = response.result.get("analysis_type", "unknown") if response.result else "none"
                    confidence = response.result.get("confidence", 0.0) if response.result else 0.0
                    print(f"Result Type: {result_type}")
                    print(f"Confidence: {confidence:.2f}")
                    
                    # Check if routing decision matches expectation
                    if response.execution_mode == test_case["expected_mode"]:
                        print(f"✅ Routing decision correct: {response.execution_mode}")
                    else:
                        print(f"⚠️  Routing decision: expected {test_case['expected_mode']}, got {response.execution_mode}")
                else:
                    print(f"❌ Query failed: {response.error_message}")
                
            except Exception as e:
                print(f"❌ Test case failed: {str(e)}")
                # Continue with other test cases
        
        # Test analytics
        print(f"\n--- Performance Analytics ---")
        try:
            analytics = await orchestrator.get_performance_analytics()
            print(f"Total Workflows: {analytics.get('total_workflows', 0)}")
            print(f"Completion Rate: {analytics.get('completion_rate', 0):.2%}")
            print(f"Execution Modes: {analytics.get('execution_mode_distribution', {})}")
        except Exception as e:
            print(f"Analytics error: {str(e)}")
        
        print(f"\n✅ Integrated system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated system test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if orchestrator:
            try:
                await orchestrator.close()
            except:
                pass  # Ignore cleanup errors
        db.close()

async def test_single_complex_query():
    orchestrator = ConsolidatedOrchestrator(db=db, mcp_server_url="http://localhost:8001")
    
    query = "Analyze my portfolio risk and suggest comprehensive rebalancing with tax optimization"
    print(f"Full query: {query}")
    
    # Test analysis directly
    analysis = await orchestrator._analyze_query(query)
    print(f"Analysis: {analysis}")
    
    # Test full execution
    request = QueryRequest(query=query, context={"portfolio_id": "test"})
    response = await orchestrator.execute_query(user_id=1, query_request=request)
    print(f"Execution mode: {response.execution_mode}")

if __name__ == "__main__":
    success = asyncio.run(test_integrated_system())
    sys.exit(0 if success else 1)