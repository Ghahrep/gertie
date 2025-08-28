# test_orchestrator.py
"""Test the ConsolidatedOrchestrator implementation"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import get_settings
from workflows.orchestrator import ConsolidatedOrchestrator
from workflows.mcp_client import MCPClient
from api.schemas import QueryRequest

async def test_orchestrator():
    """Test the orchestrator with different query types"""
    
    settings = get_settings()
    engine = create_engine(settings.database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        print("Testing ConsolidatedOrchestrator...")
        
        # Initialize orchestrator
        mcp_client = MCPClient()
        orchestrator = ConsolidatedOrchestrator(db=db, mcp_client=mcp_client)
        
        # Test queries of different complexities
        test_queries = [
            {
                "query": "What is the current price of AAPL?",
                "expected_complexity": "low"
            },
            {
                "query": "Find stocks with P/E ratio less than 15 and revenue growth above 10%",
                "expected_complexity": "medium"  
            },
            {
                "query": "Perform comprehensive portfolio optimization using Monte Carlo simulation with multiple risk factors, sector constraints, and correlation analysis for a 10-year investment horizon",
                "expected_complexity": "high"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {test_case['expected_complexity'].upper()} Complexity ---")
            print(f"Query: {test_case['query'][:80]}...")
            
            request = QueryRequest(query=test_case["query"])
            
            # Execute query
            response = await orchestrator.execute_query(user_id=1, query_request=request)
            
            print(f"Status: {response.status}")
            print(f"Execution Mode: {response.execution_mode}")
            print(f"Execution Time: {response.execution_time_ms}ms")
            print(f"Session ID: {response.session_id}")
            
            if response.status == "success":
                print(f"Result Type: {response.result.get('analysis_type', 'unknown')}")
                print(f"Confidence: {response.result.get('confidence', 0.0):.2f}")
            else:
                print(f"Error: {response.error_message}")
        
        # Test analytics
        print(f"\n--- Performance Analytics ---")
        analytics = await orchestrator.get_performance_analytics()
        print(f"Total Workflows: {analytics['total_workflows']}")
        print(f"Completion Rate: {analytics['completion_rate']:.2%}")
        print(f"Execution Modes: {analytics['execution_mode_distribution']}")
        
        # Test system health
        print(f"\n--- System Health ---")
        health = await orchestrator.get_system_health()
        print(f"Status: {health['system_status']}")
        print(f"Active Workflows: {health['active_workflows']}")
        
        print(f"\n🎉 All orchestrator tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        db.close()

if __name__ == "__main__":
    success = asyncio.run(test_orchestrator())
    sys.exit(0 if success else 1)