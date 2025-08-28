# test_routing_isolated.py
"""Isolated test to compare direct analysis vs full execution routing"""

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

async def test_routing_isolated():
    """Compare routing decisions between direct analysis and full execution"""
    
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        orchestrator = ConsolidatedOrchestrator(db=db)
        query = "Analyze my portfolio risk and suggest comprehensive rebalancing with tax optimization"
        
        print(f"Testing query: {query}")
        print("-" * 80)
        
        # Direct analysis test
        analysis = await orchestrator._analyze_query(query)
        print(f"Direct analysis - Mode: {analysis['recommended_mode'].value}, Score: {analysis['complexity_score']}")
        print(f"Workflow type: {analysis['workflow_type']}")
        print(f"Complexity level: {analysis['complexity_level'].value}")
        
        print("-" * 80)
        
        # Full execution test  
        request = QueryRequest(query=query, context={"portfolio_id": "test_123"})
        response = await orchestrator.execute_query(user_id=1, query_request=request)
        print(f"Full execution - Mode: {response.execution_mode}")
        print(f"Status: {response.status}")
        print(f"Session ID: {response.session_id}")
        
        if response.status == "error":
            print(f"Error: {response.error_message}")
        
        # Compare results
        analysis_mode = analysis['recommended_mode'].value
        execution_mode = response.execution_mode
        
        if analysis_mode == execution_mode:
            print(f"✅ Routing consistent: {analysis_mode}")
        else:
            print(f"❌ Routing inconsistent: analysis={analysis_mode}, execution={execution_mode}")
            
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await orchestrator.close()
        db.close()

if __name__ == "__main__":
    success = asyncio.run(test_routing_isolated())
    sys.exit(0 if success else 1)