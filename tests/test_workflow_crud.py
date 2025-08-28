# test_workflow_crud.py
"""Quick test script to verify workflow CRUD operations"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import models, crud
from core.config import settings # <-- CORRECTED IMPORT
import uuid

def test_workflow_crud():
    """Test basic workflow CRUD operations"""
    # settings = get_settings() # <-- REMOVE THIS LINE
    
    # Create database connection
    # Use the imported settings object directly
    engine = create_engine(settings.DATABASE_URL) 
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        print("🧪 Testing Workflow CRUD Operations...")
        
        # Test 1: Create workflow session
        session_id = str(uuid.uuid4())
        session = crud.create_workflow_session(
            db=db,
            session_id=session_id,
            user_id=1,  # Assuming user with ID 1 exists
            query="Find high-growth tech stocks",
            workflow_type="stock_screening",
            complexity_score=0.6,
            execution_mode="direct"
        )
        print(f"✅ Created workflow session: {session.session_id}")
        
        # Test 2: Update session state
        updated_session = crud.update_workflow_session_state(
            db=db,
            session_id=session_id,
            state="awaiting_screening",
            step_result={"strategy": "growth screening"},
            step_name="strategy"
        )
        print(f"✅ Updated session state: {updated_session.state}")
        
        # Test 3: Create workflow step
        step_id = str(uuid.uuid4())
        step = crud.create_workflow_step(
            db=db,
            step_id=step_id,
            session_id=session_id,
            step_number=1,
            step_name="strategy",
            agent_id="strategy_agent_v1",
            capability="stock_screening_strategy"
        )
        print(f"✅ Created workflow step: {step.step_name}")
        
        # Test 4: Record agent performance
        performance = crud.record_agent_performance(
            db=db,
            agent_id="strategy_agent_v1",
            capability="stock_screening_strategy",
            execution_time_ms=2500,
            success=True,
            execution_mode="direct",
            confidence_score=0.85,
            session_id=session_id
        )
        print(f"✅ Recorded agent performance: {performance.agent_id}")
        
        # Test 5: Create MCP job log
        job_id = str(uuid.uuid4())
        mcp_log = crud.create_mcp_job_log(
            db=db,
            job_id=job_id,
            job_request={"query": "screen stocks", "parameters": {}},
            session_id=session_id
        )
        print(f"✅ Created MCP job log: {mcp_log.job_id}")
        
        # Test 6: Get workflow analytics
        analytics = crud.get_workflow_analytics(db=db, days=7)
        print(f"✅ Retrieved analytics: {analytics['total_workflows']} workflows")
        
        # Test 7: Get system health metrics
        health = crud.get_system_health_metrics(db=db)
        print(f"✅ System health: {health['system_status']}")
        
        print("\n🎉 All workflow CRUD tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        # It's helpful to raise the exception during debugging to see the full traceback
        # raise  
        return False
        
    finally:
        db.close()

if __name__ == "__main__":
    success = test_workflow_crud()
    sys.exit(0 if success else 1)