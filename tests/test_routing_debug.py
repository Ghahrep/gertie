# test_routing_debug.py
"""Debug the routing logic specifically"""

import asyncio
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import settings
from workflows.orchestrator import ConsolidatedOrchestrator

# Enable logging to see debug messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_routing_logic():
    """Test just the routing logic without full execution"""
    
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        print("Testing Routing Logic...")
        
        orchestrator = ConsolidatedOrchestrator(db=db, mcp_server_url="http://localhost:8001")
        
        test_queries = [
            "What is my portfolio allocation?",
            "Analyze my portfolio risk and suggest comprehensive rebalancing with tax optimization and Monte Carlo simulation",
            "Compare my risk metrics to market benchmarks using statistical analysis"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query[:50]}... ---")
            
            # Just test the analysis without full execution
            analysis = await orchestrator._analyze_query(query)
            
            print(f"Complexity Score: {analysis['complexity_score']:.3f}")
            print(f"Complexity Level: {analysis['complexity_level'].value}")
            print(f"Workflow Type: {analysis['workflow_type']}")
            print(f"Recommended Mode: {analysis['recommended_mode'].value}")
            print(f"Reasoning: {analysis['reasoning']}")
        
        return True
        
    except Exception as e:
        print(f"Routing test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await orchestrator.close()
        db.close()

if __name__ == "__main__":
    success = asyncio.run(test_routing_logic())
    sys.exit(0 if success else 1)