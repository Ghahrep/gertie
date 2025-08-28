# quick_validation.py
"""
Quick Task 1.2 validation using your existing CRUD functions
Tests what you already have implemented
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from db.session import get_db
from db.crud import (
    create_risk_snapshot, get_latest_risk_snapshot, get_risk_history,
    create_workflow_session, get_workflow_session, 
    record_agent_performance, get_agent_performance_stats,
    create_mcp_job_log, update_mcp_job_log_status,
    get_workflow_analytics, get_system_health_metrics
)

def test_existing_functionality():
    """Test your existing database functionality"""
    
    print("\n" + "="*60)
    print("TASK 1.2 QUICK VALIDATION: Testing Existing Functions")
    print("="*60)
    
    try:
        db = next(get_db())
        test_results = {}
        
        # Test 1: Risk Snapshot Operations
        print("\n1. Testing Risk Snapshot Operations...")
        try:
            user_id = "test_user_123"
            portfolio_id = "test_portfolio_456"
            
            risk_result = {
                'volatility': 0.15,
                'beta': 1.2,
                'max_drawdown': -0.08,
                'var_95': -0.02,
                'var_99': -0.05,
                'sharpe_ratio': 1.5,
                'sortino_ratio': 1.8,
                'risk_score': 65.0,
                'risk_level': 'medium',
                'risk_grade': 'B'
            }
            
            # Test snapshot creation
            start_time = time.time()
            snapshot = create_risk_snapshot(
                db=db,
                user_id=user_id,
                portfolio_id=portfolio_id,
                risk_result=risk_result
            )
            create_time = (time.time() - start_time) * 1000
            
            if snapshot:
                print(f"   ✓ Risk snapshot created successfully ({create_time:.1f}ms)")
                
                # Test retrieval
                start_time = time.time()
                latest = get_latest_risk_snapshot(db, user_id, portfolio_id)
                query_time = (time.time() - start_time) * 1000
                
                if latest:
                    print(f"   ✓ Latest snapshot retrieved ({query_time:.1f}ms)")
                    test_results["risk_snapshots"] = "PASS"
                else:
                    print("   ❌ Failed to retrieve latest snapshot")
                    test_results["risk_snapshots"] = "FAIL"
            else:
                print("   ❌ Failed to create risk snapshot")
                test_results["risk_snapshots"] = "FAIL"
                
        except Exception as e:
            print(f"   ❌ Risk snapshot test error: {e}")
            test_results["risk_snapshots"] = "ERROR"
        
        # Test 2: Workflow Session Operations
        print("\n2. Testing Workflow Session Operations...")
        try:
            session_id = "test_session_789"
            user_id = 1  # Integer user ID for your model
            
            # Test session creation
            session = create_workflow_session(
                db=db,
                session_id=session_id,
                user_id=user_id,
                query="Test risk analysis query",
                workflow_type="risk_analysis",
                complexity_score=0.75
            )
            
            if session:
                print(f"   ✓ Workflow session created: {session.session_id}")
                
                # Test retrieval
                retrieved = get_workflow_session(db, session_id)
                if retrieved:
                    print(f"   ✓ Workflow session retrieved: {retrieved.workflow_type}")
                    test_results["workflow_sessions"] = "PASS"
                else:
                    print("   ❌ Failed to retrieve workflow session")
                    test_results["workflow_sessions"] = "FAIL"
            else:
                print("   ❌ Failed to create workflow session")
                test_results["workflow_sessions"] = "FAIL"
                
        except Exception as e:
            print(f"   ❌ Workflow session test error: {e}")
            test_results["workflow_sessions"] = "ERROR"
        
        # Test 3: Agent Performance Tracking
        print("\n3. Testing Agent Performance Tracking...")
        try:
            # Test performance recording
            performance = record_agent_performance(
                db=db,
                agent_id="risk_calculator",
                capability="risk_analysis",
                execution_time_ms=1500,
                success=True,
                execution_mode="direct",
                confidence_score=0.9,
                user_id=1
            )
            
            if performance:
                print(f"   ✓ Agent performance recorded: {performance.agent_id}")
                
                # Test statistics
                stats = get_agent_performance_stats(db, "risk_calculator", days=30)
                if stats and stats["total_executions"] > 0:
                    print(f"   ✓ Performance stats retrieved: {stats['total_executions']} executions")
                    test_results["agent_performance"] = "PASS"
                else:
                    print("   ❌ No performance statistics found")
                    test_results["agent_performance"] = "FAIL"
            else:
                print("   ❌ Failed to record agent performance")
                test_results["agent_performance"] = "FAIL"
                
        except Exception as e:
            print(f"   ❌ Agent performance test error: {e}")
            test_results["agent_performance"] = "ERROR"
        
        # Test 4: MCP Job Logging
        print("\n4. Testing MCP Job Logging...")
        try:
            job_id = "test_mcp_job_123"
            job_request = {
                "query": "Test MCP request",
                "agents": ["risk_calculator"],
                "complexity": 0.8
            }
            
            # Test job log creation
            log_entry = create_mcp_job_log(
                db=db,
                job_id=job_id,
                job_request=job_request
            )
            
            if log_entry:
                print(f"   ✓ MCP job logged: {log_entry.job_id}")
                
                # Test status update
                updated = update_mcp_job_log_status(
                    db=db,
                    job_id=job_id,
                    status="completed",
                    job_response={"result": "success"}
                )
                
                if updated and updated.status == "completed":
                    print(f"   ✓ MCP job status updated: {updated.status}")
                    test_results["mcp_logging"] = "PASS"
                else:
                    print("   ❌ Failed to update MCP job status")
                    test_results["mcp_logging"] = "FAIL"
            else:
                print("   ❌ Failed to create MCP job log")
                test_results["mcp_logging"] = "FAIL"
                
        except Exception as e:
            print(f"   ❌ MCP job logging test error: {e}")
            test_results["mcp_logging"] = "ERROR"
        
        # Test 5: Analytics Functions
        print("\n5. Testing Analytics Functions...")
        try:
            # Test workflow analytics
            analytics = get_workflow_analytics(db, user_id=1, days=30)
            if analytics and "total_workflows" in analytics:
                print(f"   ✓ Workflow analytics: {analytics['total_workflows']} workflows")
            
            # Test system health
            health = get_system_health_metrics(db)
            if health and "active_workflows" in health:
                print(f"   ✓ System health: {health['active_workflows']} active workflows")
                test_results["analytics"] = "PASS"
            else:
                print("   ❌ Analytics functions incomplete")
                test_results["analytics"] = "FAIL"
                
        except Exception as e:
            print(f"   ❌ Analytics test error: {e}")
            test_results["analytics"] = "ERROR"
        
        # Test 6: Database Structure Validation
        print("\n6. Testing Database Structure...")
        try:
            # Test that key tables exist and are accessible
            from db.models import (
                PortfolioRiskSnapshot, WorkflowSessionDB, 
                AgentPerformanceMetrics, MCPJobLog
            )
            
            # Count records in key tables
            snapshot_count = db.query(PortfolioRiskSnapshot).count()
            workflow_count = db.query(WorkflowSessionDB).count()
            metrics_count = db.query(AgentPerformanceMetrics).count()
            
            print(f"   ✓ Database structure valid:")
            print(f"     - Risk snapshots: {snapshot_count}")
            print(f"     - Workflow sessions: {workflow_count}")
            print(f"     - Performance metrics: {metrics_count}")
            
            test_results["database_structure"] = "PASS"
            
        except Exception as e:
            print(f"   ❌ Database structure test error: {e}")
            test_results["database_structure"] = "ERROR"
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION RESULTS SUMMARY")
        print("="*60)
        
        passed_tests = sum(1 for result in test_results.values() if result == "PASS")
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status_icon = "✓" if result == "PASS" else "❌" if result == "FAIL" else "⚠️"
            print(f"  {status_icon} {test_name}: {result}")
        
        print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 TASK 1.2 CORE FUNCTIONALITY VALIDATED!")
            print("Your existing database implementation is working correctly.")
            print("\nTo complete Task 1.2, you still need:")
            print("  1. Add performance optimization indexes")
            print("  2. Add bulk operations for better performance")
            print("  3. Add validation functions for data integrity")
            print("\nBut your core functionality is solid and ready for Sprint 2!")
            return True
        else:
            print(f"\n⚠️ {total_tests - passed_tests} tests need attention.")
            print("Address the errors above before proceeding to Sprint 2.")
            return False
            
    except Exception as e:
        print(f"\n❌ VALIDATION SETUP ERROR: {e}")
        return False
        
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    success = test_existing_functionality()
    sys.exit(0 if success else 1)