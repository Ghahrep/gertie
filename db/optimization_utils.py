# db/optimization_utils.py
"""
Simple database optimization utilities that work with your existing setup
Focuses on the specific performance improvements needed for Task 1.2.3
"""

from sqlalchemy import text, Index
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def create_performance_indexes(db: Session) -> List[str]:
    """
    Create the essential performance indexes for your existing models
    """
    indexes_sql = [
        # Portfolio Risk Snapshots - Your most queried table
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_snapshots_portfolio_date 
        ON portfolio_risk_snapshots (portfolio_id, snapshot_date DESC);
        """,
        
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_snapshots_user_date 
        ON portfolio_risk_snapshots (user_id, snapshot_date DESC);
        """,
        
        # Risk Change Events - For your alert system
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_events_portfolio_detected 
        ON risk_change_events (portfolio_id, detected_at DESC);
        """,
        
        # Proactive Alerts - For dashboard queries
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_proactive_alerts_user_active 
        ON proactive_alerts (user_id, is_active, created_at DESC);
        """,
        
        # Workflow Sessions - For analytics
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_workflow_sessions_user_state 
        ON workflow_sessions (user_id, state, created_at DESC);
        """,
        
        # Agent Performance - For optimization
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_performance_agent_capability 
        ON agent_performance_metrics (agent_id, capability, created_at DESC);
        """
    ]
    
    created_indexes = []
    
    for sql in indexes_sql:
        try:
            db.execute(text(sql))
            db.commit()
            
            # Extract index name
            index_name = sql.split("IF NOT EXISTS")[1].split("ON")[0].strip()
            created_indexes.append(index_name)
            logger.info(f"Created index: {index_name}")
            
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")
    
    return created_indexes

def analyze_table_performance(db: Session) -> Dict[str, Any]:
    """
    Analyze performance of your key tables
    """
    try:
        # Your key tables
        tables = [
            "portfolio_risk_snapshots",
            "workflow_sessions", 
            "agent_performance_metrics",
            "proactive_alerts",
            "risk_change_events"
        ]
        
        table_stats = {}
        
        for table in tables:
            # Get basic table statistics
            stats_query = f"""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    seq_scan as sequential_scans,
                    idx_scan as index_scans,
                    n_tup_hot_upd as hot_updates
                FROM pg_stat_user_tables 
                WHERE tablename = '{table}';
            """
            
            result = db.execute(text(stats_query)).fetchone()
            
            if result:
                table_stats[table] = {
                    "inserts": result[2] or 0,
                    "updates": result[3] or 0,
                    "deletes": result[4] or 0,
                    "sequential_scans": result[5] or 0,
                    "index_scans": result[6] or 0,
                    "hot_updates": result[7] or 0,
                    "scan_ratio": (result[6] or 0) / max(1, (result[5] or 0) + (result[6] or 0))
                }
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "table_statistics": table_stats
        }
        
    except Exception as e:
        return {"error": str(e)}

def optimize_queries_for_risk_monitoring(db: Session) -> Dict[str, Any]:
    """
    Optimize the specific queries used in risk monitoring
    """
    optimization_results = {}
    
    try:
        # Update statistics for your key tables
        key_tables = [
            "portfolio_risk_snapshots",
            "workflow_sessions",
            "agent_performance_metrics"
        ]
        
        for table in key_tables:
            db.execute(text(f"ANALYZE {table}"))
            optimization_results[f"{table}_analyzed"] = True
        
        db.commit()
        
        # Test query performance on common operations
        test_queries = {
            "latest_risk_snapshot": """
                SELECT * FROM portfolio_risk_snapshots 
                WHERE portfolio_id = 1 
                ORDER BY snapshot_date DESC 
                LIMIT 1
            """,
            "recent_workflows": """
                SELECT * FROM workflow_sessions 
                WHERE user_id = 1 AND created_at >= NOW() - INTERVAL '7 days' 
                ORDER BY created_at DESC 
                LIMIT 10
            """,
            "agent_performance": """
                SELECT agent_id, AVG(execution_time_ms) as avg_time
                FROM agent_performance_metrics 
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY agent_id
            """
        }
        
        query_performance = {}
        for query_name, query_sql in test_queries.items():
            try:
                start_time = datetime.now()
                db.execute(text(query_sql))
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                query_performance[query_name] = {
                    "execution_time_ms": execution_time,
                    "status": "success"
                }
            except Exception as e:
                query_performance[query_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        optimization_results["query_performance"] = query_performance
        
    except Exception as e:
        optimization_results["error"] = str(e)
    
    return optimization_results

def cleanup_old_data(db: Session, retention_days: int = 90) -> Dict[str, Any]:
    """
    Clean up old data to maintain performance
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cleanup_results = {}
    
    try:
        # Clean up old workflow sessions that are completed
        old_workflows = db.query(db.bind.dialect.name).filter(text(f"""
            DELETE FROM workflow_sessions 
            WHERE state = 'complete' 
            AND completed_at < '{cutoff_date}'
        """))
        
        # Count before deletion
        completed_count = db.execute(text(f"""
            SELECT COUNT(*) FROM workflow_sessions 
            WHERE state = 'complete' AND completed_at < '{cutoff_date}'
        """)).scalar()
        
        # Delete old completed workflows
        db.execute(text(f"""
            DELETE FROM workflow_sessions 
            WHERE state = 'complete' AND completed_at < '{cutoff_date}'
        """))
        
        cleanup_results["old_workflows_deleted"] = completed_count or 0
        
        # Clean up old agent performance metrics
        old_metrics_count = db.execute(text(f"""
            SELECT COUNT(*) FROM agent_performance_metrics 
            WHERE created_at < '{cutoff_date}'
        """)).scalar()
        
        db.execute(text(f"""
            DELETE FROM agent_performance_metrics 
            WHERE created_at < '{cutoff_date}'
        """))
        
        cleanup_results["old_metrics_deleted"] = old_metrics_count or 0
        
        db.commit()
        
        cleanup_results["status"] = "success"
        cleanup_results["retention_days"] = retention_days
        
    except Exception as e:
        cleanup_results["status"] = "error"
        cleanup_results["error"] = str(e)
        db.rollback()
    
    return cleanup_results

def get_database_health_summary(db: Session) -> Dict[str, Any]:
    """
    Get a simple database health summary for your system
    """
    try:
        # Get row counts for key tables
        table_counts = {}
        key_tables = [
            "portfolio_risk_snapshots",
            "workflow_sessions", 
            "agent_performance_metrics",
            "proactive_alerts"
        ]
        
        for table in key_tables:
            count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            table_counts[table] = count or 0
        
        # Get recent activity (last 24 hours)
        recent_activity = {}
        
        # Recent risk snapshots
        recent_snapshots = db.execute(text("""
            SELECT COUNT(*) FROM portfolio_risk_snapshots 
            WHERE snapshot_date >= NOW() - INTERVAL '24 hours'
        """)).scalar()
        recent_activity["risk_snapshots_24h"] = recent_snapshots or 0
        
        # Recent workflows
        recent_workflows = db.execute(text("""
            SELECT COUNT(*) FROM workflow_sessions 
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """)).scalar()
        recent_activity["workflows_24h"] = recent_workflows or 0
        
        # Active workflows
        active_workflows = db.execute(text("""
            SELECT COUNT(*) FROM workflow_sessions 
            WHERE state NOT IN ('complete', 'error')
        """)).scalar()
        recent_activity["active_workflows"] = active_workflows or 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "table_counts": table_counts,
            "recent_activity": recent_activity,
            "total_records": sum(table_counts.values())
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

def run_maintenance_tasks(db: Session) -> Dict[str, Any]:
    """
    Run all maintenance tasks for Task 1.2.3 requirements
    """
    maintenance_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": {}
    }
    
    # 1. Create performance indexes
    try:
        indexes = create_performance_indexes(db)
        maintenance_results["tasks"]["create_indexes"] = {
            "status": "completed",
            "indexes_created": len(indexes)
        }
    except Exception as e:
        maintenance_results["tasks"]["create_indexes"] = {
            "status": "failed", 
            "error": str(e)
        }
    
    # 2. Optimize queries
    try:
        query_results = optimize_queries_for_risk_monitoring(db)
        maintenance_results["tasks"]["optimize_queries"] = {
            "status": "completed",
            "results": query_results
        }
    except Exception as e:
        maintenance_results["tasks"]["optimize_queries"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # 3. Clean up old data
    try:
        cleanup_results = cleanup_old_data(db)
        maintenance_results["tasks"]["cleanup_data"] = cleanup_results
    except Exception as e:
        maintenance_results["tasks"]["cleanup_data"] = {
            "status": "failed",
            "error": str(e)
        }
    
    return maintenance_results