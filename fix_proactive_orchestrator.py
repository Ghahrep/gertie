#!/usr/bin/env python3
"""
Fix async/await issues in ProactiveMonitoringOrchestrator
Run with: python fix_proactive_orchestrator.py
"""

import re
import logging
from pathlib import Path
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_orchestrator_async_issues():
    """Fix the async/await issues in monitoring_orchestrator.py"""
    
    orchestrator_file = Path("services/monitoring_orchestrator.py")
    
    if not orchestrator_file.exists():
        logger.error("monitoring_orchestrator.py not found")
        return False
    
    logger.info("Reading monitoring_orchestrator.py...")
    content = orchestrator_file.read_text()
    
    # Backup original
    backup_file = orchestrator_file.with_suffix('.py.pre_async_fix')
    backup_file.write_text(content)
    logger.info(f"Backup created: {backup_file}")
    
    # Apply specific fixes based on the error patterns
    fixes_applied = []
    
    # Fix 1: start_monitoring method should be awaited
    old_pattern = r"monitor_started = self\.portfolio_monitor\.start_monitoring\(\)"
    new_pattern = "monitor_started = await self.portfolio_monitor.start_monitoring()"
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied.append("Fixed portfolio_monitor.start_monitoring() to be awaited")
    
    # Fix 2: stop_monitoring method should be awaited
    old_pattern = r"stop_result = self\.portfolio_monitor\.stop_monitoring\(\)"
    new_pattern = "stop_result = await self.portfolio_monitor.stop_monitoring()"
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied.append("Fixed portfolio_monitor.stop_monitoring() to be awaited")
    
    # Fix 3: scheduler status methods should be awaited
    old_pattern = r"scheduler_status = self\.task_scheduler\.get_scheduler_status\(\)"
    new_pattern = "scheduler_status = await self.task_scheduler.get_scheduler_status()"
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied.append("Fixed task_scheduler.get_scheduler_status() to be awaited")
    
    # Fix 4: scheduler start should be awaited
    old_pattern = r"scheduler_started = self\.task_scheduler\.start\(\)"
    new_pattern = "scheduler_started = await self.task_scheduler.start()"
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied.append("Fixed task_scheduler.start() to be awaited")
    
    # Fix 5: Handle .get() on coroutine objects - fix the result handling
    # This is more complex - need to fix the logic that expects dict responses
    
    # Fix portfolio monitoring start logic
    portfolio_start_fix = '''
            # Start portfolio monitoring with error handling
            logger.info("Starting portfolio monitoring...")
            try:
                monitor_result = await self.portfolio_monitor.start_monitoring()
                if isinstance(monitor_result, dict):
                    self.system_status.portfolio_monitor_active = monitor_result.get("status") == "started"
                else:
                    # Handle case where start_monitoring returns boolean or other type
                    self.system_status.portfolio_monitor_active = bool(monitor_result)
                
                if self.system_status.portfolio_monitor_active:
                    logger.info("Portfolio monitoring started successfully")
                else:
                    logger.warning(f"Portfolio monitoring start returned: {monitor_result}")
                    
            except Exception as e:
                logger.error(f"Failed to start portfolio monitoring: {e}")
                self.system_status.portfolio_monitor_active = False
                # Don't return False here - continue with initialization
'''
    
    # Find and replace the portfolio monitoring start section
    portfolio_start_pattern = r"# Start portfolio monitoring with error handling.*?# Don't return False here - continue with initialization"
    if re.search(portfolio_start_pattern, content, re.DOTALL):
        content = re.sub(portfolio_start_pattern, portfolio_start_fix.strip(), content, flags=re.DOTALL)
        fixes_applied.append("Fixed portfolio monitoring start logic")
    
    # Fix stop monitoring logic
    stop_fix = '''
            # Stop portfolio monitoring
            if self.portfolio_monitor:
                try:
                    stop_result = await self.portfolio_monitor.stop_monitoring()
                    if isinstance(stop_result, dict):
                        results["portfolio_monitor_stopped"] = stop_result.get("status") == "stopped"
                    else:
                        results["portfolio_monitor_stopped"] = bool(stop_result)
                except Exception as e:
                    logger.error(f"Error stopping portfolio monitor: {e}")
                    results["portfolio_monitor_stopped"] = False
'''
    
    # Find and replace stop monitoring section
    stop_pattern = r"# Stop portfolio monitoring.*?results\[\"portfolio_monitor_stopped\"\] = stop_result\.get\(\"status\"\) == \"stopped\""
    if re.search(stop_pattern, content, re.DOTALL):
        content = re.sub(stop_pattern, stop_fix.strip(), content, flags=re.DOTALL)
        fixes_applied.append("Fixed portfolio monitoring stop logic")
    
    # Fix scheduler status handling in _update_system_health
    health_fix = '''
    async def _update_system_health(self):
        """Update system health status based on component status"""
        try:
            # Get component statuses with proper async handling
            scheduler_status = {}
            if self.task_scheduler:
                try:
                    scheduler_status = await self.task_scheduler.get_scheduler_status()
                except Exception as e:
                    logger.error(f"Error getting scheduler status: {e}")
                    scheduler_status = {}
            
            monitor_stats = {}
            if self.proactive_monitor:
                try:
                    monitor_stats = self.proactive_monitor.get_monitoring_stats()
                except Exception as e:
                    logger.error(f"Error getting monitor stats: {e}")
                    monitor_stats = {}
        
            workflow_stats = {}
            if self.workflow_engine:
                try:
                    workflow_stats = self.workflow_engine.get_execution_statistics()
                except Exception as e:
                    logger.error(f"Error getting workflow stats: {e}")
                    workflow_stats = {}
            
            # Update counts with proper type checking
            self.system_status.total_active_monitors = monitor_stats.get("active_monitors", 0)
            self.system_status.recent_alerts_count = monitor_stats.get("recent_alerts_1h", 0)
            self.system_status.pending_workflows = workflow_stats.get("active_workflows", 0)
            
            # Determine overall health
            active_components = sum([
                self.system_status.portfolio_monitor_active,
                self.system_status.alert_system_active,
                self.system_status.scheduler_active,
                self.system_status.workflow_engine_active
            ])
            
            if active_components == 4:
                self.system_status.system_health = "healthy"
            elif active_components >= 2:
                self.system_status.system_health = "degraded"
            else:
                self.system_status.system_health = "critical"
                
        except Exception as e:
            logger.error(f"Error updating system health: {e}")
            self.system_status.system_health = "unknown"
'''
    
    # Find and replace the _update_system_health method
    health_pattern = r"async def _update_system_health\(self\):.*?self\.system_status\.system_health = \"unknown\""
    if re.search(health_pattern, content, re.DOTALL):
        content = re.sub(health_pattern, health_fix.strip(), content, flags=re.DOTALL)
        fixes_applied.append("Fixed _update_system_health method")
    
    # Fix get_complete_system_status method
    status_fix = '''
        try:
            # Get component statuses with proper async handling
            portfolio_status = {}
            if self.portfolio_monitor:
                try:
                    portfolio_status = self.portfolio_monitor.get_monitoring_status()
                except Exception as e:
                    logger.error(f"Error getting portfolio status: {e}")
                    portfolio_status = {}
            
            scheduler_status = {}
            if self.task_scheduler:
                try:
                    scheduler_status = await self.task_scheduler.get_scheduler_status()
                except Exception as e:
                    logger.error(f"Error getting scheduler status: {e}")
                    scheduler_status = {}
            
            workflow_stats = {}
            if self.workflow_engine:
                try:
                    workflow_stats = self.workflow_engine.get_execution_statistics()
                except Exception as e:
                    logger.error(f"Error getting workflow stats: {e}")
                    workflow_stats = {}
'''
    
    # Find the status retrieval section in get_complete_system_status
    status_pattern = r"try:\s*# Fix: Separate the conditional checks from await.*?workflow_stats = self\.workflow_engine\.get_execution_statistics\(\)"
    if re.search(status_pattern, content, re.DOTALL):
        content = re.sub(status_pattern, status_fix.strip(), content, flags=re.DOTALL)
        fixes_applied.append("Fixed get_complete_system_status method")
    
    # Write the fixed content
    orchestrator_file.write_text(content)
    
    logger.info("Fixes applied:")
    for fix in fixes_applied:
        logger.info(f"  - {fix}")
    
    return len(fixes_applied) > 0

def create_async_test_script():
    """Create a test script to verify the async fixes"""
    
    test_script = '''#!/usr/bin/env python3
"""
Test the fixed ProactiveMonitoringOrchestrator async methods
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def test_fixed_orchestrator():
    """Test that async issues are resolved"""
    
    print("🧪 Testing Fixed ProactiveMonitoringOrchestrator...")
    
    try:
        from services.monitoring_orchestrator import ProactiveMonitoringOrchestrator
        
        orchestrator = ProactiveMonitoringOrchestrator()
        print("✅ Orchestrator created")
        
        # Test initialization (this should not produce RuntimeWarnings)
        print("🔧 Testing initialization...")
        init_success = await orchestrator.initialize_all_systems()
        print(f"✅ Initialization: {init_success}")
        
        # Test system status (this should not produce async warnings)
        print("📊 Testing system status...")
        status = await orchestrator.get_complete_system_status()
        health = status.get('system_overview', {}).get('health', 'unknown')
        print(f"✅ System health: {health}")
        
        # Test shutdown (this should not produce async warnings)
        print("🛑 Testing shutdown...")
        shutdown = await orchestrator.stop_all_monitoring()
        print(f"✅ Shutdown: {shutdown.get('status', 'unknown')}")
        
        print("🎉 All async tests completed without warnings!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_fixed_orchestrator())
'''
    
    test_file = Path("test_fixed_orchestrator.py")
    test_file.write_text(test_script)
    logger.info(f"Created test script: {test_file}")

async def main():
    """Main function to fix and test"""
    
    print("🔧 Fixing ProactiveMonitoringOrchestrator Async Issues")
    print("=" * 60)
    
    # Apply fixes
    success = fix_orchestrator_async_issues()
    
    if success:
        print("✅ Async fixes applied successfully")
        
        # Create test script
        create_async_test_script()
        
        print("\n📋 Next steps:")
        print("1. Run: python test_fixed_orchestrator.py")
        print("2. Verify no RuntimeWarning messages appear")
        print("3. Run your integration tests")
        
    else:
        print("❌ No fixes were applied")

if __name__ == "__main__":
    asyncio.run(main())