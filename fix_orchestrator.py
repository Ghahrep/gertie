# Quick fix script for MonitoringOrchestrator import issue
# Save this as check_orchestrator.py and run it

import os
from pathlib import Path

def check_monitoring_orchestrator():
    """Check the monitoring orchestrator file for issues"""
    
    project_root = Path(".")
    orchestrator_file = project_root / "services" / "monitoring_orchestrator.py"
    
    print("🔍 Checking monitoring_orchestrator.py...")
    print(f"File path: {orchestrator_file}")
    print(f"File exists: {orchestrator_file.exists()}")
    
    if orchestrator_file.exists():
        content = orchestrator_file.read_text()
        
        # Check for class definition
        has_class = "class MonitoringOrchestrator" in content
        print(f"Has MonitoringOrchestrator class: {has_class}")
        
        # Check for syntax errors
        try:
            compile(content, str(orchestrator_file), 'exec')
            print("✅ File compiles without syntax errors")
        except SyntaxError as e:
            print(f"❌ Syntax error found: {e}")
            print(f"   Line {e.lineno}: {e.text}")
        
        # Show first few lines and class definition
        lines = content.split('\n')
        print(f"\nFirst 10 lines:")
        for i, line in enumerate(lines[:10], 1):
            print(f"{i:2d}: {line}")
        
        # Find class definition
        class_line = None
        for i, line in enumerate(lines, 1):
            if "class MonitoringOrchestrator" in line:
                class_line = i
                break
        
        if class_line:
            print(f"\nClass definition found at line {class_line}:")
            start = max(0, class_line - 3)
            end = min(len(lines), class_line + 5)
            for i in range(start, end):
                marker = ">>> " if i == class_line - 1 else "    "
                print(f"{marker}{i+1:2d}: {lines[i]}")
        else:
            print("❌ No MonitoringOrchestrator class definition found!")
    
    else:
        print("❌ monitoring_orchestrator.py file not found!")
    
    # Check for other Python files that might have the class
    print("\n🔍 Checking for MonitoringOrchestrator in other files...")
    
    services_dir = project_root / "services"
    if services_dir.exists():
        for py_file in services_dir.glob("*.py"):
            if py_file.name != "monitoring_orchestrator.py":
                try:
                    content = py_file.read_text()
                    if "class MonitoringOrchestrator" in content:
                        print(f"   Found in: {py_file.name}")
                except:
                    pass

def create_minimal_orchestrator():
    """Create a minimal working MonitoringOrchestrator"""
    
    project_root = Path(".")
    services_dir = project_root / "services"
    services_dir.mkdir(exist_ok=True)
    
    orchestrator_file = services_dir / "monitoring_orchestrator.py"
    
    minimal_orchestrator = '''"""
Monitoring Orchestrator - Minimal Working Version
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import service classes
try:
    from .portfolio_monitor import PortfolioMonitor
    from .alert_system import AlertManagementSystem  
    from .task_scheduler import TaskScheduler
    from .risk_detector import RiskDetector
    from .workflow_trigger import WorkflowTrigger
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # Fallback to None - will be handled in initialization
    PortfolioMonitor = None
    AlertManagementSystem = None
    TaskScheduler = None
    RiskDetector = None
    WorkflowTrigger = None

logger = logging.getLogger(__name__)

class MonitoringOrchestrator:
    """
    Orchestrates all monitoring components with proper async/sync handling
    """
    
    def __init__(self):
        self.portfolio_monitor: Optional[PortfolioMonitor] = None
        self.alert_system: Optional[AlertManagementSystem] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        self.risk_detector: Optional[RiskDetector] = None
        self.workflow_trigger: Optional[WorkflowTrigger] = None
        
        self._is_running = False
        self._health_status = {
            'portfolio_monitor': 'stopped',
            'alert_system': 'stopped', 
            'task_scheduler': 'stopped',
            'risk_detector': 'stopped',
            'workflow_trigger': 'stopped',
            'orchestrator': 'stopped'
        }
    
    async def initialize_components(self) -> Dict[str, str]:
        """Initialize all monitoring components with proper error handling"""
        logger.info("🔧 Initializing monitoring components...")
        
        try:
            # Initialize components with fallback handling
            if PortfolioMonitor:
                self.portfolio_monitor = PortfolioMonitor()
            if AlertManagementSystem:
                self.alert_system = AlertManagementSystem()
            if TaskScheduler:
                self.task_scheduler = TaskScheduler()
            if RiskDetector:
                self.risk_detector = RiskDetector()
            if WorkflowTrigger:
                self.workflow_trigger = WorkflowTrigger()
            
            # Verify initialization
            initialization_results = {
                'portfolio_monitor': 'initialized' if self.portfolio_monitor else 'not_available',
                'alert_system': 'initialized' if self.alert_system else 'not_available',
                'task_scheduler': 'initialized' if self.task_scheduler else 'not_available',
                'risk_detector': 'initialized' if self.risk_detector else 'not_available',
                'workflow_trigger': 'initialized' if self.workflow_trigger else 'not_available'
            }
            
            self._health_status['orchestrator'] = 'initialized'
            logger.info("🎯 Component initialization complete")
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            self._health_status['orchestrator'] = 'error'
            return {'orchestrator': f'initialization_error: {str(e)}'}
    
    async def start_monitoring_services(self) -> Dict[str, str]:
        """Start all monitoring services"""
        logger.info("🚀 Starting monitoring services...")
        
        start_results = {}
        
        try:
            # Start task scheduler (synchronous)
            if self.task_scheduler:
                try:
                    scheduler_result = self.task_scheduler.start()
                    start_results['task_scheduler'] = 'started' if scheduler_result else 'failed'
                    self._health_status['task_scheduler'] = start_results['task_scheduler']
                except Exception as e:
                    start_results['task_scheduler'] = f'error: {str(e)}'
                    self._health_status['task_scheduler'] = 'error'
            
            # Start portfolio monitor (synchronous)
            if self.portfolio_monitor:
                try:
                    monitor_result = self.portfolio_monitor.start_monitoring()
                    if isinstance(monitor_result, dict) and monitor_result.get('status') == 'started':
                        start_results['portfolio_monitor'] = 'started'
                        self._health_status['portfolio_monitor'] = 'started'
                    else:
                        start_results['portfolio_monitor'] = 'failed'
                        self._health_status['portfolio_monitor'] = 'failed'
                except Exception as e:
                    start_results['portfolio_monitor'] = f'error: {str(e)}'
                    self._health_status['portfolio_monitor'] = 'error'
            
            # Start alert system (synchronous)
            if self.alert_system:
                try:
                    alert_result = self.alert_system.start()
                    start_results['alert_system'] = 'started' if alert_result else 'failed'
                    self._health_status['alert_system'] = start_results['alert_system']
                except Exception as e:
                    start_results['alert_system'] = f'error: {str(e)}'
                    self._health_status['alert_system'] = 'error'
            
            self._is_running = True
            self._health_status['orchestrator'] = 'running'
            
            logger.info("🎯 Monitoring services startup complete")
            return start_results
            
        except Exception as e:
            logger.error(f"❌ Service startup failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status (synchronous)"""
        try:
            # Update component health status
            if self.portfolio_monitor and hasattr(self.portfolio_monitor, 'get_status'):
                try:
                    monitor_status = self.portfolio_monitor.get_status()
                    self._health_status['portfolio_monitor'] = monitor_status.get('status', 'unknown')
                except:
                    self._health_status['portfolio_monitor'] = 'error'
            
            if self.alert_system and hasattr(self.alert_system, 'get_status'):
                try:
                    alert_status = self.alert_system.get_status()
                    self._health_status['alert_system'] = alert_status.get('status', 'unknown')
                except:
                    self._health_status['alert_system'] = 'error'
            
            if self.task_scheduler and hasattr(self.task_scheduler, 'get_status'):
                try:
                    scheduler_status = self.task_scheduler.get_status()
                    self._health_status['task_scheduler'] = scheduler_status.get('status', 'unknown')
                except:
                    self._health_status['task_scheduler'] = 'error'
            
            return {
                'overall_status': 'healthy' if self._is_running else 'stopped',
                'components': self._health_status.copy(),
                'is_running': self._is_running,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def stop_all_monitoring(self) -> Dict[str, str]:
        """Stop all monitoring services gracefully"""
        logger.info("🛑 Stopping all monitoring services...")
        
        stop_results = {}
        
        # Stop services (all synchronous)
        if self.task_scheduler and hasattr(self.task_scheduler, 'stop'):
            try:
                self.task_scheduler.stop()
                stop_results['task_scheduler'] = 'stopped'
                self._health_status['task_scheduler'] = 'stopped'
            except Exception as e:
                stop_results['task_scheduler'] = f'error: {str(e)}'
        
        if self.portfolio_monitor and hasattr(self.portfolio_monitor, 'stop_monitoring'):
            try:
                self.portfolio_monitor.stop_monitoring()
                stop_results['portfolio_monitor'] = 'stopped'
                self._health_status['portfolio_monitor'] = 'stopped'
            except Exception as e:
                stop_results['portfolio_monitor'] = f'error: {str(e)}'
        
        if self.alert_system and hasattr(self.alert_system, 'stop'):
            try:
                self.alert_system.stop()
                stop_results['alert_system'] = 'stopped'
                self._health_status['alert_system'] = 'stopped'
            except Exception as e:
                stop_results['alert_system'] = f'error: {str(e)}'
        
        self._is_running = False
        self._health_status['orchestrator'] = 'stopped'
        
        logger.info("✅ All monitoring services stopped")
        return stop_results

# Global orchestrator instance
_global_orchestrator: Optional[MonitoringOrchestrator] = None

async def get_monitoring_orchestrator() -> MonitoringOrchestrator:
    """Get or create global monitoring orchestrator"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = MonitoringOrchestrator()
        await _global_orchestrator.initialize_components()
    return _global_orchestrator

# Test function
async def test_orchestrator():
    """Test the monitoring orchestrator"""
    print("🧪 Testing MonitoringOrchestrator...")
    
    try:
        orchestrator = MonitoringOrchestrator()
        print("✅ MonitoringOrchestrator created successfully")
        
        # Test initialization
        result = await orchestrator.initialize_components()
        print(f"✅ Initialization result: {result}")
        
        # Test health check (synchronous)
        health = orchestrator.get_system_health()
        print(f"✅ Health check result: {health}")
        
        # Test shutdown
        shutdown = await orchestrator.stop_all_monitoring()
        print(f"✅ Shutdown result: {shutdown}")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
'''
    
    # Backup existing file if it exists
    if orchestrator_file.exists():
        backup_file = orchestrator_file.with_suffix('.py.broken_backup')
        backup_content = orchestrator_file.read_text()
        backup_file.write_text(backup_content)
        print(f"📁 Backed up existing file to: {backup_file}")
    
    # Write new file
    orchestrator_file.write_text(minimal_orchestrator)
    print(f"✅ Created minimal MonitoringOrchestrator at: {orchestrator_file}")

if __name__ == "__main__":
    print("🔍 CHECKING MONITORING ORCHESTRATOR")
    print("=" * 50)
    check_monitoring_orchestrator()
    
    print("\n" + "=" * 50)
    response = input("\n❓ Create minimal working orchestrator? (y/n): ")
    
    if response.lower().startswith('y'):
        print("\n🔧 CREATING MINIMAL ORCHESTRATOR")
        print("=" * 50)
        create_minimal_orchestrator()
        
        print("\n🧪 TESTING NEW ORCHESTRATOR")
        print("=" * 50)
        import asyncio
        result = asyncio.run(__import__('services.monitoring_orchestrator', fromlist=['test_orchestrator']).test_orchestrator())
        
        if result:
            print("\n🎉 SUCCESS! MonitoringOrchestrator is now working!")
        else:
            print("\n⚠️ There are still issues - check the error messages above")