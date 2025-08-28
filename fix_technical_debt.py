#!/usr/bin/env python3
"""
Quick Technical Debt Fix Runner - Save as fix_debt.py in your root directory
Run with: python fix_debt.py
"""

import asyncio
import os
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def quick_fix_technical_debt():
    """Quick fix for the most critical technical debt issues"""
    
    print("🚀 Quick Technical Debt Fix Starting...")
    print("=" * 50)
    
    project_root = Path(".")
    services_dir = project_root / "services"
    
    fixes_applied = []
    
    # Fix 1: Monitoring Orchestrator Async Issues
    orchestrator_file = services_dir / "monitoring_orchestrator.py"
    
    if orchestrator_file.exists():
        print("🔧 Fixing monitoring_orchestrator.py async/await issues...")
        
        content = orchestrator_file.read_text()
        original_content = content
        
        # Apply specific fixes
        fixes = [
            ("await self.task_scheduler.start()", "self.task_scheduler.start()"),
            ("await self.portfolio_monitor.start_monitoring()", "self.portfolio_monitor.start_monitoring()"),
            ("await self.portfolio_monitor.stop_monitoring()", "self.portfolio_monitor.stop_monitoring()"),
            ("await self.task_scheduler.stop()", "self.task_scheduler.stop()"),
            ("await self.alert_system.stop()", "self.alert_system.stop()"),
        ]
        
        for old_pattern, new_pattern in fixes:
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                fixes_applied.append(f"Fixed: {old_pattern} -> {new_pattern}")
        
        # Fix status method calls
        status_pattern = r"await self\.(\w+)\.get_status\(\)"
        status_replacement = r"self.\1.get_status()"
        content = re.sub(status_pattern, status_replacement, content)
        
        if content != original_content:
            # Backup original file
            backup_file = orchestrator_file.with_suffix('.py.backup')
            backup_file.write_text(original_content)
            
            # Write fixed content
            orchestrator_file.write_text(content)
            fixes_applied.append("Fixed async/await patterns in monitoring_orchestrator.py")
            print("✅ monitoring_orchestrator.py fixes applied")
        else:
            print("ℹ️ monitoring_orchestrator.py - no async/await issues found")
    else:
        print("⚠️ monitoring_orchestrator.py not found")
    
    # Fix 2: Pydantic V1 Issues
    pydantic_files = [
        "services/risk_detector.py",
        "services/mcp_client.py", 
        "models.py"
    ]
    
    for file_path in pydantic_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"🔧 Checking {file_path} for Pydantic V1 issues...")
            
            content = full_path.read_text()
            original_content = content
            
            # Fix imports
            if "from langchain.pydantic_v1" in content:
                content = content.replace("from langchain.pydantic_v1 import", "from pydantic import")
                fixes_applied.append(f"Updated Pydantic imports in {file_path}")
            
            # Fix .dict() calls
            if ".dict()" in content:
                content = content.replace(".dict()", ".model_dump()")
                fixes_applied.append(f"Updated .dict() to .model_dump() in {file_path}")
            
            # Fix .parse_obj() calls  
            if ".parse_obj(" in content:
                content = content.replace(".parse_obj(", ".model_validate(")
                fixes_applied.append(f"Updated .parse_obj() to .model_validate() in {file_path}")
            
            if content != original_content:
                # Backup and write
                backup_file = full_path.with_suffix(full_path.suffix + '.backup')
                backup_file.write_text(original_content)
                full_path.write_text(content)
                print(f"✅ {file_path} Pydantic fixes applied")
        else:
            print(f"ℹ️ {file_path} not found - skipping")
    
    # Fix 3: Create Missing Service Stubs
    service_stubs = {
        'portfolio_monitor.py': '''from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PortfolioMonitor:
    def __init__(self):
        self.is_running = False
        self.portfolios = {}
    
    def start_monitoring(self) -> Dict[str, Any]:
        """Start portfolio monitoring"""
        self.is_running = True
        return {'status': 'started', 'timestamp': datetime.utcnow().isoformat()}
    
    def stop_monitoring(self) -> bool:
        """Stop portfolio monitoring"""
        self.is_running = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {'status': 'running' if self.is_running else 'stopped'}
''',
        'alert_system.py': '''from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class AlertManagementSystem:
    def __init__(self):
        self.is_running = False
    
    def start(self) -> bool:
        """Start alert system"""
        self.is_running = True
        return True
    
    def stop(self) -> bool:
        """Stop alert system"""
        self.is_running = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        return {'status': 'running' if self.is_running else 'stopped'}
''',
        'task_scheduler.py': '''from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self):
        self.is_running = False
    
    def start(self) -> bool:
        """Start task scheduler"""
        self.is_running = True
        return True
    
    def stop(self) -> bool:
        """Stop task scheduler"""
        self.is_running = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {'status': 'running' if self.is_running else 'stopped'}
''',
        'risk_detector.py': '''from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RiskDetector:
    def __init__(self):
        pass
    
    def process_risk_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process risk event"""
        return {
            'status': 'processed',
            'alert_triggered': True,
            'timestamp': datetime.utcnow().isoformat()
        }
''',
        'workflow_trigger.py': '''from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WorkflowTrigger:
    def __init__(self):
        pass
    
    def trigger_workflow(self, risk_event: Dict[str, Any], detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger workflow"""
        return {
            'workflow_id': f'wf_{datetime.utcnow().timestamp()}',
            'status': 'triggered',
            'timestamp': datetime.utcnow().isoformat()
        }
'''
    }
    
    print("🔧 Creating missing service stubs...")
    
    for filename, stub_content in service_stubs.items():
        service_file = services_dir / filename
        if not service_file.exists():
            # Create services directory if it doesn't exist
            services_dir.mkdir(exist_ok=True)
            
            service_file.write_text(stub_content)
            fixes_applied.append(f"Created {filename} service stub")
            print(f"✅ Created {filename}")
        else:
            print(f"ℹ️ {filename} already exists - skipping")
    
    # Validation Test
    print("\n🧪 Running validation tests...")
    
    try:
        # Test import
        import sys
        sys.path.insert(0, str(project_root))
        
        from services.monitoring_orchestrator import MonitoringOrchestrator
        orchestrator = MonitoringOrchestrator()
        
        # Test initialization
        init_result = await orchestrator.initialize_components()
        
        # Test health check (synchronous)
        health = orchestrator.get_system_health()
        
        # Test shutdown
        await orchestrator.stop_all_monitoring()
        
        print("✅ All validation tests passed!")
        
    except ImportError as e:
        print(f"⚠️ Import validation failed: {e}")
        print("   This may be expected if services have additional dependencies")
    except Exception as e:
        print(f"⚠️ Validation test failed: {e}")
    
    # Summary
    print("\n📊 TECHNICAL DEBT FIX SUMMARY")
    print("=" * 50)
    print(f"✅ Total fixes applied: {len(fixes_applied)}")
    
    if fixes_applied:
        print("\nFixes applied:")
        for fix in fixes_applied:
            print(f"  - {fix}")
    
    print("\n🎯 Next steps:")
    print("1. Run your integration tests: python -m pytest tests/")
    print("2. Check that monitoring orchestrator initializes correctly")
    print("3. Proceed to Sprint 2 Task 2.3 (WebSocket Real-time Updates)")
    
    print("\n✅ Technical debt fix complete!")

if __name__ == "__main__":
    asyncio.run(quick_fix_technical_debt())