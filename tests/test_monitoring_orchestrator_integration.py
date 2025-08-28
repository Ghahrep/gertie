# tests/test_monitoring_orchestrator_integration.py - CORRECTED VERSION
import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from services.monitoring_orchestrator import (
    MonitoringOrchestrator, 
    get_monitoring_orchestrator,
    monitoring_orchestrator_context
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMonitoringOrchestratorIntegration:
    """Integration tests with reduced mocking for better real-world validation"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create fresh orchestrator for each test"""
        return MonitoringOrchestrator()
    
    @pytest.mark.asyncio
    async def test_component_initialization_real(self, orchestrator):
        """Test component initialization with minimal mocking"""
        logger.info("🧪 Testing real component initialization...")
        
        # Only mock external dependencies, not core logic
        with patch('services.portfolio_monitor.PortfolioMonitor') as mock_pm_class, \
             patch('services.alert_system.AlertManagementSystem') as mock_alert_class, \
             patch('services.task_scheduler.TaskScheduler') as mock_scheduler_class, \
             patch('services.risk_detector.RiskDetector') as mock_risk_class, \
             patch('services.workflow_trigger.WorkflowTrigger') as mock_workflow_class:
            
            # Configure mocks to return actual instances
            mock_pm_class.return_value = Mock()
            mock_alert_class.return_value = Mock()
            mock_scheduler_class.return_value = Mock()
            mock_risk_class.return_value = Mock()
            mock_workflow_class.return_value = Mock()
            
            # Test actual initialization logic
            result = await orchestrator.initialize_components()
            
            # Verify initialization results
            assert isinstance(result, dict)
            assert 'portfolio_monitor' in result
            assert 'alert_system' in result
            assert 'task_scheduler' in result
            assert 'risk_detector' in result
            assert 'workflow_trigger' in result
            
            # Verify all components were created
            assert orchestrator.portfolio_monitor is not None
            assert orchestrator.alert_system is not None
            assert orchestrator.task_scheduler is not None
            assert orchestrator.risk_detector is not None
            assert orchestrator.workflow_trigger is not None
            
            logger.info("✅ Component initialization test passed")
    
    @pytest.mark.asyncio
    async def test_service_startup_sequence(self, orchestrator):
        """Test service startup with proper async/sync handling"""
        logger.info("🧪 Testing service startup sequence...")
        
        # Initialize components first
        await orchestrator.initialize_components()
        
        # Mock only the service methods, not the whole objects
        orchestrator.task_scheduler.start = Mock(return_value=True)
        orchestrator.portfolio_monitor.start_monitoring = Mock(return_value={'status': 'started'})
        orchestrator.alert_system.start = Mock(return_value=True)
        
        # Test startup sequence
        result = await orchestrator.start_monitoring_services()
        
        # Verify startup results
        assert isinstance(result, dict)
        assert result.get('task_scheduler') == 'started'
        assert result.get('portfolio_monitor') == 'started'
        assert result.get('alert_system') == 'started'
        
        # Verify methods were called correctly (not awaited incorrectly)
        orchestrator.task_scheduler.start.assert_called_once()
        orchestrator.portfolio_monitor.start_monitoring.assert_called_once()
        orchestrator.alert_system.start.assert_called_once()
        
        logger.info("✅ Service startup test passed")
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_startup(self, orchestrator):
        """Test complete monitoring system startup"""
        logger.info("🧪 Testing complete monitoring startup...")
        
        # Mock external dependencies
        with patch.object(orchestrator, 'initialize_components') as mock_init, \
             patch.object(orchestrator, 'start_monitoring_services') as mock_start:
            
            # Configure successful responses
            mock_init.return_value = {
                'portfolio_monitor': 'initialized',
                'alert_system': 'initialized',
                'task_scheduler': 'initialized',
                'risk_detector': 'initialized',
                'workflow_trigger': 'initialized'
            }
            
            mock_start.return_value = {
                'portfolio_monitor': 'started',
                'alert_system': 'started',
                'task_scheduler': 'started'
            }
            
            # Test complete startup
            result = await orchestrator.start_complete_monitoring()
            
            # Verify success response
            assert result['status'] == 'success'
            assert 'All monitoring services operational' in result['message']
            assert 'initialization_results' in result
            assert 'start_results' in result
            assert 'timestamp' in result
            
            logger.info("✅ Complete monitoring startup test passed")
    
    @pytest.mark.asyncio
    async def test_health_check_synchronous(self, orchestrator):
        """Test health check is properly synchronous"""
        logger.info("🧪 Testing synchronous health check...")
        
        # Initialize components
        await orchestrator.initialize_components()
        
        # Mock component status methods as synchronous
        orchestrator.portfolio_monitor.get_status = Mock(return_value={'status': 'running'})
        orchestrator.alert_system.get_status = Mock(return_value={'status': 'running'})
        orchestrator.task_scheduler.get_status = Mock(return_value={'status': 'running'})
        
        # Test health check (should be synchronous, not async)
        health = orchestrator.get_system_health()  # Not awaited!
        
        # Verify response structure
        assert isinstance(health, dict)
        assert 'overall_status' in health
        assert 'components' in health
        assert 'is_running' in health
        assert 'timestamp' in health
        
        # Verify component statuses were checked
        orchestrator.portfolio_monitor.get_status.assert_called_once()
        orchestrator.alert_system.get_status.assert_called_once()
        orchestrator.task_scheduler.get_status.assert_called_once()
        
        logger.info("✅ Health check synchronous test passed")
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, orchestrator):
        """Test graceful shutdown process"""
        logger.info("🧪 Testing graceful shutdown...")
        
        # Initialize and start services
        await orchestrator.initialize_components()
        orchestrator._is_running = True
        
        # Mock stop methods as synchronous
        orchestrator.task_scheduler.stop = Mock()
        orchestrator.portfolio_monitor.stop_monitoring = Mock()
        orchestrator.alert_system.stop = Mock()
        
        # Test shutdown
        result = await orchestrator.stop_all_monitoring()
        
        # Verify all services stopped
        assert isinstance(result, dict)
        assert result.get('task_scheduler') == 'stopped'
        assert result.get('portfolio_monitor') == 'stopped'
        assert result.get('alert_system') == 'stopped'
        
        # Verify orchestrator state
        assert not orchestrator._is_running
        assert orchestrator._health_status['orchestrator'] == 'stopped'
        
        # Verify stop methods were called
        orchestrator.task_scheduler.stop.assert_called_once()
        orchestrator.portfolio_monitor.stop_monitoring.assert_called_once()
        orchestrator.alert_system.stop.assert_called_once()
        
        logger.info("✅ Graceful shutdown test passed")
    
    @pytest.mark.asyncio
    async def test_risk_event_processing(self, orchestrator):
        """Test risk event processing workflow"""
        logger.info("🧪 Testing risk event processing...")
        
        # Initialize components
        await orchestrator.initialize_components()
        
        # Mock risk detector and workflow trigger
        mock_detection_result = {
            'alert_triggered': True,
            'risk_level': 'high',
            'portfolio_id': 'test_portfolio'
        }
        
        mock_workflow_result = {
            'workflow_id': 'wf_123',
            'status': 'triggered'
        }
        
        orchestrator.risk_detector.process_risk_event = Mock(return_value=mock_detection_result)
        orchestrator.workflow_trigger.trigger_workflow = Mock(return_value=mock_workflow_result)
        
        # Test risk event processing
        risk_event = {
            'portfolio_id': 'test_portfolio',
            'risk_type': 'volatility',
            'severity': 'high'
        }
        
        result = await orchestrator.process_risk_event(risk_event)
        
        # Verify processing results
        assert result['status'] == 'success'
        assert result['risk_processed'] is True
        assert result['workflow_triggered'] is True
        assert result['workflow_result'] == mock_workflow_result
        
        # Verify methods were called
        orchestrator.risk_detector.process_risk_event.assert_called_once_with(risk_event)
        orchestrator.workflow_trigger.trigger_workflow.assert_called_once()
        
        logger.info("✅ Risk event processing test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling_in_initialization(self, orchestrator):
        """Test error handling during component initialization"""
        logger.info("🧪 Testing error handling in initialization...")
        
        # Mock component class to raise exception
        with patch('services.portfolio_monitor.PortfolioMonitor', side_effect=Exception("Connection failed")):
            result = await orchestrator.initialize_components()
            
            # Should handle error gracefully
            assert isinstance(result, dict)
            assert 'orchestrator' in result
            assert 'initialization_error' in result['orchestrator']
            assert orchestrator._health_status['orchestrator'] == 'error'
        
        logger.info("✅ Error handling test passed")
    
    @pytest.mark.asyncio
    async def test_partial_service_failure(self, orchestrator):
        """Test handling of partial service failures"""
        logger.info("🧪 Testing partial service failure handling...")
        
        # Initialize components
        await orchestrator.initialize_components()
        
        # Mock one service to fail
        orchestrator.task_scheduler.start = Mock(return_value=True)
        orchestrator.portfolio_monitor.start_monitoring = Mock(side_effect=Exception("Service unavailable"))
        orchestrator.alert_system.start = Mock(return_value=True)
        
        # Test startup with one failure
        result = await orchestrator.start_monitoring_services()
        
        # Should handle partial failure gracefully
        assert isinstance(result, dict)
        assert result.get('task_scheduler') == 'started'
        assert 'error' in result.get('portfolio_monitor', '')
        assert result.get('alert_system') == 'started'
        
        logger.info("✅ Partial service failure test passed")
    
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self):
        """Test orchestrator context manager lifecycle"""
        logger.info("🧪 Testing context manager lifecycle...")
        
        # Mock startup and shutdown
        with patch('services.monitoring_orchestrator.MonitoringOrchestrator') as MockOrchestrator:
            mock_instance = MockOrchestrator.return_value
            mock_instance.start_complete_monitoring = AsyncMock(return_value={'status': 'success'})
            mock_instance.stop_all_monitoring = AsyncMock()
            
            # Test context manager
            async with monitoring_orchestrator_context() as orchestrator:
                assert orchestrator is not None
                mock_instance.start_complete_monitoring.assert_called_once()
            
            # Verify cleanup was called
            mock_instance.stop_all_monitoring.assert_called_once()
        
        logger.info("✅ Context manager lifecycle test passed")
    
    @pytest.mark.asyncio
    async def test_global_orchestrator_singleton(self):
        """Test global orchestrator singleton pattern"""
        logger.info("🧪 Testing global orchestrator singleton...")
        
        # Clear global instance
        import services.monitoring_orchestrator
        services.monitoring_orchestrator._global_orchestrator = None
        
        # Get orchestrator twice
        orchestrator1 = await get_monitoring_orchestrator()
        orchestrator2 = await get_monitoring_orchestrator()
        
        # Should be same instance
        assert orchestrator1 is orchestrator2
        assert orchestrator1 is not None
        
        logger.info("✅ Global orchestrator singleton test passed")
    
    @pytest.mark.asyncio
    async def test_getter_methods_safety(self, orchestrator):
        """Test getter methods return safely"""
        logger.info("🧪 Testing getter methods safety...")
        
        # Test getters before initialization
        assert orchestrator.get_portfolio_monitor() is None
        assert orchestrator.get_alert_management_system() is None
        assert orchestrator.get_task_scheduler() is None
        assert orchestrator.get_risk_detector() is None
        assert orchestrator.get_workflow_trigger() is None
        
        # Initialize components
        await orchestrator.initialize_components()
        
        # Test getters after initialization
        assert orchestrator.get_portfolio_monitor() is not None
        assert orchestrator.get_alert_management_system() is not None
        assert orchestrator.get_task_scheduler() is not None
        assert orchestrator.get_risk_detector() is not None
        assert orchestrator.get_workflow_trigger() is not None
        
        logger.info("✅ Getter methods safety test passed")


# Test runner for quick validation
async def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Running Monitoring Orchestrator Integration Tests...")
    
    test_instance = TestMonitoringOrchestratorIntegration()
    orchestrator = MonitoringOrchestrator()
    
    try:
        # Run key tests
        await test_instance.test_component_initialization_real(orchestrator)
        await test_instance.test_service_startup_sequence(orchestrator)
        await test_instance.test_health_check_synchronous(orchestrator)
        await test_instance.test_graceful_shutdown(orchestrator)
        await test_instance.test_error_handling_in_initialization(orchestrator)
        
        print("✅ All integration tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_integration_tests())