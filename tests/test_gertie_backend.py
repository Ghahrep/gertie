# Complete Gertie Backend Unit Test Suite
# =======================================

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional
import json

# Test imports for your services
from services.proactive_monitor import (
    ProactiveRiskMonitor, 
    AlertPriority, 
    AlertType, 
    MonitoringAlert,
    get_proactive_monitor
)
from services.risk_attribution_service import (
    RiskAttributionService,
    get_risk_attribution_service,
    calculate_and_store_portfolio_risk,
    trigger_risk_monitoring_for_user
)
from services.risk_detector import (
    RiskDetectorService,
    RiskChangeAnalysis,
    create_risk_detector
)
from services.risk_notification_service import (
    RiskNotificationService,
    get_risk_notification_service,
    notify_threshold_breach,
    notify_workflow_started
)
from services.portfolio_monitor_service import PortfolioMonitorService
from services.risk_calculator import RiskCalculatorService, RiskMetrics


class TestProactiveRiskMonitor:
    """Test suite for ProactiveRiskMonitor"""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCP client"""
        mock_client = AsyncMock()
        mock_client.submit_job = AsyncMock(return_value={"job_id": "test_job_123"})
        return mock_client
    
    @pytest.fixture
    def proactive_monitor(self):
        """Create ProactiveRiskMonitor instance"""
        return ProactiveRiskMonitor()
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Sample portfolio data for testing"""
        return {
            "portfolio_id": "portfolio_123",
            "user_id": "user_456",
            "name": "Test Portfolio",
            "total_value": 100000.0,
            "positions": [
                {"symbol": "AAPL", "weight": 0.6, "shares": 100},
                {"symbol": "MSFT", "weight": 0.4, "shares": 50}
            ]
        }

    @pytest.mark.asyncio
    async def test_start_portfolio_monitoring_success(self, proactive_monitor):
        """Test starting portfolio monitoring"""
        
        result = await proactive_monitor.start_portfolio_monitoring("portfolio_123", "user_456")
        
        assert result["status"] == "started"
        assert result["portfolio_id"] == "portfolio_123"
        assert "monitoring_intervals" in result
        assert "alert_thresholds" in result
        assert "portfolio_123" in proactive_monitor.active_monitors

    @pytest.mark.asyncio
    async def test_start_portfolio_monitoring_already_active(self, proactive_monitor):
        """Test starting monitoring for already monitored portfolio"""
        
        # Start monitoring first time
        await proactive_monitor.start_portfolio_monitoring("portfolio_123", "user_456")
        
        # Try to start again
        result = await proactive_monitor.start_portfolio_monitoring("portfolio_123", "user_456")
        
        assert result["status"] == "already_active"
        assert result["portfolio_id"] == "portfolio_123"

    @pytest.mark.asyncio
    async def test_stop_portfolio_monitoring_success(self, proactive_monitor):
        """Test stopping portfolio monitoring"""
        
        # Start monitoring first
        await proactive_monitor.start_portfolio_monitoring("portfolio_123", "user_456")
        
        # Stop monitoring
        result = await proactive_monitor.stop_portfolio_monitoring("portfolio_123")
        
        assert result["status"] == "stopped"
        assert result["portfolio_id"] == "portfolio_123"
        assert "portfolio_123" not in proactive_monitor.active_monitors

    @pytest.mark.asyncio
    async def test_stop_portfolio_monitoring_not_active(self, proactive_monitor):
        """Test stopping monitoring for non-monitored portfolio"""
        
        result = await proactive_monitor.stop_portfolio_monitoring("portfolio_123")
        
        assert result["status"] == "not_active"
        assert result["portfolio_id"] == "portfolio_123"

    @pytest.mark.asyncio
    async def test_check_risk_thresholds_with_risk_detector(self, proactive_monitor):
        """Test risk threshold checking with risk detector"""
        
        # Mock risk detector
        mock_risk_detector = MagicMock()
        mock_analysis = MagicMock()
        mock_analysis.threshold_breached = True
        mock_analysis.risk_direction = "INCREASED"
        mock_analysis.risk_magnitude_pct = 25.0
        mock_analysis.should_trigger_workflow = True
        
        mock_risk_detector.detect_portfolio_risk_changes = AsyncMock(return_value=mock_analysis)
        mock_risk_detector.integrate_with_proactive_monitor = AsyncMock(return_value=[
            {
                'portfolio_id': 'portfolio_123',
                'alert_type': AlertType.VAR_BREACH,
                'priority': AlertPriority.HIGH,
                'message': 'Risk threshold breached',
                'details': {'risk_change': 25.0}
            }
        ])
        
        # Mock the lazy loading of risk detector
        with patch.object(proactive_monitor, '_get_risk_detector', return_value=mock_risk_detector):
            with patch.object(proactive_monitor, '_create_alert') as mock_create_alert:
                with patch.object(proactive_monitor, '_send_alert') as mock_send_alert:
                    with patch.object(proactive_monitor, '_trigger_risk_workflow') as mock_trigger_workflow:
                        
                        await proactive_monitor._check_risk_thresholds("portfolio_123", "user_456")
                        
                        # Verify risk detector was called
                        mock_risk_detector.detect_portfolio_risk_changes.assert_called_once()
                        
                        # Verify alert creation and sending
                        mock_create_alert.assert_called()
                        mock_send_alert.assert_called()
                        
                        # Verify workflow was triggered
                        mock_trigger_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_risk_thresholds_no_breach(self, proactive_monitor):
        """Test risk threshold checking with no breach"""
        
        mock_risk_detector = MagicMock()
        mock_analysis = MagicMock()
        mock_analysis.threshold_breached = False
        mock_analysis.should_trigger_workflow = False
        
        mock_risk_detector.detect_portfolio_risk_changes = AsyncMock(return_value=mock_analysis)
        
        with patch.object(proactive_monitor, '_get_risk_detector', return_value=mock_risk_detector):
            with patch.object(proactive_monitor, '_trigger_risk_workflow') as mock_trigger_workflow:
                
                await proactive_monitor._check_risk_thresholds("portfolio_123", "user_456")
                
                # Verify workflow was not triggered
                mock_trigger_workflow.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_risk_workflow(self, proactive_monitor, mock_mcp_client):
        """Test triggering risk workflow"""
        
        mock_analysis = MagicMock()
        mock_analysis.risk_direction = "INCREASED"
        mock_analysis.risk_magnitude_pct = 30.0
        mock_analysis.significant_changes = {"volatility": 25.0, "risk_score": 30.0}
        mock_analysis.recommendation = "Consider reducing position sizes"
        mock_analysis.current_metrics = MagicMock()
        mock_analysis.current_metrics.risk_score = 75.0
        
        with patch('services.proactive_monitor.get_mcp_client', return_value=mock_mcp_client):
            with patch.object(proactive_monitor, '_create_alert') as mock_create_alert:
                with patch.object(proactive_monitor, '_send_alert') as mock_send_alert:
                    
                    await proactive_monitor._trigger_risk_workflow("portfolio_123", "user_456", mock_analysis)
                    
                    # Verify MCP job was submitted
                    mock_mcp_client.submit_job.assert_called_once()
                    
                    # Verify alert was created and sent
                    mock_create_alert.assert_called_once()
                    mock_send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_alert(self, proactive_monitor):
        """Test alert creation"""
        
        alert = await proactive_monitor._create_alert(
            portfolio_id="portfolio_123",
            user_id="user_456",
            alert_type=AlertType.VAR_BREACH,
            priority=AlertPriority.HIGH,
            message="Test alert",
            details={"test": "data"}
        )
        
        assert isinstance(alert, MonitoringAlert)
        assert alert.portfolio_id == "portfolio_123"
        assert alert.user_id == "user_456"
        assert alert.alert_type == AlertType.VAR_BREACH
        assert alert.priority == AlertPriority.HIGH
        assert alert.message == "Test alert"
        assert alert.details == {"test": "data"}
        assert not alert.resolved

    @pytest.mark.asyncio
    async def test_send_alert(self, proactive_monitor):
        """Test alert sending and storage"""
        
        alert = MonitoringAlert(
            alert_id="test_alert_123",
            portfolio_id="portfolio_123",
            user_id="user_456",
            alert_type=AlertType.VAR_BREACH,
            priority=AlertPriority.HIGH,
            message="Test alert",
            details={"test": "data"},
            timestamp=datetime.utcnow()
        )
        
        await proactive_monitor._send_alert(alert)
        
        # Verify alert was added to history
        assert len(proactive_monitor.alert_history) == 1
        assert proactive_monitor.alert_history[0] == alert

    def test_should_send_alert_rate_limiting(self, proactive_monitor):
        """Test alert rate limiting"""
        
        alert_key = "portfolio_123_var_breach"
        
        # First alert should be allowed
        assert proactive_monitor._should_send_alert(alert_key) is True
        
        # Second alert immediately should be blocked
        assert proactive_monitor._should_send_alert(alert_key) is False
        
        # Simulate time passing (mock the cooldown)
        past_time = datetime.utcnow() - timedelta(hours=1)
        proactive_monitor.last_alert_times[alert_key] = past_time
        
        # Should be allowed again after cooldown
        assert proactive_monitor._should_send_alert(alert_key) is True

    def test_get_monitoring_stats(self, proactive_monitor):
        """Test monitoring statistics"""
        
        # Add some test alerts
        test_alerts = [
            MonitoringAlert(
                alert_id=f"alert_{i}",
                portfolio_id="portfolio_123",
                alert_type=AlertType.VAR_BREACH,
                priority=AlertPriority.HIGH if i % 2 == 0 else AlertPriority.MEDIUM,
                message=f"Test alert {i}",
                details={},
                timestamp=datetime.utcnow() - timedelta(minutes=i*10)
            )
            for i in range(5)
        ]
        proactive_monitor.alert_history.extend(test_alerts)
        
        # Add an active monitor
        proactive_monitor.active_monitors["portfolio_123"] = MagicMock()
        
        stats = proactive_monitor.get_monitoring_stats()
        
        assert stats["active_monitors"] == 1
        assert stats["total_alerts"] == 5
        assert "alerts_by_priority" in stats
        assert "monitoring_intervals" in stats
        assert "alert_thresholds" in stats

    def test_get_portfolio_alerts(self, proactive_monitor):
        """Test getting alerts for specific portfolio"""
        
        # Add alerts for different portfolios
        alerts = [
            MonitoringAlert(
                alert_id=f"alert_{i}",
                portfolio_id="portfolio_123" if i < 3 else "portfolio_456",
                alert_type=AlertType.VAR_BREACH,
                priority=AlertPriority.HIGH,
                message=f"Test alert {i}",
                details={},
                timestamp=datetime.utcnow() - timedelta(minutes=i*10)
            )
            for i in range(5)
        ]
        proactive_monitor.alert_history.extend(alerts)
        
        portfolio_alerts = proactive_monitor.get_portfolio_alerts("portfolio_123")
        
        assert len(portfolio_alerts) == 3
        assert all(alert["portfolio_id"] == "portfolio_123" for alert in portfolio_alerts)

    @pytest.mark.asyncio
    async def test_resolve_alert(self, proactive_monitor):
        """Test resolving alerts"""
        
        alert = MonitoringAlert(
            alert_id="test_alert_123",
            portfolio_id="portfolio_123",
            alert_type=AlertType.VAR_BREACH,
            priority=AlertPriority.HIGH,
            message="Test alert",
            details={},
            timestamp=datetime.utcnow()
        )
        proactive_monitor.alert_history.append(alert)
        
        # Resolve the alert
        result = await proactive_monitor.resolve_alert("test_alert_123")
        
        assert result is True
        assert alert.resolved is True

    @pytest.mark.asyncio
    async def test_manual_risk_check(self, proactive_monitor):
        """Test manual risk check functionality"""
        
        mock_risk_detector = MagicMock()
        mock_risk_detector.detect_portfolio_risk_changes = AsyncMock(return_value=MagicMock())
        
        with patch.object(proactive_monitor, '_get_risk_detector', return_value=mock_risk_detector):
            with patch.object(proactive_monitor, '_perform_real_risk_detection') as mock_perform:
                
                result = await proactive_monitor.manual_risk_check("portfolio_123", "user_456")
                
                assert result["status"] == "completed"
                assert result["portfolio_id"] == "portfolio_123"
                mock_perform.assert_called_once()


class TestRiskDetectorService:
    """Test suite for RiskDetectorService"""
    
    @pytest.fixture
    def risk_detector(self):
        """Create RiskDetectorService instance"""
        return create_risk_detector(threshold_pct=15.0)
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return MagicMock()
    
    @pytest.fixture
    def sample_risk_metrics(self):
        """Sample risk metrics for testing"""
        return RiskMetrics(
            volatility=0.18,
            beta=1.2,
            max_drawdown=0.12,
            var_95=0.025,
            var_99=0.035,
            cvar_95=0.030,
            cvar_99=0.040,
            sharpe_ratio=1.1,
            sortino_ratio=1.3,
            calmar_ratio=2.0,
            hurst_exponent=0.55,
            dfa_alpha=0.62,
            risk_score=65.0,
            sentiment_index=70,
            regime_volatility=0.16,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_portfolio(self):
        """Sample portfolio with holdings"""
        mock_portfolio = MagicMock()
        mock_portfolio.id = 123
        
        # Mock holdings
        mock_holding1 = MagicMock()
        mock_holding1.shares = 100
        mock_holding1.purchase_price = 150.0
        mock_holding1.asset.ticker = "AAPL"
        
        mock_holding2 = MagicMock()
        mock_holding2.shares = 50
        mock_holding2.purchase_price = 300.0
        mock_holding2.asset.ticker = "MSFT"
        
        mock_portfolio.holdings = [mock_holding1, mock_holding2]
        return mock_portfolio

    @pytest.mark.asyncio
    async def test_detect_portfolio_risk_changes_with_previous_data(
        self, risk_detector, mock_db_session, sample_risk_metrics, sample_portfolio
    ):
        """Test risk change detection with previous data"""
        
        # Mock database queries
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_portfolio
        
        # Mock previous snapshot
        mock_previous_snapshot = MagicMock()
        mock_previous_snapshot.id = 456
        mock_previous_snapshot.volatility = 0.15  # Lower than current
        mock_previous_snapshot.beta = 1.0
        mock_previous_snapshot.risk_score = 50.0  # Significant increase to 65.0
        mock_previous_snapshot.max_drawdown = 0.10
        mock_previous_snapshot.var_95 = 0.020
        mock_previous_snapshot.var_99 = 0.030
        mock_previous_snapshot.cvar_95 = 0.025
        mock_previous_snapshot.cvar_99 = 0.035
        mock_previous_snapshot.sharpe_ratio = 1.2
        mock_previous_snapshot.sortino_ratio = 1.4
        mock_previous_snapshot.calmar_ratio = 2.2
        mock_previous_snapshot.sentiment_index = 75
        mock_previous_snapshot.snapshot_date = datetime.now() - timedelta(hours=1)
        
        with patch.object(risk_detector, '_get_latest_risk_snapshot', return_value=mock_previous_snapshot):
            with patch.object(risk_detector, '_calculate_current_risk_metrics', return_value=sample_risk_metrics):
                with patch.object(risk_detector, '_store_risk_snapshot', return_value=789):
                    with patch.object(risk_detector, '_store_risk_change_event', return_value=101):
                        
                        result = await risk_detector.detect_portfolio_risk_changes(123, 456, mock_db_session)
                        
                        assert result is not None
                        assert isinstance(result, RiskChangeAnalysis)
                        assert result.portfolio_id == "123"
                        assert result.risk_direction == "INCREASED"
                        assert result.threshold_breached is True
                        assert "risk_score" in result.significant_changes

    @pytest.mark.asyncio
    async def test_detect_portfolio_risk_changes_no_previous_data(
        self, risk_detector, mock_db_session, sample_risk_metrics, sample_portfolio
    ):
        """Test risk change detection with no previous data (baseline)"""
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_portfolio
        
        with patch.object(risk_detector, '_get_latest_risk_snapshot', return_value=None):
            with patch.object(risk_detector, '_calculate_current_risk_metrics', return_value=sample_risk_metrics):
                with patch.object(risk_detector, '_store_risk_snapshot', return_value=789):
                    
                    result = await risk_detector.detect_portfolio_risk_changes(123, 456, mock_db_session)
                    
                    assert result is not None
                    assert result.risk_direction == "BASELINE"
                    assert result.threshold_breached is False
                    assert result.recommendation == "Baseline risk metrics established"

    @pytest.mark.asyncio
    async def test_calculate_current_risk_metrics_success(
        self, risk_detector, mock_db_session, sample_portfolio, sample_risk_metrics
    ):
        """Test successful current risk metrics calculation"""
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_portfolio
        
        with patch.object(risk_detector.risk_calculator, 'calculate_portfolio_risk_from_tickers', return_value=sample_risk_metrics):
            
            result = await risk_detector._calculate_current_risk_metrics(123, 456, mock_db_session)
            
            assert result == sample_risk_metrics

    @pytest.mark.asyncio
    async def test_calculate_current_risk_metrics_no_portfolio(
        self, risk_detector, mock_db_session
    ):
        """Test risk metrics calculation when portfolio not found"""
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        result = await risk_detector._calculate_current_risk_metrics(123, 456, mock_db_session)
        
        assert result is None

    def test_analyze_risk_changes_significant_increase(self, risk_detector, sample_risk_metrics):
        """Test risk change analysis with significant increase"""
        
        # Create previous metrics with lower risk
        previous_metrics = RiskMetrics(
            volatility=0.12,      # 50% increase to 0.18
            beta=1.0,            # 20% increase to 1.2
            max_drawdown=0.08,   # 50% increase to 0.12
            var_95=0.020,
            var_99=0.025,
            cvar_95=0.025,
            cvar_99=0.030,
            sharpe_ratio=1.3,
            sortino_ratio=1.5,
            calmar_ratio=2.5,
            hurst_exponent=0.5,
            dfa_alpha=0.6,
            risk_score=45.0,     # 44% increase to 65.0
            sentiment_index=80,   # 12.5% decrease to 70
            regime_volatility=0.10,
            timestamp=datetime.now() - timedelta(hours=1)
        )
        
        result = risk_detector._analyze_risk_changes(sample_risk_metrics, previous_metrics, 123)
        
        assert result.risk_direction == "INCREASED"
        assert result.threshold_breached is True
        assert len(result.significant_changes) > 0
        assert "volatility" in result.significant_changes
        assert "risk_score" in result.significant_changes

    def test_analyze_risk_changes_no_significant_change(self, risk_detector, sample_risk_metrics):
        """Test risk change analysis with no significant changes"""
        
        # Create previous metrics very similar to current
        previous_metrics = RiskMetrics(
            volatility=0.175,     # Small change
            beta=1.15,           # Small change
            max_drawdown=0.115,  # Small change
            var_95=0.024,
            var_99=0.034,
            cvar_95=0.029,
            cvar_99=0.039,
            sharpe_ratio=1.12,
            sortino_ratio=1.32,
            calmar_ratio=1.95,
            hurst_exponent=0.54,
            dfa_alpha=0.61,
            risk_score=63.0,     # Small change
            sentiment_index=72,   # Small change
            regime_volatility=0.155,
            timestamp=datetime.now() - timedelta(hours=1)
        )
        
        result = risk_detector._analyze_risk_changes(sample_risk_metrics, previous_metrics, 123)
        
        assert result.threshold_breached is False
        assert len(result.significant_changes) == 0

    @pytest.mark.asyncio
    async def test_integrate_with_proactive_monitor(self, risk_detector, sample_risk_metrics):
        """Test integration with proactive monitor"""
        
        # Create analysis with threshold breach
        analysis = RiskChangeAnalysis(
            portfolio_id="123",
            current_metrics=sample_risk_metrics,
            previous_metrics=None,
            risk_direction="INCREASED",
            risk_magnitude_pct=25.0,
            threshold_breached=True,
            significant_changes={"volatility": 30.0, "risk_score": 25.0},
            all_changes={},
            recommendation="Risk increased significantly",
            should_trigger_workflow=True
        )
        
        mock_monitor = MagicMock()
        
        alerts = await risk_detector.integrate_with_proactive_monitor(analysis, mock_monitor)
        
        assert len(alerts) == 2  # One for volatility, one for risk_score
        assert all("portfolio_id" in alert for alert in alerts)
        assert all("alert_type" in alert for alert in alerts)
        assert all("priority" in alert for alert in alerts)

    def test_metric_change_to_alert_volatility_spike(self, risk_detector):
        """Test converting volatility change to alert"""
        
        alert_type, priority, message = risk_detector._metric_change_to_alert(
            "volatility", 35.0, "INCREASED"
        )
        
        assert alert_type == AlertType.VOLATILITY_SPIKE
        assert priority == AlertPriority.CRITICAL  # >30% change
        assert "volatility" in message.lower()
        assert "+35.0%" in message

    def test_metric_change_to_alert_no_increase(self, risk_detector):
        """Test that no alert is generated for decreases"""
        
        alert_type, priority, message = risk_detector._metric_change_to_alert(
            "volatility", 25.0, "DECREASED"
        )
        
        assert alert_type is None


class TestRiskAttributionService:
    """Test suite for RiskAttributionService"""
    
    @pytest.fixture
    def mock_risk_calculator(self):
        """Mock risk calculator"""
        mock_calc = MagicMock()
        mock_calc.calculate_portfolio_risk = AsyncMock(return_value={
            'status': 'success',
            'risk_score': 65.0,
            'volatility': 0.18,
            'beta': 1.2
        })
        return mock_calc
    
    @pytest.fixture
    def mock_proactive_monitor(self):
        """Mock proactive monitor"""
        mock_monitor = AsyncMock()
        mock_monitor.trigger_risk_workflow = AsyncMock(return_value={'workflow_id': 'workflow_123'})
        return mock_monitor
    
    @pytest.fixture
    async def risk_attribution_service(self, mock_risk_calculator, mock_proactive_monitor):
        """Create RiskAttributionService instance"""
        service = RiskAttributionService()
        service.risk_calculator = mock_risk_calculator
        service.proactive_monitor = mock_proactive_monitor
        service._initialized = True
        return service
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Sample portfolio data"""
        return {
            "name": "Test Portfolio",
            "total_value": 100000,
            "positions": [
                {"symbol": "AAPL", "weight": 0.6, "shares": 100},
                {"symbol": "MSFT", "weight": 0.4, "shares": 50}
            ]
        }

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful service initialization"""
        
        service = RiskAttributionService()
        
        with patch('services.risk_attribution_service.get_proactive_monitor') as mock_get_monitor:
            mock_get_monitor.return_value = MagicMock()
            
            result = await service.initialize()
            
            assert result is True
            assert service._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test service initialization failure"""
        
        service = RiskAttributionService()
        
        with patch('services.risk_attribution_service.get_proactive_monitor') as mock_get_monitor:
            mock_get_monitor.side_effect = Exception("Connection failed")
            
            result = await service.initialize()
            
            assert result is False
            assert service._initialized is False

    @pytest.mark.asyncio
    async def test_create_and_store_risk_snapshot_success(
        self, risk_attribution_service, sample_portfolio_data
    ):
        """Test successful risk snapshot creation and storage"""
        
        # Mock database operations
        mock_db = MagicMock()
        mock_snapshot = MagicMock()
        mock_snapshot.snapshot_id = 123
        mock_snapshot.risk_score = 65.0
        mock_snapshot.is_threshold_breach = False
        mock_snapshot.to_dict.return_value = {"risk_score": 65.0, "volatility": 0.18}
        
        with patch('services.risk_attribution_service.get_db') as mock_get_db:
            mock_get_db.return_value = iter([mock_db])
            
            with patch('services.risk_attribution_service.create_risk_snapshot') as mock_create:
                mock_create.return_value = mock_snapshot
                
                result = await risk_attribution_service.create_and_store_risk_snapshot(
                    "user_123", "portfolio_456", sample_portfolio_data
                )
                
                assert result['status'] == 'success'
                assert result['threshold_breach'] is False
                assert 'snapshot' in result

    @pytest.mark.asyncio
    async def test_create_and_store_risk_snapshot_with_threshold_breach(
        self, risk_attribution_service, sample_portfolio_data
    ):
        """Test risk snapshot creation with threshold breach"""
        
        # Mock database operations
        mock_db = MagicMock()
        mock_snapshot = MagicMock()
        mock_snapshot.snapshot_id = 123
        mock_snapshot.risk_score = 85.0
        mock_snapshot.risk_score_change_pct = 25.0
        mock_snapshot.is_threshold_breach = True
        mock_snapshot.to_dict.return_value = {"risk_score": 85.0, "volatility": 0.25}
        
        mock_alert = MagicMock()
        mock_alert.workflow_id = None
        
        with patch('services.risk_attribution_service.get_db') as mock_get_db:
            mock_get_db.return_value = iter([mock_db])
            
            with patch('services.risk_attribution_service.create_risk_snapshot') as mock_create:
                with patch('services.risk_attribution_service.log_risk_alert') as mock_log_alert:
                    mock_create.return_value = mock_snapshot
                    mock_log_alert.return_value = mock_alert
                    
                    result = await risk_attribution_service.create_and_store_risk_snapshot(
                        "user_123", "portfolio_456", sample_portfolio_data
                    )
                    
                    assert result['status'] == 'success'
                    assert result['threshold_breach'] is True
                    assert result['alert_triggered'] is True
                    
                    # Verify alert was logged
                    mock_log_alert.assert_called_once()
                    
                    # Verify workflow was triggered
                    risk_attribution_service.proactive_monitor.trigger_risk_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_and_store_risk_snapshot_calculation_failure(
        self, risk_attribution_service, sample_portfolio_data
    ):
        """Test risk snapshot creation with calculation failure"""
        
        # Mock failed risk calculation
        risk_attribution_service.risk_calculator.calculate_portfolio_risk.return_value = {
            'status': 'error',
            'error': 'Failed to fetch market data'
        }
        
        result = await risk_attribution_service.create_and_store_risk_snapshot(
            "user_123", "portfolio_456", sample_portfolio_data
        )
        
        assert result['status'] == 'error'
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_monitor_portfolio_risk_changes(self, risk_attribution_service):
        """Test monitoring multiple portfolios for risk changes"""
        
        user_portfolios = {
            "user_123": [
                {"id": "portfolio_1", "name": "Portfolio 1"},
                {"id": "portfolio_2", "name": "Portfolio 2"}
            ]
        }
        
        # Mock successful risk calculations
        mock_results = [
            {'status': 'success', 'portfolio_id': 'portfolio_1'},
            {'status': 'success', 'portfolio_id': 'portfolio_2'}
        ]
        
        with patch.object(
            risk_attribution_service, 
            'create_and_store_risk_snapshot',
            side_effect=mock_results
        ):
            results = await risk_attribution_service.monitor_portfolio_risk_changes(user_portfolios)
            
            assert len(results) == 2
            assert all(result['status'] == 'success' for result in results)

    def test_get_risk_dashboard_data_success(self, risk_attribution_service):
        """Test getting risk dashboard data"""
        
        mock_db = MagicMock()
        mock_rankings = [{"portfolio_id": "123", "risk_score": 65.0}]
        mock_alerts = [MagicMock()]
        mock_breaches = [MagicMock(), MagicMock()]
        
        with patch('services.risk_attribution_service.get_db') as mock_get_db:
            mock_get_db.return_value = iter([mock_db])
            
            with patch('services.risk_attribution_service.get_portfolio_rankings') as mock_get_rankings:
                with patch('services.risk_attribution_service.get_recent_alerts') as mock_get_alerts:
                    with patch('services.risk_attribution_service.get_threshold_breaches') as mock_get_breaches:
                        mock_get_rankings.return_value = mock_rankings
                        mock_get_alerts.return_value = mock_alerts
                        mock_get_breaches.return_value = mock_breaches
                        
                        result = risk_attribution_service.get_risk_dashboard_data("user_123")
                        
                        assert result['status'] == 'success'
                        assert result['portfolio_rankings'] == mock_rankings
                        assert len(result['recent_alerts']) == 1
                        assert result['threshold_breaches'] == 2
                        assert result['monitoring_active'] is True


class TestRiskNotificationService:
    """Test suite for RiskNotificationService"""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Mock WebSocket connection manager"""
        mock_manager = MagicMock()
        mock_manager.send_risk_alert = AsyncMock(return_value=True)
        mock_manager.send_workflow_update = AsyncMock(return_value=True)
        mock_manager.send_personal_message = AsyncMock(return_value=True)
        mock_manager.get_connection_stats.return_value = {"active_connections": 5}
        return mock_manager
    
    @pytest.fixture
    def notification_service(self, mock_connection_manager):
        """Create RiskNotificationService instance"""
        service = RiskNotificationService()
        service.connection_manager = mock_connection_manager
        return service
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Sample portfolio data for notifications"""
        return {
            "portfolio_id": "portfolio_123",
            "portfolio_name": "Test Portfolio",
            "name": "Test Portfolio"
        }
    
    @pytest.fixture
    def sample_risk_snapshot(self):
        """Sample risk snapshot data"""
        return {
            "risk_score": 75.0,
            "risk_score_change_pct": 25.0,
            "volatility": 0.22,
            "volatility_change_pct": 30.0,
            "is_threshold_breach": True,
            "snapshot_id": "snapshot_123"
        }

    @pytest.mark.asyncio
    async def test_send_threshold_breach_alert_success(
        self, notification_service, sample_portfolio_data, sample_risk_snapshot
    ):
        """Test successful threshold breach alert"""
        
        result = await notification_service.send_threshold_breach_alert(
            "user_123", sample_portfolio_data, sample_risk_snapshot, "workflow_456"
        )
        
        assert result is True
        
        # Verify connection manager was called
        notification_service.connection_manager.send_risk_alert.assert_called_once()
        
        # Check the alert data structure
        call_args = notification_service.connection_manager.send_risk_alert.call_args
        user_id, alert_data = call_args[0]
        
        assert user_id == "user_123"
        assert alert_data["portfolio_id"] == "portfolio_123"
        assert alert_data["risk_score"] == 75.0
        assert alert_data["workflow_id"] == "workflow_456"
        assert alert_data["threshold_breached"] is True

    @pytest.mark.asyncio
    async def test_send_threshold_breach_alert_rate_limited(
        self, notification_service, sample_portfolio_data, sample_risk_snapshot
    ):
        """Test rate limiting for threshold breach alerts"""
        
        # Send first alert
        result1 = await notification_service.send_threshold_breach_alert(
            "user_123", sample_portfolio_data, sample_risk_snapshot
        )
        assert result1 is True
        
        # Try to send second alert immediately (should be rate limited)
        result2 = await notification_service.send_threshold_breach_alert(
            "user_123", sample_portfolio_data, sample_risk_snapshot
        )
        assert result2 is False
        
        # Verify only one call was made to connection manager
        assert notification_service.connection_manager.send_risk_alert.call_count == 1

    @pytest.mark.asyncio
    async def test_send_workflow_started_notification(self, notification_service):
        """Test workflow started notification"""
        
        workflow_data = {
            "workflow_id": "workflow_123",
            "portfolio_name": "Test Portfolio",
            "current_agent": "Dr. Sarah Chen"
        }
        
        result = await notification_service.send_workflow_started_notification(
            "user_123", workflow_data
        )
        
        assert result is True
        
        # Verify connection manager was called
        notification_service.connection_manager.send_workflow_update.assert_called_once()
        
        # Check notification data
        call_args = notification_service.connection_manager.send_workflow_update.call_args
        user_id, notification_data = call_args[0]
        
        assert user_id == "user_123"
        assert notification_data["workflow_id"] == "workflow_123"
        assert notification_data["status"] == "started"
        assert notification_data["progress"] == 0

    @pytest.mark.asyncio
    async def test_send_workflow_progress_update(self, notification_service):
        """Test workflow progress update"""
        
        result = await notification_service.send_workflow_progress_update(
            "user_123", "workflow_123", 50, "Marcus Thompson", "Analyzing risk factors"
        )
        
        assert result is True
        
        # Verify connection manager was called
        notification_service.connection_manager.send_workflow_update.assert_called_once()
        
        # Check notification data
        call_args = notification_service.connection_manager.send_workflow_update.call_args
        user_id, notification_data = call_args[0]
        
        assert user_id == "user_123"
        assert notification_data["workflow_id"] == "workflow_123"
        assert notification_data["status"] == "in_progress"
        assert notification_data["progress"] == 50
        assert notification_data["current_agent"] == "Marcus Thompson"

    @pytest.mark.asyncio
    async def test_send_workflow_completed_notification(self, notification_service):
        """Test workflow completed notification"""
        
        workflow_data = {"workflow_id": "workflow_123"}
        analysis_summary = {"recommendation": "Consider reducing risk exposure"}
        
        result = await notification_service.send_workflow_completed_notification(
            "user_123", workflow_data, analysis_summary
        )
        
        assert result is True
        
        # Verify connection manager was called
        notification_service.connection_manager.send_workflow_update.assert_called_once()
        
        # Check notification data
        call_args = notification_service.connection_manager.send_workflow_update.call_args
        user_id, notification_data = call_args[0]
        
        assert user_id == "user_123"
        assert notification_data["workflow_id"] == "workflow_123"
        assert notification_data["status"] == "completed"
        assert notification_data["progress"] == 100
        assert notification_data["analysis_summary"] == analysis_summary

    @pytest.mark.asyncio
    async def test_send_portfolio_monitoring_status(self, notification_service):
        """Test portfolio monitoring status notification"""
        
        result = await notification_service.send_portfolio_monitoring_status(
            "user_123", 5, 2
        )
        
        assert result is True
        
        # Verify connection manager was called
        notification_service.connection_manager.send_personal_message.assert_called_once()
        
        # Check notification data
        call_args = notification_service.connection_manager.send_personal_message.call_args
        user_id, notification_data = call_args[0]
        
        assert user_id == "user_123"
        assert notification_data["type"] == "monitoring_status"
        assert notification_data["data"]["portfolios_monitored"] == 5
        assert notification_data["data"]["alerts_triggered"] == 2

    def test_determine_severity_critical(self, notification_service):
        """Test severity determination - critical level"""
        
        risk_snapshot = {
            "risk_score": 95.0,
            "risk_score_change_pct": 60.0,
            "volatility": 0.7
        }
        
        severity = notification_service._determine_severity(risk_snapshot)
        assert severity == "critical"

    def test_determine_severity_high(self, notification_service):
        """Test severity determination - high level"""
        
        risk_snapshot = {
            "risk_score": 85.0,
            "risk_score_change_pct": 35.0,
            "volatility": 0.45
        }
        
        severity = notification_service._determine_severity(risk_snapshot)
        assert severity == "high"

    def test_determine_severity_medium(self, notification_service):
        """Test severity determination - medium level"""
        
        risk_snapshot = {
            "risk_score": 70.0,
            "risk_score_change_pct": 20.0,
            "volatility": 0.3
        }
        
        severity = notification_service._determine_severity(risk_snapshot)
        assert severity == "medium"

    def test_determine_severity_low(self, notification_service):
        """Test severity determination - low level"""
        
        risk_snapshot = {
            "risk_score": 50.0,
            "risk_score_change_pct": 10.0,
            "volatility": 0.15
        }
        
        severity = notification_service._determine_severity(risk_snapshot)
        assert severity == "low"

    def test_get_notification_stats(self, notification_service):
        """Test getting notification statistics"""
        
        # Add some test notification history
        notification_service.notification_history = {
            "user_123_portfolio_456": datetime.now(timezone.utc),
            "user_123_portfolio_789": datetime.now(timezone.utc) - timedelta(minutes=10)
        }
        
        stats = notification_service.get_notification_stats("user_123")
        
        assert "connection_stats" in stats
        assert stats["recent_notifications"] == 2
        assert len(stats["notification_history"]) == 2

    @pytest.mark.asyncio
    async def test_helper_functions(self):
        """Test helper functions for easy integration"""
        
        with patch('services.risk_notification_service.get_risk_notification_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.send_threshold_breach_alert = AsyncMock(return_value=True)
            mock_service.send_workflow_started_notification = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service
            
            # Test notify_threshold_breach helper
            result = await notify_threshold_breach(
                "user_123", {"portfolio_id": "123"}, {"risk_score": 75.0}
            )
            assert result is True
            mock_service.send_threshold_breach_alert.assert_called_once()
            
            # Test notify_workflow_started helper
            result = await notify_workflow_started(
                "user_123", {"workflow_id": "workflow_123"}
            )
            assert result is True
            mock_service.send_workflow_started_notification.assert_called_once()


class TestPortfolioMonitorService:
    """Test suite for PortfolioMonitorService"""
    
    @pytest.fixture
    def mock_scheduler(self):
        """Mock AsyncIOScheduler"""
        return MagicMock()
    
    @pytest.fixture
    def mock_risk_detector(self):
        """Mock risk detector"""
        return MagicMock()
    
    @pytest.fixture
    def portfolio_monitor(self, mock_scheduler, mock_risk_detector):
        """Create PortfolioMonitorService instance"""
        service = PortfolioMonitorService()
        service.scheduler = mock_scheduler
        service.risk_detector = mock_risk_detector
        return service

    @pytest.mark.asyncio
    async def test_start_monitoring(self, portfolio_monitor):
        """Test starting portfolio monitoring"""
        
        await portfolio_monitor.start_monitoring()
        
        # Verify scheduler job was added
        portfolio_monitor.scheduler.add_job.assert_called_once()
        
        # Check job configuration
        call_args = portfolio_monitor.scheduler.add_job.call_args
        func, interval_type = call_args[0], call_args[1]
        kwargs = call_args.kwargs
        
        assert interval_type == 'interval'
        assert kwargs['hours'] == 1
        assert kwargs['id'] == 'risk_monitoring'
        
        # Verify scheduler was started
        portfolio_monitor.scheduler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitor_all_portfolios(self, portfolio_monitor):
        """Test monitoring all portfolios"""
        
        # Mock getting portfolios
        mock_portfolios = [
            {"id": "portfolio_1", "user_id": "user_123"},
            {"id": "portfolio_2", "user_id": "user_456"}
        ]
        
        with patch.object(portfolio_monitor, 'get_active_portfolios', return_value=mock_portfolios):
            with patch.object(portfolio_monitor.risk_detector, 'detect_portfolio_risk_changes') as mock_detect:
                mock_detect.return_value = AsyncMock()
                
                await portfolio_monitor.monitor_all_portfolios()
                
                # Verify risk detection was called for each portfolio
                assert mock_detect.call_count == 2


class TestIntegrationHelpers:
    """Test integration helper functions"""

    @pytest.mark.asyncio
    async def test_calculate_and_store_portfolio_risk(self):
        """Test the integration helper function"""
        
        portfolio_data = {
            "name": "Test Portfolio",
            "positions": [{"symbol": "AAPL", "weight": 0.6}]
        }
        
        with patch('services.risk_attribution_service.get_risk_attribution_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.create_and_store_risk_snapshot = AsyncMock(return_value={'status': 'success'})
            mock_get_service.return_value = mock_service
            
            result = await calculate_and_store_portfolio_risk(
                "user_123", "portfolio_456", portfolio_data
            )
            
            assert result['status'] == 'success'
            mock_service.create_and_store_risk_snapshot.assert_called_once_with(
                "user_123", "portfolio_456", portfolio_data
            )

    @pytest.mark.asyncio
    async def test_trigger_risk_monitoring_for_user(self):
        """Test the user monitoring helper function"""
        
        portfolios = [
            {"id": "portfolio_1", "name": "Portfolio 1"},
            {"id": "portfolio_2", "name": "Portfolio 2"}
        ]
        
        with patch('services.risk_attribution_service.get_risk_attribution_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.monitor_portfolio_risk_changes = AsyncMock(return_value=[
                {'status': 'success'}, {'status': 'success'}
            ])
            mock_get_service.return_value = mock_service
            
            result = await trigger_risk_monitoring_for_user("user_123", portfolios)
            
            assert len(result) == 2
            assert all(r['status'] == 'success' for r in result)


class TestServiceErrors:
    """Test error handling across services"""

    @pytest.mark.asyncio
    async def test_proactive_monitor_mcp_connection_error(self):
        """Test handling MCP connection errors"""
        
        monitor = ProactiveRiskMonitor()
        
        with patch('services.proactive_monitor.get_mcp_client') as mock_get_client:
            mock_get_client.side_effect = Exception("MCP connection failed")
            
            # Should not raise exception, should handle gracefully
            await monitor._check_risk_thresholds("portfolio_123", "user_456")

    @pytest.mark.asyncio
    async def test_risk_detector_database_error(self):
        """Test handling database errors in risk detector"""
        
        detector = create_risk_detector()
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("Database connection failed")
        
        result = await detector.detect_portfolio_risk_changes(123, 456, mock_db)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_notification_service_websocket_error(self):
        """Test handling WebSocket errors in notification service"""
        
        service = RiskNotificationService()
        service.connection_manager = MagicMock()
        service.connection_manager.send_risk_alert = AsyncMock(side_effect=Exception("WebSocket error"))
        
        result = await service.send_threshold_breach_alert(
            "user_123", {"portfolio_id": "123"}, {"risk_score": 75.0}
        )
        
        assert result is False


if __name__ == "__main__":
    # Run specific test classes or all tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])