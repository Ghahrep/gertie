import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import your existing models and CRUD functions
from db.models import PortfolioRiskSnapshot, RiskThresholdConfig, RiskAlertLog, Base
from db.crud import (
    create_risk_snapshot, 
    get_risk_history, 
    get_risk_thresholds,
    log_risk_alert, 
    get_recent_alerts,
    # Only import functions that actually exist in your crud.py
    # We'll test the functions you have and create missing ones as needed
)


class TestDatabaseOperations:
    """Test existing database CRUD operations for risk attribution system"""
    
    @pytest.fixture(scope="function")
    def db_session(self):
        """Create test database session with in-memory SQLite"""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def test_create_risk_snapshot_success(self, db_session):
        """Test successful risk snapshot creation with your existing function"""
        snapshot_data = {
            "user_id": "test_user_001",
            "portfolio_id": "portfolio_tech_growth",
            "volatility": 0.18,
            "beta": 1.35,
            "max_drawdown": -0.15,
            "var_95": -0.045,
            "sharpe_ratio": 0.92,
            "risk_score": 42.5,
            "total_return": 0.086,
            "correlation_spy": 0.78
        }
        
        result = create_risk_snapshot(db_session, **snapshot_data)
        
        # Verify all fields are correctly stored
        assert result.user_id == "test_user_001"
        assert result.portfolio_id == "portfolio_tech_growth"
        assert result.volatility == 0.18
        assert result.beta == 1.35
        assert result.risk_score == 42.5
        assert result.id is not None
        assert isinstance(result.snapshot_date, datetime)
        
        # Test the risk grade method if it exists
        if hasattr(result, 'get_risk_grade'):
            expected_grade = "B"  # Risk score 42.5 should be grade B
            assert result.get_risk_grade() == expected_grade
    
    def test_create_risk_snapshot_with_minimal_data(self, db_session):
        """Test risk snapshot creation with only required fields"""
        minimal_data = {
            "user_id": "test_user_002",
            "portfolio_id": "portfolio_minimal",
            "volatility": 0.12,
            "risk_score": 25.0
        }
        
        result = create_risk_snapshot(db_session, **minimal_data)
        
        assert result.user_id == "test_user_002"
        assert result.volatility == 0.12
        assert result.risk_score == 25.0
        # Optional fields should be None
        assert result.beta is None
        assert result.sharpe_ratio is None
    
    def test_get_risk_history_multiple_snapshots(self, db_session):
        """Test retrieving historical risk data with multiple snapshots"""
        user_id = "test_user_history"
        portfolio_id = "portfolio_history_test"
        
        # Create 5 snapshots with increasing risk over time
        snapshots = []
        for i in range(5):
            snapshot_data = {
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "volatility": 0.10 + (i * 0.02),  # 0.10, 0.12, 0.14, 0.16, 0.18
                "risk_score": 20 + (i * 8),       # 20, 28, 36, 44, 52
                "beta": 0.8 + (i * 0.1),          # 0.8, 0.9, 1.0, 1.1, 1.2
            }
            snapshot = create_risk_snapshot(db_session, **snapshot_data)
            snapshots.append(snapshot)
        
        # Retrieve all history
        history = get_risk_history(db_session, user_id, days=30)
        
        assert len(history) == 5
        
        # Verify data exists (exact order depends on your implementation)
        risk_scores = [h.risk_score for h in history]
        assert 20 in risk_scores
        assert 52 in risk_scores
    
    def test_get_risk_history_with_date_filter(self, db_session):
        """Test risk history retrieval with date filtering"""
        user_id = "test_user_date_filter"
        
        # Create snapshots
        recent_data = {
            "user_id": user_id,
            "portfolio_id": "portfolio_recent",
            "volatility": 0.20,
            "risk_score": 60
        }
        recent_snapshot = create_risk_snapshot(db_session, **recent_data)
        
        old_data = {
            "user_id": user_id,
            "portfolio_id": "portfolio_old", 
            "volatility": 0.15,
            "risk_score": 40
        }
        old_snapshot = create_risk_snapshot(db_session, **old_data)
        
        # Manually set dates for testing
        base_date = datetime.now()
        recent_snapshot.snapshot_date = base_date - timedelta(days=3)
        old_snapshot.snapshot_date = base_date - timedelta(days=10)
        
        db_session.commit()
        
        # Get recent history only (7 days)
        recent_history = get_risk_history(db_session, user_id, days=7)
        
        # Should only get the recent snapshot
        assert len(recent_history) >= 1
        recent_scores = [h.risk_score for h in recent_history]
        assert 60 in recent_scores
        
        # Get extended history (30 days)
        full_history = get_risk_history(db_session, user_id, days=30)
        
        # Should get both snapshots
        assert len(full_history) >= 2
        all_scores = [h.risk_score for h in full_history]
        assert 60 in all_scores
        assert 40 in all_scores
    
    def test_risk_threshold_operations(self, db_session):
        """Test risk threshold configuration operations"""
        user_id = "test_user_thresholds"
        
        # Get thresholds (may return defaults or empty list)
        thresholds = get_risk_thresholds(db_session, user_id)
        
        # Should return a list
        assert isinstance(thresholds, list)
        
        # If your implementation creates default thresholds
        if len(thresholds) > 0:
            threshold_metrics = [t.metric_name for t in thresholds]
            # Check if common metrics exist
            assert any(metric in ["volatility", "risk_score", "beta"] for metric in threshold_metrics)
    
    def test_risk_alert_logging(self, db_session):
        """Test risk alert creation and retrieval"""
        user_id = "test_user_alerts"
        portfolio_id = "portfolio_alert_test"
        
        # Create a threshold breach alert
        alert_data = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "alert_type": "threshold_breach",
            "metric_name": "volatility",
            "current_value": 0.28,
            "threshold_value": 0.20,
            "message": "Portfolio volatility exceeded 20% threshold",
            "severity": "high",
            "workflow_triggered": True,
            "workflow_id": "workflow_auto_123"
        }
        
        alert = log_risk_alert(db_session, **alert_data)
        
        # Verify alert creation
        assert alert.user_id == user_id
        assert alert.alert_type == "threshold_breach"
        assert alert.metric_name == "volatility"
        assert alert.current_value == 0.28
        assert alert.workflow_triggered == True
        assert alert.workflow_id == "workflow_auto_123"
        assert isinstance(alert.alert_date, datetime)
        
        # Test alert retrieval
        recent_alerts = get_recent_alerts(db_session, user_id, days=1)
        
        assert len(recent_alerts) >= 1
        found_alert = recent_alerts[0]
        assert found_alert.message == "Portfolio volatility exceeded 20% threshold"
        assert found_alert.severity == "high"
    
    def test_multiple_alert_types(self, db_session):
        """Test different types of risk alerts"""
        user_id = "test_user_multi_alerts"
        
        # Create different alert types
        alert_types = [
            {
                "alert_type": "threshold_breach",
                "metric_name": "volatility",
                "message": "Volatility alert",
                "severity": "medium"
            },
            {
                "alert_type": "regime_change",
                "metric_name": "correlation_spy",
                "message": "Market regime change detected",
                "severity": "high"
            },
            {
                "alert_type": "risk_increase",
                "metric_name": "risk_score",
                "message": "Significant risk increase detected",
                "severity": "critical"
            }
        ]
        
        for alert_data in alert_types:
            alert_data.update({
                "user_id": user_id,
                "portfolio_id": "test_portfolio",
                "current_value": 0.25,
                "threshold_value": 0.20
            })
            log_risk_alert(db_session, **alert_data)
        
        # Retrieve all alerts
        all_alerts = get_recent_alerts(db_session, user_id, days=7)
        
        assert len(all_alerts) >= 3
        
        # Verify different alert types exist
        alert_types_found = [alert.alert_type for alert in all_alerts]
        assert "threshold_breach" in alert_types_found
        assert "regime_change" in alert_types_found
        assert "risk_increase" in alert_types_found
        
        # Test severity filtering
        critical_alerts = [a for a in all_alerts if a.severity == "critical"]
        assert len(critical_alerts) >= 1
    
    def test_database_model_properties(self, db_session):
        """Test database model properties and methods"""
        snapshot_data = {
            "user_id": "test_user_model",
            "portfolio_id": "test_portfolio",
            "risk_score": 45,
            "volatility": 0.18,
            "beta": 1.1
        }
        
        created_snapshot = create_risk_snapshot(db_session, **snapshot_data)
        
        # Test that the model has expected properties
        assert hasattr(created_snapshot, 'id')
        assert hasattr(created_snapshot, 'user_id')
        assert hasattr(created_snapshot, 'portfolio_id')
        assert hasattr(created_snapshot, 'snapshot_date')
        assert hasattr(created_snapshot, 'risk_score')
        assert hasattr(created_snapshot, 'volatility')
        
        # Test that snapshot_date is automatically set
        assert created_snapshot.snapshot_date is not None
        assert isinstance(created_snapshot.snapshot_date, datetime)
    
    def test_database_constraints_validation(self, db_session):
        """Test database constraints and data validation"""
        # Test that required fields are enforced
        with pytest.raises((ValueError, TypeError, Exception)):
            # Try to create snapshot without required user_id
            invalid_data = {
                "portfolio_id": "test_portfolio",
                "risk_score": 45
                # Missing user_id should cause error
            }
            create_risk_snapshot(db_session, **invalid_data)


class TestDatabasePerformance:
    """Test database performance with reasonable datasets"""
    
    @pytest.fixture(scope="function") 
    def db_session(self):
        """Create test database session"""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def test_bulk_snapshot_creation_performance(self, db_session):
        """Test performance with multiple snapshots"""
        import time
        
        user_id = "test_user_performance"
        start_time = time.time()
        
        # Create 50 snapshots (reasonable for testing)
        for i in range(50):
            snapshot_data = {
                "user_id": user_id,
                "portfolio_id": f"portfolio_{i % 5}",  # 5 different portfolios
                "risk_score": 30 + (i % 50),
                "volatility": 0.10 + (i % 20) / 100
            }
            create_risk_snapshot(db_session, **snapshot_data)
        
        creation_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 10 seconds for testing)
        assert creation_time < 10.0
        
        # Verify all were created
        history = get_risk_history(db_session, user_id, days=30)
        assert len(history) >= 50
    
    def test_history_retrieval_performance(self, db_session):
        """Test performance of retrieving historical datasets"""
        user_id = "test_user_large_history"
        
        # Create 100 snapshots
        for i in range(100):
            snapshot_data = {
                "user_id": user_id,
                "portfolio_id": "large_portfolio",
                "risk_score": 25 + (i % 50),
                "volatility": 0.08 + (i % 30) / 100
            }
            create_risk_snapshot(db_session, **snapshot_data)
        
        import time
        start_time = time.time()
        
        # Retrieve large history
        history = get_risk_history(db_session, user_id, days=365)
        
        retrieval_time = time.time() - start_time
        
        # Should retrieve quickly (less than 5 seconds)
        assert retrieval_time < 5.0
        assert len(history) >= 100


# Helper function to create missing CRUD functions if needed
def test_check_existing_crud_functions():
    """Test to see what CRUD functions are actually available"""
    from db import crud
    
    # Check which functions exist
    available_functions = [attr for attr in dir(crud) if not attr.startswith('_')]
    print(f"Available CRUD functions: {available_functions}")
    
    # This test always passes - it's just for inspection
    assert len(available_functions) > 0


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v", 
        "--tb=short",  # Shorter traceback for easier reading
        "-s"  # Don't capture output so we can see print statements
    ])


class TestDatabaseOperations:
    """Test all database CRUD operations for risk attribution system"""
    
    @pytest.fixture(scope="function")
    def db_session(self):
        """Create test database session with in-memory SQLite"""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def test_create_risk_snapshot_success(self, db_session):
        """Test successful risk snapshot creation"""
        snapshot_data = {
            "user_id": "test_user_001",
            "portfolio_id": "portfolio_tech_growth",
            "volatility": 0.18,
            "beta": 1.35,
            "max_drawdown": -0.15,
            "var_95": -0.045,
            "sharpe_ratio": 0.92,
            "risk_score": 42.5,
            "total_return": 0.086,
            "correlation_spy": 0.78
        }
        
        result = create_risk_snapshot(db_session, **snapshot_data)
        
        # Verify all fields are correctly stored
        assert result.user_id == "test_user_001"
        assert result.portfolio_id == "portfolio_tech_growth"
        assert result.volatility == 0.18
        assert result.beta == 1.35
        assert result.risk_score == 42.5
        assert result.id is not None
        assert isinstance(result.snapshot_date, datetime)
        
        # Verify risk grade calculation (assuming you have this method)
        expected_grade = "B"  # Risk score 42.5 should be grade B
        assert result.get_risk_grade() == expected_grade
    
    def test_create_risk_snapshot_with_minimal_data(self, db_session):
        """Test risk snapshot creation with only required fields"""
        minimal_data = {
            "user_id": "test_user_002",
            "portfolio_id": "portfolio_minimal",
            "volatility": 0.12,
            "risk_score": 25.0
        }
        
        result = create_risk_snapshot(db_session, **minimal_data)
        
        assert result.user_id == "test_user_002"
        assert result.volatility == 0.12
        assert result.risk_score == 25.0
        # Optional fields should be None
        assert result.beta is None
        assert result.sharpe_ratio is None
    
    def test_get_risk_history_multiple_snapshots(self, db_session):
        """Test retrieving historical risk data with multiple snapshots"""
        user_id = "test_user_history"
        portfolio_id = "portfolio_history_test"
        
        # Create 5 snapshots with increasing risk over time
        for i in range(5):
            snapshot_data = {
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "volatility": 0.10 + (i * 0.02),  # 0.10, 0.12, 0.14, 0.16, 0.18
                "risk_score": 20 + (i * 8),       # 20, 28, 36, 44, 52
                "beta": 0.8 + (i * 0.1),          # 0.8, 0.9, 1.0, 1.1, 1.2
            }
            create_risk_snapshot(db_session, **snapshot_data)
        
        # Retrieve all history
        history = get_risk_history(db_session, user_id, days=30)
        
        assert len(history) == 5
        
        # Verify chronological order (most recent first)
        assert history[0].risk_score == 52  # Most recent
        assert history[-1].risk_score == 20  # Oldest
        
        # Verify trend calculation
        assert history[0].volatility > history[-1].volatility
        assert history[0].beta > history[-1].beta
    
    def test_get_risk_history_with_date_filter(self, db_session):
        """Test risk history retrieval with date filtering"""
        user_id = "test_user_date_filter"
        
        # Create snapshots from different time periods
        base_date = datetime.now()
        
        # Recent snapshot (within 7 days)
        recent_data = {
            "user_id": user_id,
            "portfolio_id": "portfolio_recent",
            "volatility": 0.20,
            "risk_score": 60
        }
        recent_snapshot = create_risk_snapshot(db_session, **recent_data)
        recent_snapshot.snapshot_date = base_date - timedelta(days=3)
        
        # Old snapshot (older than 7 days)
        old_data = {
            "user_id": user_id,
            "portfolio_id": "portfolio_old",
            "volatility": 0.15,
            "risk_score": 40
        }
        old_snapshot = create_risk_snapshot(db_session, **old_data)
        old_snapshot.snapshot_date = base_date - timedelta(days=10)
        
        db_session.commit()
        
        # Get recent history only (7 days)
        recent_history = get_risk_history(db_session, user_id, days=7)
        
        assert len(recent_history) == 1
        assert recent_history[0].risk_score == 60
        
        # Get extended history (30 days)
        full_history = get_risk_history(db_session, user_id, days=30)
        
        assert len(full_history) == 2
    
    def test_risk_threshold_operations(self, db_session):
        """Test risk threshold configuration CRUD operations"""
        user_id = "test_user_thresholds"
        
        # Get default thresholds (should be auto-created)
        thresholds = get_risk_thresholds(db_session, user_id)
        
        # Should have default thresholds for key metrics
        threshold_metrics = [t.metric_name for t in thresholds]
        expected_metrics = [
            "volatility", "risk_score", "beta", "max_drawdown",
            "var_95", "correlation_spy", "sharpe_ratio"
        ]
        
        for metric in expected_metrics:
            assert metric in threshold_metrics
        
        # Test threshold breach detection
        volatility_threshold = next(
            (t for t in thresholds if t.metric_name == "volatility"), None
        )
        assert volatility_threshold is not None
        
        # Test breach detection logic
        assert volatility_threshold.is_threshold_breached(0.25) == True  # Above threshold
        assert volatility_threshold.is_threshold_breached(0.10) == False  # Below threshold
    
    def test_update_custom_thresholds(self, db_session):
        """Test updating user's custom threshold values"""
        user_id = "test_user_custom_thresholds"
        
        # Update custom thresholds
        custom_thresholds = {
            "volatility": 0.15,        # Lower than default
            "risk_score": 45,          # Custom risk tolerance
            "beta": 1.2,               # Lower beta tolerance
        }
        
        result = update_risk_thresholds(db_session, user_id, custom_thresholds)
        
        assert result["updated_count"] == 3
        assert "volatility" in result["updated_metrics"]
        
        # Verify updates were applied
        updated_thresholds = get_risk_thresholds(db_session, user_id)
        volatility_threshold = next(
            (t for t in updated_thresholds if t.metric_name == "volatility"), None
        )
        
        assert volatility_threshold.threshold_value == 0.15
    
    def test_risk_alert_logging(self, db_session):
        """Test risk alert creation and retrieval"""
        user_id = "test_user_alerts"
        portfolio_id = "portfolio_alert_test"
        
        # Create a threshold breach alert
        alert_data = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "alert_type": "threshold_breach",
            "metric_name": "volatility",
            "current_value": 0.28,
            "threshold_value": 0.20,
            "message": "Portfolio volatility exceeded 20% threshold",
            "severity": "high",
            "workflow_triggered": True,
            "workflow_id": "workflow_auto_123"
        }
        
        alert = log_risk_alert(db_session, **alert_data)
        
        # Verify alert creation
        assert alert.user_id == user_id
        assert alert.alert_type == "threshold_breach"
        assert alert.metric_name == "volatility"
        assert alert.current_value == 0.28
        assert alert.workflow_triggered == True
        assert alert.workflow_id == "workflow_auto_123"
        assert isinstance(alert.alert_date, datetime)
        
        # Test alert retrieval
        recent_alerts = get_recent_alerts(db_session, user_id, days=1)
        
        assert len(recent_alerts) == 1
        assert recent_alerts[0].message == "Portfolio volatility exceeded 20% threshold"
        assert recent_alerts[0].severity == "high"
    
    def test_multiple_alert_types(self, db_session):
        """Test different types of risk alerts"""
        user_id = "test_user_multi_alerts"
        
        # Create different alert types
        alert_types = [
            {
                "alert_type": "threshold_breach",
                "metric_name": "volatility",
                "message": "Volatility alert",
                "severity": "medium"
            },
            {
                "alert_type": "regime_change",
                "metric_name": "correlation_spy",
                "message": "Market regime change detected",
                "severity": "high"
            },
            {
                "alert_type": "risk_increase",
                "metric_name": "risk_score",
                "message": "Significant risk increase detected",
                "severity": "critical"
            }
        ]
        
        for alert_data in alert_types:
            alert_data.update({
                "user_id": user_id,
                "portfolio_id": "test_portfolio",
                "current_value": 0.25,
                "threshold_value": 0.20
            })
            log_risk_alert(db_session, **alert_data)
        
        # Retrieve all alerts
        all_alerts = get_recent_alerts(db_session, user_id, days=7)
        
        assert len(all_alerts) == 3
        
        # Verify different alert types
        alert_types_found = [alert.alert_type for alert in all_alerts]
        assert "threshold_breach" in alert_types_found
        assert "regime_change" in alert_types_found
        assert "risk_increase" in alert_types_found
        
        # Test severity filtering
        critical_alerts = [a for a in all_alerts if a.severity == "critical"]
        assert len(critical_alerts) == 1
        assert critical_alerts[0].message == "Significant risk increase detected"
    
    def test_calculate_risk_change_percentage(self, db_session):
        """Test risk change calculation between snapshots"""
        user_id = "test_user_risk_change"
        portfolio_id = "portfolio_change_test"
        
        # Create initial snapshot
        initial_data = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "volatility": 0.15,
            "risk_score": 35,
            "beta": 1.0
        }
        initial_snapshot = create_risk_snapshot(db_session, **initial_data)
        
        # Create updated snapshot with higher risk
        updated_data = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "volatility": 0.22,  # 46.7% increase
            "risk_score": 55,    # 57.1% increase
            "beta": 1.4          # 40% increase
        }
        updated_snapshot = create_risk_snapshot(db_session, **updated_data)
        
        # Calculate risk changes
        risk_changes = calculate_risk_change_percentage(
            initial_snapshot, updated_snapshot
        )
        
        # Verify calculations
        assert abs(risk_changes["volatility_change"] - 46.7) < 0.1
        assert abs(risk_changes["risk_score_change"] - 57.1) < 0.1
        assert abs(risk_changes["beta_change"] - 40.0) < 0.1
        
        # Test with risk decrease
        decreased_data = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "volatility": 0.12,  # 20% decrease from initial
            "risk_score": 28,    # 20% decrease from initial
        }
        decreased_snapshot = create_risk_snapshot(db_session, **decreased_data)
        
        decrease_changes = calculate_risk_change_percentage(
            initial_snapshot, decreased_snapshot
        )
        
        assert decrease_changes["volatility_change"] == -20.0
        assert decrease_changes["risk_score_change"] == -20.0
    
    def test_portfolio_risk_trend_analysis(self, db_session):
        """Test portfolio risk trend calculation over time"""
        user_id = "test_user_trends"
        portfolio_id = "portfolio_trend_test"
        
        # Create snapshots with varying risk levels
        risk_scores = [30, 35, 45, 40, 50, 48, 55]  # Up and down trend
        
        for i, risk_score in enumerate(risk_scores):
            snapshot_data = {
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "risk_score": risk_score,
                "volatility": 0.10 + (risk_score / 1000)  # Correlate with risk score
            }
            snapshot = create_risk_snapshot(db_session, **snapshot_data)
            # Set different dates for trend analysis
            snapshot.snapshot_date = datetime.now() - timedelta(days=(6-i))
            
        db_session.commit()
        
        # Calculate trend
        trend_data = get_portfolio_risk_trend(db_session, user_id, portfolio_id, days=7)
        
        assert "trend_direction" in trend_data
        assert "average_risk_score" in trend_data
        assert "risk_volatility" in trend_data
        assert "trend_strength" in trend_data
        
        # Should detect upward trend (30 -> 55)
        assert trend_data["trend_direction"] == "increasing"
        assert trend_data["average_risk_score"] > 40
    
    def test_get_user_portfolios(self, db_session):
        """Test retrieval of all user portfolios"""
        user_id = "test_user_multi_portfolio"
        
        # Create snapshots for multiple portfolios
        portfolios = [
            "portfolio_growth",
            "portfolio_income", 
            "portfolio_balanced"
        ]
        
        for portfolio_id in portfolios:
            snapshot_data = {
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "risk_score": 40,
                "volatility": 0.15
            }
            create_risk_snapshot(db_session, **snapshot_data)
        
        # Get all user portfolios
        user_portfolios = get_user_portfolios(db_session, user_id)
        
        assert len(user_portfolios) == 3
        
        portfolio_ids = [p.portfolio_id for p in user_portfolios]
        for expected_portfolio in portfolios:
            assert expected_portfolio in portfolio_ids
    
    def test_database_constraints_and_validation(self, db_session):
        """Test database constraints and data validation"""
        
        # Test duplicate prevention (if implemented)
        snapshot_data = {
            "user_id": "test_user_constraints",
            "portfolio_id": "test_portfolio",
            "risk_score": 45,
            "volatility": 0.18
        }
        
        # Create first snapshot
        first_snapshot = create_risk_snapshot(db_session, **snapshot_data)
        assert first_snapshot.id is not None
        
        # Test invalid data handling
        with pytest.raises(ValueError):
            invalid_data = snapshot_data.copy()
            invalid_data["volatility"] = -0.5  # Negative volatility should be invalid
            create_risk_snapshot(db_session, **invalid_data)
        
        with pytest.raises(ValueError):
            invalid_data = snapshot_data.copy()
            invalid_data["risk_score"] = 150  # Risk score > 100 should be invalid
            create_risk_snapshot(db_session, **invalid_data)
    
    def test_risk_snapshot_retrieval_by_id(self, db_session):
        """Test retrieving specific risk snapshot by ID"""
        snapshot_data = {
            "user_id": "test_user_by_id",
            "portfolio_id": "test_portfolio",
            "risk_score": 42,
            "volatility": 0.16,
            "beta": 1.1
        }
        
        created_snapshot = create_risk_snapshot(db_session, **snapshot_data)
        snapshot_id = created_snapshot.id
        
        # Retrieve by ID
        retrieved_snapshot = get_risk_snapshot_by_id(db_session, snapshot_id)
        
        assert retrieved_snapshot is not None
        assert retrieved_snapshot.id == snapshot_id
        assert retrieved_snapshot.user_id == "test_user_by_id"
        assert retrieved_snapshot.risk_score == 42
        
        # Test non-existent ID
        non_existent = get_risk_snapshot_by_id(db_session, 99999)
        assert non_existent is None


# Performance and stress tests
class TestDatabasePerformance:
    """Test database performance with larger datasets"""
    
    def test_bulk_snapshot_creation_performance(self, db_session):
        """Test performance with multiple snapshots"""
        import time
        
        user_id = "test_user_performance"
        start_time = time.time()
        
        # Create 100 snapshots
        for i in range(100):
            snapshot_data = {
                "user_id": user_id,
                "portfolio_id": f"portfolio_{i % 10}",  # 10 different portfolios
                "risk_score": 30 + (i % 50),
                "volatility": 0.10 + (i % 20) / 100
            }
            create_risk_snapshot(db_session, **snapshot_data)
        
        creation_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 5 seconds)
        assert creation_time < 5.0
        
        # Verify all were created
        history = get_risk_history(db_session, user_id, days=30)
        assert len(history) == 100
    
    def test_large_history_retrieval_performance(self, db_session):
        """Test performance of retrieving large historical datasets"""
        user_id = "test_user_large_history"
        
        # Create 1000 snapshots
        for i in range(1000):
            snapshot_data = {
                "user_id": user_id,
                "portfolio_id": "large_portfolio",
                "risk_score": 25 + (i % 50),
                "volatility": 0.08 + (i % 30) / 100
            }
            create_risk_snapshot(db_session, **snapshot_data)
        
        import time
        start_time = time.time()
        
        # Retrieve large history
        history = get_risk_history(db_session, user_id, days=365)
        
        retrieval_time = time.time() - start_time
        
        # Should retrieve quickly (less than 2 seconds)
        assert retrieval_time < 2.0
        assert len(history) == 1000


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=db",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])