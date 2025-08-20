# check_and_add_crud.py
"""
Check if CRUD functions are added, and add them if missing
"""

def check_crud_functions():
    """Check if risk CRUD functions are in db/crud.py"""
    
    try:
        from db.crud import create_risk_snapshot, get_risk_history, log_risk_alert
        print("✅ Risk CRUD functions already present!")
        return True
    except ImportError as e:
        print(f"❌ Risk CRUD functions missing: {e}")
        return False

def add_crud_functions():
    """Add risk CRUD functions to db/crud.py"""
    
    crud_file = 'db/crud.py'
    
    # Read current content
    with open(crud_file, 'r') as f:
        content = f.read()
    
    # Check if already added
    if 'create_risk_snapshot' in content:
        print("✅ CRUD functions already in file")
        return True
    
    print("📝 Adding risk CRUD functions to db/crud.py...")
    
    # Add the risk CRUD functions
    crud_additions = '''

# Risk Attribution CRUD Functions
from sqlalchemy import desc, and_, or_
from typing import List, Optional, Dict
from datetime import datetime, timedelta, timezone

# Import the risk models
from db.models import (
    PortfolioRiskSnapshot, 
    RiskThresholdConfig, 
    RiskAlertLog,
    create_risk_snapshot_from_calculator,
    get_default_risk_thresholds
)

def create_risk_snapshot(
    db, 
    user_id: str, 
    portfolio_id: str, 
    risk_result: dict,
    portfolio_data: dict = None
):
    """Create and save risk snapshot from calculator results"""
    # Get previous snapshot for change calculation
    previous = get_latest_risk_snapshot(db, user_id, portfolio_id)
    
    # Create new snapshot
    snapshot = create_risk_snapshot_from_calculator(
        user_id, portfolio_id, risk_result, portfolio_data
    )
    
    # Calculate changes from previous snapshot
    if previous:
        snapshot.volatility_change_pct = _calculate_percentage_change(
            previous.volatility, snapshot.volatility
        )
        snapshot.risk_score_change_pct = _calculate_percentage_change(
            previous.risk_score, snapshot.risk_score
        )
        
        # Check if threshold is breached
        snapshot.is_threshold_breach = _check_threshold_breach(
            db, user_id, portfolio_id, snapshot, previous
        )
    
    # Save to database
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    
    return snapshot

def get_latest_risk_snapshot(db, user_id: str, portfolio_id: str):
    """Get the most recent risk snapshot for a portfolio"""
    return db.query(PortfolioRiskSnapshot)\\
             .filter(and_(
                 PortfolioRiskSnapshot.user_id == user_id,
                 PortfolioRiskSnapshot.portfolio_id == portfolio_id
             ))\\
             .order_by(desc(PortfolioRiskSnapshot.snapshot_date))\\
             .first()

def get_risk_history(db, user_id: str, portfolio_id: str = None, days: int = 30, limit: int = 100):
    """Get risk history for user/portfolio"""
    query = db.query(PortfolioRiskSnapshot)\\
              .filter(PortfolioRiskSnapshot.user_id == user_id)
    
    if portfolio_id:
        query = query.filter(PortfolioRiskSnapshot.portfolio_id == portfolio_id)
    
    if days > 0:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        query = query.filter(PortfolioRiskSnapshot.snapshot_date >= cutoff_date)
    
    return query.order_by(desc(PortfolioRiskSnapshot.snapshot_date))\\
               .limit(limit)\\
               .all()

def get_risk_thresholds(db, user_id: str, portfolio_id: str = None):
    """Get risk thresholds for user/portfolio"""
    # Try portfolio-specific config first
    if portfolio_id:
        config = db.query(RiskThresholdConfig)\\
                   .filter(and_(
                       RiskThresholdConfig.user_id == user_id,
                       RiskThresholdConfig.portfolio_id == portfolio_id
                   ))\\
                   .first()
        if config:
            return _config_to_dict(config)
    
    # Try user default config
    config = db.query(RiskThresholdConfig)\\
               .filter(and_(
                   RiskThresholdConfig.user_id == user_id,
                   RiskThresholdConfig.portfolio_id.is_(None)
               ))\\
               .first()
    if config:
        return _config_to_dict(config)
    
    # Fall back to system defaults
    return get_default_risk_thresholds()

def log_risk_alert(db, user_id: str, portfolio_id: str, alert_type: str, alert_message: str, 
                   snapshot_id: str = None, severity: str = "medium", triggered_metrics: Dict = None, workflow_id: str = None):
    """Log a risk alert to database"""
    alert = RiskAlertLog(
        user_id=user_id,
        portfolio_id=portfolio_id,
        snapshot_id=snapshot_id,
        alert_type=alert_type,
        alert_severity=severity,
        alert_message=alert_message,
        triggered_metrics=triggered_metrics,
        workflow_id=workflow_id,
        notification_methods=['websocket'],
        notification_status={'websocket': 'pending'}
    )
    
    db.add(alert)
    db.commit()
    db.refresh(alert)
    
    return alert

def get_recent_alerts(db, user_id: str, days: int = 7, limit: int = 50):
    """Get recent alerts for a user"""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    return db.query(RiskAlertLog)\\
             .filter(and_(
                 RiskAlertLog.user_id == user_id,
                 RiskAlertLog.created_at >= cutoff_date
             ))\\
             .order_by(desc(RiskAlertLog.created_at))\\
             .limit(limit)\\
             .all()

# Helper functions
def _calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / abs(old_value)) * 100

def _check_threshold_breach(db, user_id: str, portfolio_id: str, current_snapshot, previous_snapshot) -> bool:
    """Check if current snapshot breaches any thresholds"""
    thresholds = get_risk_thresholds(db, user_id, portfolio_id)
    
    # Check percentage change thresholds
    if current_snapshot.volatility_change_pct and \\
       abs(current_snapshot.volatility_change_pct) > (thresholds['volatility_threshold'] * 100):
        return True
    
    if current_snapshot.risk_score_change_pct and \\
       abs(current_snapshot.risk_score_change_pct) > (thresholds['risk_score_threshold'] * 100):
        return True
    
    # Check absolute thresholds
    if current_snapshot.risk_score > thresholds['max_acceptable_risk_score']:
        return True
    
    if current_snapshot.volatility > thresholds['max_acceptable_volatility']:
        return True
    
    return False

def _config_to_dict(config) -> Dict:
    """Convert RiskThresholdConfig to dictionary"""
    return {
        'volatility_threshold': config.volatility_threshold,
        'beta_threshold': config.beta_threshold,
        'max_drawdown_threshold': config.max_drawdown_threshold,
        'var_threshold': config.var_threshold,
        'risk_score_threshold': config.risk_score_threshold,
        'max_acceptable_risk_score': config.max_acceptable_risk_score,
        'max_acceptable_volatility': config.max_acceptable_volatility,
        'monitoring_enabled': config.monitoring_enabled,
        'alert_frequency': config.alert_frequency,
        'notification_methods': config.notification_methods
    }
'''
    
    # Append to file
    with open(crud_file, 'a') as f:
        f.write(crud_additions)
    
    print("✅ Risk CRUD functions added to db/crud.py")
    return True

def test_crud_functions():
    """Test that CRUD functions work"""
    print("🧪 Testing CRUD functions...")
    
    try:
        from db.crud import create_risk_snapshot, get_risk_history, log_risk_alert
        from db.crud import get_risk_thresholds, get_recent_alerts
        print("✅ All risk CRUD functions imported successfully")
        
        # Test getting default thresholds
        from db.session import get_db
        db = next(get_db())
        thresholds = get_risk_thresholds(db, "test_user")
        db.close()
        print(f"✅ Default thresholds retrieved: {len(thresholds)} settings")
        
        return True
        
    except Exception as e:
        print(f"❌ CRUD test failed: {e}")
        return False

def main():
    print("🔧 CHECKING AND ADDING CRUD FUNCTIONS")
    print("=" * 40)
    
    # Check if already present
    if check_crud_functions():
        print("✅ CRUD functions already working!")
        return True
    
    # Add them
    if add_crud_functions():
        # Test them
        if test_crud_functions():
            print("\n🎉 CRUD FUNCTIONS ADDED SUCCESSFULLY!")
            print("✅ All risk CRUD operations available")
            print("✅ Ready for service integration")
            return True
    
    print("\n❌ CRUD setup failed")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)