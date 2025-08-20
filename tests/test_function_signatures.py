import pytest
import inspect
from db.crud import create_risk_snapshot, log_risk_alert

def test_create_risk_snapshot_signature():
    """Discover the exact signature of create_risk_snapshot"""
    sig = inspect.signature(create_risk_snapshot)
    print(f"\n=== create_risk_snapshot signature ===")
    print(f"create_risk_snapshot{sig}")
    
    for param_name, param in sig.parameters.items():
        print(f"  - {param_name}: {param.annotation} = {param.default}")
    
    assert True

def test_log_risk_alert_signature():
    """Discover the exact signature of log_risk_alert"""
    sig = inspect.signature(log_risk_alert)
    print(f"\n=== log_risk_alert signature ===")
    print(f"log_risk_alert{sig}")
    
    for param_name, param in sig.parameters.items():
        print(f"  - {param_name}: {param.annotation} = {param.default}")
    
    assert True

def test_simple_create_risk_snapshot():
    """Try to call create_risk_snapshot with minimal parameters to see what it needs"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from db.models import Base
    
    # Create test database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()
    
    try:
        # Try with just the required parameters based on inspection
        result = create_risk_snapshot(
            db=db,
            user_id="test_user",
            portfolio_id="test_portfolio", 
            risk_result={"volatility": 0.15, "risk_score": 35}
        )
        print(f"\n✅ SUCCESS: Created snapshot with ID {result.id}")
        print(f"   - user_id: {result.user_id}")
        print(f"   - portfolio_id: {result.portfolio_id}")
        print(f"   - volatility: {result.volatility}")
        print(f"   - risk_score: {result.risk_score}")
        
    except Exception as e:
        print(f"\n❌ ERROR calling create_risk_snapshot: {e}")
        print(f"   Error type: {type(e).__name__}")
        
    finally:
        db.close()
    
    assert True

def test_simple_log_risk_alert():
    """Try to call log_risk_alert with minimal parameters to see what it needs"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from db.models import Base
    
    # Create test database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()
    
    try:
        # Try with just the required parameters based on inspection
        result = log_risk_alert(
            db=db,
            user_id="test_user",
            portfolio_id="test_portfolio",
            alert_type="threshold_breach",
            alert_message="Test alert message"
        )
        print(f"\n✅ SUCCESS: Created alert with ID {result.id}")
        print(f"   - user_id: {result.user_id}")
        print(f"   - alert_type: {result.alert_type}")
        print(f"   - alert_message: {result.alert_message}")
        
    except Exception as e:
        print(f"\n❌ ERROR calling log_risk_alert: {e}")
        print(f"   Error type: {type(e).__name__}")
        
    finally:
        db.close()
    
    assert True

if __name__ == "__main__":
    test_create_risk_snapshot_signature()
    test_log_risk_alert_signature()
    test_simple_create_risk_snapshot()
    test_simple_log_risk_alert()