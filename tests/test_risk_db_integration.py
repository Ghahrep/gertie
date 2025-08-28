from db.session import get_db
from services.risk_detector import create_risk_detector

async def test_risk_detection():
    db = next(get_db())
    detector = create_risk_detector()
    
    # Test with actual portfolio from your database
    risk_analysis = await detector.detect_portfolio_risk_changes(
        portfolio_id=1,  # Use your test portfolio ID
        user_id=1,       # Use your test user ID
        db=db
    )
    
    if risk_analysis:
        print("✅ Risk Detection Working!")
        print(f"Risk Direction: {risk_analysis.risk_direction}")
        print(f"Threshold Breached: {risk_analysis.threshold_breached}")
        return True
    return False