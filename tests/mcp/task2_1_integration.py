#!/usr/bin/env python3
"""
Test Task 2.1 Risk Detection Pipeline Database Setup
"""

from db.session import get_db
from db.models import RiskThreshold, RiskTrend, PriceDataCache, PortfolioRiskSnapshot, Portfolio

def test_database_setup():
    """Test that all Task 2.1 tables and data exist"""
    
    db = next(get_db())
    
    print("=== Task 2.1 Database Setup Test ===")
    
    # Test table existence and counts
    risk_thresholds_count = db.query(RiskThreshold).count()
    risk_trends_count = db.query(RiskTrend).count()
    price_cache_count = db.query(PriceDataCache).count()
    
    print(f"Risk thresholds: {risk_thresholds_count}")
    print(f"Risk trends: {risk_trends_count}")
    print(f"Price cache entries: {price_cache_count}")
    
    # Test that default thresholds were inserted
    if risk_thresholds_count > 0:
        print("\nDefault Risk Thresholds:")
        thresholds = db.query(RiskThreshold).all()
        for t in thresholds:
            print(f"  {t.metric_name}: Warning={t.warning_threshold}, Critical={t.critical_threshold}, Direction={t.direction}")
    else:
        print("WARNING: No default risk thresholds found!")
    
    # Test portfolio_risk_snapshots table has new columns
    try:
        snapshots = db.query(PortfolioRiskSnapshot).limit(1).all()
        if snapshots:
            snapshot = snapshots[0]
            has_new_columns = hasattr(snapshot, 'compressed_metrics') and hasattr(snapshot, 'metrics_summary')
            print(f"\nPortfolioRiskSnapshot has new columns: {has_new_columns}")
        else:
            print("\nNo risk snapshots exist yet (this is normal)")
    except Exception as e:
        print(f"Error checking portfolio risk snapshots: {e}")
    
    # Test relationships work
    try:
        portfolios = db.query(Portfolio).limit(1).all()
        if portfolios:
            portfolio = portfolios[0]
            risk_snapshots_rel = hasattr(portfolio, 'risk_snapshots')
            risk_trends_rel = hasattr(portfolio, 'risk_trends')
            print(f"Portfolio relationships work: snapshots={risk_snapshots_rel}, trends={risk_trends_rel}")
        else:
            print("No portfolios exist to test relationships")
    except Exception as e:
        print(f"Error testing relationships: {e}")
    
    print("\n=== Test Complete ===")
    
    if risk_thresholds_count > 0:
        print("✅ Task 2.1 database setup appears successful!")
        return True
    else:
        print("❌ Task 2.1 database setup incomplete - missing default thresholds")
        return False

if __name__ == "__main__":
    test_database_setup()