# setup_phase1_simple.py
"""
Simple Phase 1 Setup for Risk Attribution
Works with your exact file structure: db/models.py, db/crud.py, db/session.py
"""

import asyncio
import sys
import os

async def test_phase1_setup():
    """
    Test if Phase 1 is set up correctly with your file structure
    """
    print("🧪 TESTING PHASE 1 SETUP")
    print("=" * 40)
    print("Checking your database structure...")
    print()
    
    results = []
    
    # Test 1: Check existing files
    print("TEST 1: Your File Structure")
    print("-" * 30)
    
    required_files = ['db/models.py', 'db/crud.py', 'db/session.py']
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found {file_path}")
        else:
            print(f"❌ Missing {file_path}")
            return False
    
    # Test 2: Check if models were added
    print("\nTEST 2: Risk Models in db/models.py")
    print("-" * 38)
    try:
        from db.models import PortfolioRiskSnapshot, RiskThresholdConfig, RiskAlertLog
        print("✅ PortfolioRiskSnapshot model found")
        print("✅ RiskThresholdConfig model found") 
        print("✅ RiskAlertLog model found")
        results.append(("Risk Models", True))
    except ImportError as e:
        print(f"❌ Risk models not found in db/models.py")
        print(f"   Error: {e}")
        print("   → You need to add the risk models to db/models.py")
        results.append(("Risk Models", False))
    
    # Test 3: Check if CRUD functions were added
    print("\nTEST 3: Risk CRUD in db/crud.py")
    print("-" * 33)
    try:
        from db.crud import create_risk_snapshot, get_risk_history, log_risk_alert
        print("✅ create_risk_snapshot function found")
        print("✅ get_risk_history function found")
        print("✅ log_risk_alert function found")
        results.append(("Risk CRUD", True))
    except ImportError as e:
        print(f"❌ Risk CRUD functions not found in db/crud.py")
        print(f"   Error: {e}")
        print("   → You need to add the risk CRUD functions to db/crud.py")
        results.append(("Risk CRUD", False))
    
    # Test 4: Check database connection
    print("\nTEST 4: Database Connection")
    print("-" * 29)
    try:
        from db.session import get_db
        db = next(get_db())
        db.execute("SELECT 1")
        db.close()
        print("✅ Database connection working")
        results.append(("Database", True))
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   → Check your database settings in db/session.py")
        results.append(("Database", False))
    
    # Test 5: Check existing services
    print("\nTEST 5: Your Existing Services")
    print("-" * 32)
    services_found = 0
    
    try:
        from services.risk_calculator import RiskCalculator
        print("✅ risk_calculator.py found")
        services_found += 1
    except ImportError:
        print("⚠️  risk_calculator.py not found (needed for integration)")
    
    try:
        from services.proactive_monitor import get_proactive_monitor
        print("✅ proactive_monitor.py found")
        services_found += 1
    except ImportError:
        print("⚠️  proactive_monitor.py not found (optional)")
    
    results.append(("Services", services_found > 0))
    
    # Test 6: Test risk attribution service
    if len([r for r in results if r[1]]) >= 3:  # Only if basic components work
        print("\nTEST 6: Risk Attribution Service")
        print("-" * 35)
        try:
            if os.path.exists('services/risk_attribution_service.py'):
                from services.risk_attribution_service import get_risk_attribution_service
                service = await get_risk_attribution_service()
                print("✅ Risk Attribution Service initialized")
                results.append(("Integration", True))
            else:
                print("❌ services/risk_attribution_service.py not found")
                print("   → You need to add the risk attribution service file")
                results.append(("Integration", False))
        except Exception as e:
            print(f"❌ Service initialization failed: {e}")
            results.append(("Integration", False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 SETUP RESULTS")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:15} | {status}")
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print("-" * 40)
    print(f"SCORE: {passed}/{total} ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print("\n🎉 PHASE 1 SETUP SUCCESSFUL!")
        print("✅ Your platform is ready for risk attribution")
        print("\n📋 What works now:")
        print("   - Store portfolio risk snapshots")
        print("   - Track risk changes over time")
        print("   - Automatic threshold detection")
        print("   - Risk alerts and workflow triggers")
        
        # Quick demo
        await run_quick_demo()
        
        return True
    else:
        print("\n⚠️  PHASE 1 SETUP INCOMPLETE")
        print(f"🔧 Need to fix {total - passed} components")
        print("\n📋 Next steps:")
        print("1. Copy the risk models to db/models.py")
        print("2. Copy the risk CRUD functions to db/crud.py")
        print("3. Copy services/risk_attribution_service.py")
        print("4. Run this script again to verify")
        
        return False


async def run_quick_demo():
    """Run a quick demo of the risk attribution system"""
    print("\n🎬 QUICK DEMO")
    print("-" * 15)
    
    try:
        from services.risk_attribution_service import calculate_and_store_portfolio_risk
        
        # Demo portfolio
        demo_portfolio = {
            "id": "demo_portfolio",
            "name": "Demo Conservative Portfolio", 
            "positions": [
                {"symbol": "SPY", "weight": 0.7, "shares": 100},
                {"symbol": "BND", "weight": 0.3, "shares": 150}
            ],
            "total_value": 35000
        }
        
        print("📊 Calculating risk for demo portfolio...")
        result = await calculate_and_store_portfolio_risk(
            user_id="demo_user",
            portfolio_id=demo_portfolio["id"],
            portfolio_data=demo_portfolio
        )
        
        if result['status'] == 'success':
            snapshot = result['snapshot']
            print(f"✅ Risk calculation successful!")
            print(f"   Risk Score: {snapshot['risk_score']:.1f}/100")
            print(f"   Risk Level: {snapshot['risk_level']}")
            print(f"   Volatility: {snapshot['volatility']:.1%}")
            print(f"   Alert Triggered: {'Yes' if result['alert_triggered'] else 'No'}")
        else:
            print(f"❌ Demo failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")


def show_setup_instructions():
    """Show setup instructions if components are missing"""
    
    print("\n📋 PHASE 1 SETUP INSTRUCTIONS")
    print("=" * 35)
    print()
    print("To complete Phase 1, you need to:")
    print()
    print("1. ADD MODELS TO db/models.py:")
    print("   - Copy the 3 risk model classes")
    print("   - Add them to the bottom of your existing db/models.py")
    print()
    print("2. ADD CRUD FUNCTIONS TO db/crud.py:")
    print("   - Copy all the risk CRUD functions") 
    print("   - Add them to the bottom of your existing db/crud.py")
    print()
    print("3. ADD SERVICE FILE:")
    print("   - Copy services/risk_attribution_service.py to your services/ folder")
    print()
    print("4. CREATE DATABASE TABLES:")
    print("   - Run the SQL migration in your PostgreSQL database")
    print("   - Or let SQLAlchemy create them automatically")
    print()
    print("5. TEST AGAIN:")
    print("   - Run: python setup_phase1_simple.py")
    print()
    print("📄 All the code you need is in the artifacts I provided!")


async def main():
    """Main setup function"""
    print("🚀 PHASE 1 SETUP CHECK")
    print("Checking your exact file structure...")
    print()
    
    success = await test_phase1_setup()
    
    if not success:
        show_setup_instructions()
    
    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Setup check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Setup error: {e}")
        sys.exit(1)