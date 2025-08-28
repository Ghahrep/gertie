# test_risk_integration.py
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.risk_calculator import create_risk_calculator

async def test_risk_calculator():
    print("🧪 Testing Risk Calculator...")
    
    try:
        calculator = create_risk_calculator()
        print("✅ Risk calculator instance created")
        
        # Test with actual portfolio tickers
        print("📊 Calculating risk metrics for AAPL, MSFT, GOOGL...")
        risk_metrics = calculator.calculate_portfolio_risk_from_tickers(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            weights=[0.4, 0.3, 0.3],
            lookback_days=252
        )
        
        if risk_metrics:
            print("✅ Risk Calculator Working!")
            print(f"📈 Risk Score: {risk_metrics.risk_score}/100")
            print(f"📊 Volatility: {risk_metrics.volatility:.2%}")
            print(f"📉 Beta: {risk_metrics.beta:.2f}")
            print(f"💥 Max Drawdown: {risk_metrics.max_drawdown:.2%}")
            print(f"⚠️  VaR (99%): {risk_metrics.var_99:.2%}")
            print(f"🎯 Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
            print(f"🌊 Sentiment Index: {risk_metrics.sentiment_index}/100")
            return True
        else:
            print("❌ Risk calculation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error in risk calculation: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_risk_detector():
    print("\n🧪 Testing Risk Detector Integration...")
    
    try:
        from services.risk_detector import create_risk_detector
        from db.session import get_db
        
        detector = create_risk_detector()
        print("✅ Risk detector instance created")
        
        # This would require actual database setup
        # For now, just test the creation
        print("✅ Risk detector test passed (DB integration requires setup)")
        return True
        
    except ImportError as e:
        print(f"⚠️  Risk detector not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in risk detector test: {e}")
        return False

async def test_monitoring_integration():
    print("\n🧪 Testing Proactive Monitor Integration...")
    
    try:
        from services.proactive_monitor import get_proactive_monitor
        
        monitor = await get_proactive_monitor()
        print("✅ Proactive monitor instance created")
        
        # Test the enhanced stats
        stats = monitor.get_monitoring_stats()
        print(f"📊 Enhanced risk detection: {stats.get('enhanced_risk_detection', False)}")
        print(f"🔍 Active monitors: {stats['active_monitors']}")
        print(f"⚠️  Total alerts: {stats['total_alerts']}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Proactive monitor not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in monitor test: {e}")
        return False

async def run_all_tests():
    print("🚀 Starting Risk Integration Tests...\n")
    
    tests = [
        ("Risk Calculator", test_risk_calculator),
        ("Risk Detector", test_risk_detector),
        ("Monitor Integration", test_monitoring_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "="*50)
    print("📋 TEST RESULTS SUMMARY")
    print("="*50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

# 🔧 FIX: Add this to make the script runnable
def main():
    """Main function to run the async tests"""
    try:
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()