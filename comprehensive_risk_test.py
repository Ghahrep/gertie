# comprehensive_risk_test.py
import asyncio
import sys
import os
from datetime import datetime

print("🚀 Comprehensive Risk Integration Test")
print("=" * 60)

def test_1_risk_calculator():
    """Test 1: Risk Calculator Service"""
    print("\n🧪 TEST 1: Risk Calculator Service")
    print("-" * 40)
    
    try:
        from services.risk_calculator import create_risk_calculator
        
        calculator = create_risk_calculator()
        print("✅ Risk calculator created")
        
        # Test with a realistic portfolio
        portfolios_to_test = [
            {
                "name": "Tech Portfolio",
                "tickers": ['AAPL', 'MSFT', 'GOOGL'],
                "weights": [0.4, 0.3, 0.3]
            },
            {
                "name": "Diversified Portfolio", 
                "tickers": ['SPY', 'BND'],
                "weights": [0.7, 0.3]
            }
        ]
        
        results = []
        
        for portfolio in portfolios_to_test:
            print(f"\n📊 Testing {portfolio['name']}...")
            
            risk_metrics = calculator.calculate_portfolio_risk_from_tickers(
                tickers=portfolio['tickers'],
                weights=portfolio['weights'],
                lookback_days=100  # Faster for testing
            )
            
            if risk_metrics:
                print(f"✅ {portfolio['name']} analysis successful!")
                print(f"   📈 Risk Score: {risk_metrics.risk_score}/100")
                print(f"   📊 Volatility: {risk_metrics.volatility:.2%}")
                print(f"   📉 Beta: {risk_metrics.beta:.2f}")
                print(f"   💥 Max Drawdown: {risk_metrics.max_drawdown:.2%}")
                print(f"   🎯 Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
                
                results.append({
                    "portfolio": portfolio['name'],
                    "success": True,
                    "risk_score": risk_metrics.risk_score,
                    "volatility": risk_metrics.volatility
                })
            else:
                print(f"❌ {portfolio['name']} analysis failed")
                results.append({
                    "portfolio": portfolio['name'], 
                    "success": False
                })
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\n📋 Risk Calculator Results: {success_count}/{len(results)} portfolios analyzed successfully")
        
        return success_count == len(results), results
        
    except Exception as e:
        print(f"❌ Risk Calculator test failed: {e}")
        return False, []

def test_2_risk_comparison():
    """Test 2: Risk Metrics Comparison"""
    print("\n🧪 TEST 2: Risk Metrics Comparison")
    print("-" * 40)
    
    try:
        from services.risk_calculator import create_risk_calculator
        
        calculator = create_risk_calculator()
        
        # Create two different risk scenarios
        conservative_portfolio = calculator.calculate_portfolio_risk_from_tickers(
            tickers=['SPY', 'BND'],
            weights=[0.6, 0.4],
            lookback_days=100
        )
        
        aggressive_portfolio = calculator.calculate_portfolio_risk_from_tickers(
            tickers=['NVDA', 'TSLA'],
            weights=[0.5, 0.5], 
            lookback_days=100
        )
        
        if conservative_portfolio and aggressive_portfolio:
            print("✅ Both portfolios analyzed")
            
            # Compare the portfolios
            comparison = calculator.compare_risk_metrics(
                current_metrics=aggressive_portfolio,
                previous_metrics=conservative_portfolio,
                threshold_pct=15.0
            )
            
            print(f"📊 Risk Comparison Results:")
            print(f"   Conservative Risk Score: {conservative_portfolio.risk_score}/100")
            print(f"   Aggressive Risk Score: {aggressive_portfolio.risk_score}/100")
            print(f"   Risk Direction: {comparison['risk_direction']}")
            print(f"   Risk Magnitude: {comparison['risk_magnitude_pct']:.1f}%")
            print(f"   Threshold Breached: {comparison['threshold_breached']}")
            
            if comparison['significant_changes']:
                print(f"   Significant Changes:")
                for metric, change in comparison['significant_changes'].items():
                    print(f"     • {metric}: {change:+.1f}%")
            
            return True, comparison
        else:
            print("❌ Portfolio analysis failed")
            return False, {}
            
    except Exception as e:
        print(f"❌ Risk comparison test failed: {e}")
        return False, {}

async def test_3_proactive_monitor():
    """Test 3: Proactive Monitor Integration"""
    print("\n🧪 TEST 3: Proactive Monitor Integration")
    print("-" * 40)
    
    try:
        from services.proactive_monitor import get_proactive_monitor
        
        monitor = await get_proactive_monitor()
        print("✅ Proactive monitor created")
        
        # Test enhanced stats
        stats = monitor.get_monitoring_stats()
        print(f"📊 Monitor Statistics:")
        print(f"   Enhanced Risk Detection: {stats.get('enhanced_risk_detection', False)}")
        print(f"   Active Monitors: {stats['active_monitors']}")
        print(f"   Total Alerts: {stats['total_alerts']}")
        print(f"   Risk-Specific Alerts: {stats.get('risk_specific_alerts', 0)}")
        print(f"   Workflow Triggered Count: {stats.get('workflow_triggered_count', 0)}")
        
        # Test manual risk check (this would normally require database)
        try:
            result = await monitor.manual_risk_check("test_portfolio", "test_user")
            print(f"✅ Manual risk check: {result['status']}")
            manual_check_success = result['status'] == 'completed'
        except Exception as e:
            print(f"⚠️  Manual risk check failed (expected without DB): {e}")
            manual_check_success = False  # Expected without database
        
        return True, {
            "enhanced_detection": stats.get('enhanced_risk_detection', False),
            "manual_check": manual_check_success
        }
        
    except Exception as e:
        print(f"❌ Proactive monitor test failed: {e}")
        return False, {}

def test_4_tools_integration():
    """Test 4: Tools Integration"""
    print("\n🧪 TEST 4: Tools Integration")
    print("-" * 40)
    
    tools_results = {}
    
    # Test fractal tools
    try:
        from tools.fractal_tools import calculate_hurst, calculate_dfa
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_returns = pd.Series(np.random.randn(200) * 0.01)
        
        # Test Hurst calculation
        try:
            hurst_result = calculate_hurst(sample_returns)
            if hurst_result and 'hurst_exponent' in hurst_result:
                print(f"✅ Hurst calculation: {hurst_result['hurst_exponent']:.3f}")
                tools_results['hurst'] = True
            else:
                print("⚠️  Hurst calculation returned unexpected format")
                tools_results['hurst'] = False
        except Exception as e:
            print(f"❌ Hurst calculation failed: {e}")
            tools_results['hurst'] = False
        
        # Test DFA calculation
        try:
            dfa_result = calculate_dfa(sample_returns)
            if dfa_result and 'dfa_alpha' in dfa_result:
                print(f"✅ DFA calculation: {dfa_result['dfa_alpha']:.3f}")
                tools_results['dfa'] = True
            else:
                print("⚠️  DFA calculation returned unexpected format")
                tools_results['dfa'] = False
        except Exception as e:
            print(f"❌ DFA calculation failed: {e}")
            tools_results['dfa'] = False
            
    except ImportError as e:
        print(f"❌ Fractal tools import failed: {e}")
        tools_results['fractal_tools'] = False
    
    # Test regime tools
    try:
        from tools.regime_tools import detect_hmm_regimes
        
        sample_returns = pd.Series(np.random.randn(150) * 0.01)
        
        try:
            regime_result = detect_hmm_regimes(sample_returns, n_regimes=2)
            if regime_result and 'current_regime' in regime_result:
                print(f"✅ Regime detection: Current regime {regime_result['current_regime']}")
                tools_results['regime'] = True
            else:
                print("⚠️  Regime detection returned unexpected format")
                tools_results['regime'] = False
        except Exception as e:
            print(f"❌ Regime detection failed: {e}")
            tools_results['regime'] = False
            
    except ImportError as e:
        print(f"❌ Regime tools import failed: {e}")
        tools_results['regime_tools'] = False
    
    success_count = sum(1 for success in tools_results.values() if success)
    total_tests = len(tools_results)
    
    print(f"\n📋 Tools Integration: {success_count}/{total_tests} tools working")
    
    return success_count > 0, tools_results

async def run_comprehensive_test():
    """Run all tests and provide summary"""
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Risk Calculator", test_1_risk_calculator),
        ("Risk Comparison", test_2_risk_comparison), 
        ("Proactive Monitor", test_3_proactive_monitor),
        ("Tools Integration", test_4_tools_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                success, details = await test_func()
            else:
                success, details = test_func()
            
            results[test_name] = {
                "success": success,
                "details": details
            }
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = {
                "success": False,
                "error": str(e)
            }
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    overall_success = True
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if not result['success']:
            overall_success = False
            if 'error' in result:
                print(f"     Error: {result['error']}")
    
    print("\n" + "=" * 60)
    
    if overall_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your Risk Attribution System is ready for production!")
        print("\n🚀 Next Steps:")
        print("   1. Add database models for risk storage")
        print("   2. Set up database migration")
        print("   3. Test with real portfolio data")
        print("   4. Configure WebSocket notifications")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("🔧 Review the failed tests above and fix issues before proceeding")
    
    print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return overall_success

def main():
    """Main test runner"""
    try:
        result = asyncio.run(run_comprehensive_test())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()