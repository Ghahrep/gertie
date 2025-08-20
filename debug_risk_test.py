# debug_risk_test.py
import sys
import os

print("🔍 Debug: Script starting...")
print(f"🔍 Python version: {sys.version}")
print(f"🔍 Current directory: {os.getcwd()}")
print(f"🔍 Python path: {sys.path[:3]}...")  # Show first 3 paths

try:
    print("🔍 Step 1: Testing basic imports...")
    import asyncio
    print("✅ asyncio imported")
    
    import pandas as pd
    print("✅ pandas imported")
    
    import numpy as np
    print("✅ numpy imported")
    
    print("🔍 Step 2: Testing yfinance...")
    import yfinance as yf
    print("✅ yfinance imported")
    
    print("🔍 Step 3: Testing basic yfinance functionality...")
    # Quick test to see if yfinance works
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    print(f"✅ yfinance working - AAPL current price: ~${info.get('currentPrice', 'N/A')}")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    print("🔍 Error details:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🔍 Step 4: Testing project structure...")
print(f"🔍 Looking for services directory: {os.path.exists('services')}")
print(f"🔍 Looking for tools directory: {os.path.exists('tools')}")

if os.path.exists('services'):
    print(f"🔍 Services contents: {os.listdir('services')}")
else:
    print("❌ Services directory not found!")

if os.path.exists('tools'):
    print(f"🔍 Tools contents: {os.listdir('tools')}")
else:
    print("❌ Tools directory not found!")

print("\n🔍 Step 5: Testing project imports...")
try:
    print("🔍 Attempting to import risk_calculator...")
    from services.risk_calculator import create_risk_calculator
    print("✅ risk_calculator imported successfully")
    
    print("🔍 Creating risk calculator instance...")
    calculator = create_risk_calculator()
    print("✅ Risk calculator instance created")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔍 This suggests the services/risk_calculator.py file doesn't exist or has issues")
    
    # Let's check what files actually exist
    if os.path.exists('services'):
        files = [f for f in os.listdir('services') if f.endswith('.py')]
        print(f"🔍 Python files in services/: {files}")
    
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()

print("\n🔍 Step 6: Simple risk calculation test...")
try:
    # Only proceed if we got this far
    if 'calculator' in locals():
        print("🔍 Testing portfolio risk calculation...")
        print("⏳ This may take 30-60 seconds to fetch data...")
        
        risk_metrics = calculator.calculate_portfolio_risk_from_tickers(
            tickers=['AAPL', 'MSFT'],  # Reduced to 2 tickers for faster testing
            weights=[0.6, 0.4],
            lookback_days=100  # Reduced lookback for faster testing
        )
        
        if risk_metrics:
            print("✅ SUCCESS! Risk calculation working!")
            print(f"📊 Risk Score: {risk_metrics.risk_score}")
            print(f"📈 Volatility: {risk_metrics.volatility:.2%}")
            print(f"📉 Beta: {risk_metrics.beta:.2f}")
        else:
            print("❌ Risk calculation returned None")
    else:
        print("⏭️  Skipping risk calculation (import failed)")
        
except Exception as e:
    print(f"❌ Risk calculation error: {e}")
    import traceback
    traceback.print_exc()

print("\n🔍 Debug complete!")
print("=" * 50)