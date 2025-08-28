# Test file: test_risk_integration.py
from services.risk_calculator import create_risk_calculator

async def test_risk_calculator():
    calculator = create_risk_calculator()
    
    # Test with your actual portfolio tickers
    risk_metrics = calculator.calculate_portfolio_risk_from_tickers(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        weights=[0.4, 0.3, 0.3],
        lookback_days=252
    )
    
    if risk_metrics:
        print("✅ Risk Calculator Working!")
        print(f"Risk Score: {risk_metrics.risk_score}/100")
        print(f"Volatility: {risk_metrics.volatility:.2%}")
        return True
    return False