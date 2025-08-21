"""
Test utilities and helpers
=========================
"""

import json
import tempfile
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock
from contextlib import contextmanager

class MockAgent:
    """Mock agent for testing"""
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.call_count = 0
        self.last_request = None
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """Mock capability execution"""
        self.call_count += 1
        self.last_request = {"capability": capability, "data": data, "context": context}
        
        # Return capability-specific mock responses
        if capability == "risk_analysis":
            return {
                "var_95": 0.025,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.15,
                "risk_grade": "Moderate"
            }
        elif capability == "tax_optimization":
            return {
                "potential_savings": 2500.00,
                "recommendations": ["Harvest TSLA losses", "Move bonds to 401k"]
            }
        elif capability == "portfolio_analysis":
            return {
                "total_value": 50000.00,
                "allocation": {"stocks": 0.7, "bonds": 0.25, "cash": 0.05},
                "performance": {"ytd_return": 0.12, "annual_volatility": 0.18}
            }
        else:
            return {"result": f"Mock result for {capability}"}

class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_portfolio_data(size: str = "medium") -> Dict:
        """Generate portfolio test data"""
        sizes = {
            "small": {"holdings": 5, "value": 10000},
            "medium": {"holdings": 15, "value": 50000}, 
            "large": {"holdings": 50, "value": 250000}
        }
        
        config = sizes.get(size, sizes["medium"])
        
        holdings = []
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.B", "JNJ", "V"]
        
        for i in range(config["holdings"]):
            symbol = symbols[i % len(symbols)] + (f".{i//len(symbols)}" if i >= len(symbols) else "")
            holdings.append({
                "symbol": symbol,
                "shares": 10 + i * 5,
                "cost_basis": 100 + i * 50,
                "current_price": 120 + i * 60
            })
        
        return {
            "portfolio_id": f"test_portfolio_{size}",
            "user_id": "test_user",
            "holdings": holdings,
            "cash_balance": config["value"] * 0.1,
            "total_value": config["value"],
            "risk_tolerance": "moderate",
            "investment_goals": ["retirement", "growth"]
        }
    
    @staticmethod
    def generate_agent_positions(num_agents: int = 3, agreement_level: str = "mixed") -> List[Dict]:
        """Generate agent positions for consensus testing"""
        positions = []
        
        base_stances = {
            "high": ["bullish", "bullish", "bullish"],  # High agreement
            "medium": ["bullish", "bullish", "bearish"], # Medium agreement  
            "low": ["bullish", "bearish", "neutral"],   # Low agreement
            "mixed": ["conservative", "aggressive", "balanced"]  # Mixed approaches
        }
        
        stances = base_stances.get(agreement_level, base_stances["mixed"])
        
        for i in range(num_agents):
            stance = stances[i % len(stances)]
            positions.append({
                "agent_id": f"agent_{i+1}",
                "stance": f"{stance} approach recommended",
                "key_arguments": [
                    f"{stance.capitalize()} market outlook",
                    f"Risk management from {stance} perspective"
                ],
                "supporting_evidence": [
                    {
                        "type": "analytical",
                        "data": f"Analysis supporting {stance} view",
                        "confidence": 0.7 + (i * 0.1)
                    }
                ],
                "confidence_score": 0.75 + (i * 0.05),
                "risk_assessment": {
                    "primary_risks": [f"{stance}_risk", "market_risk"]
                }
            })
        
        return positions

@contextmanager
def temporary_config_file(config_data: Dict):
    """Create temporary configuration file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name
    
    try:
        yield config_file
    finally:
        import os
        os.unlink(config_file)

def assert_response_structure(response: Dict, required_fields: List[str], optional_fields: List[str] = None):
    """Assert response has required structure"""
    # Check required fields
    for field in required_fields:
        assert field in response, f"Required field '{field}' missing from response"
    
    # Check field types if specified
    if optional_fields:
        for field in optional_fields:
            if field in response:
                assert response[field] is not None, f"Optional field '{field}' should not be None if present"

def measure_execution_time(func, *args, **kwargs):
    """Measure function execution time"""
    import time
    start_time = time.time()
    
    if asyncio.iscoroutinefunction(func):
        result = asyncio.run(func(*args, **kwargs))
    else:
        result = func(*args, **kwargs)
    
    execution_time = time.time() - start_time
    return result, execution_time

class AsyncTestHelper:
    """Helper for async testing patterns"""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
        """Wait for condition to become true"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(f"Operation timed out after {timeout} seconds")


if __name__ == "__main__":
    exit(main())