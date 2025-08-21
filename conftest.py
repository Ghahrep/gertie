"""
Pytest configuration and shared fixtures for MCP testing
========================================================
Provides shared test configuration, fixtures, and utilities
for all MCP test modules.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch
from typing import Dict, Any
import tempfile
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure asyncio for testing
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def clean_registries():
    """Clean up global registries before each test"""
    from mcp.server import agent_registry, workflow_engine
    
    # Clear agent registry
    agent_registry.clear()
    
    # Clean workflow engine jobs
    workflow_engine.jobs.clear()
    workflow_engine.active_jobs.clear()
    
    yield
    
    # Clean up after test
    agent_registry.clear()
    workflow_engine.jobs.clear()
    workflow_engine.active_jobs.clear()

@pytest.fixture
def mock_agent_endpoint():
    """Mock agent endpoint for testing"""
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "step_id": "test_step",
            "status": "completed",
            "result": {"mock": "agent_response"},
            "confidence_score": 0.85
        }
        mock_post.return_value = mock_response
        yield mock_post

@pytest.fixture
def temp_config_file():
    """Create temporary configuration file for testing"""
    config_data = {
        "mcp_server": {
            "host": "localhost",
            "port": 8001,
            "timeout": 300
        },
        "agents": {
            "max_concurrent_jobs": 5,
            "default_timeout": 60
        },
        "workflow": {
            "max_rounds": 3,
            "consensus_threshold": 0.6
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file_path = f.name
    
    yield config_file_path
    
    # Clean up
    os.unlink(config_file_path)

@pytest.fixture
def sample_portfolio_context():
    """Sample portfolio context for testing"""
    return {
        "portfolio_id": "test_portfolio_123",
        "user_id": "test_user_456", 
        "holdings": [
            {"symbol": "AAPL", "shares": 100, "cost_basis": 150.00},
            {"symbol": "MSFT", "shares": 50, "cost_basis": 300.00},
            {"symbol": "TSLA", "shares": 25, "cost_basis": 800.00}
        ],
        "cash_balance": 10000.00,
        "risk_tolerance": "moderate",
        "investment_goals": ["retirement", "growth"],
        "time_horizon": "long_term"
    }

@pytest.fixture
def mock_market_data():
    """Mock market data for testing"""
    return {
        "AAPL": {"price": 175.50, "change": 2.50, "volume": 50000000},
        "MSFT": {"price": 350.25, "change": -1.75, "volume": 30000000},
        "TSLA": {"price": 250.00, "change": 15.00, "volume": 80000000},
        "VTI": {"price": 220.00, "change": 0.50, "volume": 10000000},
        "BND": {"price": 85.50, "change": -0.25, "volume": 5000000}
    }

@pytest.fixture
def comprehensive_agent_response():
    """Comprehensive mock agent response"""
    return {
        "step_id": "comprehensive_analysis",
        "status": "completed",
        "result": {
            "risk_metrics": {
                "var_95": 0.0245,
                "max_drawdown": 0.187,
                "sharpe_ratio": 1.34,
                "sortino_ratio": 1.67,
                "beta": 0.89
            },
            "portfolio_analytics": {
                "total_value": 45000.00,
                "asset_allocation": {
                    "stocks": 0.75,
                    "bonds": 0.15,
                    "cash": 0.10
                },
                "sector_exposure": {
                    "technology": 0.60,
                    "healthcare": 0.20,
                    "financials": 0.20
                }
            },
            "recommendations": [
                {
                    "type": "rebalancing",
                    "priority": "high",
                    "description": "Reduce technology overweight",
                    "actions": [
                        {"symbol": "AAPL", "action": "sell", "quantity": 25},
                        {"symbol": "VTI", "action": "buy", "quantity": 50}
                    ]
                },
                {
                    "type": "tax_optimization",
                    "priority": "medium", 
                    "description": "Harvest tax losses",
                    "potential_savings": 1250.00
                }
            ]
        },
        "confidence_score": 0.88,
        "execution_time_ms": 1450,
        "metadata": {
            "model_version": "v2.1",
            "data_sources": ["real_time_market", "portfolio_history"],
            "analysis_timestamp": "2024-01-15T14:30:00Z"
        }
    }

# Test markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower, multi-component)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async tests"
    )