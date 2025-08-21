"""
MCP Testing Setup Script
========================
Sets up the complete testing environment for the MCP system.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_test_structure():
    """Create test directory structure"""
    print("📁 Creating test directory structure...")
    
    directories = [
        "tests",
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "tests/fixtures",
        "tests/data",
        "htmlcov"  # Coverage reports
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created: {directory}")

def create_requirements_file():
    """Create requirements-test.txt file"""
    print("📝 Creating test requirements file...")
    
    requirements = """# Core testing framework
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-xdist>=3.0.0
pytest-html>=3.1.0
pytest-timeout>=2.1.0

# HTTP testing
httpx>=0.24.0
requests>=2.31.0

# FastAPI testing
fastapi[all]>=0.100.0
uvicorn>=0.22.0

# Data validation
pydantic>=2.0.0

# Performance monitoring
psutil>=5.9.0
memory-profiler>=0.60.0

# Async utilities
aiofiles>=23.1.0
"""
    
    with open("requirements-test.txt", "w") as f:
        f.write(requirements)
    
    print("   ✅ Created: requirements-test.txt")

def create_pytest_config():
    """Create pytest.ini configuration"""
    print("⚙️  Creating pytest configuration...")
    
    config = """[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    --strict-markers
    --strict-config
    --disable-warnings
    --tb=short
testpaths = tests .
asyncio_mode = auto
timeout = 300
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, multi-component)
    performance: Performance tests (may be slow)
    slow: Slow running tests
    asyncio: Async tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::pytest.PytestUnraisableExceptionWarning
"""
    
    with open("pytest.ini", "w") as f:
        f.write(config)
    
    print("   ✅ Created: pytest.ini")

def create_conftest():
    """Create conftest.py if it doesn't exist"""
    if not Path("conftest.py").exists():
        print("🔧 Creating conftest.py...")
        
        conftest_content = '''"""Pytest configuration and shared fixtures"""
import pytest
import asyncio
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment before each test"""
    # Clean up any global state
    yield
    # Cleanup after test
'''
        
        with open("conftest.py", "w") as f:
            f.write(conftest_content)
        
        print("   ✅ Created: conftest.py")

def install_dependencies():
    """Install test dependencies"""
    print("📦 Installing test dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
        ], check=True)
        print("   ✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        return False

def create_test_data():
    """Create sample test data files"""
    print("📊 Creating test data files...")
    
    # Sample portfolio data
    portfolio_data = {
        "test_portfolio_1": {
            "portfolio_id": "test_portfolio_1",
            "user_id": "test_user_123",
            "holdings": [
                {"symbol": "AAPL", "shares": 100, "cost_basis": 150.00},
                {"symbol": "MSFT", "shares": 50, "cost_basis": 300.00},
                {"symbol": "TSLA", "shares": 25, "cost_basis": 800.00}
            ],
            "cash_balance": 10000.00,
            "risk_tolerance": "moderate"
        }
    }
    
    import json
    with open("tests/data/sample_portfolios.json", "w") as f:
        json.dump(portfolio_data, f, indent=2)
    
    # Sample market data
    market_data = {
        "AAPL": {"price": 175.50, "change": 2.50, "volume": 50000000},
        "MSFT": {"price": 350.25, "change": -1.75, "volume": 30000000},
        "TSLA": {"price": 250.00, "change": 15.00, "volume": 80000000}
    }
    
    with open("tests/data/sample_market_data.json", "w") as f:
        json.dump(market_data, f, indent=2)
    
    print("   ✅ Created test data files")

def create_github_actions():
    """Create GitHub Actions workflow for CI/CD"""
    print("🚀 Creating GitHub Actions workflow...")
    
    # Create .github/workflows directory
    Path(".github/workflows").mkdir(parents=True, exist_ok=True)
    
    workflow = """name: MCP Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install -r requirements.txt
    
    - name: Run unit tests
      run: |
        python run_tests.py unit --coverage
    
    - name: Run integration tests
      run: |
        python run_tests.py integration
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install -r requirements.txt
    
    - name: Run performance tests
      run: |
        python run_tests.py performance
"""
    
    with open(".github/workflows/test.yml", "w") as f:
        f.write(workflow)
    
    print("   ✅ Created: .github/workflows/test.yml")

def create_makefile():
    """Create Makefile for common test commands"""
    print("🛠️  Creating Makefile...")
    
    makefile_content = """# MCP Testing Makefile

.PHONY: help test test-unit test-integration test-performance test-all coverage clean

help:
	@echo "Available commands:"
	@echo "  make test-unit       - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-performance - Run performance tests"
	@echo "  make test-all        - Run all tests"
	@echo "  make coverage        - Run tests with coverage"
	@echo "  make clean          - Clean test artifacts"

test-unit:
	python run_tests.py unit

test-integration:
	python run_tests.py integration

test-performance:
	python run_tests.py performance

test-all:
	python run_tests.py all

coverage:
	python run_tests.py all --coverage

smoke:
	python run_tests.py smoke

ci:
	python run_tests.py ci

clean:
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf test-results.xml
	rm -rf ci-test-report.json
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

install-deps:
	pip install -r requirements-test.txt

setup:
	python setup_testing.py
"""
    
    with open("Makefile", "w") as f:
        f.write(makefile_content)
    
    print("   ✅ Created: Makefile")

def verify_setup():
    """Verify the testing setup"""
    print("🔍 Verifying test setup...")
    
    # Check if key files exist
    required_files = [
    "requirements-test.txt",
    "pytest.ini", 
    "conftest.py",
    "run_mcp_tests.py",
    "mcp/test_mcp_comprehensive.py"  # Updated path
]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ❌ Missing files: {', '.join(missing_files)}")
        return False
    
    # Try importing pytest
    try:
        import pytest
        print(f"   ✅ pytest {pytest.__version__} available")
    except ImportError:
        print("   ❌ pytest not available")
        return False
    
    # Check if we can run a simple test
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "--version"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ pytest executable working")
        else:
            print("   ❌ pytest not working properly")
            return False
    except Exception as e:
        print(f"   ❌ Error testing pytest: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🧪 MCP Testing Environment Setup")
    print("=" * 50)
    
    # Create directory structure
    create_test_structure()
    
    # Create configuration files
    create_requirements_file()
    create_pytest_config()
    create_conftest()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return 1
    
    # Create test data and additional files
    create_test_data()
    create_github_actions()
    create_makefile()
    
    # Verify setup
    if verify_setup():
        print("\n✅ Test environment setup complete!")
        print("\nNext steps:")
        print("1. Run smoke test: python run_tests.py smoke")
        print("2. Run all tests: python run_tests.py all")
        print("3. Interactive mode: python run_tests.py")
        print("4. Or use Makefile: make test-unit")
        return 0
    else:
        print("\n❌ Setup verification failed")
        return 1

if __name__ == "__main__":
    exit(main())