"""
MCP Test Suite Runner
====================
Comprehensive test runner for the MCP system with multiple test categories,
reporting, and CI/CD integration capabilities.
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
import json

class MCPTestRunner:
    """Comprehensive test runner for MCP system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_categories = {
            "unit": {
                "description": "Fast unit tests for individual components",
                "marker": "unit",
                "timeout": 300  # 5 minutes
            },
            "integration": {
                "description": "Integration tests between components", 
                "marker": "integration",
                "timeout": 600  # 10 minutes
            },
            "performance": {
                "description": "Performance and load tests",
                "marker": "performance", 
                "timeout": 900  # 15 minutes
            },
            "all": {
                "description": "All tests including slow tests",
                "marker": None,
                "timeout": 1800  # 30 minutes
            }
        }
    
    def setup_environment(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        
        # Check if in virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("⚠️  Warning: Not in virtual environment")
        
        # Install test dependencies
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
            ], check=True, capture_output=True)
            print("✅ Test dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        
        # Create test directories
        test_dirs = ["tests", "tests/unit", "tests/integration", "tests/performance"]
        for dir_name in test_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        return True
    
    def run_test_category(self, category: str, verbose: bool = False, coverage: bool = False) -> Dict:
        """Run specific test category"""
        if category not in self.test_categories:
            raise ValueError(f"Unknown test category: {category}")
        
        config = self.test_categories[category]
        print(f"\n🧪 Running {category} tests: {config['description']}")
        print("=" * 60)
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]
        
        # Add test files
        if category == "unit":
            cmd.extend([
                "test_mcp_comprehensive.py::TestMCPServer",
                "test_mcp_comprehensive.py::TestWorkflowEngine", 
                "test_mcp_comprehensive.py::TestSchemas",
                "test_server_endpoints.py",
                "test_workflow_engine_detailed.py"
            ])
        elif category == "integration": 
            cmd.extend([
                "test_mcp_comprehensive.py::TestIntegration",
                "test_integration_scenarios.py"
            ])
        elif category == "performance":
            cmd.extend([
                "test_mcp_comprehensive.py::TestPerformance",
                "test_performance_benchmarks.py"
            ])
        else:  # all tests
            cmd.append(".")
        
        # Add markers
        if config["marker"]:
            cmd.extend(["-m", config["marker"]])
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Add coverage
        if coverage:
            cmd.extend([
                "--cov=mcp",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-fail-under=80"
            ])
        
        # Add other options
        cmd.extend([
            "--tb=short",
            "--durations=5",
            f"--timeout={config['timeout']}",
            "--junit-xml=test-results.xml"
        ])
        
        # Run tests
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=config["timeout"])
            execution_time = time.time() - start_time
            
            return {
                "category": category,
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "category": category,
                "success": False,
                "returncode": -1,
                "execution_time": config["timeout"],
                "stdout": "",
                "stderr": f"Tests timed out after {config['timeout']} seconds"
            }
    
    def run_specific_tests(self, test_patterns: List[str], verbose: bool = False) -> Dict:
        """Run specific test patterns"""
        print(f"\n🎯 Running specific tests: {', '.join(test_patterns)}")
        print("=" * 60)
        
        cmd = [sys.executable, "-m", "pytest"]
        cmd.extend(test_patterns)
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend(["--tb=short", "--durations=5"])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        return {
            "category": "specific",
            "success": result.returncode == 0,
            "returncode": result.returncode, 
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def generate_report(self, results: List[Dict], output_file: Optional[str] = None):
        """Generate test report"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_categories": len(results),
            "successful_categories": sum(1 for r in results if r["success"]),
            "total_execution_time": sum(r["execution_time"] for r in results),
            "results": results
        }
        
        # Console report
        print("\n" + "=" * 80)
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        for result in results:
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            print(f"{result['category'].upper():>12}: {status} ({result['execution_time']:.1f}s)")
        
        print(f"\nOverall: {report['successful_categories']}/{report['total_categories']} categories passed")
        print(f"Total time: {report['total_execution_time']:.1f} seconds")
        
        # Failed tests details
        failed_results = [r for r in results if not r["success"]]
        if failed_results:
            print(f"\n❌ FAILED TESTS ({len(failed_results)} categories):")
            for result in failed_results:
                print(f"\n{result['category'].upper()} FAILURES:")
                if result["stderr"]:
                    print(result["stderr"][:500] + "..." if len(result["stderr"]) > 500 else result["stderr"])
        
        # Save JSON report
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Detailed report saved to: {output_file}")
        
        return report
    
    def run_ci_pipeline(self) -> bool:
        """Run CI/CD pipeline tests"""
        print("🚀 Running CI/CD Pipeline Tests")
        print("=" * 80)
        
        pipeline_stages = [
            ("unit", True, False),      # Unit tests with verbose
            ("integration", False, True), # Integration with coverage
            ("performance", False, False) # Performance tests
        ]
        
        results = []
        overall_success = True
        
        for stage, verbose, coverage in pipeline_stages:
            result = self.run_test_category(stage, verbose=verbose, coverage=coverage)
            results.append(result)
            
            if not result["success"]:
                overall_success = False
                print(f"❌ Pipeline failed at {stage} stage")
                break
            else:
                print(f"✅ {stage} stage passed")
        
        self.generate_report(results, "ci-test-report.json")
        
        return overall_success
    
    def run_quick_smoke_test(self) -> bool:
        """Run quick smoke test to verify basic functionality"""
        print("💨 Running Quick Smoke Test...")
        
        smoke_tests = [
            "test_mcp_comprehensive.py::TestMCPServer::test_health_check",
            "test_mcp_comprehensive.py::TestWorkflowEngine::test_create_job",
            "test_mcp_comprehensive.py::TestConsensusBuilder::test_calculate_agent_weights"
        ]
        
        result = self.run_specific_tests(smoke_tests, verbose=True)
        
        if result["success"]:
            print("✅ Smoke test passed - basic functionality working")
        else:
            print("❌ Smoke test failed - basic functionality broken")
        
        return result["success"]
    
    def interactive_test_menu(self):
        """Interactive test selection menu"""
        while True:
            print("\n" + "=" * 60)
            print("🧪 MCP TEST SUITE - INTERACTIVE MENU")
            print("=" * 60)
            print("1. Run all tests")
            print("2. Run unit tests only")
            print("3. Run integration tests only") 
            print("4. Run performance tests only")
            print("5. Run quick smoke test")
            print("6. Run CI/CD pipeline")
            print("7. Run specific test pattern")
            print("8. Exit")
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == "1":
                result = self.run_test_category("all", verbose=True, coverage=True)
                self.generate_report([result])
            elif choice == "2":
                result = self.run_test_category("unit", verbose=True)
                self.generate_report([result])
            elif choice == "3":
                result = self.run_test_category("integration", verbose=True)
                self.generate_report([result])
            elif choice == "4":
                result = self.run_test_category("performance", verbose=True)
                self.generate_report([result])
            elif choice == "5":
                self.run_quick_smoke_test()
            elif choice == "6":
                success = self.run_ci_pipeline()
                print(f"\n🚀 CI Pipeline: {'✅ PASSED' if success else '❌ FAILED'}")
            elif choice == "7":
                pattern = input("Enter test pattern (e.g., test_file.py::TestClass::test_method): ")
                if pattern:
                    result = self.run_specific_tests([pattern], verbose=True)
                    self.generate_report([result])
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MCP Test Suite Runner")
    parser.add_argument(
        "category", 
        nargs="?", 
        choices=["unit", "integration", "performance", "all", "smoke", "ci", "interactive"],
        default="interactive",
        help="Test category to run"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--pattern", "-p", action="append", help="Specific test patterns")
    
    args = parser.parse_args()
    
    runner = MCPTestRunner()
    
    # Setup environment
    if not runner.setup_environment():
        print("❌ Environment setup failed")
        return 1
    
    try:
        if args.category == "interactive":
            runner.interactive_test_menu()
            return 0
        elif args.category == "smoke":
            success = runner.run_quick_smoke_test()
            return 0 if success else 1
        elif args.category == "ci":
            success = runner.run_ci_pipeline()
            return 0 if success else 1
        elif args.pattern:
            result = runner.run_specific_tests(args.pattern, verbose=args.verbose)
            runner.generate_report([result], args.output)
            return 0 if result["success"] else 1
        else:
            result = runner.run_test_category(
                args.category, 
                verbose=args.verbose, 
                coverage=args.coverage
            )
            runner.generate_report([result], args.output)
            return 0 if result["success"] else 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        return 1