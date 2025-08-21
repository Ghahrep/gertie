# run_mcp_tests.py
"""
MCP Test Runner
==============
Simplified test runner specifically for MCP components
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

class MCPTestRunner:
    """Simple test runner for MCP system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        
    def run_smoke_test(self):
        """Quick smoke test to verify basic functionality"""
        print("💨 Running MCP Smoke Test...")
        
        # Test if we can import MCP modules
        try:
            sys.path.insert(0, str(self.project_root))
            from mcp.schemas import JobRequest, AgentRegistration
            print("   ✅ MCP schemas import working")
        except ImportError as e:
            print(f"   ❌ MCP import failed: {e}")
            return False
        
        # Test if we can run pytest
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "--version"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ pytest working")
            else:
                print("   ❌ pytest not working")
                return False
        except Exception as e:
            print(f"   ❌ pytest error: {e}")
            return False
        
        print("✅ Smoke test passed!")
        return True
    
    def run_comprehensive_test(self, verbose=False):
        """Run the comprehensive MCP test"""
        print("🧪 Running MCP Comprehensive Test...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        # Check multiple possible locations for the comprehensive test
        test_locations = [
            "tests/mcp/test_mcp_comprehensive.py",
            "tests\\mcp\\test_mcp_comprehensive.py", 
            "mcp/test_mcp_comprehensive.py",
            "test_mcp_comprehensive.py"
        ]
        
        test_file = None
        for location in test_locations:
            if Path(location).exists():
                test_file = location
                break
        
        if not test_file:
            print("❌ test_mcp_comprehensive.py not found in any expected location")
            print("   Searched locations:")
            for loc in test_locations:
                print(f"     {loc}")
            return False
        
        cmd.append(test_file)
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend(["--tb=short"])
        
        try:
            result = subprocess.run(cmd, text=True)
            if result.returncode == 0:
                print("✅ Comprehensive test passed!")
                return True
            else:
                print("❌ Some tests failed")
                return False
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
    
    def run_all_mcp_tests(self, verbose=False):
        """Run all MCP-related tests"""
        print("🚀 Running All MCP Tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        # Check for test directories
        test_paths = []
        
        # Check tests/mcp directory (most likely location)
        if Path("tests/mcp").exists():
            test_paths.append("tests/mcp/")
        elif Path("tests\\mcp").exists():
            test_paths.append("tests\\mcp\\")
        
        # Check mcp directory for any test files
        if Path("mcp").exists():
            mcp_tests = list(Path("mcp").glob("test*.py"))
            if mcp_tests:
                test_paths.extend([str(f) for f in mcp_tests])
        
        # Check root directory for comprehensive test
        if Path("test_mcp_comprehensive.py").exists():
            test_paths.append("test_mcp_comprehensive.py")
        
        if not test_paths:
            print("❌ No MCP test files found")
            print("   Searched locations:")
            print("     tests/mcp/")
            print("     mcp/test*.py")
            print("     test_mcp_comprehensive.py")
            return False
        
        print(f"   Found test paths: {test_paths}")
        cmd.extend(test_paths)
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend(["--tb=short", "--durations=5"])
        
        try:
            result = subprocess.run(cmd, text=True)
            if result.returncode == 0:
                print("✅ All MCP tests passed!")
                return True
            else:
                print("❌ Some tests failed")
                return False
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
    
    def run_with_coverage(self):
        """Run tests with coverage"""
        print("📊 Running MCP Tests with Coverage...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        # Find test paths for coverage
        test_paths = []
        
        if Path("tests/mcp").exists():
            test_paths.append("tests/mcp/")
        elif Path("tests\\mcp").exists():
            test_paths.append("tests\\mcp\\")
        
        if Path("mcp/test_mcp_comprehensive.py").exists():
            test_paths.append("mcp/test_mcp_comprehensive.py")
        
        if not test_paths:
            print("❌ No test files found for coverage")
            return False
        
        cmd.extend(test_paths)
        cmd.extend([
            "--cov=mcp",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "-v"
        ])
        
        try:
            result = subprocess.run(cmd, text=True)
            if result.returncode == 0:
                print("✅ Tests with coverage completed!")
                print("📄 Coverage report: htmlcov/index.html")
                return True
            else:
                print("❌ Some tests failed")
                return False
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
    
    def list_available_tests(self):
        """List all available test files"""
        print("📋 Available MCP Test Files:")
        
        # Check tests/mcp directory
        tests_mcp = Path("tests/mcp")
        if tests_mcp.exists():
            print("  📁 tests/mcp/:")
            for test_file in tests_mcp.glob("test*.py"):
                print(f"     ✅ {test_file}")
        
        # Check mcp directory
        mcp_dir = Path("mcp")
        if mcp_dir.exists():
            mcp_tests = list(mcp_dir.glob("test*.py"))
            if mcp_tests:
                print("  📁 mcp/:")
                for test_file in mcp_tests:
                    print(f"     ✅ {test_file}")
        
        # Check root directory
        root_tests = list(Path(".").glob("test_mcp*.py"))
        if root_tests:
            print("  📁 root directory:")
            for test_file in root_tests:
                print(f"     ✅ {test_file}")
    
    def interactive_menu(self):
        """Interactive test menu"""
        while True:
            print("\n" + "=" * 50)
            print("🧪 MCP TEST RUNNER")
            print("=" * 50)
            print("1. Smoke test (quick)")
            print("2. Comprehensive test")
            print("3. All MCP tests")
            print("4. Tests with coverage")
            print("5. List available tests")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                self.run_smoke_test()
            elif choice == "2":
                self.run_comprehensive_test(verbose=True)
            elif choice == "3":
                self.run_all_mcp_tests(verbose=True)
            elif choice == "4":
                self.run_with_coverage()
            elif choice == "5":
                self.list_available_tests()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MCP Test Runner")
    parser.add_argument(
        "test_type",
        nargs="?",
        choices=["smoke", "comprehensive", "all", "coverage", "list"],
        default="interactive",
        help="Type of test to run"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = MCPTestRunner()
    
    try:
        if args.test_type == "smoke":
            success = runner.run_smoke_test()
            return 0 if success else 1
        elif args.test_type == "comprehensive":
            success = runner.run_comprehensive_test(verbose=args.verbose)
            return 0 if success else 1
        elif args.test_type == "all":
            success = runner.run_all_mcp_tests(verbose=args.verbose)
            return 0 if success else 1
        elif args.test_type == "coverage":
            success = runner.run_with_coverage()
            return 0 if success else 1
        elif args.test_type == "list":
            runner.list_available_tests()
            return 0
        else:
            runner.interactive_menu()
            return 0
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())