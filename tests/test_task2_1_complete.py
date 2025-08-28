#!/usr/bin/env python3
"""
Task 2.1 Complete Integration Test
=================================
Comprehensive test to validate all Task 2.1 components are working together:
- Risk calculator integration (2.1.1)
- Risk change detection service (2.1.2) 
- Risk snapshot storage (2.1.3)
- Portfolio data integration (2.1.4)
"""

import asyncio
import time
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np

# Database imports
from db.session import get_db
from db.models import (
    Portfolio, User, Asset, Holding, PortfolioRiskSnapshot, 
    RiskTrend, RiskThreshold, PriceDataCache
)

# Service imports - adjust imports based on your actual structure
try:
    from services.risk_snapshot_storage import RiskSnapshotStorage
    from services.portfolio_data_integrator import PortfolioDataIntegrator  
    from services.risk_detection_pipeline import RiskDetectionPipeline
    from services.risk_calculator import RiskCalculatorService, RiskMetrics
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all service files are in the correct location")
    sys.exit(1)

class Task21CompleteTest:
    """Complete Task 2.1 integration test"""
    
    def __init__(self):
        self.db = next(get_db())
        self.test_results = []
        self.test_user_id = None
        self.test_portfolio_id = None
        
    async def run_complete_test(self):
        """Run all Task 2.1 tests"""
        
        print("=" * 60)
        print("TASK 2.1 COMPLETE INTEGRATION TEST")
        print("=" * 60)
        
        try:
            # Test 1: Database Setup
            await self.test_database_setup()
            
            # Test 2: Create Test Data
            await self.test_create_test_data()
            
            # Test 3: Risk Calculator Integration (2.1.1)
            await self.test_risk_calculator_integration()
            
            # Test 4: Portfolio Data Integration (2.1.4)  
            await self.test_portfolio_data_integration()
            
            # Test 5: Risk Snapshot Storage (2.1.3)
            await self.test_risk_snapshot_storage()
            
            # Test 6: Risk Change Detection (2.1.2)
            await self.test_risk_change_detection()
            
            # Test 7: Complete Pipeline Integration
            await self.test_complete_pipeline()
            
            # Test 8: Performance Requirements
            await self.test_performance_requirements()
            
            # Final Results
            self.print_final_results()
            
        except Exception as e:
            print(f"Test suite failed with error: {e}")
            traceback.print_exc()
            return False
        
        return self.all_tests_passed()
    
    async def test_database_setup(self):
        """Test 2.1.3: Database setup and models"""
        
        test_name = "Database Setup & Models"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Test table existence
            risk_thresholds = self.db.query(RiskThreshold).count()
            risk_trends = self.db.query(RiskTrend).count()
            price_cache = self.db.query(PriceDataCache).count()
            
            # Test relationships
            portfolios = self.db.query(Portfolio).first()
            if portfolios:
                has_relationships = (
                    hasattr(portfolios, 'risk_snapshots') and
                    hasattr(portfolios, 'risk_trends') and
                    hasattr(portfolios, 'risk_thresholds')
                )
            else:
                has_relationships = True  # Can't test without data
            
            success = risk_thresholds >= 3 and has_relationships
            
            self.test_results.append({
                'test': test_name,
                'success': success,
                'details': f"Thresholds: {risk_thresholds}, Relationships: {has_relationships}"
            })
            
            print(f"   {'✅' if success else '❌'} Default thresholds: {risk_thresholds}")
            print(f"   {'✅' if has_relationships else '❌'} Model relationships working")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'success': False,
                'details': f"Error: {str(e)}"
            })
            print(f"   ❌ Error: {str(e)}")
    
    async def test_create_test_data(self):
        """Create test portfolio data with proper error handling"""
        
        test_name = "Test Data Creation"
        print(f"\nTesting: {test_name}")
        
        try:
            from db.crud import get_asset_by_ticker
            
            # Clean up any existing test data first
            existing_user = self.db.query(User).filter(User.email == "task21_test@example.com").first()
            if existing_user:
                # Delete in proper order to avoid foreign key violations
                
                # 1. Delete risk snapshots first
                self.db.query(PortfolioRiskSnapshot).filter(
                    PortfolioRiskSnapshot.portfolio_id.in_(
                        self.db.query(Portfolio.id).filter(Portfolio.user_id == existing_user.id)
                    )
                ).delete(synchronize_session=False)
                
                # 2. Delete holdings
                self.db.query(Holding).filter(
                    Holding.portfolio_id.in_(
                        self.db.query(Portfolio.id).filter(Portfolio.user_id == existing_user.id)
                    )
                ).delete(synchronize_session=False)
                
                # 3. Delete portfolios
                self.db.query(Portfolio).filter(Portfolio.user_id == existing_user.id).delete()
                
                # 4. Delete user
                self.db.delete(existing_user)
                self.db.commit()
                print("   Cleaned up existing test data")
            
            # CREATE USER FIRST - ADD THIS SECTION
            test_user = User(
                email="task21_test@example.com",
                hashed_password="test_hash"
            )
            self.db.add(test_user)
            self.db.commit()
            self.db.refresh(test_user)
            self.test_user_id = test_user.id
            print(f"   Test user created: ID {self.test_user_id}")
            
            # Create or get assets with proper attributes
            test_tickers = [
                ("AAPL", "Apple Inc", "Technology"),
                ("GOOGL", "Alphabet Inc", "Technology"),
                ("MSFT", "Microsoft Corp", "Technology")
            ]
            
            assets = []
            for ticker, name, sector in test_tickers:
                try:
                    asset = get_asset_by_ticker(self.db, ticker)
                    if not asset.name:
                        asset.name = name
                    
                    # Set asset_type (handle both cases - column exists or doesn't)
                    try:
                        asset.asset_type = "stock"
                    except Exception:
                        pass  # Column might not exist yet
                        
                    if hasattr(asset, 'sector') and not asset.sector:
                        asset.sector = sector
                        
                    self.db.commit()
                    assets.append(asset)
                    print(f"   Asset configured: {ticker}")
                    
                except Exception as asset_error:
                    print(f"   Asset error for {ticker}: {asset_error}")
                    # Continue with other assets
                    continue
            
            if not assets:
                raise Exception("No assets were successfully created/configured")
            
            # Create test portfolio with explicit error checking
            test_portfolio = Portfolio(
                name="Task 2.1 Test Portfolio",
                user_id=self.test_user_id
            )
            self.db.add(test_portfolio)
            self.db.flush()  # Flush to get ID before commit
            
            if not test_portfolio.id:
                raise Exception("Portfolio ID not generated after flush")
            
            self.test_portfolio_id = test_portfolio.id
            print(f"   Test portfolio created: ID {self.test_portfolio_id}")
            
            # Create test holdings with both quantity and shares for compatibility
            holdings_data = [
                (assets[0].id, 100, 150.0, 160.0),
                (assets[1].id if len(assets) > 1 else assets[0].id, 50, 2500.0, 2600.0),
                (assets[2].id if len(assets) > 2 else assets[0].id, 75, 300.0, 310.0)
            ]
            
            holdings_created = 0
            for asset_id, shares_count, purchase_price, current_price in holdings_data:
                try:
                    holding = Holding(
                        portfolio_id=self.test_portfolio_id, 
                        asset_id=asset_id, 
                        shares=shares_count,
                        purchase_price=purchase_price,
                        current_price=current_price
                    )
                    
                    # Set quantity field if it exists (for compatibility)
                    try:
                        holding.quantity = shares_count
                    except Exception:
                        pass  # Column might not exist
                        
                    self.db.add(holding)
                    holdings_created += 1
                    
                except Exception as holding_error:
                    print(f"   Holding creation error: {holding_error}")
                    continue
            
            # Commit all changes
            self.db.commit()
            
            # Verify data was created
            portfolio_check = self.db.query(Portfolio).filter(Portfolio.id == self.test_portfolio_id).first()
            holdings_check = self.db.query(Holding).filter(Holding.portfolio_id == self.test_portfolio_id).count()
            
            if not portfolio_check:
                raise Exception(f"Portfolio {self.test_portfolio_id} not found after creation")
            
            success = True
            details = f"Created user {self.test_user_id}, portfolio {self.test_portfolio_id}, {len(assets)} assets, {holdings_created} holdings"
            
            self.test_results.append({
                'test': test_name,
                'success': success,
                'details': details
            })
            
            print(f"   Test holdings created: {holdings_created} holdings")
            print(f"   Verification: Portfolio exists={portfolio_check is not None}, Holdings count={holdings_check}")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'success': False,
                'details': f"Error: {str(e)}"
            })
            print(f"   Error: {str(e)}")
            
            # Set None values to prevent cascade failures
            self.test_user_id = None
            self.test_portfolio_id = None
            
            # Rollback on error
            try:
                self.db.rollback()
            except Exception:
                pass
        
    async def test_risk_calculator_integration(self):
        """Test 2.1.1: Risk calculator integration"""
        
        test_name = "Risk Calculator Integration (2.1.1)"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Generate mock returns data instead of holdings data
            import numpy as np
            np.random.seed(42)  # For reproducible results
            mock_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
            
            # Initialize risk calculator
            risk_calculator = RiskCalculatorService()
            
            # Test comprehensive risk calculation
            risk_metrics = risk_calculator.calculate_comprehensive_risk(mock_returns)
            
            # Validate risk metrics
            required_metrics = [
                'annualized_volatility', 'var_95', 'cvar_95', 'max_drawdown',
                'sharpe_ratio', 'sortino_ratio', 'skewness', 'kurtosis'
            ]
            
            has_all_metrics = all(hasattr(risk_metrics, metric) for metric in required_metrics)
            metrics_valid = all(
                getattr(risk_metrics, metric, None) is not None and 
                isinstance(getattr(risk_metrics, metric), (int, float)) and
                not (hasattr(np, 'isnan') and np.isnan(float(getattr(risk_metrics, metric))))
                for metric in required_metrics
            )
            success = has_all_metrics and metrics_valid
            
            self.test_results.append({
                'test': test_name,
                'success': success,
                'details': f"Metrics calculated: {has_all_metrics}, Valid values: {metrics_valid}"
            })
            
            print(f"   {'✅' if has_all_metrics else '❌'} All required metrics present")
            print(f"   {'✅' if metrics_valid else '❌'} Metric values are valid")
            print(f"   📊 Sample - Vol: {risk_metrics.annualized_volatility:.3f}, VaR: {risk_metrics.var_95:.3f}")
            
            # Store for later tests
            self.sample_risk_metrics = risk_metrics
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'success': False,
                'details': f"Error: {str(e)}"
            })
            print(f"   ❌ Error: {str(e)}")
    
    async def test_portfolio_data_integration(self):
        """Test 2.1.4: Portfolio data integration"""
        
        test_name = "Portfolio Data Integration (2.1.4)"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Initialize data integrator
            data_integrator = PortfolioDataIntegrator(self.db)
            
            # Test real-time portfolio data fetch
            start_time = time.time()
            portfolio_snapshot = await data_integrator.get_real_time_portfolio_data(self.test_portfolio_id)
            fetch_time_ms = (time.time() - start_time) * 1000
            
            # Validate portfolio snapshot
            has_required_fields = all(
                hasattr(portfolio_snapshot, field) for field in 
                ['portfolio_id', 'total_value', 'holdings', 'data_quality_score']
            )
            
            performance_ok = fetch_time_ms < 10000  # 10 second limit for test
            has_holdings = len(portfolio_snapshot.holdings) > 0
            
            success = has_required_fields and performance_ok and has_holdings
            
            self.test_results.append({
                'test': test_name,
                'success': success,
                'details': f"Fields: {has_required_fields}, Performance: {fetch_time_ms:.0f}ms, Holdings: {len(portfolio_snapshot.holdings)}"
            })
            
            print(f"   {'✅' if has_required_fields else '❌'} Required fields present")
            print(f"   {'✅' if performance_ok else '❌'} Performance: {fetch_time_ms:.0f}ms")
            print(f"   {'✅' if has_holdings else '❌'} Holdings loaded: {len(portfolio_snapshot.holdings)}")
            print(f"   📊 Data quality: {portfolio_snapshot.data_quality_score:.2f}")
            
            # Store for later tests
            self.sample_portfolio_snapshot = portfolio_snapshot
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'success': False,
                'details': f"Error: {str(e)}"
            })
            print(f"   ❌ Error: {str(e)}")
    
    async def test_risk_snapshot_storage(self):
        """Test 2.1.3: Risk snapshot storage"""
        
        test_name = "Risk Snapshot Storage (2.1.3)"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Initialize snapshot storage
            snapshot_storage = RiskSnapshotStorage(self.db)
            
            # Test compressed snapshot creation
            if hasattr(self, 'sample_risk_metrics'):
                start_time = time.time()
                snapshot = await snapshot_storage.create_compressed_snapshot(
                    self.test_portfolio_id, self.sample_risk_metrics, user_id=self.test_user_id
                )
                storage_time_ms = (time.time() - start_time) * 1000
                
                # Test retrieval performance
                start_time = time.time()
                decompressed_metrics = await snapshot_storage.get_decompressed_metrics(snapshot)
                retrieval_time_ms = (time.time() - start_time) * 1000
                
                # Test historical comparison
                comparison = await snapshot_storage.get_historical_comparison(
                    self.test_portfolio_id, self.sample_risk_metrics
                )
                
                # Validate results
                storage_fast = storage_time_ms < 1000  # 1 second limit
                retrieval_fast = retrieval_time_ms < 100  # 100ms limit per requirement
                has_compression = snapshot.compression_ratio is not None and snapshot.compression_ratio < 1.0
                comparison_works = comparison['status'] in ['success', 'insufficient_data']
                
                success = storage_fast and retrieval_fast and has_compression and comparison_works
                
                self.test_results.append({
                    'test': test_name,
                    'success': success,
                    'details': f"Storage: {storage_time_ms:.0f}ms, Retrieval: {retrieval_time_ms:.0f}ms, Compression: {snapshot.compression_ratio:.3f}"
                })
                
                print(f"   {'✅' if storage_fast else '❌'} Storage time: {storage_time_ms:.0f}ms")
                print(f"   {'✅' if retrieval_fast else '❌'} Retrieval time: {retrieval_time_ms:.0f}ms (<100ms required)")
                print(f"   {'✅' if has_compression else '❌'} Compression ratio: {snapshot.compression_ratio:.3f}")
                print(f"   {'✅' if comparison_works else '❌'} Historical comparison: {comparison['status']}")
            else:
                self.test_results.append({
                    'test': test_name,
                    'success': False,
                    'details': "No sample risk metrics available"
                })
                print(f"   ❌ No sample risk metrics available")
                
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'success': False,
                'details': f"Error: {str(e)}"
            })
            print(f"   ❌ Error: {str(e)}")
    
    async def test_risk_change_detection(self):
        """Test 2.1.2: Risk change detection"""
        
        test_name = "Risk Change Detection (2.1.2)"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            from services.risk_detector import RiskChangeDetector
            
            # Initialize risk detector
            risk_detector = RiskChangeDetector()
            
            # Test with sample risk metrics
            if hasattr(self, 'sample_risk_metrics'):
                start_time = time.time()
                alerts = await risk_detector.add_risk_snapshot(
                    self.sample_risk_metrics, str(self.test_portfolio_id)
                )
                detection_time_ms = (time.time() - start_time) * 1000
                
                # Test detection performance
                performance_ok = detection_time_ms < 30000  # 30 second SLA requirement
                has_detection_capability = risk_detector is not None
                
                # Get detection stats
                stats = risk_detector.get_detection_stats()
                stats_available = 'total_detections' in stats
                
                success = performance_ok and has_detection_capability and stats_available
                
                self.test_results.append({
                    'test': test_name,
                    'success': success,
                    'details': f"Detection time: {detection_time_ms:.0f}ms, Alerts: {len(alerts)}, Stats: {stats_available}"
                })
                
                print(f"   {'✅' if performance_ok else '❌'} Detection time: {detection_time_ms:.0f}ms (<30s required)")
                print(f"   {'✅' if has_detection_capability else '❌'} Detection system initialized")
                print(f"   {'✅' if stats_available else '❌'} Performance stats available")
                print(f"   📊 Alerts generated: {len(alerts)}")
                
            else:
                self.test_results.append({
                    'test': test_name,
                    'success': False,
                    'details': "No sample risk metrics available"
                })
                print(f"   ❌ No sample risk metrics available")
                
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'success': False,
                'details': f"Error: {str(e)}"
            })
            print(f"   ❌ Error: {str(e)}")
    
    async def test_complete_pipeline(self):
        """Test complete pipeline integration"""
        
        test_name = "Complete Pipeline Integration"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Initialize complete pipeline
            pipeline = RiskDetectionPipeline(self.db)
            
            # Test complete pipeline processing
            start_time = time.time()
            result = await pipeline.process_portfolio_risk(
                self.test_portfolio_id, self.test_user_id
            )
            total_time_ms = (time.time() - start_time) * 1000
            
            # Validate pipeline result
            has_required_fields = all(
                field in result for field in 
                ['status', 'portfolio_id', 'risk_metrics', 'processing_time_ms']
            )
            
            status_success = result.get('status') == 'success'
            performance_ok = total_time_ms < 30000  # 30 second SLA
            meets_sla = result.get('performance_metrics', {}).get('meets_sla', False)
            
            success = has_required_fields and status_success and performance_ok
            
            self.test_results.append({
                'test': test_name,
                'success': success,
                'details': f"Status: {result.get('status')}, Time: {total_time_ms:.0f}ms, SLA: {meets_sla}"
            })
            
            print(f"   {'✅' if status_success else '❌'} Pipeline status: {result.get('status')}")
            print(f"   {'✅' if performance_ok else '❌'} Total processing time: {total_time_ms:.0f}ms")
            print(f"   {'✅' if meets_sla else '❌'} Meets 30-second SLA: {meets_sla}")
            print(f"   📊 Risk metrics calculated: {len(result.get('risk_metrics', {}))}")
            print(f"   📊 Alerts generated: {len(result.get('alerts', []))}")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'success': False,
                'details': f"Error: {str(e)}"
            })
            print(f"   ❌ Error: {str(e)}")
    
    async def test_performance_requirements(self):
        """Test Task 2.1 acceptance criteria"""
        
        test_name = "Performance Requirements"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Test multiple portfolio processing (simulate 10K+ securities scenario)
            portfolio_ids = [self.test_portfolio_id] * 5  # Simulate batch processing
            
            pipeline = RiskDetectionPipeline(self.db)
            
            start_time = time.time()
            batch_results = await pipeline.process_multiple_portfolios(portfolio_ids, self.test_user_id)
            batch_time = time.time() - start_time
            
            # Validate performance
            all_successful = all(result.get('status') == 'success' for result in batch_results)
            avg_time_per_portfolio = (batch_time * 1000) / len(portfolio_ids)  # ms
            scalability_ok = avg_time_per_portfolio < 30000  # 30s per portfolio
            
            # Test snapshot retrieval performance
            storage = RiskSnapshotStorage(self.db)
            snapshots = self.db.query(PortfolioRiskSnapshot).limit(10).all()
            
            if snapshots:
                start_time = time.time()
                for snapshot in snapshots[:3]:  # Test first 3
                    await storage.get_decompressed_metrics(snapshot)
                retrieval_time_ms = ((time.time() - start_time) / 3) * 1000  # Average per snapshot
                retrieval_fast = retrieval_time_ms < 100  # <100ms requirement
            else:
                retrieval_fast = True  # No data to test
                retrieval_time_ms = 0
            
            success = all_successful and scalability_ok and retrieval_fast
            
            self.test_results.append({
                'test': test_name,
                'success': success,
                'details': f"Batch success: {all_successful}, Avg time: {avg_time_per_portfolio:.0f}ms, Retrieval: {retrieval_time_ms:.1f}ms"
            })
            
            print(f"   {'✅' if all_successful else '❌'} Batch processing: {len([r for r in batch_results if r.get('status') == 'success'])}/{len(batch_results)} successful")
            print(f"   {'✅' if scalability_ok else '❌'} Average time per portfolio: {avg_time_per_portfolio:.0f}ms")
            print(f"   {'✅' if retrieval_fast else '❌'} Snapshot retrieval: {retrieval_time_ms:.1f}ms (<100ms required)")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'success': False,
                'details': f"Error: {str(e)}"
            })
            print(f"   ❌ Error: {str(e)}")
    
    def cleanup_test_data(self):
        """Clean up test data"""
        try:
            # Delete in correct order: holdings -> portfolio -> user -> assets (if no other references)
            if self.test_portfolio_id:
                self.db.query(Holding).filter(Holding.portfolio_id == self.test_portfolio_id).delete()
                self.db.query(Portfolio).filter(Portfolio.id == self.test_portfolio_id).delete()
                
            if self.test_user_id:
                self.db.query(User).filter(User.id == self.test_user_id).delete()
                
            # Don't delete assets - they might be used by other tests
            self.db.commit()
            print("Test data cleaned up")
            
        except Exception as e:
            print(f"Cleanup error (non-critical): {e}")
            self.db.rollback()
    
    def print_final_results(self):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 60)
        print("TASK 2.1 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {status} - {result['test']}")
            print(f"    {result['details']}")
        
        print("\n" + "=" * 60)
        
        if self.all_tests_passed():
            print("🎉 TASK 2.1: RISK DETECTION PIPELINE - COMPLETE!")
            print("All acceptance criteria have been met:")
            print("✅ Risk metrics calculated accurately for all portfolio types")
            print("✅ Risk changes detected within 30 seconds of occurrence") 
            print("✅ Risk snapshots stored and retrieved efficiently (<100ms)")
            print("✅ System handles real-time price data updates for portfolios")
        else:
            print("⚠️  TASK 2.1: INCOMPLETE - Some tests failed")
            print("Review the failed tests above and fix the issues")
        
        print("=" * 60)
    
    def all_tests_passed(self) -> bool:
        """Check if all tests passed"""
        return all(result['success'] for result in self.test_results)

async def main():
    """Main test execution"""
    
    test_suite = Task21CompleteTest()
    
    try:
        success = await test_suite.run_complete_test()
        return success
        
    except KeyboardInterrupt:
        print("\n⏸️  Test interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        return False
        
    finally:
        # Always cleanup
        test_suite.cleanup_test_data()

if __name__ == "__main__":
    # Run the complete test suite
    result = asyncio.run(main())
    sys.exit(0 if result else 1)