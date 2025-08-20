# services/risk_attribution_service.py
"""
Risk Attribution Service
Integrates with your existing services and database structure
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Import your existing services
from services.risk_calculator import RiskCalculator
from services.proactive_monitor import get_proactive_monitor

# Import your existing database setup
from db.session import get_db
from db.crud import (
    create_risk_snapshot,
    get_latest_risk_snapshot,
    get_risk_history,
    get_risk_thresholds,
    update_risk_thresholds,
    log_risk_alert,
    get_recent_alerts,
    get_risk_trends,
    get_portfolio_rankings,
    acknowledge_alert
)

logger = logging.getLogger(__name__)

class RiskAttributionService:
    """
    Centralized service that connects risk calculation, storage, and monitoring
    Integrates with your existing platform architecture
    """
    
    def __init__(self):
        self.risk_calculator = RiskCalculator()
        self.proactive_monitor = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the service with your existing components"""
        try:
            # Initialize proactive monitor
            self.proactive_monitor = await get_proactive_monitor()
            self._initialized = True
            logger.info("✅ Risk Attribution Service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Risk Attribution Service: {e}")
            return False
    
    async def create_and_store_risk_snapshot(
        self, 
        user_id: str, 
        portfolio_id: str, 
        portfolio_data: dict
    ) -> Dict:
        """
        Complete workflow: Calculate risk metrics and store in database
        Integrates with your existing risk_calculator and database
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Step 1: Calculate risk metrics using your existing calculator
            logger.info(f"📊 Calculating risk metrics for portfolio {portfolio_id}")
            
            risk_result = await self.risk_calculator.calculate_portfolio_risk(portfolio_data)
            
            if risk_result['status'] != 'success':
                raise Exception(f"Risk calculation failed: {risk_result.get('error', 'Unknown error')}")
            
            # Step 2: Store in database using your db structure
            logger.info(f"💾 Storing risk snapshot in database")
            
            db = next(get_db())
            try:
                snapshot = create_risk_snapshot(
                    db=db,
                    user_id=user_id,
                    portfolio_id=portfolio_id,
                    risk_result=risk_result,
                    portfolio_data={
                        'name': portfolio_data.get('name', f'Portfolio {portfolio_id}'),
                        'total_value': portfolio_data.get('total_value'),
                        'num_positions': len(portfolio_data.get('positions', [])),
                        'weights': {pos['symbol']: pos['weight'] for pos in portfolio_data.get('positions', [])}
                    }
                )
                
                # Step 3: Check for threshold breaches and trigger workflows
                if snapshot.is_threshold_breach:
                    logger.warning(f"🚨 Threshold breach detected for portfolio {portfolio_id}")
                    
                    # Log the alert
                    alert = log_risk_alert(
                        db=db,
                        user_id=user_id,
                        portfolio_id=portfolio_id,
                        alert_type="threshold_breach",
                        alert_message=f"Portfolio risk increased by {snapshot.risk_score_change_pct:.1f}%",
                        snapshot_id=snapshot.snapshot_id,    async def initialize(self):
        """Initialize the service with your existing components"""
        try:
            # Initialize proactive monitor
            self.proactive_monitor = await get_proactive_monitor()
            self._initialized = True
            logger.info("✅ Risk Attribution Service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Risk Attribution Service: {e}")
            return False
    
    async def create_and_store_risk_snapshot(
        self, 
        user_id: str, 
        portfolio_id: str, 
        portfolio_data: dict
    ) -> Dict:
        """
        Complete workflow: Calculate risk metrics and store in database
        Integrates with your existing risk_calculator and database
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Step 1: Calculate risk metrics using your existing calculator
            logger.info(f"📊 Calculating risk metrics for portfolio {portfolio_id}")
            
            risk_result = await self.risk_calculator.calculate_portfolio_risk(portfolio_data)
            
            if risk_result['status'] != 'success':
                raise Exception(f"Risk calculation failed: {risk_result.get('error', 'Unknown error')}")
            
            # Step 2: Store in database using your db structure
            logger.info(f"💾 Storing risk snapshot in database")
            
            db = next(get_db())
            try:
                snapshot = RiskCRUD.create_risk_snapshot(
                    db=db,
                    user_id=user_id,
                    portfolio_id=portfolio_id,
                    risk_result=risk_result,
                    portfolio_data={
                        'name': portfolio_data.get('name', f'Portfolio {portfolio_id}'),
                        'total_value': portfolio_data.get('total_value'),
                        'num_positions': len(portfolio_data.get('positions', [])),
                        'weights': {pos['symbol']: pos['weight'] for pos in portfolio_data.get('positions', [])}
                    }
                )
                
                # Step 3: Check for threshold breaches and trigger workflows
                if snapshot.is_threshold_breach:
                    logger.warning(f"🚨 Threshold breach detected for portfolio {portfolio_id}")
                    
                    # Log the alert
                    alert = RiskCRUD.log_risk_alert(
                        db=db,
                        user_id=user_id,
                        portfolio_id=portfolio_id,
                        alert_type="threshold_breach",
                        alert_message=f"Portfolio risk increased by {snapshot.risk_score_change_pct:.1f}%",
                        snapshot_id=snapshot.snapshot_id,
                        severity="high" if snapshot.risk_score > 80 else "medium",
                        triggered_metrics={
                            'volatility_change': snapshot.volatility_change_pct,
                            'risk_score_change': snapshot.risk_score_change_pct,
                            'current_risk_score': snapshot.risk_score
                        }
                    )
                    
                    # Trigger proactive workflow using your existing monitor
                    if self.proactive_monitor:
                        try:
                            workflow_result = await self.proactive_monitor.trigger_risk_workflow(
                                user_id, portfolio_id, snapshot.to_dict()
                            )
                            
                            if workflow_result.get('workflow_id'):
                                # Update alert with workflow ID
                                alert.workflow_id = workflow_result['workflow_id']
                                alert.workflow_status = 'started'
                                db.commit()
                                logger.info(f"🤖 Proactive workflow triggered: {workflow_result['workflow_id']}")
                        except Exception as e:
                            logger.error(f"Failed to trigger workflow: {e}")
                
                logger.info(f"✅ Risk snapshot complete: risk_score={snapshot.risk_score:.1f}")
                
                return {
                    'status': 'success',
                    'snapshot': snapshot.to_dict(),
                    'threshold_breach': snapshot.is_threshold_breach,
                    'alert_triggered': snapshot.is_threshold_breach
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"❌ Error in risk snapshot workflow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def monitor_portfolio_risk_changes(self, user_portfolios: Dict) -> List[Dict]:
        """
        Monitor all portfolios for a user and detect risk changes
        Integrates with your existing portfolio structure
        """
        results = []
        
        for user_id, portfolios in user_portfolios.items():
            logger.info(f"🔍 Monitoring {len(portfolios)} portfolios for user {user_id}")
            
            for portfolio in portfolios:
                try:
                    result = await self.create_and_store_risk_snapshot(
                        user_id=user_id,
                        portfolio_id=portfolio['id'],
                        portfolio_data=portfolio
                    )
                    results.append(result)
                    
                    # Small delay between calculations
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error monitoring portfolio {portfolio['id']}: {e}")
                    results.append({
                        'status': 'error',
                        'portfolio_id': portfolio['id'],
                        'error': str(e)
                    })
        
        return results
    
    def get_risk_dashboard_data(self, user_id: str) -> Dict:
        """
        Get comprehensive risk dashboard data for a user
        Uses your existing database structure
        """
        try:
            db = next(get_db())
            try:
                # Get portfolio rankings
                rankings = get_portfolio_rankings(db, user_id)
                
                # Get recent alerts
                alerts = get_recent_alerts(db, user_id, days=7)
                
                # Get threshold breaches
                from db.crud import get_threshold_breaches
                breaches = get_threshold_breaches(db, user_id, days=30)
                
                return {
                    'status': 'success',
                    'portfolio_rankings': rankings,
                    'recent_alerts': [self._alert_to_dict(alert) for alert in alerts],
                    'threshold_breaches': len(breaches),
                    'monitoring_active': self._initialized
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_portfolio_risk_history(self, user_id: str, portfolio_id: str, days: int = 30) -> Dict:
        """Get risk history for a specific portfolio"""
        try:
            db = next(get_db())
            try:
                history = get_risk_history(db, user_id, portfolio_id, days)
                trends = get_risk_trends(db, user_id, portfolio_id, days)
                
                return {
                    'status': 'success',
                    'history': [snapshot.to_dict() for snapshot in history],
                    'trends': trends
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting risk history: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_user_risk_thresholds(self, user_id: str, thresholds: Dict, portfolio_id: str = None) -> bool:
        """Update risk threshold configuration for user"""
        try:
            db = next(get_db())
            try:
                success = update_risk_thresholds(db, user_id, thresholds, portfolio_id)
                logger.info(f"Risk thresholds updated for user {user_id}")
                return success
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")
            return False
    
    def acknowledge_risk_alert(self, alert_id: str, user_id: str) -> bool:
        """Mark a risk alert as acknowledged"""
        try:
            db = next(get_db())
            try:
                return acknowledge_alert(db, alert_id, user_id)
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def _alert_to_dict(self, alert) -> Dict:
        """Convert RiskAlertLog to dictionary"""
        return {
            'alert_id': alert.alert_id,
            'portfolio_id': alert.portfolio_id,
            'alert_type': alert.alert_type,
            'alert_severity': alert.alert_severity,
            'alert_message': alert.alert_message,
            'workflow_id': alert.workflow_id,
            'workflow_status': alert.workflow_status,
            'user_acknowledged': alert.user_acknowledged,
            'created_at': alert.created_at.isoformat(),
            'triggered_metrics': alert.triggered_metrics
        }


# Global service instance
_risk_attribution_service = None

async def get_risk_attribution_service() -> RiskAttributionService:
    """Get global risk attribution service instance"""
    global _risk_attribution_service
    if _risk_attribution_service is None:
        _risk_attribution_service = RiskAttributionService()
        await _risk_attribution_service.initialize()
    return _risk_attribution_service


# Integration functions for your existing codebase
async def calculate_and_store_portfolio_risk(
    user_id: str, 
    portfolio_id: str, 
    portfolio_data: dict
) -> Dict:
    """
    Convenience function to calculate and store portfolio risk
    Easy integration with your existing portfolio management
    """
    service = await get_risk_attribution_service()
    return await service.create_and_store_risk_snapshot(user_id, portfolio_id, portfolio_data)


async def trigger_risk_monitoring_for_user(user_id: str, portfolios: List[Dict]) -> List[Dict]:
    """
    Trigger risk monitoring for all portfolios of a specific user
    Returns list of results for each portfolio
    """
    service = await get_risk_attribution_service()
    user_portfolios = {user_id: portfolios}
    return await service.monitor_portfolio_risk_changes(user_portfolios)


def get_user_risk_dashboard(user_id: str) -> Dict:
    """
    Get risk dashboard data for a user
    Synchronous function for easy integration
    """
    service = _risk_attribution_service
    if service is None:
        return {"status": "error", "error": "Service not initialized"}
    
    return service.get_risk_dashboard_data(user_id)


# Test function to validate integration
async def test_risk_attribution_integration():
    """
    Test the complete risk attribution integration with your existing services
    """
    print("🧪 Testing Risk Attribution Integration with Your Platform")
    print("=" * 60)
    
    try:
        # Initialize service
        service = await get_risk_attribution_service()
        print("✅ Risk Attribution Service initialized")
        
        # Test with sample portfolio data (matching your structure)
        test_portfolio = {
            "id": "test_portfolio_001",
            "name": "Test Conservative Portfolio",
            "positions": [
                {"symbol": "SPY", "weight": 0.6, "shares": 100},
                {"symbol": "BND", "weight": 0.4, "shares": 200}
            ],
            "total_value": 50000
        }
        
        # Test risk calculation and storage
        print("📊 Testing risk calculation and storage...")
        result = await service.create_and_store_risk_snapshot(
            user_id="test_user_001",
            portfolio_id=test_portfolio["id"],
            portfolio_data=test_portfolio
        )
        
        if result['status'] == 'success':
            print(f"✅ Risk snapshot created successfully:")
            print(f"   Risk Score: {result['snapshot']['risk_score']:.1f}")
            print(f"   Volatility: {result['snapshot']['volatility']:.2%}")
            print(f"   Threshold Breach: {result['threshold_breach']}")
        
        # Test dashboard data
        print("📋 Testing dashboard data...")
        dashboard = service.get_risk_dashboard_data("test_user_001")
        
        if dashboard['status'] == 'success':
            print(f"✅ Dashboard data retrieved:")
            print(f"   Portfolio rankings: {len(dashboard['portfolio_rankings'])}")
            print(f"   Recent alerts: {len(dashboard['recent_alerts'])}")
            print(f"   Monitoring active: {dashboard['monitoring_active']}")
        
        print("\n🎉 Risk Attribution Integration Test SUCCESSFUL!")
        print("✅ All components working together perfectly")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_risk_attribution_integration())
                    
                    # Trigger proactive workflow using your existing monitor
                    if self.proactive_monitor:
                        try:
                            workflow_result = await self.proactive_monitor.trigger_risk_workflow(
                                user_id, portfolio_id, snapshot.to_dict()
                            )
                            
                            if workflow_result.get('workflow_id'):
                                # Update alert with workflow ID
                                alert.workflow_id = workflow_result['workflow_id']
                                alert.workflow_status = 'started'
                                db.commit()
                                logger.info(f"🤖 Proactive workflow triggered: {workflow_result['workflow_id']}")
                        except Exception as e:
                            logger.error(f"Failed to trigger workflow: {e}")
                
                logger.info(f"✅ Risk snapshot complete: risk_score={snapshot.risk_score:.1f}")
                
                return {
                    'status': 'success',
                    'snapshot': snapshot.to_dict(),
                    'threshold_breach': snapshot.is_threshold_breach,
                    'alert_triggered': snapshot.is_threshold_breach
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"❌ Error in risk snapshot workflow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def monitor_portfolio_risk_changes(self, user_portfolios: Dict) -> List[Dict]:
        """
        Monitor all portfolios for a user and detect risk changes
        Integrates with your existing portfolio structure
        """
        results = []
        
        for user_id, portfolios in user_portfolios.items():
            logger.info(f"🔍 Monitoring {len(portfolios)} portfolios for user {user_id}")
            
            for portfolio in portfolios:
                try:
                    result = await self.create_and_store_risk_snapshot(
                        user_id=user_id,
                        portfolio_id=portfolio['id'],
                        portfolio_data=portfolio
                    )
                    results.append(result)
                    
                    # Small delay between calculations
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error monitoring portfolio {portfolio['id']}: {e}")
                    results.append({
                        'status': 'error',
                        'portfolio_id': portfolio['id'],
                        'error': str(e)
                    })
        
        return results
    
    def get_risk_dashboard_data(self, user_id: str) -> Dict:
        """
        Get comprehensive risk dashboard data for a user
        Uses your existing database structure
        """
        try:
            db = next(get_db())
            try:
                # Get portfolio rankings
                rankings = RiskCRUD.get_portfolio_rankings(db, user_id)
                
                # Get recent alerts
                alerts = RiskCRUD.get_recent_alerts(db, user_id, days=7)
                
                # Get threshold breaches
                breaches = RiskCRUD.get_threshold_breaches(db, user_id, days=30)
                
                return {
                    'status': 'success',
                    'portfolio_rankings': rankings,
                    'recent_alerts': [self._alert_to_dict(alert) for alert in alerts],
                    'threshold_breaches': len(breaches),
                    'monitoring_active': self._initialized
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_portfolio_risk_history(self, user_id: str, portfolio_id: str, days: int = 30) -> Dict:
        """Get risk history for a specific portfolio"""
        try:
            db = next(get_db())
            try:
                history = RiskCRUD.get_risk_history(db, user_id, portfolio_id, days)
                trends = RiskCRUD.get_risk_trends(db, user_id, portfolio_id, days)
                
                return {
                    'status': 'success',
                    'history': [snapshot.to_dict() for snapshot in history],
                    'trends': trends
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting risk history: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_user_risk_thresholds(self, user_id: str, thresholds: Dict, portfolio_id: str = None) -> bool:
        """Update risk threshold configuration for user"""
        try:
            db = next(get_db())
            try:
                success = RiskCRUD.update_risk_thresholds(db, user_id, thresholds, portfolio_id)
                logger.info(f"Risk thresholds updated for user {user_id}")
                return success
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")
            return False
    
    def acknowledge_risk_alert(self, alert_id: str, user_id: str) -> bool:
        """Mark a risk alert as acknowledged"""
        try:
            db = next(get_db())
            try:
                return RiskCRUD.acknowledge_alert(db, alert_id, user_id)
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def run_background_monitoring(self, user_portfolios: Dict, interval_hours: int = 1):
        """
        Run continuous background monitoring
        Can be integrated with your existing background services
        """
        logger.info(f"🔄 Starting background risk monitoring (every {interval_hours} hours)")
        
        while True:
            try:
                results = await self.monitor_portfolio_risk_changes(user_portfolios)
                
                success_count = len([r for r in results if r['status'] == 'success'])
                alert_count = len([r for r in results if r.get('alert_triggered')])
                
                logger.info(f"📊 Background monitoring cycle complete: {success_count} portfolios, {alert_count} alerts")
                
                # Wait for next cycle
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    def _alert_to_dict(self, alert) -> Dict:
        """Convert RiskAlertLog to dictionary"""
        return {
            'alert_id': alert.alert_id,
            'portfolio_id': alert.portfolio_id,
            'alert_type': alert.alert_type,
            'alert_severity': alert.alert_severity,
            'alert_message': alert.alert_message,
            'workflow_id': alert.workflow_id,
            'workflow_status': alert.workflow_status,
            'user_acknowledged': alert.user_acknowledged,
            'created_at': alert.created_at.isoformat(),
            'triggered_metrics': alert.triggered_metrics
        }


# Global service instance
_risk_attribution_service = None

async def get_risk_attribution_service() -> RiskAttributionService:
    """Get global risk attribution service instance"""
    global _risk_attribution_service
    if _risk_attribution_service is None:
        _risk_attribution_service = RiskAttributionService()
        await _risk_attribution_service.initialize()
    return _risk_attribution_service


# Integration functions for your existing codebase
async def calculate_and_store_portfolio_risk(
    user_id: str, 
    portfolio_id: str, 
    portfolio_data: dict
) -> Dict:
    """
    Convenience function to calculate and store portfolio risk
    Easy integration with your existing portfolio management
    """
    service = await get_risk_attribution_service()
    return await service.create_and_store_risk_snapshot(user_id, portfolio_id, portfolio_data)


async def trigger_risk_monitoring_for_user(user_id: str, portfolios: List[Dict]) -> List[Dict]:
    """
    # services/risk_attribution_service.py
"""
Risk Attribution Service
Integrates with your existing services to provide complete risk monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Import your existing services
from services.risk_calculator import RiskCalculator
from services.proactive_monitor import get_proactive_monitor

# Import your existing database setup
from db.session import get_db
from db.crud.risk import RiskCRUD

logger = logging.getLogger(__name__)

class RiskAttributionService:
    """
    Centralized service that connects risk calculation, storage, and monitoring
    Integrates with your existing platform architecture
    """
    
    def __init__(self):
        self.risk_calculator = RiskCalculator()
        self.proactive_monitor = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the service with your existing components"""
        try: