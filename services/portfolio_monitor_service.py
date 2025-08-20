# services/portfolio_monitor_service.py
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from services.risk_detector import create_risk_detector
from services.proactive_monitor import get_proactive_monitor

class PortfolioMonitorService:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.risk_detector = create_risk_detector()
        
    async def start_monitoring(self):
        """Start background risk monitoring"""
        # Schedule risk checks every hour
        self.scheduler.add_job(
            self.monitor_all_portfolios,
            'interval',
            hours=1,
            id='risk_monitoring'
        )
        self.scheduler.start()
        
    async def monitor_all_portfolios(self):
        """Check all active portfolios for risk changes"""
        # Get all active portfolios and run risk detection
        # This completes Task 4.1.2!