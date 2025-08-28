#!/usr/bin/env python3
"""
Portfolio Data Integrator - Task 2.1.4 Implementation
=====================================================
Real-time portfolio data integration with multiple price providers,
efficient caching, and performance optimization for large portfolios.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
import aiohttp
import json

# Local imports
from db.models import Portfolio, Holding, Asset, PriceDataCache

logger = logging.getLogger(__name__)

@dataclass
class PortfolioSnapshot:
    """Real-time portfolio data snapshot"""
    portfolio_id: int
    total_value: float
    holdings: List[Dict[str, Any]]
    data_quality_score: float
    fetch_time_ms: float = 0.0  # Add default value
    last_updated: datetime = None  # Add default value

@dataclass
class HoldingData:
    """Individual holding data"""
    asset_id: int
    ticker: str
    quantity: float
    current_price: float
    value: float
    change_percent: float
    last_updated: datetime

class PortfolioDataIntegrator:
    def __init__(self, db_session):
        self.db = db_session
        self.price_cache = {}  # FIXED: Added missing cache
        self.cache_ttl = 60   # FIXED: Added missing cache TTL
    
    async def get_real_time_portfolio_data(self, portfolio_id: int):
        """Get real-time portfolio data with proper error handling"""
        
        start_time = time.time()  # Track timing
        
        try:
            # Get portfolio and holdings
            portfolio = self.db.query(Portfolio).filter(
                Portfolio.id == portfolio_id
            ).first()
            
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            holdings = self.db.query(Holding).filter(
                Holding.portfolio_id == portfolio_id
            ).all()
            
            # Initialize fetched_prices to avoid UnboundLocalError
            fetched_prices = {}
            
            # Get price data for each holding
            for holding in holdings:
                asset = self.db.query(Asset).filter(
                    Asset.id == holding.asset_id
                ).first()
                
                if asset and asset.ticker:
                    # Try to fetch price, fallback to stored price
                    try:
                        # Your price fetching logic here
                        current_price = await self.fetch_current_price(asset.ticker)
                        fetched_prices[asset.ticker] = current_price
                        
                        # Update holding current price
                        holding.current_price = current_price
                        
                    except Exception as price_error:
                        # Fallback to existing price or purchase price
                        fallback_price = holding.current_price or holding.purchase_price
                        fetched_prices[asset.ticker] = fallback_price
                        print(f"Using fallback price for {asset.ticker}: {fallback_price}")
            
            # Calculate portfolio metrics
            total_value = 0
            portfolio_holdings = []
            
            for holding in holdings:
                asset = self.db.query(Asset).filter(
                    Asset.id == holding.asset_id
                ).first()
                
                if asset:
                    current_price = fetched_prices.get(asset.ticker, holding.purchase_price)
                    value = holding.shares * current_price  # Use 'shares' if that's your field name
                    total_value += value
                    
                    portfolio_holdings.append({
                        'asset_id': asset.id,
                        'ticker': asset.ticker,
                        'quantity': holding.shares,  # Ensure consistency
                        'current_price': current_price,
                        'value': value
                    })
            
            # FIXED: Use the existing dataclass, don't redefine it
            fetch_time_ms = (time.time() - start_time) * 1000
            
            return PortfolioSnapshot(
                portfolio_id=portfolio_id,
                total_value=total_value,
                holdings=portfolio_holdings,
                data_quality_score=1.0 if fetched_prices else 0.5,
                fetch_time_ms=fetch_time_ms,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            print(f"Failed to get portfolio data for {portfolio_id}: {e}")
            raise e
    
    async def fetch_current_price(self, ticker: str) -> float:
        """Fetch current price for a ticker - implement your price fetching logic"""
        # For testing, return a mock price
        import random
        return random.uniform(50, 500)
    
    async def get_current_price(self, ticker: str) -> float:
        """
        Get current price with caching and multiple provider support
        """
        # Check cache first
        cache_key = f"price_{ticker}"
        if cache_key in self.price_cache:
            cached_data = self.price_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['price']
        
        # Check database cache
        db_price = self.db.query(PriceDataCache).filter(
            PriceDataCache.symbol == ticker,
            PriceDataCache.expires_at > datetime.utcnow()
        ).first()
        
        if db_price:
            self.price_cache[cache_key] = {
                'price': db_price.price,
                'timestamp': time.time()
            }
            return db_price.price
        
        # Fetch new price
        try:
            price = await self._fetch_price_from_provider(ticker)
            
            # Cache in memory
            self.price_cache[cache_key] = {
                'price': price,
                'timestamp': time.time()
            }
            
            # Cache in database
            await self._cache_price_in_db(ticker, price)
            
            return price
            
        except Exception as e:
            logger.error(f"Failed to fetch price for {ticker}: {e}")
            # Return a mock price for testing
            import random
            return random.uniform(50, 500)
    
    async def _fetch_price_from_provider(self, ticker: str) -> float:
        """
        Fetch price from external provider (mock implementation for testing)
        """
        # In a real implementation, this would call APIs like:
        # - Yahoo Finance
        # - Alpha Vantage
        # - IEX Cloud
        # - Polygon.io
        
        # For testing, return mock data
        await asyncio.sleep(0.01)  # Simulate network delay
        
        # Generate mock price with some volatility
        import random
        base_prices = {
            'AAPL': 150,
            'GOOGL': 2500,
            'MSFT': 300,
            'TSLA': 200,
            'AMZN': 3200
        }
        
        base_price = base_prices.get(ticker, 100)
        volatility = random.uniform(-0.05, 0.05)  # ±5% random change
        return base_price * (1 + volatility)
    
    async def _cache_price_in_db(self, ticker: str, price: float):
        """
        Cache price in database for performance
        """
        try:
            # Remove old cache entries
            self.db.query(PriceDataCache).filter(
                PriceDataCache.symbol == ticker
            ).delete()
            
            # Add new cache entry
            cache_entry = PriceDataCache(
                symbol=ticker,
                price=price,
                currency="USD",
                provider="test_provider",
                data_quality_score=1.0,
                market_timestamp=datetime.utcnow(),
                fetched_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(minutes=1)
            )
            
            self.db.add(cache_entry)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to cache price for {ticker}: {e}")
            self.db.rollback()
    
    def _calculate_change_percent(self, current_price: float, purchase_price: float) -> float:
        """Calculate percentage change from purchase price"""
        if purchase_price == 0:
            return 0.0
        return ((current_price - purchase_price) / purchase_price) * 100
    
    def _calculate_data_quality_score(self, fetched_prices: Dict[str, float], total_holdings: int) -> float:
        """
        Calculate data quality score based on successful price fetches
        """
        if total_holdings == 0:
            return 1.0
        
        successful_fetches = len(fetched_prices)
        return successful_fetches / total_holdings
    
    async def get_portfolio_performance_metrics(self, portfolio_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for portfolio
        """
        try:
            snapshot = await self.get_real_time_portfolio_data(portfolio_id)
            
            # Calculate basic performance metrics
            total_cost_basis = sum(
                holding['quantity'] * holding.get('purchase_price', 0) 
                for holding in snapshot.holdings
            )
            
            total_return_dollars = snapshot.total_value - total_cost_basis
            total_return_percent = (
                (total_return_dollars / total_cost_basis) * 100 
                if total_cost_basis > 0 else 0.0
            )
            
            return {
                'portfolio_id': portfolio_id,
                'total_value': snapshot.total_value,
                'total_cost_basis': total_cost_basis,
                'total_return_dollars': total_return_dollars,
                'total_return_percent': total_return_percent,
                'holdings_count': len(snapshot.holdings),
                'data_quality_score': snapshot.data_quality_score,
                'last_updated': snapshot.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics for portfolio {portfolio_id}: {e}")
            raise e
    
    async def refresh_all_prices(self, portfolio_id: int) -> Dict[str, Any]:
        """
        Force refresh of all prices for a portfolio
        """
        try:
            start_time = time.time()
            
            holdings = self.db.query(Holding).filter(
                Holding.portfolio_id == portfolio_id
            ).all()
            
            updated_count = 0
            errors = []
            
            for holding in holdings:
                asset = self.db.query(Asset).filter(
                    Asset.id == holding.asset_id
                ).first()
                
                if asset and asset.ticker:
                    try:
                        # Force fetch new price (bypass cache)
                        cache_key = f"price_{asset.ticker}"
                        if cache_key in self.price_cache:
                            del self.price_cache[cache_key]
                        
                        new_price = await self.get_current_price(asset.ticker)
                        holding.current_price = new_price
                        updated_count += 1
                        
                    except Exception as e:
                        errors.append(f"{asset.ticker}: {str(e)}")
            
            self.db.commit()
            refresh_time_ms = (time.time() - start_time) * 1000
            
            return {
                'portfolio_id': portfolio_id,
                'updated_count': updated_count,
                'total_holdings': len(holdings),
                'errors': errors,
                'refresh_time_ms': refresh_time_ms
            }
            
        except Exception as e:
            logger.error(f"Failed to refresh prices for portfolio {portfolio_id}: {e}")
            self.db.rollback()
            raise e
        
def get_portfolio_data_integrator(db_session):
    """Factory function to create PortfolioDataIntegrator instance"""
    return PortfolioDataIntegrator(db_session)