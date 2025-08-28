#!/usr/bin/env python3
"""
Risk Snapshot Storage Service - Task 2.1.3 Implementation
=========================================================
Compressed storage and retrieval of portfolio risk snapshots with
efficient compression, historical comparison, and performance optimization.
"""

import asyncio
import json
import gzip
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
import pandas as pd
import numpy as np

# Local imports
from db.models import PortfolioRiskSnapshot, Portfolio
from services.risk_calculator import RiskMetrics

logger = logging.getLogger(__name__)

@dataclass
class SnapshotMetrics:
    """Metrics for snapshot performance"""
    storage_time_ms: float
    retrieval_time_ms: float
    compression_ratio: float
    data_size_bytes: int
    compressed_size_bytes: int

@dataclass
class HistoricalComparison:
    """Historical risk comparison result"""
    status: str
    current_risk_score: float
    previous_risk_score: float
    change_percent: float
    trend: str
    comparison_period_days: int

class RiskSnapshotStorage:
    """
    Risk snapshot storage service with compression and historical analysis
    """
    
    def __init__(self, db: Session):
        self.db = db
        logger.info("RiskSnapshotStorage initialized")
    
    async def create_compressed_snapshot(
        self, 
        portfolio_id: int, 
        risk_metrics: RiskMetrics,
        user_id: int = None
    ) -> PortfolioRiskSnapshot:
        """
        Create compressed risk snapshot with user_id support
        """
        try:
            start_time = datetime.now()
            
            # Get portfolio and user_id if not provided
            if user_id is None:
                portfolio = self.db.query(Portfolio).filter(
                    Portfolio.id == portfolio_id
                ).first()
                if portfolio:
                    user_id = portfolio.user_id
                else:
                    raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Convert metrics to dictionary
            metrics_dict = {
                'annualized_volatility': getattr(risk_metrics, 'annualized_volatility', 0.0),
                'var_95': getattr(risk_metrics, 'var_95', 0.0),
                'cvar_95': getattr(risk_metrics, 'cvar_95', 0.0),
                'max_drawdown': getattr(risk_metrics, 'max_drawdown', 0.0),
                'sharpe_ratio': getattr(risk_metrics, 'sharpe_ratio', 0.0),
                'sortino_ratio': getattr(risk_metrics, 'sortino_ratio', 0.0),
                'calmar_ratio': getattr(risk_metrics, 'calmar_ratio', 0.0),
                'skewness': getattr(risk_metrics, 'skewness', 0.0),
                'kurtosis': getattr(risk_metrics, 'kurtosis', 0.0),
                'beta': getattr(risk_metrics, 'beta', 1.0),
                'calculation_date': getattr(risk_metrics, 'calculation_date', datetime.utcnow()).isoformat(),
            }
            
            # Compress the data
            json_data = json.dumps(metrics_dict, default=str)
            compressed_data = gzip.compress(json_data.encode('utf-8'), compresslevel=6)
            
            # Create snapshot with all required fields
            snapshot = PortfolioRiskSnapshot(
                portfolio_id=portfolio_id,
                user_id=user_id,
                snapshot_date=getattr(risk_metrics, 'calculation_date', datetime.utcnow()),
                
                # Compressed data
                compressed_metrics=compressed_data,
                compression_ratio=len(compressed_data) / len(json_data.encode('utf-8')),
                
                # Individual fields (required by your schema)
                volatility=getattr(risk_metrics, 'annualized_volatility', 0.0),
                beta=getattr(risk_metrics, 'beta', 1.0),
                max_drawdown=getattr(risk_metrics, 'max_drawdown', 0.0),
                var_95=getattr(risk_metrics, 'var_95', 0.0),
                var_99=getattr(risk_metrics, 'var_99', 0.0),
                cvar_95=getattr(risk_metrics, 'cvar_95', 0.0),
                cvar_99=getattr(risk_metrics, 'cvar_99', 0.0),
                sharpe_ratio=getattr(risk_metrics, 'sharpe_ratio', 0.0),
                sortino_ratio=getattr(risk_metrics, 'sortino_ratio', 0.0),
                calmar_ratio=getattr(risk_metrics, 'calmar_ratio', 0.0),
                hurst_exponent=getattr(risk_metrics, 'hurst_exponent', None),
                dfa_alpha=getattr(risk_metrics, 'dfa_alpha', None),
                risk_score=self._calculate_risk_score(risk_metrics),
                sentiment_index=50,  # Default neutral sentiment
                regime_volatility=getattr(risk_metrics, 'regime_volatility', None),
                
                # Summary for quick access
                metrics_summary={
                    'volatility': getattr(risk_metrics, 'annualized_volatility', 0.0),
                    'var_95': getattr(risk_metrics, 'var_95', 0.0),
                    'sharpe_ratio': getattr(risk_metrics, 'sharpe_ratio', 0.0),
                    'max_drawdown': getattr(risk_metrics, 'max_drawdown', 0.0)
                },
                
                # Metadata
                data_quality_score=1.0,
                calculation_method='comprehensive',
                data_window_days=252,
                created_at=datetime.utcnow()
            )
            
            self.db.add(snapshot)
            self.db.commit()
            self.db.refresh(snapshot)
            
            storage_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Created snapshot {snapshot.id} in {storage_time_ms:.1f}ms")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            self.db.rollback()
            raise e
    
    def _calculate_risk_score(self, risk_metrics: RiskMetrics) -> float:
        """Calculate composite risk score (0-100 scale)"""
        try:
            volatility = getattr(risk_metrics, 'annualized_volatility', 0.0)
            max_drawdown = abs(getattr(risk_metrics, 'max_drawdown', 0.0))
            var_95 = abs(getattr(risk_metrics, 'var_95', 0.0))
            sharpe = getattr(risk_metrics, 'sharpe_ratio', 0.0)
            
            # Normalize components (0-100 scale)
            vol_score = min(volatility * 100 * 2, 100)  # 0.5 vol = 100 points
            drawdown_score = min(max_drawdown * 100 * 4, 100)  # 25% drawdown = 100 points
            var_score = min(var_95 * 100 * 10, 100)  # 10% VaR = 100 points
            sharpe_score = max(100 - (sharpe * 20), 0) if sharpe > 0 else 100  # Lower is better
            
            # Weighted composite (higher = riskier)
            risk_score = (
                vol_score * 0.3 +
                drawdown_score * 0.3 +
                var_score * 0.25 +
                sharpe_score * 0.15
            )
            
            return min(max(risk_score, 0.0), 100.0)
            
        except Exception:
            return 50.0  # Default moderate risk
    
    async def get_decompressed_metrics(self, snapshot: PortfolioRiskSnapshot) -> Dict[str, Any]:
        """
        Decompress and return metrics from snapshot
        """
        try:
            start_time = datetime.now()
            
            if not snapshot.compressed_metrics:
                # Fall back to individual fields
                result = {
                    'volatility': snapshot.volatility,
                    'var_95': snapshot.var_95,
                    'sharpe_ratio': snapshot.sharpe_ratio,
                    'max_drawdown': snapshot.max_drawdown,
                    'beta': snapshot.beta,
                    'sortino_ratio': snapshot.sortino_ratio,
                    'calmar_ratio': snapshot.calmar_ratio
                }
            else:
                # Decompress data
                decompressed_data = gzip.decompress(snapshot.compressed_metrics)
                result = json.loads(decompressed_data.decode('utf-8'))
            
            retrieval_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug(f"Decompressed snapshot {snapshot.id} in {retrieval_time_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to decompress snapshot {snapshot.id}: {e}")
            raise e
    
    async def get_historical_comparison(
        self, 
        portfolio_id: int, 
        current_metrics: RiskMetrics,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare current metrics with historical data
        """
        try:
            # Get historical snapshot
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            historical_snapshot = self.db.query(PortfolioRiskSnapshot).filter(
                and_(
                    PortfolioRiskSnapshot.portfolio_id == portfolio_id,
                    PortfolioRiskSnapshot.snapshot_date <= cutoff_date
                )
            ).order_by(desc(PortfolioRiskSnapshot.snapshot_date)).first()
            
            if not historical_snapshot:
                return {
                    'status': 'insufficient_data',
                    'message': f'No historical data available for {lookback_days} day comparison',
                    'current_risk_score': self._calculate_risk_score(current_metrics)
                }
            
            # Calculate current risk score
            current_risk_score = self._calculate_risk_score(current_metrics)
            previous_risk_score = historical_snapshot.risk_score
            
            # Calculate change
            if previous_risk_score > 0:
                change_percent = ((current_risk_score - previous_risk_score) / previous_risk_score) * 100
            else:
                change_percent = 0.0
            
            # Determine trend
            if abs(change_percent) < 5:
                trend = 'stable'
            elif change_percent > 0:
                trend = 'increasing_risk'
            else:
                trend = 'decreasing_risk'
            
            return {
                'status': 'success',
                'current_risk_score': current_risk_score,
                'previous_risk_score': previous_risk_score,
                'change_percent': change_percent,
                'trend': trend,
                'comparison_period_days': lookback_days,
                'historical_date': historical_snapshot.snapshot_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Historical comparison failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'current_risk_score': self._calculate_risk_score(current_metrics)
            }
    
    async def get_snapshots_for_portfolio(
        self, 
        portfolio_id: int, 
        limit: int = 100,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[PortfolioRiskSnapshot]:
        """
        Get historical snapshots for portfolio
        """
        try:
            query = self.db.query(PortfolioRiskSnapshot).filter(
                PortfolioRiskSnapshot.portfolio_id == portfolio_id
            )
            
            if start_date:
                query = query.filter(PortfolioRiskSnapshot.snapshot_date >= start_date)
            
            if end_date:
                query = query.filter(PortfolioRiskSnapshot.snapshot_date <= end_date)
            
            snapshots = query.order_by(
                desc(PortfolioRiskSnapshot.snapshot_date)
            ).limit(limit).all()
            
            return snapshots
            
        except Exception as e:
            logger.error(f"Failed to get snapshots for portfolio {portfolio_id}: {e}")
            return []
    
    async def cleanup_old_snapshots(self, retention_days: int = 365) -> int:
        """
        Clean up snapshots older than retention period
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            deleted_count = self.db.query(PortfolioRiskSnapshot).filter(
                PortfolioRiskSnapshot.snapshot_date < cutoff_date
            ).delete()
            
            self.db.commit()
            
            logger.info(f"Cleaned up {deleted_count} snapshots older than {retention_days} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            self.db.rollback()
            return 0

# Singleton instance
_storage_instance = None

def get_risk_snapshot_storage(db: Session) -> RiskSnapshotStorage:
    """Get or create risk snapshot storage instance"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = RiskSnapshotStorage(db)
    return _storage_instance