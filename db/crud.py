# in db/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from db import models
from api import schemas
from core.security import get_password_hash
from typing import List,Optional,Dict
from datetime import datetime, timedelta, timezone

from db.models import (
    PortfolioRiskSnapshot, 
    RiskThresholdConfig, 
    RiskAlertLog,
    create_risk_snapshot_from_calculator,
    get_default_risk_thresholds
)

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_asset_by_ticker(db: Session, ticker: str):
    """Finds an asset by its ticker, or creates it if it doesn't exist."""
    db_asset = db.query(models.Asset).filter(models.Asset.ticker == ticker.upper()).first()
    if not db_asset:
        db_asset = models.Asset(ticker=ticker.upper())
        db.add(db_asset)
        db.commit()
        db.refresh(db_asset)
    return db_asset

def create_portfolio(db: Session, portfolio: schemas.PortfolioCreate, user_id: int):
    db_portfolio = models.Portfolio(**portfolio.model_dump(), user_id=user_id)
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

def get_user_portfolios(db: Session, user_id: int):
    return db.query(models.Portfolio).filter(models.Portfolio.user_id == user_id).all()

def add_holding_to_portfolio(db: Session, portfolio_id: int, holding: schemas.HoldingCreate):
    asset = get_asset_by_ticker(db, ticker=holding.ticker)
    db_holding = models.Holding(
        portfolio_id=portfolio_id,
        asset_id=asset.id,
        shares=holding.shares
    )
    db.add(db_holding)
    db.commit()
    db.refresh(db_holding)
    return db_holding

def add_holdings_to_portfolio_bulk(db: Session, portfolio_id: int, holdings: List[schemas.HoldingCreate]):
    """
    Adds a list of new asset holdings to a specific portfolio in a single transaction.
    """
    new_holdings = []
    for holding in holdings:
        # get_asset_by_ticker will find or create the master asset entry
        asset = get_asset_by_ticker(db, ticker=holding.ticker)
        db_holding = models.Holding(
            portfolio_id=portfolio_id,
            asset_id=asset.id,
            shares=holding.shares
        )
        new_holdings.append(db_holding)
    
    # db.add_all() is more efficient for adding multiple items
    db.add_all(new_holdings)
    db.commit()
    
    # We need to refresh each object individually to get its new ID from the DB
    for h in new_holdings:
        db.refresh(h)
        
    return new_holdings

def create_risk_snapshot(
    db: Session, 
    user_id: str, 
    portfolio_id: str, 
    risk_result: dict,
    portfolio_data: dict = None
) -> PortfolioRiskSnapshot:
    """
    Create and save risk snapshot from calculator results
    """
    # Get previous snapshot for change calculation
    previous = get_latest_risk_snapshot(db, user_id, portfolio_id)
    
    # Create new snapshot
    snapshot = create_risk_snapshot_from_calculator(
        user_id, portfolio_id, risk_result, portfolio_data
    )
    
    # Calculate changes from previous snapshot
    if previous:
        snapshot.volatility_change_pct = _calculate_percentage_change(
            previous.volatility, snapshot.volatility
        )
        snapshot.risk_score_change_pct = _calculate_percentage_change(
            previous.risk_score, snapshot.risk_score
        )
        
        # Check if threshold is breached
        snapshot.is_threshold_breach = _check_threshold_breach(
            db, user_id, portfolio_id, snapshot, previous
        )
    
    # Save to database
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    
    return snapshot

def get_latest_risk_snapshot(
    db: Session, 
    user_id: str, 
    portfolio_id: str
) -> Optional[PortfolioRiskSnapshot]:
    """Get the most recent risk snapshot for a portfolio"""
    return db.query(PortfolioRiskSnapshot)\
             .filter(and_(
                 PortfolioRiskSnapshot.user_id == user_id,
                 PortfolioRiskSnapshot.portfolio_id == portfolio_id
             ))\
             .order_by(desc(PortfolioRiskSnapshot.snapshot_date))\
             .first()

def get_risk_history(
    db: Session, 
    user_id: str, 
    portfolio_id: str = None, 
    days: int = 30,
    limit: int = 100
) -> List[PortfolioRiskSnapshot]:
    """
    Get risk history for user/portfolio
    """
    query = db.query(PortfolioRiskSnapshot)\
              .filter(PortfolioRiskSnapshot.user_id == user_id)
    
    if portfolio_id:
        query = query.filter(PortfolioRiskSnapshot.portfolio_id == portfolio_id)
    
    if days > 0:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        query = query.filter(PortfolioRiskSnapshot.snapshot_date >= cutoff_date)
    
    return query.order_by(desc(PortfolioRiskSnapshot.snapshot_date))\
               .limit(limit)\
               .all()

def get_threshold_breaches(
    db: Session, 
    user_id: str = None, 
    days: int = 7
) -> List[PortfolioRiskSnapshot]:
    """Get all recent threshold breaches"""
    query = db.query(PortfolioRiskSnapshot)\
              .filter(PortfolioRiskSnapshot.is_threshold_breach == True)
    
    if user_id:
        query = query.filter(PortfolioRiskSnapshot.user_id == user_id)
    
    if days > 0:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        query = query.filter(PortfolioRiskSnapshot.snapshot_date >= cutoff_date)
    
    return query.order_by(desc(PortfolioRiskSnapshot.snapshot_date)).all()

def get_portfolio_rankings(db: Session, user_id: str) -> List[Dict]:
    """Get risk rankings for all user portfolios"""
    # Get all unique portfolio_ids for user
    portfolio_ids = db.query(PortfolioRiskSnapshot.portfolio_id)\
                     .filter(PortfolioRiskSnapshot.user_id == user_id)\
                     .distinct()\
                     .all()
    
    latest_snapshots = []
    for (portfolio_id,) in portfolio_ids:
        latest = get_latest_risk_snapshot(db, user_id, portfolio_id)
        if latest:
            latest_snapshots.append(latest)
    
    # Sort by risk score and convert to dict
    rankings = []
    for snapshot in sorted(latest_snapshots, key=lambda x: x.risk_score):
        rankings.append({
            "portfolio_id": snapshot.portfolio_id,
            "portfolio_name": snapshot.portfolio_name,
            "risk_score": snapshot.risk_score,
            "risk_level": snapshot.risk_level,
            "risk_grade": snapshot.risk_grade,
            "volatility": snapshot.volatility,
            "last_updated": snapshot.snapshot_date.isoformat()
        })
    
    return rankings

def get_risk_thresholds(
    db: Session, 
    user_id: str, 
    portfolio_id: str = None
) -> Dict:
    """
    Get risk thresholds for user/portfolio
    Falls back to user default, then system default
    """
    # Try portfolio-specific config first
    if portfolio_id:
        config = db.query(RiskThresholdConfig)\
                   .filter(and_(
                       RiskThresholdConfig.user_id == user_id,
                       RiskThresholdConfig.portfolio_id == portfolio_id
                   ))\
                   .first()
        if config:
            return _config_to_dict(config)
    
    # Try user default config
    config = db.query(RiskThresholdConfig)\
               .filter(and_(
                   RiskThresholdConfig.user_id == user_id,
                   RiskThresholdConfig.portfolio_id.is_(None)
               ))\
               .first()
    if config:
        return _config_to_dict(config)
    
    # Fall back to system defaults
    return get_default_risk_thresholds()

def update_risk_thresholds(
    db: Session, 
    user_id: str, 
    thresholds: Dict, 
    portfolio_id: str = None
) -> bool:
    """Update or create risk threshold configuration"""
    # Try to find existing config
    config = db.query(RiskThresholdConfig)\
               .filter(and_(
                   RiskThresholdConfig.user_id == user_id,
                   RiskThresholdConfig.portfolio_id == portfolio_id
               ))\
               .first()
    
    if config:
        # Update existing
        for key, value in thresholds.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.updated_at = datetime.now(timezone.utc)
    else:
        # Create new
        config = RiskThresholdConfig(
            user_id=user_id,
            portfolio_id=portfolio_id,
            **thresholds
        )
        db.add(config)
    
    db.commit()
    return True

def log_risk_alert(
    db: Session,
    user_id: str,
    portfolio_id: str,
    alert_type: str,
    alert_message: str,
    snapshot_id: str = None,
    severity: str = "medium",
    triggered_metrics: Dict = None,
    workflow_id: str = None
) -> RiskAlertLog:
    """Log a risk alert to database"""
    alert = RiskAlertLog(
        user_id=user_id,
        portfolio_id=portfolio_id,
        snapshot_id=snapshot_id,
        alert_type=alert_type,
        alert_severity=severity,
        alert_message=alert_message,
        triggered_metrics=triggered_metrics,
        workflow_id=workflow_id,
        notification_methods=['websocket'],  # Default
        notification_status={'websocket': 'pending'}
    )
    
    db.add(alert)
    db.commit()
    db.refresh(alert)
    
    return alert

def get_recent_alerts(
    db: Session, 
    user_id: str, 
    days: int = 7, 
    limit: int = 50
) -> List[RiskAlertLog]:
    """Get recent alerts for a user"""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    return db.query(RiskAlertLog)\
             .filter(and_(
                 RiskAlertLog.user_id == user_id,
                 RiskAlertLog.created_at >= cutoff_date
             ))\
             .order_by(desc(RiskAlertLog.created_at))\
             .limit(limit)\
             .all()

def acknowledge_alert(db: Session, alert_id: str, user_id: str) -> bool:
    """Mark an alert as acknowledged by user"""
    alert = db.query(RiskAlertLog)\
              .filter(and_(
                  RiskAlertLog.alert_id == alert_id,
                  RiskAlertLog.user_id == user_id
              ))\
              .first()
    
    if alert:
        alert.user_acknowledged = True
        alert.acknowledged_at = datetime.now(timezone.utc)
        db.commit()
        return True
    
    return False

def get_risk_trends(
    db: Session, 
    user_id: str, 
    portfolio_id: str = None, 
    days: int = 30
) -> Dict:
    """Get risk trend analysis"""
    snapshots = get_risk_history(db, user_id, portfolio_id, days)
    
    if len(snapshots) < 2:
        return {"status": "insufficient_data", "snapshots_count": len(snapshots)}
    
    # Calculate trends
    latest = snapshots[0]
    oldest = snapshots[-1]
    
    volatility_trend = _calculate_percentage_change(
        oldest.volatility, latest.volatility
    )
    risk_score_trend = _calculate_percentage_change(
        oldest.risk_score, latest.risk_score
    )
    
    # Risk level changes
    risk_levels = [s.risk_score for s in snapshots]
    avg_risk = sum(risk_levels) / len(risk_levels)
    max_risk = max(risk_levels)
    min_risk = min(risk_levels)
    
    return {
        "status": "success",
        "period_days": days,
        "snapshots_count": len(snapshots),
        "latest_risk_score": latest.risk_score,
        "volatility_trend_pct": volatility_trend,
        "risk_score_trend_pct": risk_score_trend,
        "average_risk_score": avg_risk,
        "max_risk_score": max_risk,
        "min_risk_score": min_risk,
        "risk_stability": max_risk - min_risk,  # Lower = more stable
        "threshold_breaches": len([s for s in snapshots if s.is_threshold_breach])
    }

# Helper functions (add these too)
def _calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / abs(old_value)) * 100

def _check_threshold_breach(
    db: Session,
    user_id: str, 
    portfolio_id: str, 
    current_snapshot: PortfolioRiskSnapshot,
    previous_snapshot: PortfolioRiskSnapshot
) -> bool:
    """Check if current snapshot breaches any thresholds"""
    
    thresholds = get_risk_thresholds(db, user_id, portfolio_id)
    
    # Check percentage change thresholds
    if current_snapshot.volatility_change_pct and \
       abs(current_snapshot.volatility_change_pct) > (thresholds['volatility_threshold'] * 100):
        return True
    
    if current_snapshot.risk_score_change_pct and \
       abs(current_snapshot.risk_score_change_pct) > (thresholds['risk_score_threshold'] * 100):
        return True
    
    # Check absolute thresholds
    if current_snapshot.risk_score > thresholds['max_acceptable_risk_score']:
        return True
    
    if current_snapshot.volatility > thresholds['max_acceptable_volatility']:
        return True
    
    return False

def _config_to_dict(config: RiskThresholdConfig) -> Dict:
    """Convert RiskThresholdConfig to dictionary"""
    return {
        'volatility_threshold': config.volatility_threshold,
        'beta_threshold': config.beta_threshold,
        'max_drawdown_threshold': config.max_drawdown_threshold,
        'var_threshold': config.var_threshold,
        'risk_score_threshold': config.risk_score_threshold,
        'max_acceptable_risk_score': config.max_acceptable_risk_score,
        'max_acceptable_volatility': config.max_acceptable_volatility,
        'monitoring_enabled': config.monitoring_enabled,
        'alert_frequency': config.alert_frequency,
        'notification_methods': config.notification_methods
    }