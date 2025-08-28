# in db/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func
from db import models
from api import schemas
from core.security import get_password_hash
from typing import List,Optional,Dict, Any
from datetime import datetime, timedelta, timezone

from db.models import (
    PortfolioRiskSnapshot, 
    RiskThresholdConfig, 
    RiskAlertLog,
    RiskTrend, 
    RiskChangeEvent,
    RiskThreshold,
     PriceDataCache, 
     Portfolio, 
     User,
     Holding,
    WorkflowSessionDB, 
    WorkflowStepDB, 
    AgentPerformanceMetrics,
    MCPJobLog
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
    """Get all portfolios for a user"""
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

# ===============================================
# WORKFLOW SESSION CRUD OPERATIONS
# ===============================================

def create_workflow_session(
    db: Session,
    session_id: str,
    user_id: int,
    query: str,
    workflow_type: str,
    complexity_score: float = 0.0,
    execution_mode: str = "direct"
) -> WorkflowSessionDB:
    """Create a new workflow session"""
    db_session = WorkflowSessionDB(
        session_id=session_id,
        user_id=user_id,
        query=query,
        state="awaiting_strategy",
        workflow_type=workflow_type,
        complexity_score=complexity_score,
        execution_mode=execution_mode
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_workflow_session(db: Session, session_id: str) -> Optional[WorkflowSessionDB]:
    """Get workflow session by ID"""
    return db.query(WorkflowSessionDB).filter(WorkflowSessionDB.session_id == session_id).first()

def get_user_workflow_sessions(
    db: Session, 
    user_id: int, 
    limit: int = 20,
    state: Optional[str] = None,
    workflow_type: Optional[str] = None
) -> List[WorkflowSessionDB]:
    """Get workflow sessions for a user with filters"""
    query = db.query(WorkflowSessionDB).filter(WorkflowSessionDB.user_id == user_id)
    
    if state:
        query = query.filter(WorkflowSessionDB.state == state)
    
    if workflow_type:
        query = query.filter(WorkflowSessionDB.workflow_type == workflow_type)
    
    return query.order_by(WorkflowSessionDB.created_at.desc()).limit(limit).all()

def update_workflow_session_state(
    db: Session,
    session_id: str,
    state: str,
    result: Optional[Dict[str, Any]] = None,
    step_result: Optional[Dict[str, Any]] = None,
    step_name: Optional[str] = None
) -> Optional[WorkflowSessionDB]:
    """Update workflow session state and results"""
    db_session = db.query(WorkflowSessionDB).filter(WorkflowSessionDB.session_id == session_id).first()
    
    if not db_session:
        return None
    
    db_session.state = state
    db_session.updated_at = datetime.utcnow()
    
    # Update step-specific results
    if step_result and step_name:
        if step_name == "strategy":
            db_session.strategy_result = step_result
            db_session.current_step = 2
        elif step_name == "screening":
            db_session.screening_result = step_result
            db_session.current_step = 3
        elif step_name == "analysis":
            db_session.analysis_result = step_result
            db_session.current_step = 4
        elif step_name == "synthesis":
            db_session.final_synthesis = step_result
            db_session.completed_at = datetime.utcnow()
    
    # Update final result
    if result:
        db_session.final_synthesis = result
        if state == "complete":
            db_session.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_session)
    return db_session

def update_workflow_session_mcp_job(
    db: Session,
    session_id: str,
    mcp_job_id: str
) -> Optional[WorkflowSessionDB]:
    """Associate MCP job ID with workflow session"""
    db_session = db.query(WorkflowSessionDB).filter(WorkflowSessionDB.session_id == session_id).first()
    
    if not db_session:
        return None
    
    db_session.mcp_job_id = mcp_job_id
    db.commit()
    db.refresh(db_session)
    return db_session

def add_workflow_session_error(
    db: Session,
    session_id: str,
    error_message: str
) -> Optional[WorkflowSessionDB]:
    """Add error to workflow session"""
    db_session = db.query(WorkflowSessionDB).filter(WorkflowSessionDB.session_id == session_id).first()
    
    if not db_session:
        return None
    
    if db_session.errors is None:
        db_session.errors = []
    
    db_session.errors.append({
        "timestamp": datetime.utcnow().isoformat(),
        "message": error_message
    })
    
    db_session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_session)
    return db_session

def complete_workflow_session_step(
    db: Session,
    session_id: str,
    step_name: str
) -> Optional[WorkflowSessionDB]:
    """Mark a workflow step as completed"""
    db_session = db.query(WorkflowSessionDB).filter(WorkflowSessionDB.session_id == session_id).first()
    
    if not db_session:
        return None
    
    if db_session.steps_completed is None:
        db_session.steps_completed = []
    
    if step_name not in db_session.steps_completed:
        db_session.steps_completed.append(step_name)
    
    db_session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_session)
    return db_session

def delete_workflow_session(
    db: Session,
    session_id: str,
    user_id: int
) -> bool:
    """Delete a workflow session (with user verification)"""
    db_session = db.query(WorkflowSessionDB).filter(
        WorkflowSessionDB.session_id == session_id,
        WorkflowSessionDB.user_id == user_id
    ).first()
    
    if not db_session:
        return False
    
    db.delete(db_session)
    db.commit()
    return True

# ===============================================
# WORKFLOW STEP CRUD OPERATIONS
# ===============================================

def create_workflow_step(
    db: Session,
    step_id: str,
    session_id: str,
    step_number: int,
    step_name: str,
    agent_id: str,
    capability: str,
    input_data: Optional[Dict[str, Any]] = None
) -> WorkflowStepDB:
    """Create a new workflow step"""
    db_step = WorkflowStepDB(
        step_id=step_id,
        session_id=session_id,
        step_number=step_number,
        step_name=step_name,
        agent_id=agent_id,
        capability=capability,
        status="pending",
        input_data=input_data
    )
    db.add(db_step)
    db.commit()
    db.refresh(db_step)
    return db_step

def update_workflow_step_status(
    db: Session,
    step_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    confidence_score: Optional[float] = None,
    execution_time_ms: Optional[int] = None
) -> Optional[WorkflowStepDB]:
    """Update workflow step status and results"""
    db_step = db.query(WorkflowStepDB).filter(WorkflowStepDB.step_id == step_id).first()
    
    if not db_step:
        return None
    
    db_step.status = status
    
    if status == "running":
        db_step.started_at = datetime.utcnow()
    elif status in ["completed", "failed"]:
        db_step.completed_at = datetime.utcnow()
        
        if db_step.started_at:
            execution_time = (db_step.completed_at - db_step.started_at).total_seconds() * 1000
            db_step.execution_time_ms = int(execution_time)
    
    if result:
        db_step.result = result
        db_step.success = True
    
    if error_message:
        db_step.error_message = error_message
        db_step.success = False
    
    if confidence_score is not None:
        db_step.confidence_score = confidence_score
    
    if execution_time_ms is not None:
        db_step.execution_time_ms = execution_time_ms
    
    db.commit()
    db.refresh(db_step)
    return db_step

def get_workflow_steps(db: Session, session_id: str) -> List[WorkflowStepDB]:
    """Get all steps for a workflow session"""
    return db.query(WorkflowStepDB).filter(
        WorkflowStepDB.session_id == session_id
    ).order_by(WorkflowStepDB.step_number).all()

def get_workflow_step(db: Session, step_id: str) -> Optional[WorkflowStepDB]:
    """Get a specific workflow step by ID"""
    return db.query(WorkflowStepDB).filter(WorkflowStepDB.step_id == step_id).first()

# ===============================================
# AGENT PERFORMANCE CRUD OPERATIONS
# ===============================================

def record_agent_performance(
    db: Session,
    agent_id: str,
    capability: str,
    execution_time_ms: int,
    success: bool,
    execution_mode: str,
    confidence_score: Optional[float] = None,
    query_complexity: Optional[float] = None,
    user_rating: Optional[int] = None,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    workflow_type: Optional[str] = None
) -> AgentPerformanceMetrics:
    """Record agent performance metrics"""
    performance = AgentPerformanceMetrics(
        agent_id=agent_id,
        capability=capability,
        execution_time_ms=execution_time_ms,
        success=success,
        confidence_score=confidence_score,
        user_rating=user_rating,
        query_complexity=query_complexity,
        execution_mode=execution_mode,
        workflow_type=workflow_type,
        user_id=user_id,
        session_id=session_id
    )
    db.add(performance)
    db.commit()
    db.refresh(performance)
    return performance

def get_agent_performance_stats(
    db: Session,
    agent_id: str,
    capability: Optional[str] = None,
    execution_mode: Optional[str] = None,
    days: int = 30
) -> Dict[str, Any]:
    """Get performance statistics for an agent"""
    since_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(AgentPerformanceMetrics).filter(
        AgentPerformanceMetrics.agent_id == agent_id,
        AgentPerformanceMetrics.created_at >= since_date
    )
    
    if capability:
        query = query.filter(AgentPerformanceMetrics.capability == capability)
    
    if execution_mode:
        query = query.filter(AgentPerformanceMetrics.execution_mode == execution_mode)
    
    metrics = query.all()
    
    if not metrics:
        return {
            "total_executions": 0,
            "success_rate": 0.0,
            "average_execution_time_ms": 0,
            "average_confidence_score": 0.0,
            "average_user_rating": 0.0
        }
    
    total_executions = len(metrics)
    successful_executions = sum(1 for m in metrics if m.success)
    success_rate = successful_executions / total_executions
    
    avg_execution_time = sum(m.execution_time_ms for m in metrics) / total_executions
    
    confidence_scores = [m.confidence_score for m in metrics if m.confidence_score is not None]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    user_ratings = [m.user_rating for m in metrics if m.user_rating is not None]
    avg_rating = sum(user_ratings) / len(user_ratings) if user_ratings else 0.0
    
    return {
        "total_executions": total_executions,
        "success_rate": success_rate,
        "average_execution_time_ms": avg_execution_time,
        "average_confidence_score": avg_confidence,
        "average_user_rating": avg_rating,
        "period_days": days
    }

def get_best_agent_for_capability(
    db: Session,
    capability: str,
    execution_mode: Optional[str] = None,
    days: int = 30
) -> Optional[str]:
    """Get the best performing agent for a capability"""
    since_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(AgentPerformanceMetrics).filter(
        AgentPerformanceMetrics.capability == capability,
        AgentPerformanceMetrics.created_at >= since_date
    )
    
    if execution_mode:
        query = query.filter(AgentPerformanceMetrics.execution_mode == execution_mode)
    
    metrics = query.all()
    
    if not metrics:
        return None
    
    # Group by agent and calculate performance scores
    agent_stats = {}
    for metric in metrics:
        if metric.agent_id not in agent_stats:
            agent_stats[metric.agent_id] = {
                "executions": 0,
                "successes": 0,
                "total_time": 0,
                "total_confidence": 0,
                "confidence_count": 0
            }
        
        stats = agent_stats[metric.agent_id]
        stats["executions"] += 1
        if metric.success:
            stats["successes"] += 1
        stats["total_time"] += metric.execution_time_ms
        
        if metric.confidence_score:
            stats["total_confidence"] += metric.confidence_score
            stats["confidence_count"] += 1
    
    # Calculate performance score for each agent
    best_agent = None
    best_score = 0
    
    for agent_id, stats in agent_stats.items():
        if stats["executions"] < 3:  # Need minimum executions for reliability
            continue
            
        success_rate = stats["successes"] / stats["executions"]
        avg_time = stats["total_time"] / stats["executions"]
        avg_confidence = stats["total_confidence"] / stats["confidence_count"] if stats["confidence_count"] > 0 else 0.5
        
        # Performance score: weighted combination of success rate, speed, and confidence
        speed_score = max(0, (10000 - avg_time) / 10000)  # Normalize execution time
        performance_score = (success_rate * 0.5) + (speed_score * 0.3) + (avg_confidence * 0.2)
        
        if performance_score > best_score:
            best_score = performance_score
            best_agent = agent_id
    
    return best_agent

def get_agent_performance_trends(
    db: Session,
    agent_id: str,
    days: int = 30
) -> Dict[str, Any]:
    """Get performance trends for an agent over time"""
    since_date = datetime.utcnow() - timedelta(days=days)
    
    metrics = db.query(AgentPerformanceMetrics).filter(
        AgentPerformanceMetrics.agent_id == agent_id,
        AgentPerformanceMetrics.created_at >= since_date
    ).order_by(AgentPerformanceMetrics.created_at).all()
    
    if not metrics:
        return {"trend_data": [], "summary": {}}
    
    # Group by day
    daily_stats = {}
    for metric in metrics:
        day = metric.created_at.date()
        if day not in daily_stats:
            daily_stats[day] = {
                "executions": 0,
                "successes": 0,
                "total_time": 0,
                "total_confidence": 0,
                "confidence_count": 0
            }
        
        stats = daily_stats[day]
        stats["executions"] += 1
        if metric.success:
            stats["successes"] += 1
        stats["total_time"] += metric.execution_time_ms
        if metric.confidence_score:
            stats["total_confidence"] += metric.confidence_score
            stats["confidence_count"] += 1
    
    # Calculate daily metrics
    trend_data = []
    for day, stats in sorted(daily_stats.items()):
        success_rate = stats["successes"] / stats["executions"] if stats["executions"] > 0 else 0
        avg_time = stats["total_time"] / stats["executions"] if stats["executions"] > 0 else 0
        avg_confidence = stats["total_confidence"] / stats["confidence_count"] if stats["confidence_count"] > 0 else 0
        
        trend_data.append({
            "date": day.isoformat(),
            "executions": stats["executions"],
            "success_rate": success_rate,
            "avg_execution_time_ms": avg_time,
            "avg_confidence": avg_confidence
        })
    
    return {
        "trend_data": trend_data,
        "summary": {
            "total_days": len(daily_stats),
            "total_executions": sum(stats["executions"] for stats in daily_stats.values()),
            "overall_success_rate": sum(stats["successes"] for stats in daily_stats.values()) / sum(stats["executions"] for stats in daily_stats.values()) if sum(stats["executions"] for stats in daily_stats.values()) > 0 else 0
        }
    }

# ===============================================
# MCP JOB LOG CRUD OPERATIONS
# ===============================================

def create_mcp_job_log(
    db: Session,
    job_id: str,
    job_request: Dict[str, Any],
    session_id: Optional[str] = None
) -> MCPJobLog:
    """Create MCP job log entry"""
    log_entry = MCPJobLog(
        job_id=job_id,
        session_id=session_id,
        job_request=job_request,
        status="submitted"
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry

def update_mcp_job_log_status(
    db: Session,
    job_id: str,
    status: str,
    job_response: Optional[Dict[str, Any]] = None,
    agents_involved: Optional[List[str]] = None,
    error_details: Optional[Dict[str, Any]] = None
) -> Optional[MCPJobLog]:
    """Update MCP job log status"""
    log_entry = db.query(MCPJobLog).filter(MCPJobLog.job_id == job_id).first()
    
    if not log_entry:
        return None
    
    log_entry.status = status
    
    if status == "running":
        log_entry.started_at = datetime.utcnow()
    elif status in ["completed", "failed"]:
        log_entry.completed_at = datetime.utcnow()
        
        if log_entry.submitted_at:
            total_time = (log_entry.completed_at - log_entry.submitted_at).total_seconds() * 1000
            log_entry.total_execution_time_ms = int(total_time)
    
    if job_response:
        log_entry.job_response = job_response
    
    if agents_involved:
        log_entry.agents_involved = agents_involved
    
    if error_details:
        log_entry.error_details = error_details
    
    db.commit()
    db.refresh(log_entry)
    return log_entry

def get_mcp_job_logs(
    db: Session,
    job_id: Optional[str] = None,
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
) -> List[MCPJobLog]:
    """Get MCP job logs with filters"""
    query = db.query(MCPJobLog)
    
    if job_id:
        query = query.filter(MCPJobLog.job_id == job_id)
    if session_id:
        query = query.filter(MCPJobLog.session_id == session_id)
    if status:
        query = query.filter(MCPJobLog.status == status)
    
    return query.order_by(MCPJobLog.submitted_at.desc()).limit(limit).all()

def get_mcp_job_log(db: Session, job_id: str) -> Optional[MCPJobLog]:
    """Get a specific MCP job log by job ID"""
    return db.query(MCPJobLog).filter(MCPJobLog.job_id == job_id).first()

# ===============================================
# ANALYTICS AND REPORTING FUNCTIONS
# ===============================================

def get_workflow_analytics(
    db: Session,
    user_id: Optional[int] = None,
    days: int = 30
) -> Dict[str, Any]:
    """Get workflow execution analytics"""
    since_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(WorkflowSessionDB).filter(
        WorkflowSessionDB.created_at >= since_date
    )
    
    if user_id:
        query = query.filter(WorkflowSessionDB.user_id == user_id)
    
    sessions = query.all()
    
    if not sessions:
        return {
            "total_workflows": 0,
            "completion_rate": 0.0,
            "average_duration_minutes": 0,
            "workflow_type_distribution": {},
            "success_by_complexity": {},
            "execution_mode_distribution": {}
        }
    
    total_workflows = len(sessions)
    completed_workflows = len([s for s in sessions if s.state == "complete"])
    completion_rate = completed_workflows / total_workflows
    
    # Calculate average duration for completed workflows
    completed_sessions = [s for s in sessions if s.completed_at and s.created_at]
    avg_duration = 0
    if completed_sessions:
        total_duration = sum(
            (s.completed_at - s.created_at).total_seconds() 
            for s in completed_sessions
        )
        avg_duration = (total_duration / len(completed_sessions)) / 60  # Convert to minutes
    
    # Workflow type distribution
    type_distribution = {}
    for session in sessions:
        workflow_type = session.workflow_type
        type_distribution[workflow_type] = type_distribution.get(workflow_type, 0) + 1
    
    # Execution mode distribution
    execution_mode_distribution = {}
    for session in sessions:
        execution_mode = session.execution_mode
        execution_mode_distribution[execution_mode] = execution_mode_distribution.get(execution_mode, 0) + 1
    
    # Success rate by complexity
    complexity_buckets = {"low": [], "medium": [], "high": []}
    for session in sessions:
        if session.complexity_score < 0.3:
            complexity_buckets["low"].append(session)
        elif session.complexity_score < 0.7:
            complexity_buckets["medium"].append(session)
        else:
            complexity_buckets["high"].append(session)
    
    success_by_complexity = {}
    for level, sessions_list in complexity_buckets.items():
        if sessions_list:
            success_rate = len([s for s in sessions_list if s.state == "complete"]) / len(sessions_list)
            success_by_complexity[level] = success_rate
        else:
            success_by_complexity[level] = 0.0
    
    return {
        "total_workflows": total_workflows,
        "completion_rate": completion_rate,
        "average_duration_minutes": avg_duration,
        "workflow_type_distribution": type_distribution,
        "execution_mode_distribution": execution_mode_distribution,
        "success_by_complexity": success_by_complexity,
        "period_days": days
    }

def get_system_health_metrics(db: Session) -> Dict[str, Any]:
    """Get overall system health metrics"""
    now = datetime.utcnow()
    last_hour = now - timedelta(hours=1)
    last_day = now - timedelta(days=1)
    
    # Active workflows
    active_workflows = db.query(WorkflowSessionDB).filter(
        WorkflowSessionDB.state.in_(["awaiting_strategy", "awaiting_screening", "awaiting_deep_analysis", "awaiting_final_synthesis"])
    ).count()
    
    # Recent completions
    recent_completions = db.query(WorkflowSessionDB).filter(
        WorkflowSessionDB.completed_at >= last_hour
    ).count()
    
    # Error rate
    recent_workflows = db.query(WorkflowSessionDB).filter(
        WorkflowSessionDB.created_at >= last_day
    ).count()
    
    failed_workflows = db.query(WorkflowSessionDB).filter(
        WorkflowSessionDB.created_at >= last_day,
        WorkflowSessionDB.state == "error"
    ).count()
    
    error_rate = failed_workflows / recent_workflows if recent_workflows > 0 else 0
    
    # Average response time
    completed_today = db.query(WorkflowSessionDB).filter(
        WorkflowSessionDB.completed_at >= last_day,
        WorkflowSessionDB.state == "complete"
    ).all()
    
    avg_response_time = 0
    if completed_today:
        total_duration = sum(
            (w.completed_at - w.created_at).total_seconds()
            for w in completed_today
            if w.completed_at and w.created_at
        )
        avg_response_time = total_duration / len(completed_today)
    
    # MCP job statistics
    mcp_jobs_today = db.query(MCPJobLog).filter(
        MCPJobLog.submitted_at >= last_day
    ).all()
    
    mcp_success_rate = 0
    if mcp_jobs_today:
        successful_mcp = len([j for j in mcp_jobs_today if j.status == "completed"])
        mcp_success_rate = successful_mcp / len(mcp_jobs_today)
    
    return {
        "active_workflows": active_workflows,
        "completions_last_hour": recent_completions,
        "workflows_last_24h": recent_workflows,
        "error_rate_24h": error_rate,
        "avg_response_time_seconds": avg_response_time,
        "mcp_jobs_last_24h": len(mcp_jobs_today),
        "mcp_success_rate": mcp_success_rate,
        "system_status": "healthy" if error_rate < 0.05 and active_workflows < 100 else "degraded"
    }

def get_agent_leaderboard(
    db: Session,
    days: int = 30,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get agent performance leaderboard"""
    since_date = datetime.utcnow() - timedelta(days=days)
    
    metrics = db.query(AgentPerformanceMetrics).filter(
        AgentPerformanceMetrics.created_at >= since_date
    ).all()
    
    if not metrics:
        return []
    
    # Group by agent
    agent_stats = {}
    for metric in metrics:
        if metric.agent_id not in agent_stats:
            agent_stats[metric.agent_id] = {
                "agent_id": metric.agent_id,
                "executions": 0,
                "successes": 0,
                "total_time": 0,
                "total_confidence": 0,
                "confidence_count": 0,
                "total_rating": 0,
                "rating_count": 0
            }
        
        stats = agent_stats[metric.agent_id]
        stats["executions"] += 1
        if metric.success:
            stats["successes"] += 1
        stats["total_time"] += metric.execution_time_ms
        
        if metric.confidence_score:
            stats["total_confidence"] += metric.confidence_score
            stats["confidence_count"] += 1
        
        if metric.user_rating:
            stats["total_rating"] += metric.user_rating
            stats["rating_count"] += 1
    
    # Calculate performance metrics
    leaderboard = []
    for agent_id, stats in agent_stats.items():
        if stats["executions"] < 3:  # Minimum executions required
            continue
        
        success_rate = stats["successes"] / stats["executions"]
        avg_time = stats["total_time"] / stats["executions"]
        avg_confidence = stats["total_confidence"] / stats["confidence_count"] if stats["confidence_count"] > 0 else 0
        avg_rating = stats["total_rating"] / stats["rating_count"] if stats["rating_count"] > 0 else 0
        
        # Calculate overall score
        speed_score = max(0, (10000 - avg_time) / 10000)
        overall_score = (success_rate * 0.4) + (speed_score * 0.3) + (avg_confidence * 0.2) + (avg_rating / 5 * 0.1)
        
        leaderboard.append({
            "agent_id": agent_id,
            "overall_score": overall_score,
            "executions": stats["executions"],
            "success_rate": success_rate,
            "avg_execution_time_ms": avg_time,
            "avg_confidence_score": avg_confidence,
            "avg_user_rating": avg_rating,
            "rating_count": stats["rating_count"]
        })
    
    # Sort by overall score
    leaderboard.sort(key=lambda x: x["overall_score"], reverse=True)
    
    return leaderboard[:limit]

# ===============================================
# CLEANUP FUNCTIONS
# ===============================================

def cleanup_old_workflow_sessions(db: Session, days_old: int = 30) -> int:
    """Clean up old workflow sessions"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    
    old_sessions = db.query(WorkflowSessionDB).filter(
        WorkflowSessionDB.created_at < cutoff_date,
        WorkflowSessionDB.state.in_(["complete", "error"])
    ).all()
    
    count = len(old_sessions)
    
    for session in old_sessions:
        db.delete(session)
    
    db.commit()
    return count

def cleanup_old_performance_metrics(db: Session, days_old: int = 90) -> int:
    """Clean up old performance metrics"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    
    count = db.query(AgentPerformanceMetrics).filter(
        AgentPerformanceMetrics.created_at < cutoff_date
    ).count()
    
    db.query(AgentPerformanceMetrics).filter(
        AgentPerformanceMetrics.created_at < cutoff_date
    ).delete()
    
    db.commit()
    return count

def cleanup_old_mcp_job_logs(db: Session, days_old: int = 60) -> int:
    """Clean up old MCP job logs"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    
    count = db.query(MCPJobLog).filter(
        MCPJobLog.submitted_at < cutoff_date
    ).count()
    
    db.query(MCPJobLog).filter(
        MCPJobLog.submitted_at < cutoff_date
    ).delete()
    
    db.commit()
    return count

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def get_workflow_session_summary(db: Session, session_id: str) -> Optional[Dict[str, Any]]:
    """Get a comprehensive summary of a workflow session"""
    session = get_workflow_session(db, session_id)
    if not session:
        return None
    
    steps = get_workflow_steps(db, session_id)
    mcp_logs = get_mcp_job_logs(db, session_id=session_id)
    
    return {
        "session": session.to_dict(),
        "steps": [
            {
                "step_id": step.step_id,
                "step_name": step.step_name,
                "agent_id": step.agent_id,
                "status": step.status,
                "execution_time_ms": step.execution_time_ms,
                "confidence_score": step.confidence_score,
                "success": step.success
            } for step in steps
        ],
        "mcp_logs": [
            {
                "job_id": log.job_id,
                "status": log.status,
                "execution_time_ms": log.total_execution_time_ms,
                "agents_involved": log.agents_involved
            } for log in mcp_logs
        ],
        "summary_stats": {
            "total_steps": len(steps),
            "completed_steps": len([s for s in steps if s.status == "completed"]),
            "failed_steps": len([s for s in steps if s.status == "failed"]),
            "total_mcp_jobs": len(mcp_logs)
        }
    }

def create_risk_snapshot_from_calculator(
    user_id: str, 
    portfolio_id: str, 
    risk_result: dict,
    portfolio_data: dict = None
) -> PortfolioRiskSnapshot:
    """
    Create a PortfolioRiskSnapshot object from risk calculator results
    """
    return PortfolioRiskSnapshot(
        user_id=user_id,
        portfolio_id=portfolio_id,
        portfolio_name=portfolio_data.get('name', f'Portfolio {portfolio_id}') if portfolio_data else f'Portfolio {portfolio_id}',
        snapshot_date=datetime.now(timezone.utc),
        volatility=risk_result.get('volatility', 0.0),
        beta=risk_result.get('beta', 1.0),
        max_drawdown=risk_result.get('max_drawdown', 0.0),
        var_95=risk_result.get('var_95', 0.0),
        risk_score=risk_result.get('risk_score', 0.5),
        risk_level=risk_result.get('risk_level', 'medium'),
        risk_grade=risk_result.get('risk_grade', 'B'),
        total_value=portfolio_data.get('total_value', 0.0) if portfolio_data else 0.0,
        asset_count=portfolio_data.get('asset_count', 0) if portfolio_data else 0,
        concentration_risk=risk_result.get('concentration_risk', 0.0),
        sector_diversification=risk_result.get('sector_diversification', 0.5),
        geographic_diversification=risk_result.get('geographic_diversification', 0.5)
    )

def get_default_risk_thresholds() -> Dict:
    """
    Return default risk threshold configuration
    """
    return {
        'volatility_threshold': 0.15,          # 15% volatility change threshold
        'beta_threshold': 0.20,                # 20% beta change threshold  
        'max_drawdown_threshold': 0.10,        # 10% max drawdown threshold
        'var_threshold': 0.15,                 # 15% VaR change threshold
        'risk_score_threshold': 0.10,          # 10% risk score change threshold
        'max_acceptable_risk_score': 0.80,     # Maximum acceptable risk score
        'max_acceptable_volatility': 0.35,     # Maximum acceptable volatility (35%)
        'monitoring_enabled': True,
        'alert_frequency': 'immediate',
        'notification_methods': ['websocket', 'email']
    }

# Add these functions to your existing db/crud.py
# These fill the gaps for Task 1.2 completion

from sqlalchemy import func, text
import asyncio
from typing import Generator

# ===============================================
# MISSING BULK OPERATIONS FOR PERFORMANCE
# ===============================================

def bulk_create_risk_snapshots(
    db: Session,
    snapshots_data: List[Dict[str, Any]]
) -> List[PortfolioRiskSnapshot]:
    """
    Efficiently create multiple risk snapshots - missing from your CRUD
    """
    db_snapshots = []
    for snapshot_data in snapshots_data:
        db_snapshot = PortfolioRiskSnapshot(**snapshot_data)
        db_snapshots.append(db_snapshot)
    
    db.bulk_save_objects(db_snapshots, return_defaults=True)
    db.commit()
    
    return db_snapshots

def bulk_update_alert_delivery_status(
    db: Session,
    updates: List[Dict[str, Any]]
) -> int:
    """
    Bulk update alert delivery statuses for performance
    """
    updated_count = 0
    
    for update in updates:
        result = db.query(ProactiveAlert).filter(
            ProactiveAlert.id == update['alert_id']
        ).update({
            'delivery_status': update['delivery_status'],
            'sent_at': update.get('sent_at')
        })
        updated_count += result
    
    db.commit()
    return updated_count

# ===============================================
# MISSING ADVANCED QUERY OPERATIONS
# ===============================================

def get_portfolio_risk_correlation_matrix(
    db: Session,
    user_id: int,
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Calculate risk correlation between user's portfolios
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    snapshots = db.query(PortfolioRiskSnapshot).filter(
        and_(
            PortfolioRiskSnapshot.user_id == user_id,
            PortfolioRiskSnapshot.snapshot_date >= cutoff_date
        )
    ).all()
    
    if len(snapshots) < 10:  # Need minimum data
        return {"status": "insufficient_data"}
    
    # Group by portfolio
    portfolio_risks = {}
    for snapshot in snapshots:
        if snapshot.portfolio_id not in portfolio_risks:
            portfolio_risks[snapshot.portfolio_id] = []
        portfolio_risks[snapshot.portfolio_id].append(snapshot.risk_score)
    
    # Calculate correlations (simplified)
    correlations = {}
    portfolio_ids = list(portfolio_risks.keys())
    
    for i, p1 in enumerate(portfolio_ids):
        correlations[p1] = {}
        for j, p2 in enumerate(portfolio_ids):
            if i <= j:
                # Simple correlation calculation (you could use numpy.corrcoef)
                if len(portfolio_risks[p1]) == len(portfolio_risks[p2]):
                    corr = _calculate_simple_correlation(
                        portfolio_risks[p1], 
                        portfolio_risks[p2]
                    )
                    correlations[p1][p2] = corr
                    if p1 != p2:
                        if p2 not in correlations:
                            correlations[p2] = {}
                        correlations[p2][p1] = corr
    
    return {
        "status": "success",
        "correlations": correlations,
        "portfolio_count": len(portfolio_ids),
        "data_points": len(snapshots)
    }

def get_risk_forecast(
    db: Session,
    user_id: int,
    portfolio_id: int,
    forecast_days: int = 7
) -> Dict[str, Any]:
    """
    Generate risk forecast based on historical trends
    """
    # Get historical data
    snapshots = get_risk_history(db, str(user_id), str(portfolio_id), days=60, limit=60)
    
    if len(snapshots) < 10:
        return {"status": "insufficient_data"}
    
    # Extract risk scores and dates
    risk_scores = [s.risk_score for s in reversed(snapshots)]  # Chronological order
    
    # Simple trend analysis (you could use more sophisticated forecasting)
    if len(risk_scores) >= 5:
        recent_trend = sum(risk_scores[-5:]) / 5 - sum(risk_scores[-10:-5]) / 5
    else:
        recent_trend = 0
    
    # Generate forecast
    current_risk = risk_scores[-1]
    forecast = []
    
    for day in range(1, forecast_days + 1):
        # Simple linear trend projection with volatility
        forecasted_risk = current_risk + (recent_trend * day)
        
        # Add some realistic bounds
        forecasted_risk = max(0, min(100, forecasted_risk))
        
        forecast.append({
            "day": day,
            "forecasted_risk_score": round(forecasted_risk, 2),
            "confidence": max(0.3, 0.9 - (day * 0.1))  # Decreasing confidence
        })
    
    return {
        "status": "success",
        "current_risk_score": current_risk,
        "trend_direction": "increasing" if recent_trend > 0 else "decreasing",
        "trend_magnitude": abs(recent_trend),
        "forecast": forecast
    }

# ===============================================
# MISSING USER PREFERENCE MANAGEMENT
# ===============================================

def get_user_notification_preferences(
    db: Session,
    user_id: int
) -> Dict[str, Any]:
    """
    Get user notification preferences with fallbacks
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {}
    
    preferences = user.preferences or {}
    
    # Default notification preferences
    default_prefs = {
        "risk_alerts": {
            "enabled": True,
            "channels": ["websocket", "email"],
            "threshold": "medium",
            "frequency": "immediate"
        },
        "workflow_updates": {
            "enabled": True,
            "channels": ["websocket"],
            "include_progress": True
        },
        "portfolio_reports": {
            "enabled": True,
            "frequency": "weekly",
            "channels": ["email"]
        }
    }
    
    # Merge user preferences with defaults
    notification_prefs = preferences.get("notifications", {})
    
    for category, default_settings in default_prefs.items():
        if category not in notification_prefs:
            notification_prefs[category] = default_settings
        else:
            # Merge with defaults for missing keys
            for key, value in default_settings.items():
                if key not in notification_prefs[category]:
                    notification_prefs[category][key] = value
    
    return notification_prefs

def update_user_notification_preferences(
    db: Session,
    user_id: int,
    preferences: Dict[str, Any]
) -> bool:
    """
    Update user notification preferences
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return False
    
    current_prefs = user.preferences or {}
    
    # Update notification preferences
    if "notifications" not in current_prefs:
        current_prefs["notifications"] = {}
    
    current_prefs["notifications"].update(preferences)
    user.preferences = current_prefs
    
    db.commit()
    return True

# ===============================================
# MISSING DATA VALIDATION FUNCTIONS
# ===============================================

def validate_risk_snapshot_data(snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate risk snapshot data before insertion
    """
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ['volatility', 'risk_score', 'user_id', 'portfolio_id']
    for field in required_fields:
        if field not in snapshot_data or snapshot_data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Value range validations
    if 'volatility' in snapshot_data:
        vol = snapshot_data['volatility']
        if vol < 0 or vol > 2.0:  # 200% volatility seems extreme
            warnings.append(f"Volatility {vol:.2f} is outside normal range [0, 2.0]")
    
    if 'risk_score' in snapshot_data:
        score = snapshot_data['risk_score']
        if score < 0 or score > 100:
            errors.append(f"Risk score {score} must be between 0 and 100")
    
    if 'sharpe_ratio' in snapshot_data and snapshot_data['sharpe_ratio']:
        sharpe = snapshot_data['sharpe_ratio']
        if sharpe < -5 or sharpe > 10:
            warnings.append(f"Sharpe ratio {sharpe:.2f} is outside typical range [-5, 10]")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def validate_workflow_session_data(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate workflow session data
    """
    errors = []
    
    required_fields = ['session_id', 'user_id', 'query', 'workflow_type']
    for field in required_fields:
        if field not in session_data or not session_data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate workflow_type
    valid_types = ['risk_analysis', 'portfolio_optimization', 'market_analysis', 'custom']
    if 'workflow_type' in session_data:
        if session_data['workflow_type'] not in valid_types:
            errors.append(f"Invalid workflow_type. Must be one of: {valid_types}")
    
    # Validate complexity_score
    if 'complexity_score' in session_data:
        score = session_data['complexity_score']
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            errors.append("complexity_score must be a number between 0 and 1")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

# ===============================================
# MISSING DATABASE MAINTENANCE FUNCTIONS
# ===============================================

def optimize_database_performance(db: Session) -> Dict[str, Any]:
    """
    Run database optimization tasks
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": {}
    }
    
    try:
        # Update table statistics
        tables_to_analyze = [
            "portfolio_risk_snapshots",
            "risk_change_events", 
            "proactive_alerts",
            "workflow_sessions",
            "agent_performance_metrics"
        ]
        
        for table in tables_to_analyze:
            db.execute(text(f"ANALYZE {table}"))
        
        results["tasks"]["analyze_tables"] = {
            "status": "completed",
            "tables_analyzed": len(tables_to_analyze)
        }
        
        # Get table sizes
        table_sizes = {}
        for table in tables_to_analyze:
            size_result = db.execute(text(f"""
                SELECT pg_size_pretty(pg_total_relation_size('{table}')) as size,
                       pg_total_relation_size('{table}') as bytes
            """)).fetchone()
            
            if size_result:
                table_sizes[table] = {
                    "size": size_result[0],
                    "bytes": size_result[1]
                }
        
        results["tasks"]["table_sizes"] = table_sizes
        
        db.commit()
        
    except Exception as e:
        results["tasks"]["error"] = str(e)
    
    return results

def get_database_performance_metrics(db: Session) -> Dict[str, Any]:
    """
    Get comprehensive database performance metrics
    """
    try:
        # Connection stats
        conn_stats = db.execute(text("""
            SELECT 
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active_connections,
                count(*) FILTER (WHERE state = 'idle') as idle_connections
            FROM pg_stat_activity
        """)).fetchone()
        
        # Cache hit ratio
        cache_stats = db.execute(text("""
            SELECT 
                sum(blks_read) as total_read,
                sum(blks_hit) as total_hit,
                CASE 
                    WHEN sum(blks_read + blks_hit) = 0 THEN 0 
                    ELSE round((sum(blks_hit) * 100.0) / sum(blks_read + blks_hit), 2) 
                END as cache_hit_ratio
            FROM pg_stat_database
            WHERE datname = current_database()
        """)).fetchone()
        
        # Recent query performance
        slow_queries = db.execute(text("""
            SELECT 
                count(*) as slow_query_count
            FROM pg_stat_statements 
            WHERE mean_exec_time > 1000 
            LIMIT 1
        """)).fetchone()
        
        return {
            "connection_stats": {
                "total": conn_stats[0] if conn_stats else 0,
                "active": conn_stats[1] if conn_stats else 0,
                "idle": conn_stats[2] if conn_stats else 0
            },
            "cache_performance": {
                "hit_ratio_percent": cache_stats[2] if cache_stats else 0,
                "total_reads": cache_stats[0] if cache_stats else 0,
                "total_hits": cache_stats[1] if cache_stats else 0
            },
            "query_performance": {
                "slow_queries_count": slow_queries[0] if slow_queries else 0
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# ===============================================
# MISSING HELPER FUNCTIONS
# ===============================================

def _calculate_simple_correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate simple correlation coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

async def async_create_risk_snapshot(
    db: Session,
    user_id: str,
    portfolio_id: str,
    risk_result: dict,
    portfolio_data: dict = None
) -> PortfolioRiskSnapshot:
    """
    Async version of risk snapshot creation for high-throughput scenarios
    """
    # This would be useful for processing many snapshots concurrently
    # For now, just call the sync version
    return create_risk_snapshot(db, user_id, portfolio_id, risk_result, portfolio_data)

def batch_process_risk_snapshots(
    db: Session,
    snapshot_requests: List[Dict[str, Any]],
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Process risk snapshots in batches for better performance
    """
    total_requests = len(snapshot_requests)
    processed = 0
    errors = []
    
    for i in range(0, total_requests, batch_size):
        batch = snapshot_requests[i:i + batch_size]
        
        try:
            # Process batch
            batch_data = []
            for request in batch:
                validation = validate_risk_snapshot_data(request)
                if validation["valid"]:
                    snapshot = create_risk_snapshot_from_calculator(
                        request["user_id"],
                        request["portfolio_id"], 
                        request["risk_result"],
                        request.get("portfolio_data")
                    )
                    batch_data.append(snapshot)
                else:
                    errors.append({
                        "request": request,
                        "errors": validation["errors"]
                    })
            
            # Bulk insert valid snapshots
            if batch_data:
                db.bulk_save_objects([s.__dict__ for s in batch_data])
                db.commit()
                processed += len(batch_data)
                
        except Exception as e:
            errors.append({
                "batch_start": i,
                "error": str(e)
            })
    
    return {
        "total_requests": total_requests,
        "processed_successfully": processed,
        "error_count": len(errors),
        "errors": errors
    }

def ensure_integer_id(id_value):
    """Convert string ID to integer if needed"""
    if isinstance(id_value, str):
        try:
            return int(id_value)
        except ValueError:
            return None
    return id_value

def create_risk_snapshot(
    db: Session,
    portfolio_id: int,
    risk_metrics,
    compression_level: int = 6
) -> PortfolioRiskSnapshot:
    """Create a new risk snapshot with compressed metrics"""
    
    snapshot = PortfolioRiskSnapshot.create_from_risk_metrics(
        portfolio_id=portfolio_id,
        risk_metrics=risk_metrics,
        compression_level=compression_level
    )
    
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot

def get_latest_risk_snapshot(
    db: Session,
    user_id: str,         # 1. Added user_id: str parameter to match the call
    portfolio_id: str     # 2. Changed portfolio_id to str for consistency with the call
) -> Optional[PortfolioRiskSnapshot]:
    """Get the most recent risk snapshot for a specific user's portfolio"""
    
    return db.query(PortfolioRiskSnapshot).filter(
        PortfolioRiskSnapshot.user_id == user_id,             # 3. Added user_id to the filter (important!)
        PortfolioRiskSnapshot.portfolio_id == portfolio_id
    ).order_by(PortfolioRiskSnapshot.snapshot_date.desc()).first() # 4. Completed the query

def get_risk_snapshots_range(
    db: Session,
    portfolio_id: int,
    start_date: datetime,
    end_date: datetime,
    limit: Optional[int] = None
) -> List[PortfolioRiskSnapshot]:
    """Get risk snapshots within a date range"""
    
    query = db.query(PortfolioRiskSnapshot).filter(
        and_(
            PortfolioRiskSnapshot.portfolio_id == portfolio_id,
            PortfolioRiskSnapshot.snapshot_date >= start_date,
            PortfolioRiskSnapshot.snapshot_date <= end_date
        )
    ).order_by(PortfolioRiskSnapshot.snapshot_date)
    
    if limit:
        query = query.limit(limit)
    
    return query.all()

def get_risk_snapshots_count(db: Session, portfolio_id: int) -> int:
    """Get total count of risk snapshots for a portfolio"""
    
    return db.query(PortfolioRiskSnapshot).filter(
        PortfolioRiskSnapshot.portfolio_id == portfolio_id
    ).count()

def delete_old_snapshots(
    db: Session, 
    older_than_days: int,
    batch_size: int = 1000
) -> int:
    """Delete old risk snapshots beyond retention period"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
    
    # Delete in batches to avoid locking issues
    deleted_count = 0
    while True:
        snapshots_to_delete = db.query(PortfolioRiskSnapshot).filter(
            PortfolioRiskSnapshot.snapshot_date < cutoff_date
        ).limit(batch_size).all()
        
        if not snapshots_to_delete:
            break
        
        for snapshot in snapshots_to_delete:
            db.delete(snapshot)
        
        db.commit()
        deleted_count += len(snapshots_to_delete)
    
    return deleted_count

# Risk Trend CRUD Operations
def create_or_update_risk_trend(
    db: Session,
    portfolio_id: int,
    metric_name: str,
    trend_data: Dict[str, Any]
) -> RiskTrend:
    """Create or update risk trend analysis"""
    
    # Check if trend already exists
    existing_trend = db.query(RiskTrend).filter(
        and_(
            RiskTrend.portfolio_id == portfolio_id,
            RiskTrend.metric_name == metric_name
        )
    ).first()
    
    if existing_trend:
        # Update existing trend
        for key, value in trend_data.items():
            if hasattr(existing_trend, key):
                setattr(existing_trend, key, value)
        existing_trend.last_updated = datetime.utcnow()
        
        db.commit()
        db.refresh(existing_trend)
        return existing_trend
    else:
        # Create new trend
        trend = RiskTrend(
            portfolio_id=portfolio_id,
            metric_name=metric_name,
            **trend_data
        )
        
        db.add(trend)
        db.commit()
        db.refresh(trend)
        return trend

def get_risk_trends(
    db: Session,
    portfolio_id: int,
    metric_names: Optional[List[str]] = None
) -> List[RiskTrend]:
    """Get risk trends for a portfolio"""
    
    query = db.query(RiskTrend).filter(
        RiskTrend.portfolio_id == portfolio_id
    )
    
    if metric_names:
        query = query.filter(RiskTrend.metric_name.in_(metric_names))
    
    return query.order_by(RiskTrend.metric_name).all()

def get_latest_risk_trends(db: Session, portfolio_id: int) -> List[RiskTrend]:
    """Get the most recent risk trends for a portfolio"""
    
    return db.query(RiskTrend).filter(
        RiskTrend.portfolio_id == portfolio_id
    ).order_by(desc(RiskTrend.last_updated)).all()

# Risk Change Event CRUD Operations
def create_risk_change_event(
    db: Session,
    portfolio_id: int,
    event_data: Dict[str, Any]
) -> RiskChangeEvent:
    """Create a new risk change event"""
    
    event = RiskChangeEvent(
        portfolio_id=portfolio_id,
        **event_data
    )
    
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

def get_recent_risk_events(
    db: Session,
    portfolio_id: Optional[int] = None,
    user_id: Optional[int] = None,
    hours: int = 24,
    severity_filter: Optional[List[str]] = None,
    limit: int = 100
) -> List[RiskChangeEvent]:
    """Get recent risk change events"""
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    query = db.query(RiskChangeEvent).filter(
        RiskChangeEvent.detected_at >= cutoff_time
    )
    
    if portfolio_id:
        query = query.filter(RiskChangeEvent.portfolio_id == portfolio_id)
    
    if user_id:
        query = query.filter(RiskChangeEvent.user_id == user_id)
    
    if severity_filter:
        query = query.filter(RiskChangeEvent.severity.in_(severity_filter))
    
    return query.order_by(desc(RiskChangeEvent.detected_at)).all()

# Risk Threshold CRUD Operations
def create_risk_threshold(
    db: Session,
    threshold_data: Dict[str, Any]
) -> RiskThreshold:
    """Create a new risk threshold configuration"""
    
    threshold = RiskThreshold(**threshold_data)
    
    db.add(threshold)
    db.commit()
    db.refresh(threshold)
    return threshold

def get_risk_thresholds(
    db: Session,
    portfolio_id: Optional[int] = None,
    user_id: Optional[int] = None,
    metric_name: Optional[str] = None
) -> List[RiskThreshold]:
    """Get risk thresholds with optional filtering"""
    
    query = db.query(RiskThreshold).filter(RiskThreshold.enabled == True)
    
    if portfolio_id:
        # Get portfolio-specific thresholds, with global defaults as fallback
        query = query.filter(
            (RiskThreshold.portfolio_id == portfolio_id) | 
            (RiskThreshold.portfolio_id.is_(None))
        )
    
    if user_id:
        query = query.filter(
            (RiskThreshold.user_id == user_id) | 
            (RiskThreshold.user_id.is_(None))
        )
    
    if metric_name:
        query = query.filter(RiskThreshold.metric_name == metric_name)
    
    return query.order_by(
        RiskThreshold.portfolio_id.desc(),  # Portfolio-specific first
        RiskThreshold.user_id.desc(),      # User-specific first
        RiskThreshold.metric_name
    ).all()

def update_risk_threshold(
    db: Session,
    threshold_id: int,
    threshold_data: Dict[str, Any]
) -> Optional[RiskThreshold]:
    """Update an existing risk threshold"""
    
    threshold = db.query(RiskThreshold).filter(
        RiskThreshold.id == threshold_id
    ).first()
    
    if threshold:
        for key, value in threshold_data.items():
            if hasattr(threshold, key):
                setattr(threshold, key, value)
        
        threshold.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(threshold)
    
    return threshold

def delete_risk_threshold(db: Session, threshold_id: int) -> bool:
    """Delete a risk threshold configuration"""
    
    threshold = db.query(RiskThreshold).filter(
        RiskThreshold.id == threshold_id
    ).first()
    
    if threshold:
        db.delete(threshold)
        db.commit()
        return True
    
    return False

def get_effective_thresholds(
    db: Session,
    portfolio_id: int,
    user_id: Optional[int] = None
) -> Dict[str, RiskThreshold]:
    """Get the effective thresholds for a portfolio (with priority resolution)"""
    
    # Get all applicable thresholds
    thresholds = get_risk_thresholds(db, portfolio_id, user_id)
    
    # Resolve priorities: portfolio-specific > user-specific > global defaults
    effective_thresholds = {}
    
    for threshold in thresholds:
        metric = threshold.metric_name
        
        # Priority logic: more specific configurations override general ones
        if metric not in effective_thresholds:
            effective_thresholds[metric] = threshold
        else:
            current = effective_thresholds[metric]
            
            # Portfolio-specific beats all
            if threshold.portfolio_id is not None and current.portfolio_id is None:
                effective_thresholds[metric] = threshold
            # User-specific beats global default
            elif (threshold.user_id is not None and current.user_id is None and 
                  threshold.portfolio_id == current.portfolio_id):
                effective_thresholds[metric] = threshold
    
    return effective_thresholds

# Price Data Cache CRUD Operations
def cache_price_data(
    db: Session,
    symbol: str,
    price_data: Dict[str, Any],
    ttl_minutes: int = 5
) -> PriceDataCache:
    """Cache price data for performance optimization"""
    
    expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    
    # Check if entry already exists
    existing_cache = db.query(PriceDataCache).filter(
        PriceDataCache.symbol == symbol
    ).first()
    
    if existing_cache:
        # Update existing cache
        for key, value in price_data.items():
            if hasattr(existing_cache, key):
                setattr(existing_cache, key, value)
        existing_cache.fetched_at = datetime.utcnow()
        existing_cache.expires_at = expires_at
        
        db.commit()
        db.refresh(existing_cache)
        return existing_cache
    else:
        # Create new cache entry
        cache_entry = PriceDataCache(
            symbol=symbol,
            expires_at=expires_at,
            **price_data
        )
        
        db.add(cache_entry)
        db.commit()
        db.refresh(cache_entry)
        return cache_entry

def get_cached_price(db: Session, symbol: str) -> Optional[PriceDataCache]:
    """Get cached price data if not expired"""
    
    return db.query(PriceDataCache).filter(
        and_(
            PriceDataCache.symbol == symbol,
            PriceDataCache.expires_at > datetime.utcnow()
        )
    ).first()

def get_cached_prices(db: Session, symbols: List[str]) -> List[PriceDataCache]:
    """Get cached price data for multiple symbols"""
    
    return db.query(PriceDataCache).filter(
        and_(
            PriceDataCache.symbol.in_(symbols),
            PriceDataCache.expires_at > datetime.utcnow()
        )
    ).all()

def cleanup_expired_price_cache(db: Session, batch_size: int = 1000) -> int:
    """Remove expired price cache entries"""
    
    cutoff_time = datetime.utcnow()
    
    deleted_count = 0
    while True:
        expired_entries = db.query(PriceDataCache).filter(
            PriceDataCache.expires_at <= cutoff_time
        ).limit(batch_size).all()
        
        if not expired_entries:
            break
        
        for entry in expired_entries:
            db.delete(entry)
        
        db.commit()
        deleted_count += len(expired_entries)
    
    return deleted_count

# Portfolio Holdings CRUD (Enhanced for Risk Detection)
def get_portfolio_holdings_with_prices(
    db: Session,
    portfolio_id: int,
    include_cached_prices: bool = True
) -> List[Dict[str, Any]]:
    """Get portfolio holdings with current price information"""
    
    from sqlalchemy.orm import joinedload
    
    # Get holdings with asset information
    holdings = db.query(Holding).filter(
        Holding.portfolio_id == portfolio_id
    ).options(joinedload(Holding.asset)).all()
    
    holdings_data = []
    
    for holding in holdings:
        holding_data = {
            'holding_id': holding.id,
            'symbol': holding.asset.symbol,
            'quantity': holding.quantity,
            'asset_type': holding.asset.asset_type,
            'current_price': None,
            'position_value': 0.0,
            'price_timestamp': None,
            'price_provider': None
        }
        
        # Get cached price if available
        if include_cached_prices and holding.asset.symbol:
            cached_price = get_cached_price(db, holding.asset.symbol)
            if cached_price:
                holding_data.update({
                    'current_price': cached_price.price,
                    'position_value': holding.quantity * cached_price.price,
                    'price_timestamp': cached_price.fetched_at,
                    'price_provider': cached_price.provider,
                    'currency': cached_price.currency,
                    'change_percent': cached_price.change_percent
                })
        
        holdings_data.append(holding_data)
    
    return holdings_data

# Analytics and Reporting Functions
def get_portfolio_risk_summary(
    db: Session,
    portfolio_id: int,
    days_back: int = 30
) -> Dict[str, Any]:
    """Get comprehensive risk summary for a portfolio"""
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    
    # Get latest snapshot
    latest_snapshot = get_latest_risk_snapshot(db, portfolio_id)
    
    # Get recent snapshots for trend analysis
    recent_snapshots = get_risk_snapshots_range(
        db, portfolio_id, start_date, end_date
    )
    
    # Get recent events
    recent_events = get_recent_risk_events(
        db, portfolio_id=portfolio_id, hours=days_back * 24
    )
    
    # Get current trends
    current_trends = get_latest_risk_trends(db, portfolio_id)
    
    # Calculate summary statistics
    summary = {
        'portfolio_id': portfolio_id,
        'analysis_period_days': days_back,
        'latest_snapshot': {
            'snapshot_date': latest_snapshot.snapshot_date if latest_snapshot else None,
            'metrics_summary': latest_snapshot.metrics_summary if latest_snapshot else None,
            'data_quality': latest_snapshot.data_quality_score if latest_snapshot else 0.0
        },
        'historical_data': {
            'snapshots_count': len(recent_snapshots),
            'data_coverage_percent': (len(recent_snapshots) / days_back) * 100 if days_back > 0 else 0
        },
        'recent_events': {
            'total_events': len(recent_events),
            'critical_events': len([e for e in recent_events if e.severity in ['critical', 'emergency']]),
            'unacknowledged_events': len([e for e in recent_events if e.acknowledged_at is None])
        },
        'trends': {
            'metrics_tracked': len(current_trends),
            'deteriorating_trends': len([t for t in current_trends if t.trend_direction == 'increasing' and t.metric_name in ['annualized_volatility', 'max_drawdown']]),
            'improving_trends': len([t for t in current_trends if t.trend_direction == 'decreasing' and t.metric_name in ['annualized_volatility', 'max_drawdown']])
        },
        'generated_at': datetime.utcnow()
    }
    
    return summary

def get_system_health_metrics(db: Session) -> Dict[str, Any]:
    """Get system health metrics for monitoring"""
    
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    last_hour = now - timedelta(hours=1)
    
    # Snapshot metrics
    recent_snapshots = db.query(func.count(PortfolioRiskSnapshot.id)).filter(
        PortfolioRiskSnapshot.created_at >= last_24h
    ).scalar()
    
    # Event metrics
    recent_events = db.query(func.count(RiskChangeEvent.id)).filter(
        RiskChangeEvent.detected_at >= last_24h
    ).scalar()
    
    critical_events_last_hour = db.query(func.count(RiskChangeEvent.id)).filter(
    and_(
        RiskChangeEvent.detected_at >= last_hour,
        RiskChangeEvent.severity.in_(['critical', 'emergency'])
    )
).scalar()
    
    # Cache metrics
    cache_entries = db.query(func.count(PriceDataCache.id)).scalar()
    expired_cache_entries = db.query(func.count(PriceDataCache.id)).filter(
        PriceDataCache.expires_at <= now
    ).scalar()
    
    # Portfolio coverage
    portfolios_with_recent_snapshots = db.query(func.count(func.distinct(PortfolioRiskSnapshot.portfolio_id))).filter(
        PortfolioRiskSnapshot.created_at >= last_24h
    ).scalar()
    
    total_portfolios = db.query(func.count(Portfolio.id)).scalar()
    
    return {
        'snapshots_last_24h': recent_snapshots,
        'events_last_24h': recent_events,
        'critical_events_last_hour': critical_events_last_hour,
        'cache_entries_total': cache_entries,
        'cache_entries_expired': expired_cache_entries,
        'cache_hit_rate_estimate': max(0, (cache_entries - expired_cache_entries) / max(cache_entries, 1)),
        'portfolio_coverage_24h': portfolios_with_recent_snapshots,
        'total_portfolios': total_portfolios,
        'coverage_percentage': (portfolios_with_recent_snapshots / max(total_portfolios, 1)) * 100,
        'system_status': 'healthy' if (
            recent_snapshots > 0 and 
            critical_events_last_hour < 5 and 
            portfolios_with_recent_snapshots > 0
        ) else 'degraded',
        'generated_at': now
    }

# Bulk Operations for Performance
def bulk_create_risk_snapshots(
    db: Session,
    snapshots_data: List[Dict[str, Any]]
) -> List[PortfolioRiskSnapshot]:
    """Create multiple risk snapshots in bulk for performance"""
    
    snapshots = []
    for data in snapshots_data:
        snapshot = PortfolioRiskSnapshot(**data)
        snapshots.append(snapshot)
    
    db.add_all(snapshots)
    db.commit()
    
    for snapshot in snapshots:
        db.refresh(snapshot)
    
    return snapshots

def acknowledge_risk_event(
    db: Session,
    event_id: int,
    acknowledged_by: int
) -> Optional[RiskChangeEvent]:
    """Acknowledge a risk change event"""
    
    event = db.query(RiskChangeEvent).filter(
        RiskChangeEvent.id == event_id
    ).first()
    
    if event:
        event.acknowledged_at = datetime.utcnow()
        event.acknowledged_by = acknowledged_by
        
        db.commit()
        db.refresh(event)
    
    return event

def resolve_risk_event(
    db: Session,
    event_id: int
) -> Optional[RiskChangeEvent]:
    """Mark a risk change event as resolved"""
    
    event = db.query(RiskChangeEvent).filter(
        RiskChangeEvent.id == event_id
    ).first()
    
    if event:
        event.resolved_at = datetime.utcnow()
        
        db.commit()
        db.refresh(event)
    
    return event

def get_unacknowledged_critical_events(
    db: Session,
    portfolio_id: Optional[int] = None,
    hours: int = 4
) -> List[RiskChangeEvent]:
    """Get critical events that haven't been acknowledged"""
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    query = db.query(RiskChangeEvent).filter(
        and_(
            RiskChangeEvent.detected_at >= cutoff_time,
            RiskChangeEvent.severity.in_(["critical", "emergency"]),
            RiskChangeEvent.acknowledged_at.is_(None)
        )
    )
    
    if portfolio_id:
        query = query.filter(RiskChangeEvent.portfolio_id == portfolio_id)
    
    return query.order_by(desc(RiskChangeEvent.detected_at)).all()


def log_risk_alert(
    db: Session,
    portfolio_id: int,
    alert_data: Dict[str, Any]
) -> RiskChangeEvent:
    """Convenience function to log a risk alert"""
    return create_risk_change_event(db, portfolio_id, alert_data)


def get_portfolio_holdings(db: Session, portfolio_id: int):
    """Get all holdings for a specific portfolio with asset information"""
    from sqlalchemy.orm import joinedload
    
    return db.query(models.Holding).filter(
        models.Holding.portfolio_id == portfolio_id
    ).options(joinedload(models.Holding.asset)).all()

def get_portfolio_by_id(db: Session, portfolio_id: int):
    """Get portfolio by ID"""
    return db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()