# db/models.py

from sqlalchemy import (
    Column, Integer, String, Float, ForeignKey, Text, 
    JSON, Boolean, DateTime, Enum as SQLAlchemyEnum, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime,timezone
import enum

# This is our local Python Enum, used for defining choices
from .session import Base

# --- Enums for the Alert Model ---
class AlertType(enum.Enum):
    PRICE = "price"
    PORTFOLIO_VALUE = "portfolio_value"
    RISK_METRIC = "risk_metric"

class Operator(enum.Enum):
    GREATER_THAN = ">"
    LESS_THAN = "<"

# --- Database Models ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    preferences = Column(JSON, nullable=True)

    portfolios = relationship("Portfolio", back_populates="owner")
    alerts = relationship("Alert", back_populates="owner", cascade="all, delete-orphan")

class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")

class Holding(Base):
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    shares = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=True)
    
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    asset_id = Column(Integer, ForeignKey("assets.id"))

    portfolio = relationship("Portfolio", back_populates="holdings")
    asset = relationship("Asset")

    @hybrid_property
    def ticker(self):
        return self.asset.ticker

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=True)
    
    # ### THE FIX IS HERE ###
    # Use the imported and aliased SQLAlchemyEnum, not the standard Python Enum
    alert_type = Column(SQLAlchemyEnum(AlertType), nullable=False)
    metric = Column(String, nullable=True)
    ticker = Column(String, nullable=True)
    operator = Column(SQLAlchemyEnum(Operator), nullable=False)
    # ### END FIX ###

    value = Column(Float, nullable=False)
    
    is_triggered = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    triggered_at = Column(DateTime, nullable=True)

    owner = relationship("User", back_populates="alerts")
    portfolio = relationship("Portfolio")

class PortfolioRiskSnapshot(Base):
    """
    Store historical risk metrics for portfolio monitoring.
    Enables trend analysis and risk change detection.
    """
    __tablename__ = "portfolio_risk_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Core Risk Metrics
    volatility = Column(Float, nullable=False)  # Annualized volatility (0.15 = 15%)
    beta = Column(Float, nullable=False)  # Market beta
    max_drawdown = Column(Float, nullable=False)  # Maximum drawdown (0.20 = 20%)
    
    # Value at Risk Metrics
    var_95 = Column(Float, nullable=False)  # 95% VaR (daily)
    var_99 = Column(Float, nullable=False)  # 99% VaR (daily)
    cvar_95 = Column(Float, nullable=False)  # 95% CVaR/Expected Shortfall
    cvar_99 = Column(Float, nullable=False)  # 99% CVaR/Expected Shortfall
    
    # Risk-Adjusted Performance
    sharpe_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float, nullable=False)
    calmar_ratio = Column(Float, nullable=False)
    
    # Advanced Risk Metrics
    hurst_exponent = Column(Float, nullable=True)  # Fractal analysis
    dfa_alpha = Column(Float, nullable=True)  # Detrended Fluctuation Analysis
    
    # Composite Metrics
    risk_score = Column(Float, nullable=False)  # 0-100 composite risk score
    sentiment_index = Column(Integer, nullable=False)  # Risk sentiment (0-100)
    regime_volatility = Column(Float, nullable=True)  # Regime-aware volatility
    
    # Metadata
    snapshot_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    calculation_method = Column(String, default="comprehensive", nullable=False)
    data_window_days = Column(Integer, default=252, nullable=False)  # Lookback period
    
    # Additional Context (JSON for flexibility)
    risk_factors = Column(JSON, nullable=True)  # Detailed risk breakdown
    market_conditions = Column(JSON, nullable=True)  # Market context at time of snapshot
    
    # Relationships
    portfolio = relationship("Portfolio")
    user = relationship("User")
    
    def __repr__(self):
        return f"<RiskSnapshot(portfolio_id={self.portfolio_id}, date={self.snapshot_date}, risk_score={self.risk_score})>"

class RiskChangeEvent(Base):
    """
    Track significant risk changes that trigger alerts or workflows.
    Provides audit trail for risk monitoring system.
    """
    __tablename__ = "risk_change_events"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Risk Change Details
    current_snapshot_id = Column(Integer, ForeignKey("portfolio_risk_snapshots.id"), nullable=False)
    previous_snapshot_id = Column(Integer, ForeignKey("portfolio_risk_snapshots.id"), nullable=True)
    
    # Change Metrics
    risk_direction = Column(String, nullable=False)  # "INCREASED" or "DECREASED"
    risk_magnitude_pct = Column(Float, nullable=False)  # Percentage change in risk score
    threshold_breached = Column(Boolean, default=False, nullable=False)
    
    # Specific Changes (JSON for detailed breakdown)
    risk_changes = Column(JSON, nullable=False)  # All metric changes
    significant_changes = Column(JSON, nullable=False)  # Only changes above threshold
    
    # Response Actions
    workflow_triggered = Column(Boolean, default=False, nullable=False)
    workflow_session_id = Column(String, nullable=True)  # If workflow was started
    alerts_generated = Column(JSON, nullable=True)  # Alert IDs generated
    
    # Metadata
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    threshold_pct = Column(Float, default=15.0, nullable=False)  # Threshold used
    
    # Relationships
    portfolio = relationship("Portfolio")
    user = relationship("User")
    current_snapshot = relationship("PortfolioRiskSnapshot", foreign_keys=[current_snapshot_id])
    previous_snapshot = relationship("PortfolioRiskSnapshot", foreign_keys=[previous_snapshot_id])
    
    def __repr__(self):
        return f"<RiskChangeEvent(portfolio_id={self.portfolio_id}, direction={self.risk_direction}, magnitude={self.risk_magnitude_pct}%)>"

class ProactiveAlert(Base):
    """
    Enhanced alert model specifically for proactive risk monitoring.
    Extends your existing Alert model with additional risk-specific fields.
    """
    __tablename__ = "proactive_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Alert Classification
    alert_type = Column(String, nullable=False)  # From AlertType enum in proactive_monitor.py
    priority = Column(String, nullable=False)  # From AlertPriority enum
    
    # Alert Content
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)  # Structured alert details
    
    # Risk Context
    risk_change_event_id = Column(Integer, ForeignKey("risk_change_events.id"), nullable=True)
    triggered_risk_score = Column(Float, nullable=True)
    
    # Alert Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    sent_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_sent = Column(Boolean, default=False, nullable=False)
    is_acknowledged = Column(Boolean, default=False, nullable=False)
    is_resolved = Column(Boolean, default=False, nullable=False)
    
    # Delivery Tracking
    delivery_channels = Column(JSON, nullable=True)  # email, websocket, etc.
    delivery_status = Column(JSON, nullable=True)  # Delivery confirmation
    
    # Relationships
    portfolio = relationship("Portfolio")
    user = relationship("User")
    risk_change_event = relationship("RiskChangeEvent")
    
    def __repr__(self):
        return f"<ProactiveAlert(id={self.id}, type={self.alert_type}, priority={self.priority})>"
    


# Risk Attribution Models - Clean Version
from sqlalchemy.dialects.postgresql import JSON
import uuid

class PortfolioRiskSnapshot(Base):
    """
    Stores point-in-time risk metrics for portfolios
    """
    __tablename__ = "portfolio_risk_snapshots"
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    
    # Portfolio identification
    user_id = Column(String(50), index=True, nullable=False)
    portfolio_id = Column(String(50), index=True, nullable=False)
    portfolio_name = Column(String(100))
    
    # Timestamp information
    snapshot_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Core risk metrics
    volatility = Column(Float, nullable=False)
    beta = Column(Float)
    max_drawdown = Column(Float)
    var_95 = Column(Float)
    var_99 = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    
    # Composite metrics
    risk_score = Column(Float, nullable=False)
    risk_grade = Column(String(2))
    
    # Portfolio composition metrics
    portfolio_value = Column(Float)
    num_positions = Column(Integer)
    concentration_risk = Column(Float)
    
    # Advanced risk metrics
    tail_dependency = Column(Float)
    hurst_exponent = Column(Float)
    regime_state = Column(Integer)
    
    # Change detection
    volatility_change_pct = Column(Float)
    risk_score_change_pct = Column(Float)
    is_threshold_breach = Column(Boolean, default=False)
    
    # Metadata
    calculation_method = Column(String(20), default='standard')
    data_quality_score = Column(Float, default=1.0)
    notes = Column(Text)
    
    # Raw data storage
    raw_metrics = Column(JSON)
    portfolio_weights = Column(JSON)
    correlation_matrix = Column(JSON)
    
    # Table configuration
    __table_args__ = (
        Index('idx_user_portfolio_date', 'user_id', 'portfolio_id', 'snapshot_date'),
        Index('idx_risk_score_date', 'risk_score', 'snapshot_date'),
        Index('idx_threshold_breach', 'is_threshold_breach', 'snapshot_date'),
        Index('idx_user_recent', 'user_id', 'snapshot_date'),
        {'extend_existing': True}
    )
    
    def __repr__(self):
        return f"<RiskSnapshot(user={self.user_id}, portfolio={self.portfolio_id}, risk_score={self.risk_score})>"
    
    @property
    def risk_level(self):
        """Convert risk score to descriptive level"""
        if self.risk_score <= 20:
            return "Very Low"
        elif self.risk_score <= 40:
            return "Low"
        elif self.risk_score <= 60:
            return "Moderate"
        elif self.risk_score <= 80:
            return "High"
        else:
            return "Very High"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'snapshot_id': self.snapshot_id,
            'user_id': self.user_id,
            'portfolio_id': self.portfolio_id,
            'portfolio_name': self.portfolio_name,
            'snapshot_date': self.snapshot_date.isoformat() if self.snapshot_date else None,
            'volatility': self.volatility,
            'beta': self.beta,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'sharpe_ratio': self.sharpe_ratio,
            'risk_score': self.risk_score,
            'risk_grade': self.risk_grade,
            'risk_level': self.risk_level,
            'portfolio_value': self.portfolio_value,
            'volatility_change_pct': self.volatility_change_pct,
            'risk_score_change_pct': self.risk_score_change_pct,
            'is_threshold_breach': self.is_threshold_breach,
            'tail_dependency': self.tail_dependency,
            'hurst_exponent': self.hurst_exponent
        }


class RiskThresholdConfig(Base):
    """
    Configurable risk thresholds per user/portfolio
    """
    __tablename__ = "risk_threshold_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), index=True, nullable=False)
    portfolio_id = Column(String(50), index=True)
    
    # Threshold percentages
    volatility_threshold = Column(Float, default=0.15)
    beta_threshold = Column(Float, default=0.20)
    max_drawdown_threshold = Column(Float, default=0.10)
    var_threshold = Column(Float, default=0.15)
    risk_score_threshold = Column(Float, default=0.20)
    
    # Absolute thresholds
    max_acceptable_risk_score = Column(Float, default=80.0)
    max_acceptable_volatility = Column(Float, default=0.50)
    
    # Monitoring settings
    monitoring_enabled = Column(Boolean, default=True)
    alert_frequency = Column(String(20), default='immediate')
    notification_methods = Column(JSON, default=lambda: ['websocket', 'email'])
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    __table_args__ = (
        Index('idx_user_portfolio_config', 'user_id', 'portfolio_id', unique=True),
        {'extend_existing': True}
    )


class RiskAlertLog(Base):
    """
    Log of all risk alerts sent to users
    """
    __tablename__ = "risk_alert_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    
    # Alert identification
    user_id = Column(String(50), index=True, nullable=False)
    portfolio_id = Column(String(50), index=True, nullable=False)
    snapshot_id = Column(String(36))
    
    # Alert details
    alert_type = Column(String(30), nullable=False)
    alert_severity = Column(String(10), default='medium')
    alert_message = Column(Text, nullable=False)
    
    # Risk metrics
    triggered_metrics = Column(JSON)
    risk_change_summary = Column(JSON)
    
    # Workflow integration
    workflow_id = Column(String(36))
    workflow_status = Column(String(20))
    
    # Notification tracking
    notification_methods = Column(JSON)
    notification_status = Column(JSON)
    user_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    __table_args__ = (
        Index('idx_alert_user_portfolio', 'user_id', 'portfolio_id'),
        Index('idx_alert_workflow', 'workflow_id'),
        {'extend_existing': True}
    )


# Helper functions
def create_risk_snapshot_from_calculator(
    user_id: str,
    portfolio_id: str,
    risk_result: dict,
    portfolio_data: dict = None
) -> PortfolioRiskSnapshot:
    """Create risk snapshot from risk calculator output"""
    
    if risk_result['status'] != 'success':
        raise ValueError(f"Risk calculation failed: {risk_result.get('error')}")
    
    metrics = risk_result['metrics']
    
    # Calculate risk grade
    risk_score = metrics.get('risk_score', 0)
    if risk_score <= 20:
        risk_grade = 'A'
    elif risk_score <= 40:
        risk_grade = 'B'
    elif risk_score <= 60:
        risk_grade = 'C'
    elif risk_score <= 80:
        risk_grade = 'D'
    else:
        risk_grade = 'F'
    
    snapshot = PortfolioRiskSnapshot(
        user_id=user_id,
        portfolio_id=portfolio_id,
        portfolio_name=portfolio_data.get('name') if portfolio_data else None,
        volatility=metrics.get('volatility'),
        beta=metrics.get('beta'),
        max_drawdown=metrics.get('max_drawdown'),
        var_95=metrics.get('var_95'),
        var_99=metrics.get('var_99'),
        sharpe_ratio=metrics.get('sharpe_ratio'),
        sortino_ratio=metrics.get('sortino_ratio'),
        risk_score=risk_score,
        risk_grade=risk_grade,
        portfolio_value=portfolio_data.get('total_value') if portfolio_data else None,
        num_positions=len(portfolio_data.get('positions', [])) if portfolio_data else None,
        concentration_risk=metrics.get('concentration_risk'),
        tail_dependency=metrics.get('tail_dependency'),
        hurst_exponent=metrics.get('hurst_exponent'),
        regime_state=metrics.get('regime_state'),
        raw_metrics=metrics,
        portfolio_weights=portfolio_data.get('weights') if portfolio_data else None,
        correlation_matrix=metrics.get('correlation_matrix'),
        calculation_method=risk_result.get('method', 'standard'),
        data_quality_score=metrics.get('data_quality', 1.0)
    )
    
    return snapshot


def get_default_risk_thresholds() -> dict:
    """Get default risk threshold configuration"""
    return {
        'volatility_threshold': 0.15,
        'beta_threshold': 0.20,
        'max_drawdown_threshold': 0.10,
        'var_threshold': 0.15,
        'risk_score_threshold': 0.20,
        'max_acceptable_risk_score': 80.0,
        'max_acceptable_volatility': 0.50
    }
