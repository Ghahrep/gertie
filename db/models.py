# db/models.py

from sqlalchemy import (
    Column, Integer, String, Float, ForeignKey, Text, LargeBinary,
    JSON, Boolean, DateTime, Enum as SQLAlchemyEnum, Index
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import enum
from typing import Dict, Any, List, Optional
import uuid
import gzip

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

# --- Enums for the Debate System ---
class DebateStatus(enum.Enum):
    """Debate execution status"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    TIMEOUT = "timeout" # CORRECTED: Changed to lowercase for consistency

class DebateStage(enum.Enum):
    """Stages of debate progression"""
    POSITION_FORMATION = "position_formation"
    CHALLENGE_ROUND = "challenge_round" 
    RESPONSE_ROUND = "response_round"
    CONSENSUS_BUILDING = "consensus_building"
    FINAL_STATEMENTS = "final_statements"
    COMPLETED = "completed"

class MessageType(enum.Enum):
    """Types of messages in debates"""
    POSITION = "position"
    CHALLENGE = "challenge"
    RESPONSE = "response"
    EVIDENCE = "evidence"
    CONSENSUS_CHECK = "consensus_check"
    FINAL_STATEMENT = "final_statement"

class ConsensusType(enum.Enum):
    """Types of consensus reached"""
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    PLURALITY = "plurality"
    NO_CONSENSUS = "no_consensus"

# --- Database Models ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    preferences = Column(JSON, nullable=True)

    portfolios = relationship("Portfolio", back_populates="owner")
    alerts = relationship("Alert", back_populates="owner", cascade="all, delete-orphan")
    csv_jobs = relationship("CSVImportJob", back_populates="user")
    workflow_sessions = relationship("WorkflowSessionDB", back_populates="user")
    risk_events = relationship("RiskChangeEvent", back_populates="user", foreign_keys="RiskChangeEvent.user_id")
    risk_thresholds = relationship("RiskThreshold", back_populates="user", foreign_keys="RiskThreshold.user_id")
    debates = relationship("Debate", back_populates="user")
    created_templates = relationship("DebateTemplate", back_populates="creator")

class Asset(Base):
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    
    asset_type = Column(String, nullable=True, default="stock")
    sector = Column(String, nullable=True)
    
    holdings = relationship("Holding", back_populates="asset")

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    risk_snapshots = relationship("PortfolioRiskSnapshot", back_populates="portfolio")
    risk_trends = relationship("RiskTrend", back_populates="portfolio") 
    risk_events = relationship("RiskChangeEvent", back_populates="portfolio")
    risk_thresholds = relationship("RiskThreshold", back_populates="portfolio")
    debates = relationship("Debate", back_populates="portfolio")

class Holding(Base):
    __tablename__ = "holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    
    shares = Column(Float, nullable=False)
    # FIXED: Removed redundant `quantity` column from the database model.
    purchase_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    
    portfolio = relationship("Portfolio", back_populates="holdings")
    asset = relationship("Asset", back_populates="holdings")
    
    @property
    def quantity(self):
        """Alias for shares for API consistency."""
        return self.shares
    
    @quantity.setter
    def quantity(self, value):
        self.shares = value

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=True)
    
    alert_type = Column(SQLAlchemyEnum(AlertType), nullable=False)
    metric = Column(String, nullable=True)
    ticker = Column(String, nullable=True)
    operator = Column(SQLAlchemyEnum(Operator), nullable=False)

    value = Column(Float, nullable=False)
    
    is_triggered = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    triggered_at = Column(DateTime, nullable=True)

    owner = relationship("User", back_populates="alerts")
    portfolio = relationship("Portfolio")

# --- Debate System Models ---

class Debate(Base):
    __tablename__ = "debates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=True, index=True)
    query = Column(Text, nullable=False)
    description = Column(Text)
    max_rounds = Column(Integer, default=3)
    require_unanimous_consensus = Column(Boolean, default=False)
    include_minority_report = Column(Boolean, default=True)
    urgency_level = Column(String(20), default="medium")
    max_duration_seconds = Column(Integer, default=900)
    status = Column(SQLAlchemyEnum(DebateStatus), default=DebateStatus.PENDING, index=True)
    current_stage = Column(SQLAlchemyEnum(DebateStage), default=DebateStage.POSITION_FORMATION)
    current_round = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    last_activity_at = Column(DateTime, default=datetime.utcnow)
    consensus_type = Column(SQLAlchemyEnum(ConsensusType))
    confidence_score = Column(Float)
    final_recommendation = Column(JSON)
    minority_opinions = Column(JSON)
    implementation_guidance = Column(JSON)
    total_messages = Column(Integer, default=0)
    total_evidence_items = Column(Integer, default=0)
    participant_count = Column(Integer, default=0)
    duration_seconds = Column(Float)
    
    participants = relationship("DebateParticipant", back_populates="debate", cascade="all, delete-orphan")
    messages = relationship("DebateMessage", back_populates="debate", cascade="all, delete-orphan")
    consensus_items = relationship("ConsensusItem", back_populates="debate", cascade="all, delete-orphan")
    analytics = relationship("DebateAnalytics", back_populates="debate", uselist=False, cascade="all, delete-orphan")
    portfolio = relationship("Portfolio", back_populates="debates")
    user = relationship("User", back_populates="debates")
    
    __table_args__ = (
        Index("idx_debates_user_status", "user_id", "status"),
        Index("idx_debates_created_status", "created_at", "status"),
        Index("idx_debates_portfolio_status", "portfolio_id", "status"),
    )

class DebateParticipant(Base):
    __tablename__ = "debate_participants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    debate_id = Column(UUID(as_uuid=True), ForeignKey("debates.id"), nullable=False, index=True)
    
    agent_id = Column(String(100), nullable=False, index=True)
    agent_name = Column(String(200))
    agent_type = Column(String(100))
    agent_specialization = Column(String(200))
    
    role = Column(String(50), default="participant")
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime, default=datetime.utcnow)
    
    messages_sent = Column(Integer, default=0)
    evidence_provided = Column(Integer, default=0)
    challenges_issued = Column(Integer, default=0)
    challenges_received = Column(Integer, default=0)
    responses_given = Column(Integer, default=0)
    avg_confidence_score = Column(Float, default=0.0)
    avg_response_time_seconds = Column(Float, default=0.0)
    
    final_position = Column(JSON)
    consensus_agreement = Column(Boolean)
    minority_position = Column(JSON)
    
    debate = relationship("Debate", back_populates="participants")
    messages = relationship("DebateMessage", back_populates="sender",
                            foreign_keys="DebateMessage.sender_id", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_participants_debate_agent", "debate_id", "agent_id"),
        Index("idx_participants_agent_joined", "agent_id", "joined_at"),
    )

class DebateMessage(Base):
    __tablename__ = "debate_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    debate_id = Column(UUID(as_uuid=True), ForeignKey("debates.id"), nullable=False, index=True)
    sender_id = Column(UUID(as_uuid=True), ForeignKey("debate_participants.id"), nullable=False, index=True)
    
    message_type = Column(SQLAlchemyEnum(MessageType), nullable=False, index=True)
    round_number = Column(Integer, nullable=False, index=True)
    stage = Column(SQLAlchemyEnum(DebateStage), nullable=False)
    sequence_number = Column(Integer, nullable=False)
    
    content = Column(JSON, nullable=False)
    evidence_sources = Column(JSON)
    confidence_score = Column(Float, default=0.8)
    
    parent_message_id = Column(UUID(as_uuid=True), ForeignKey("debate_messages.id"), index=True)
    recipient_id = Column(UUID(as_uuid=True), ForeignKey("debate_participants.id"), index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime)
    response_deadline = Column(DateTime)
    
    sentiment_score = Column(Float)
    complexity_score = Column(Float)
    evidence_quality_score = Column(Float)
    
    responses_received = Column(Integer, default=0)
    challenges_generated = Column(Integer, default=0)
    consensus_impact = Column(Float)
    
    debate = relationship("Debate", back_populates="messages")
    sender = relationship("DebateParticipant", back_populates="messages", 
                          foreign_keys=[sender_id])
    recipient = relationship("DebateParticipant", foreign_keys=[recipient_id])
    parent_message = relationship("DebateMessage", remote_side=[id], backref="child_messages")
    
    __table_args__ = (
        Index("idx_messages_debate_round", "debate_id", "round_number"),
        Index("idx_messages_debate_type", "debate_id", "message_type"),
        Index("idx_messages_sender_created", "sender_id", "created_at"),
        Index("idx_messages_stage_sequence", "stage", "sequence_number"),
    )

class ConsensusItem(Base):
    __tablename__ = "consensus_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    debate_id = Column(UUID(as_uuid=True), ForeignKey("debates.id"), nullable=False, index=True)
    
    topic = Column(String(500), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    
    support_count = Column(Integer, default=0)
    oppose_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    total_participants = Column(Integer, nullable=False)
    
    agreement_percentage = Column(Float)
    consensus_strength = Column(String(50))
    
    supporting_evidence = Column(JSON)
    dissenting_evidence = Column(JSON)
    
    first_mentioned_at = Column(DateTime, default=datetime.utcnow)
    consensus_reached_at = Column(DateTime)
    final_vote_at = Column(DateTime)
    
    implementation_priority = Column(String(20))
    implementation_complexity = Column(String(20))
    expected_impact = Column(Text)
    
    debate = relationship("Debate", back_populates="consensus_items")
    votes = relationship("ConsensusVote", back_populates="consensus_item", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_consensus_debate_category", "debate_id", "category"),
        Index("idx_consensus_agreement", "agreement_percentage"),
    )

class ConsensusVote(Base):
    __tablename__ = "consensus_votes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    consensus_item_id = Column(UUID(as_uuid=True), ForeignKey("consensus_items.id"), nullable=False, index=True)
    participant_id = Column(UUID(as_uuid=True), ForeignKey("debate_participants.id"), nullable=False, index=True)
    
    vote = Column(String(20), nullable=False)
    confidence = Column(Float, default=0.8)
    reasoning = Column(Text)
    
    voted_at = Column(DateTime, default=datetime.utcnow)
    changed_vote = Column(Boolean, default=False)
    final_vote = Column(Boolean, default=True)
    
    supporting_evidence = Column(JSON)
    concerns = Column(JSON)
    
    consensus_item = relationship("ConsensusItem", back_populates="votes")
    participant = relationship("DebateParticipant")
    
    __table_args__ = (
        Index("idx_votes_item_participant", "consensus_item_id", "participant_id"),
    )

class DebateAnalytics(Base):
    __tablename__ = "debate_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    debate_id = Column(UUID(as_uuid=True), ForeignKey("debates.id"), nullable=False, unique=True, index=True)
    
    avg_messages_per_participant = Column(Float)
    avg_response_time_seconds = Column(Float)
    participation_balance_score = Column(Float)
    
    avg_evidence_quality = Column(Float)
    avg_confidence_score = Column(Float)
    argument_coherence_score = Column(Float)
    
    time_to_consensus_seconds = Column(Float)
    consensus_stability_score = Column(Float)
    minority_accommodation_score = Column(Float)
    
    messages_per_consensus_point = Column(Float)
    debate_efficiency_score = Column(Float)
    rounds_to_completion = Column(Integer)
    
    avg_sentiment_score = Column(Float)
    sentiment_volatility = Column(Float)
    civility_score = Column(Float)
    
    topics_covered = Column(JSON)
    topic_depth_scores = Column(JSON)
    evidence_diversity_score = Column(Float)
    
    agent_performance_scores = Column(JSON)
    agent_synergy_scores = Column(JSON)
    
    implementation_clarity_score = Column(Float)
    recommendation_specificity = Column(Float)
    actionability_score = Column(Float)
    
    debate = relationship("Debate", back_populates="analytics")

class DebateTemplate(Base):
    __tablename__ = "debate_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    category = Column(String(100), index=True)
    
    query_template = Column(Text, nullable=False)
    recommended_agents = Column(JSON)
    default_rounds = Column(Integer, default=3)
    urgency_level = Column(String(20), default="medium")
    
    usage_count = Column(Integer, default=0)
    avg_success_rate = Column(Float, default=0.0)
    avg_user_rating = Column(Float, default=0.0)
    
    required_parameters = Column(JSON)
    optional_parameters = Column(JSON)
    
    validation_rules = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))
    
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True)
    
    creator = relationship("User", back_populates="created_templates")
    
    __table_args__ = (
        Index("idx_templates_category_active", "category", "is_active"),
        Index("idx_templates_usage_rating", "usage_count", "avg_user_rating"),
    )

class AgentPerformanceHistory(Base):
    __tablename__ = "agent_performance_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    agent_id = Column(String(100), nullable=False, index=True)
    agent_name = Column(String(200))
    agent_version = Column(String(50))
    
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    
    debates_participated = Column(Integer, default=0)
    total_messages_sent = Column(Integer, default=0)
    avg_messages_per_debate = Column(Float, default=0.0)
    
    avg_confidence_score = Column(Float, default=0.0)
    avg_evidence_quality = Column(Float, default=0.0)
    avg_response_time = Column(Float, default=0.0)
    
    consensus_contributions = Column(Integer, default=0)
    successful_challenges = Column(Integer, default=0)
    positions_defended = Column(Integer, default=0)
    minority_positions_held = Column(Integer, default=0)
    
    accuracy_score = Column(Float, default=0.0)
    user_satisfaction_score = Column(Float, default=0.0)
    implementation_success_rate = Column(Float, default=0.0)
    
    specialization_areas = Column(JSON)
    weakness_areas = Column(JSON)
    
    synergy_scores = Column(JSON)
    avg_civility_score = Column(Float, default=0.0)
    
    __table_args__ = (
        Index("idx_performance_agent_period", "agent_id", "period_start"),
        Index("idx_performance_accuracy", "accuracy_score"),
    )

# --- Existing Risk Management Models ---

class PortfolioRiskSnapshot(Base):
    __tablename__ = "portfolio_risk_snapshots"
    # FIXED: Removed {'extend_existing': True} as it's not a best practice for production
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    snapshot_date = Column(DateTime, nullable=False, index=True)
    
    compressed_metrics = Column(LargeBinary, nullable=True)
    compression_ratio = Column(Float, nullable=True)
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    volatility = Column(Float, nullable=False)
    beta = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    var_95 = Column(Float, nullable=False)
    var_99 = Column(Float, nullable=False)
    cvar_95 = Column(Float, nullable=False)
    cvar_99 = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float, nullable=False)
    calmar_ratio = Column(Float, nullable=False)
    hurst_exponent = Column(Float, nullable=True)
    dfa_alpha = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=False)
    is_threshold_breach = Column(Boolean, default=False, nullable=False, index=True)
    sentiment_index = Column(Integer, nullable=False)
    regime_volatility = Column(Float, nullable=True)
    
    metrics_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    data_quality_score = Column(Float, default=1.0)
    calculation_time_ms = Column(Float, nullable=True)
    calculation_method = Column(String, default="comprehensive", nullable=False)
    data_window_days = Column(Integer, default=252, nullable=False)
    risk_factors = Column(JSON, nullable=True)
    market_conditions = Column(JSON, nullable=True)
    
    portfolio = relationship("Portfolio", back_populates="risk_snapshots")
    user = relationship("User")

    @classmethod
    def create_from_risk_metrics(
        cls, 
        portfolio_id: int, 
        user_id: int,           # FIXED: Added user_id
        risk_score: float,        # FIXED: Added risk_score
        sentiment_index: int,     # FIXED: Added sentiment_index
        risk_metrics, 
        compression_level: int = 6
    ):
        """Create snapshot from risk metrics with compression"""
        import json
        import gzip
        
        metrics_dict = {
            'annualized_volatility': getattr(risk_metrics, 'annualized_volatility', 0),
            'var_95': getattr(risk_metrics, 'var_95', 0),
            'cvar_95': getattr(risk_metrics, 'cvar_95', 0),
            'max_drawdown': getattr(risk_metrics, 'max_drawdown', 0),
            'sharpe_ratio': getattr(risk_metrics, 'sharpe_ratio', 0),
            'sortino_ratio': getattr(risk_metrics, 'sortino_ratio', 0),
            'skewness': getattr(risk_metrics, 'skewness', 0),
            'kurtosis': getattr(risk_metrics, 'kurtosis', 0),
            'calculation_date': getattr(risk_metrics, 'calculation_date', datetime.utcnow()).isoformat(),
        }
        
        json_data = json.dumps(metrics_dict, default=str)
        compressed_data = gzip.compress(json_data.encode('utf-8'), compresslevel=compression_level)
        
        return cls(
            portfolio_id=portfolio_id,
            snapshot_date=getattr(risk_metrics, 'calculation_date', datetime.utcnow()),
            compressed_metrics=compressed_data,
            compression_ratio=len(compressed_data) / len(json_data.encode('utf-8')),
            metrics_summary={
                'volatility': getattr(risk_metrics, 'annualized_volatility', 0),
                'var_95': getattr(risk_metrics, 'var_95', 0),
                'sharpe_ratio': getattr(risk_metrics, 'sharpe_ratio', 0),
                'max_drawdown': getattr(risk_metrics, 'max_drawdown', 0)
            },
            user_id=user_id,                      # FIXED: Use parameter
            volatility=getattr(risk_metrics, 'annualized_volatility', 0),
            beta=getattr(risk_metrics, 'beta', 1.0),
            max_drawdown=getattr(risk_metrics, 'max_drawdown', 0),
            var_95=getattr(risk_metrics, 'var_95', 0),
            var_99=getattr(risk_metrics, 'var_99', 0),
            cvar_95=getattr(risk_metrics, 'cvar_95', 0),
            cvar_99=getattr(risk_metrics, 'cvar_99', 0),
            sharpe_ratio=getattr(risk_metrics, 'sharpe_ratio', 0),
            sortino_ratio=getattr(risk_metrics, 'sortino_ratio', 0),
            calmar_ratio=getattr(risk_metrics, 'calmar_ratio', 0),
            risk_score=risk_score,                # FIXED: Use parameter
            sentiment_index=sentiment_index       # FIXED: Use parameter
        )

class RiskChangeEvent(Base):
    __tablename__ = "risk_change_events"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    current_snapshot_id = Column(Integer, ForeignKey("portfolio_risk_snapshots.id"), nullable=False)
    previous_snapshot_id = Column(Integer, ForeignKey("portfolio_risk_snapshots.id"), nullable=True)
    
    risk_direction = Column(String, nullable=False)
    risk_magnitude_pct = Column(Float, nullable=False)
    threshold_breached = Column(Boolean, default=False, nullable=False)
    
    risk_changes = Column(JSON, nullable=False)
    significant_changes = Column(JSON, nullable=False)
    
    workflow_triggered = Column(Boolean, default=False, nullable=False)
    workflow_session_id = Column(String, nullable=True)
    alerts_generated = Column(JSON, nullable=True)
    
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    threshold_pct = Column(Float, default=15.0, nullable=False)
    
    portfolio = relationship("Portfolio")
    user = relationship("User")
    current_snapshot = relationship("PortfolioRiskSnapshot", foreign_keys=[current_snapshot_id])
    previous_snapshot = relationship("PortfolioRiskSnapshot", foreign_keys=[previous_snapshot_id])

class ProactiveAlert(Base):
    __tablename__ = "proactive_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    alert_type = Column(String, nullable=False)
    priority = Column(String, nullable=False)
    
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    
    risk_change_event_id = Column(Integer, ForeignKey("risk_change_events.id"), nullable=True)
    triggered_risk_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    sent_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    is_sent = Column(Boolean, default=False, nullable=False)
    is_acknowledged = Column(Boolean, default=False, nullable=False)
    is_resolved = Column(Boolean, default=False, nullable=False)
    
    delivery_channels = Column(JSON, nullable=True)
    delivery_status = Column(JSON, nullable=True)
    
    portfolio = relationship("Portfolio")
    user = relationship("User")
    risk_change_event = relationship("RiskChangeEvent")

class RiskThresholdConfig(Base):
    __tablename__ = "risk_threshold_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=True)
    
    volatility_threshold = Column(Float, default=0.15)
    beta_threshold = Column(Float, default=0.20)
    max_drawdown_threshold = Column(Float, default=0.10)
    var_threshold = Column(Float, default=0.15)
    risk_score_threshold = Column(Float, default=0.20)
    
    max_acceptable_risk_score = Column(Float, default=80.0)
    max_acceptable_volatility = Column(Float, default=0.50)
    
    monitoring_enabled = Column(Boolean, default=True)
    alert_frequency = Column(String(20), default='immediate')
    notification_methods = Column(JSON, default=lambda: ['websocket', 'email'])
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User")
    portfolio = relationship("Portfolio")
    
    __table_args__ = (
        Index('idx_user_portfolio_config', 'user_id', 'portfolio_id', unique=True),
    )

class RiskAlertLog(Base):
    __tablename__ = "risk_alert_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    snapshot_id = Column(String(36))
    
    alert_type = Column(String(30), nullable=False)
    alert_severity = Column(String(10), default='medium')
    alert_message = Column(Text, nullable=False)
    
    triggered_metrics = Column(JSON)
    risk_change_summary = Column(JSON)
    
    workflow_id = Column(String(36))
    workflow_status = Column(String(20))
    
    notification_methods = Column(JSON)
    notification_status = Column(JSON)
    user_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    user = relationship("User")
    portfolio = relationship("Portfolio")
    
    __table_args__ = (
        Index('idx_alert_user_portfolio', 'user_id', 'portfolio_id'),
        Index('idx_alert_workflow', 'workflow_id'),
    )

# --- A/B Testing System Models (NEW) ---

class ABTest(Base):
    __tablename__ = "ab_tests"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String(20), default="active", nullable=False, index=True)
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)
    
    variants = relationship("ABTestVariant", back_populates="test", cascade="all, delete-orphan")

class ABTestVariant(Base):
    __tablename__ = "ab_test_variants"
    
    id = Column(Integer, primary_key=True)
    test_id = Column(String, ForeignKey("ab_tests.id"), nullable=False, index=True)
    variant_id = Column(String(50), nullable=False)
    name = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    
    test = relationship("ABTest", back_populates="variants")

class ABTestAssignment(Base):
    __tablename__ = "ab_test_assignments"

    id = Column(Integer, primary_key=True)
    test_id = Column(String, ForeignKey("ab_tests.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    variant_id = Column(String(50), nullable=False)
    
    __table_args__ = (Index("idx_ab_test_user_assignment", "test_id", "user_id", unique=True),)


# --- Workflow Models ---

class CSVImportJob(Base):
    __tablename__ = "csv_import_jobs"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    status = Column(String)
    total_rows = Column(Integer)
    processed_rows = Column(Integer) 
    success_count = Column(Integer)
    error_count = Column(Integer)
    errors = Column(JSON)
    warnings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    user = relationship("User", back_populates="csv_jobs")

class WorkflowSessionDB(Base):
    __tablename__ = "workflow_sessions"
    
    session_id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    query = Column(Text, nullable=False)
    state = Column(String(50), nullable=False, default="awaiting_strategy")
    workflow_type = Column(String(20), nullable=False)
    complexity_score = Column(Float, default=0.0)
    
    mcp_job_id = Column(String(36), nullable=True)
    execution_mode = Column(String(20), nullable=False, default="direct")
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    current_step = Column(Integer, default=1)
    total_steps = Column(Integer, default=4)
    steps_completed = Column(JSON, default=lambda: [])
    
    strategy_result = Column(JSON, nullable=True)
    screening_result = Column(JSON, nullable=True)
    analysis_result = Column(JSON, nullable=True)
    final_synthesis = Column(JSON, nullable=True)
    
    errors = Column(JSON, default=lambda: [])
    retry_count = Column(Integer, default=0)
    
    confidence_score = Column(Float, nullable=True)
    user_rating = Column(Integer, nullable=True)
    
    user = relationship("User", back_populates="workflow_sessions")
    workflow_steps = relationship("WorkflowStepDB", back_populates="session", cascade="all, delete-orphan")
    mcp_job_logs = relationship("MCPJobLog", back_populates="workflow_session")
    
    __table_args__ = (
        Index('idx_workflow_sessions_user_id', 'user_id'),
        Index('idx_workflow_sessions_state', 'state'), 
        Index('idx_workflow_sessions_created_at', 'created_at'),
        Index('idx_workflow_sessions_workflow_type', 'workflow_type'),
        Index('idx_workflow_sessions_execution_mode', 'execution_mode'),
    )

class WorkflowStepDB(Base):
    __tablename__ = "workflow_steps"
    
    step_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey('workflow_sessions.session_id'), nullable=False)
    
    step_number = Column(Integer, nullable=False)
    step_name = Column(String(50), nullable=False)
    agent_id = Column(String(100), nullable=False)
    capability = Column(String(100), nullable=False)
    
    status = Column(String(20), nullable=False, default="pending")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    
    input_data = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    confidence_score = Column(Float, nullable=True)
    success = Column(Boolean, default=False)
    
    dependencies = Column(JSON, default=lambda: [])
    
    mcp_job_step_id = Column(String(36), nullable=True)
    
    session = relationship("WorkflowSessionDB", back_populates="workflow_steps")
    
    __table_args__ = (
        Index('idx_workflow_steps_session_id', 'session_id'),
        Index('idx_workflow_steps_status', 'status'),
        Index('idx_workflow_steps_agent_id', 'agent_id'),
        Index('idx_workflow_steps_step_number', 'step_number'),
    )

class AgentPerformanceMetrics(Base):
    __tablename__ = "agent_performance_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    agent_id = Column(String(100), nullable=False)
    capability = Column(String(100), nullable=False)
    
    execution_time_ms = Column(Integer, nullable=False)
    success = Column(Boolean, nullable=False)
    confidence_score = Column(Float, nullable=True)
    
    user_rating = Column(Integer, nullable=True)
    user_feedback = Column(Text, nullable=True)
    
    query_complexity = Column(Float, nullable=True)
    execution_mode = Column(String(20), nullable=False)
    workflow_type = Column(String(20), nullable=True)
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    session_id = Column(String(36), ForeignKey('workflow_sessions.session_id'), nullable=True)
    
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    user = relationship("User")
    workflow_session = relationship("WorkflowSessionDB")
    
    __table_args__ = (
        Index('idx_agent_performance_agent_id', 'agent_id'),
        Index('idx_agent_performance_capability', 'capability'),
        Index('idx_agent_performance_execution_mode', 'execution_mode'),
        Index('idx_agent_performance_created_at', 'created_at'),
        Index('idx_agent_performance_composite', 'agent_id', 'capability', 'execution_mode'),
    )

class MCPJobLog(Base):
    __tablename__ = "mcp_job_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    job_id = Column(String(36), nullable=False, index=True)
    session_id = Column(String(36), ForeignKey('workflow_sessions.session_id'), nullable=True)
    
    job_request = Column(JSON, nullable=False)
    job_response = Column(JSON, nullable=True)
    
    status = Column(String(20), nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    total_execution_time_ms = Column(Integer, nullable=True)
    queue_wait_time_ms = Column(Integer, nullable=True)
    network_latency_ms = Column(Integer, nullable=True)
    
    agents_involved = Column(JSON, default=lambda: [])
    workflow_steps_count = Column(Integer, nullable=True)
    
    error_details = Column(JSON, nullable=True)
    retry_count = Column(Integer, default=0)
    retry_reason = Column(String(255), nullable=True)
    
    mcp_server_load = Column(Float, nullable=True)
    priority_level = Column(Integer, nullable=True)
    
    workflow_session = relationship("WorkflowSessionDB", back_populates="mcp_job_logs")
    
    __table_args__ = (
        Index('idx_mcp_job_logs_job_id', 'job_id'),
        Index('idx_mcp_job_logs_status', 'status'),
        Index('idx_mcp_job_logs_submitted_at', 'submitted_at'),
        Index('idx_mcp_job_logs_session_id', 'session_id'),
    )

class SystemHealthMetrics(Base):
    __tablename__ = "system_health_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    active_workflows = Column(Integer, nullable=False, default=0)
    completed_workflows_last_hour = Column(Integer, nullable=False, default=0)
    failed_workflows_last_hour = Column(Integer, nullable=False, default=0)
    average_workflow_duration_seconds = Column(Float, nullable=True)
    
    mcp_server_available = Column(Boolean, nullable=False)
    mcp_active_jobs = Column(Integer, nullable=True)
    mcp_average_response_time_ms = Column(Float, nullable=True)
    mcp_error_rate_percent = Column(Float, nullable=True)
    
    fastest_agent = Column(String(100), nullable=True)
    slowest_agent = Column(String(100), nullable=True)
    most_reliable_agent = Column(String(100), nullable=True)
    
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_percent = Column(Float, nullable=True)
    disk_usage_percent = Column(Float, nullable=True)
    
    avg_query_time_ms = Column(Float, nullable=True)
    database_connections_active = Column(Integer, nullable=True)
    
    __table_args__ = (
        Index('idx_system_health_recorded_at', 'recorded_at'),
    )

class RiskTrend(Base):
    __tablename__ = "risk_trends"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    metric_name = Column(String(50), nullable=False, index=True)
    
    current_value = Column(Float, nullable=False)
    trend_direction = Column(String(20), nullable=False)
    trend_strength = Column(Float, nullable=False)
    
    forecast_1d = Column(Float, nullable=True)
    forecast_7d = Column(Float, nullable=True)
    forecast_30d = Column(Float, nullable=True)
    
    confidence_score = Column(Float, nullable=False)
    r_squared = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    
    analysis_period_days = Column(Integer, default=90)
    data_points_used = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="risk_trends")

class RiskThreshold(Base):
    __tablename__ = "risk_thresholds"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    metric_name = Column(String(50), nullable=False, index=True)
    warning_threshold = Column(Float, nullable=False)
    critical_threshold = Column(Float, nullable=False)
    emergency_threshold = Column(Float, nullable=True)
    
    direction = Column(String(20), default="increase", nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    lookback_periods = Column(Integer, default=5)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    portfolio = relationship("Portfolio", back_populates="risk_thresholds")
    user = relationship("User", back_populates="risk_thresholds", foreign_keys=[user_id])
    created_by_user = relationship("User", foreign_keys=[created_by])

class PriceDataCache(Base):
    __tablename__ = "price_data_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    price = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    volume = Column(Float, nullable=True)
    change = Column(Float, nullable=True)
    change_percent = Column(Float, nullable=True)
    
    provider = Column(String(30), nullable=False)
    data_quality_score = Column(Float, default=1.0)
    
    market_timestamp = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    
    bid_price = Column(Float, nullable=True)
    ask_price = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)

# --- Utility Functions ---

def create_risk_snapshot_from_calculator(portfolio_id: int, risk_metrics) -> PortfolioRiskSnapshot:
    return PortfolioRiskSnapshot.create_from_risk_metrics(portfolio_id, risk_metrics)

def create_debate_from_template(template: DebateTemplate, user_id: int, 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
    query = template.query_template
    for param_name, param_value in parameters.items():
        placeholder = f"{{{param_name}}}"
        query = query.replace(placeholder, str(param_value))
    
    return {
        "user_id": user_id,
        "query": query,
        "description": template.description,
        "max_rounds": template.default_rounds,
        "urgency_level": template.urgency_level,
        "recommended_agents": template.recommended_agents,
        "template_id": str(template.id)
    }

def calculate_debate_quality_score(debate: Debate) -> float:
    if not debate.analytics:
        return 0.0
    
    analytics = debate.analytics
    
    weights = {
        "participation_balance": 0.2, "evidence_quality": 0.2,
        "consensus_stability": 0.2, "efficiency": 0.15,
        "civility": 0.15, "actionability": 0.1
    }
    
    scores = {
        "participation_balance": analytics.participation_balance_score or 0.5,
        "evidence_quality": analytics.avg_evidence_quality or 0.5,
        "consensus_stability": analytics.consensus_stability_score or 0.5,
        "efficiency": analytics.debate_efficiency_score or 0.5,
        "civility": analytics.civility_score or 0.5,
        "actionability": analytics.actionability_score or 0.5
    }
    
    quality_score = sum(weights[factor] * scores[factor] for factor in weights)
    
    return min(max(quality_score, 0.0), 1.0)

def get_agent_specialization_match_score(agent_id: str, query: str) -> float:
    specialization_keywords = {
        "quantitative_analyst": ["risk", "analysis", "statistical", "var", "volatility"],
        "tax_strategist": ["tax", "optimization", "harvest", "deduction"],
        "market_intelligence": ["market", "timing", "sentiment", "economic"],
        "options_analyst": ["options", "derivatives", "hedging", "volatility"]
    }
    
    keywords = specialization_keywords.get(agent_id, [])
    query_lower = query.lower()
    
    matches = sum(1 for keyword in keywords if keyword in query_lower)
    return min(matches / max(len(keywords), 1), 1.0)

# --- Export all models and key functions ---
__all__ = [
    # Core models
    "User", "Asset", "Portfolio", "Holding", "Alert",
    # Debate system models
    "Debate", "DebateParticipant", "DebateMessage", "ConsensusItem", "ConsensusVote",
    "DebateAnalytics", "DebateTemplate", "AgentPerformanceHistory",
    # Debate enums
    "DebateStatus", "DebateStage", "MessageType", "ConsensusType",
    # Risk management models
    "PortfolioRiskSnapshot", "RiskChangeEvent", "ProactiveAlert", "RiskThresholdConfig",
    "RiskAlertLog", "RiskTrend", "RiskThreshold", "PriceDataCache",
    # A/B Testing models (NEW)
    "ABTest", "ABTestVariant", "ABTestAssignment",
    # Workflow models
    "CSVImportJob", "WorkflowSessionDB", "WorkflowStepDB", "AgentPerformanceMetrics",
    "MCPJobLog", "SystemHealthMetrics",
    # Utility functions
    "create_risk_snapshot_from_calculator", "create_debate_from_template",
    "calculate_debate_quality_score", "get_agent_specialization_match_score"
]
