# api/schemas.py

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
import enum
from db.models import AlertType, Operator # Import the enums from your models

# --- Enums for API Routes (B4 Fix) ---
# These are used for query parameter validation in FastAPI routes
class AlertTypeEnum(str, enum.Enum):
    PRICE = "price"
    PORTFOLIO_VALUE = "portfolio_value"
    RISK_METRIC = "risk_metric"

class AlertStatusEnum(str, enum.Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    PAUSED = "paused"

# --- Token and User Schemas ---

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[EmailStr] = None

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

# --- User Preferences Schemas (B2) ---

class DisplayPreferences(BaseModel):
    theme: str = "dark"
    timezone: str = "America/New_York"

class NotificationPreferences(BaseModel):
    email_alerts: bool = True
    daily_summary: bool = False

class UserPreferences(BaseModel):
    display: DisplayPreferences = Field(default_factory=DisplayPreferences)
    notifications: NotificationPreferences = Field(default_factory=NotificationPreferences)

# NEW: Schemas for updating preferences with optional fields
class DisplayPreferencesUpdate(BaseModel):
    theme: Optional[str] = None
    timezone: Optional[str] = None

class NotificationPreferencesUpdate(BaseModel):
    email_alerts: Optional[bool] = None
    daily_summary: Optional[bool] = None

class UserPreferencesUpdate(BaseModel):
    display: Optional[DisplayPreferencesUpdate] = None
    notifications: Optional[NotificationPreferencesUpdate] = None


class User(UserBase):
    id: int
    preferences: Optional[UserPreferences] = None

    class Config:
        from_attributes = True

# --- Holding and Portfolio Schemas ---

class HoldingBase(BaseModel):
    ticker: str
    shares: float

class HoldingCreate(HoldingBase):
    pass

class Holding(HoldingBase):
    id: int
    asset_id: int
    purchase_price: Optional[float] = None

    class Config:
        from_attributes = True

class PortfolioBase(BaseModel):
    name: str

class PortfolioCreate(PortfolioBase):
    pass

class Portfolio(PortfolioBase):
    id: int
    user_id: int
    holdings: List[Holding] = []

    class Config:
        from_attributes = True

# --- Alert Schemas (B4) ---

class AlertBase(BaseModel):
    alert_type: AlertType
    operator: Operator
    value: float
    portfolio_id: Optional[int] = None
    metric: Optional[str] = None
    ticker: Optional[str] = None
    is_active: bool = True

class AlertCreate(AlertBase):
    pass

class AlertUpdate(BaseModel):
    # Schema for updating an existing alert, all fields are optional
    alert_type: Optional[AlertType] = None
    operator: Optional[Operator] = None
    value: Optional[float] = None
    metric: Optional[str] = None
    ticker: Optional[str] = None
    is_active: Optional[bool] = None

class Alert(AlertBase):
    id: int
    user_id: int
    is_triggered: bool
    created_at: datetime
    triggered_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# --- Analytics Schemas (B1) ---

class RiskMetrics(BaseModel):
    # This is a simplified example; you'd flesh this out to match your agent's output
    sharpe_ratio: float
    annualized_volatility_pct: float
    max_drawdown_pct: float

class AnalyticsReport(BaseModel):
    success: bool
    summary: str
    data: Optional[RiskMetrics] = None
    agent_used: str

# --- AI Analysis Schemas (Sprint 1 - NEW) ---

class AIAnalysisRequest(BaseModel):
    """Schema for AI analysis requests"""
    query: str = Field(..., description="Natural language query for AI analysis")
    analysis_depth: str = Field(
        default="standard", 
        description="Depth of analysis: 'quick', 'standard', or 'comprehensive'"
    )
    include_recommendations: bool = Field(
        default=True, 
        description="Whether to include actionable recommendations"
    )
    focus_areas: Optional[List[str]] = Field(
        None, 
        description="Specific areas to focus on: 'risk', 'tax', 'performance', 'allocation'"
    )
    custom_parameters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Custom parameters for specialized analysis"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Analyze the risk profile of my portfolio and suggest optimizations",
                "analysis_depth": "comprehensive",
                "include_recommendations": True,
                "focus_areas": ["risk", "allocation"],
                "custom_parameters": {
                    "risk_tolerance": "moderate",
                    "time_horizon": "long_term"
                }
            }
        }

class AutonomousAnalysisResponse(BaseModel):
    """Schema for autonomous analysis responses"""
    job_id: str = Field(..., description="Unique identifier for the analysis job")
    status: str = Field(..., description="Current status of the analysis")
    progress: float = Field(..., ge=0, le=100, description="Analysis progress percentage")
    workflow_used: List[str] = Field(..., description="Workflow steps that were executed")
    agents_consulted: List[str] = Field(..., description="AI agents that participated in the analysis")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence in the analysis")
    result: Optional[Dict[str, Any]] = Field(None, description="Analysis results (when completed)")
    message: str = Field(..., description="Human-readable status message")
    completed_at: Optional[str] = Field(None, description="Completion timestamp (ISO format)")
    recommendations: Optional[List[Dict[str, Any]]] = Field(None, description="Actionable recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "analysis_job_12345",
                "status": "completed",
                "progress": 100.0,
                "workflow_used": ["data_collection", "risk_analysis", "optimization", "synthesis"],
                "agents_consulted": ["QuantitativeAnalyst", "RiskManager", "TaxOptimizer"],
                "confidence_score": 0.92,
                "result": {
                    "risk_metrics": {
                        "var_95": 0.024,
                        "max_drawdown": 0.187,
                        "sharpe_ratio": 1.34
                    },
                    "optimization_suggestions": {
                        "reduce_concentration_risk": True,
                        "increase_diversification": True
                    }
                },
                "message": "Analysis completed successfully",
                "completed_at": "2024-01-15T14:32:15Z",
                "recommendations": [
                    {
                        "type": "risk_management",
                        "priority": "high",
                        "description": "Reduce single-stock concentration by selling 25% of AAPL position"
                    }
                ]
            }
        }

class TaxOptimizationRequest(BaseModel):
    """Schema for tax optimization requests"""
    tax_year: int = Field(..., description="Tax year for optimization")
    estimated_tax_bracket: float = Field(..., ge=0, le=1, description="Estimated marginal tax rate")
    state_tax_rate: Optional[float] = Field(None, ge=0, le=1, description="State tax rate")
    account_types: Optional[List[str]] = Field(
        None, 
        description="Account types to consider: 'taxable', 'traditional_401k', 'roth_ira', etc."
    )
    exclude_holdings: Optional[List[str]] = Field(
        None, 
        description="Holdings to exclude from tax optimization"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "tax_year": 2024,
                "estimated_tax_bracket": 0.24,
                "state_tax_rate": 0.05,
                "account_types": ["taxable", "traditional_401k", "roth_ira"],
                "exclude_holdings": ["AAPL"]  # Don't suggest selling AAPL
            }
        }

class TaxOptimizationResponse(BaseModel):
    """Schema for tax optimization responses"""
    potential_tax_savings: float = Field(..., description="Estimated annual tax savings")
    recommended_actions: List[Dict[str, Any]] = Field(..., description="Recommended tax optimization actions")
    harvesting_opportunities: List[Dict[str, Any]] = Field(..., description="Tax-loss harvesting opportunities")
    asset_location_suggestions: Dict[str, Any] = Field(..., description="Optimal asset location recommendations")
    year_end_planning: Dict[str, Any] = Field(..., description="Year-end tax planning strategies")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "potential_tax_savings": 3420.50,
                "recommended_actions": [
                    {
                        "action": "harvest_losses",
                        "symbol": "TSLA",
                        "estimated_savings": 1200.00,
                        "urgency": "high"
                    }
                ],
                "harvesting_opportunities": [
                    {
                        "symbol": "NVDA",
                        "unrealized_loss": -2220.00,
                        "tax_savings": 532.80
                    }
                ],
                "asset_location_suggestions": {
                    "move_to_401k": ["BND", "VTEB"],
                    "move_to_roth": ["QQQ", "VTI"]
                },
                "year_end_planning": {
                    "charitable_giving": 5000.00,
                    "retirement_contributions": 6500.00
                },
                "confidence_score": 0.89
            }
        }

class RebalancingRequest(BaseModel):
    """Schema for portfolio rebalancing requests"""
    target_allocation: Optional[Dict[str, float]] = Field(
        None, 
        description="Desired target allocation percentages"
    )
    rebalancing_threshold: float = Field(
        default=0.05, 
        ge=0.01, 
        le=0.20, 
        description="Threshold for triggering rebalancing"
    )
    minimize_taxes: bool = Field(default=True, description="Whether to minimize tax impact")
    account_constraints: Optional[Dict[str, Any]] = Field(
        None, 
        description="Account-specific constraints"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_allocation": {
                    "stocks": 0.70,
                    "bonds": 0.25,
                    "cash": 0.05
                },
                "rebalancing_threshold": 0.05,
                "minimize_taxes": True,
                "account_constraints": {
                    "401k_only_funds": ["VTIAX", "VBTLX"]
                }
            }
        }

class RebalancingResponse(BaseModel):
    """Schema for portfolio rebalancing responses"""
    rebalancing_needed: bool = Field(..., description="Whether rebalancing is recommended")
    current_allocation: Dict[str, float] = Field(..., description="Current portfolio allocation")
    target_allocation: Dict[str, float] = Field(..., description="Target portfolio allocation")
    recommended_trades: List[Dict[str, Any]] = Field(..., description="Recommended trades to execute")
    estimated_cost: float = Field(..., description="Estimated trading costs")
    tax_impact: Optional[float] = Field(None, description="Estimated tax impact")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rebalancing_needed": True,
                "current_allocation": {
                    "stocks": 0.75,
                    "bonds": 0.20,
                    "cash": 0.05
                },
                "target_allocation": {
                    "stocks": 0.70,
                    "bonds": 0.25,
                    "cash": 0.05
                },
                "recommended_trades": [
                    {
                        "symbol": "VTI",
                        "action": "sell",
                        "shares": 50,
                        "estimated_value": 12750.00
                    },
                    {
                        "symbol": "BND",
                        "action": "buy",
                        "shares": 150,
                        "estimated_value": 12750.00
                    }
                ],
                "estimated_cost": 2.00,
                "tax_impact": 0.00,
                "confidence_score": 0.94
            }
        }

class MarketIntelligenceRequest(BaseModel):
    """Schema for market intelligence requests"""
    focus_sectors: Optional[List[str]] = Field(None, description="Specific sectors to analyze")
    time_horizon: str = Field(default="medium", description="Analysis time horizon: 'short', 'medium', 'long'")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_technical: bool = Field(default=True, description="Include technical analysis")
    
class MarketIntelligenceResponse(BaseModel):
    """Schema for market intelligence responses"""
    market_outlook: Dict[str, Any] = Field(..., description="Overall market outlook")
    sector_analysis: Dict[str, Any] = Field(..., description="Sector-specific analysis")
    risk_factors: List[Dict[str, Any]] = Field(..., description="Identified risk factors")
    opportunities: List[Dict[str, Any]] = Field(..., description="Identified opportunities")
    sentiment_score: float = Field(..., ge=-1, le=1, description="Overall market sentiment")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in analysis")

class AnalysisJobSummary(BaseModel):
    """Schema for analysis job summaries"""
    job_id: str = Field(..., description="Job identifier")
    query: str = Field(..., description="Original query")
    status: str = Field(..., description="Job status")
    confidence_score: Optional[float] = Field(None, description="Analysis confidence score")
    started_at: str = Field(..., description="Job start time")
    completed_at: Optional[str] = Field(None, description="Job completion time")
    agents_used: List[str] = Field(..., description="Agents that participated")


class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None

class WorkflowResponse(BaseModel):
    session_id: str
    status: str  # "success", "error", "processing"
    execution_mode: Optional[str] = None
    execution_time_ms: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    reasoning: Optional[str] = None

class WorkflowStatus(BaseModel):
    session_id: str
    state: str
    progress: Optional[float] = None
    current_step: Optional[int] = None
    steps_completed: Optional[List[str]] = None
    estimated_completion_time: Optional[int] = None

class PerformanceMetrics(BaseModel):
    total_executions: int
    success_rate: float
    average_execution_time_ms: float
    average_confidence_score: float
    period_days: int


class BaseResponse(BaseModel):
    """Base response schema"""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)