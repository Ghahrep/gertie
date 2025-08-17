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
