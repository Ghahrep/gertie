# db/models.py

from sqlalchemy import (
    Column, Integer, String, Float, ForeignKey, 
    JSON, Boolean, DateTime, Enum as SQLAlchemyEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime
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