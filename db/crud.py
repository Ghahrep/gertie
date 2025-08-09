# in db/crud.py
from sqlalchemy.orm import Session
from db import models
from api import schemas
from core.security import get_password_hash
from typing import List

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